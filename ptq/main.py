#!/usr/bin/env python3
"""
PTQ for Spike-Driven Transformer
所有参数从 config.yml 读取, 输出时直接复制 config.yml 到日志目录
"""
import os
import sys
import shutil
import yaml
import random
import numpy as np
import torch
import logging
import argparse
from datetime import datetime

torch.serialization.add_safe_globals([argparse.Namespace])

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from timm.models import create_model, load_checkpoint
from timm.data import create_dataset, create_loader
from timm.utils import accuracy, AverageMeter
from spikingjelly.clock_driven import functional
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from torch.utils.data import DataLoader
import model  # noqa: F401 - 注册 sdt
import dvs_utils

from ptq.quantize import (
    quantize_model, PTQConfig, get_model_size,
    calculate_quantized_model_size, print_quantized_model,
)


# ============================================================
# 工具函数
# ============================================================

def set_seed(seed):
    """设置随机数种子，保证实验可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置 cuDNN 确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(output_dir):
    """设置日志 (控制台 + 文件)"""
    logger = logging.getLogger('PTQ')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter('%(message)s')

    fh = logging.FileHandler(os.path.join(output_dir, 'ptq.log'), encoding='utf-8')
    import sys, io
    sh = logging.StreamHandler(
        stream=io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    )
    for handler in [fh, sh]:
        handler.setLevel(logging.INFO)
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger


def validate(mdl, loader, device='cuda', logger=None, use_tet=False):
    """验证模型精度"""
    log = logger.info if logger else print
    mdl.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images = images.float().to(device)
            target = target.to(device)

            output = mdl(images)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if use_tet:
                output = output.mean(0)

            a1, a5 = accuracy(output, target, topk=(1, 5))
            top1.update(a1.item(), images.size(0))
            top5.update(a5.item(), images.size(0))
            functional.reset_net(mdl)

            if i % 50 == 0:
                log(f'  [{i}/{len(loader)}] Acc@1: {top1.avg:.2f}%  Acc@5: {top5.avg:.2f}%')

    return top1.avg, top5.avg


def abs_path(path):
    """相对路径 → 基于项目根目录的绝对路径"""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_DIR, path)


# ============================================================
# 数据集相关参数
# ============================================================

DATASET_STATS = {
    'torch/cifar10': {
        'input_size': (3, 32, 32),
        'mean': (0.4914, 0.4822, 0.4465),
        'std':  (0.247,  0.2435, 0.2616),
    },
    'torch/cifar100': {
        'input_size': (3, 32, 32),
        'mean': (0.5071, 0.4867, 0.4408),
        'std':  (0.2675, 0.2565, 0.2761),
    },
    'imagenet': {
        'input_size': (3, 224, 224),
        'mean': (0.485, 0.456, 0.406),
        'std':  (0.229, 0.224, 0.225),
    },
}


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config.yml')
    args = parser.parse_args()

    # ---------- 读配置 ----------
    config_path = os.path.abspath(args.config)
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 所有参数从 cfg 取, 给默认值
    checkpoint    = abs_path(cfg['checkpoint'])
    dataset_name  = cfg.get('dataset', 'torch/cifar10')
    data_dir      = cfg.get('data_dir', '/mnt/disk2/xh2/data')
    batch_size    = cfg.get('batch_size', 64)
    crop_pct      = cfg.get('crop_pct', 1.0)
    weight_bit    = cfg.get('weight_bit', 4)
    first_layer   = cfg.get('first_layer_bit', 16)
    last_layer    = cfg.get('last_layer_bit', 8)
    mem_bit       = cfg.get('mem_bit', 32)
    first_mem_bit = cfg.get('first_mem_bit', 16)
    fold_bn       = cfg.get('fold_bn', True)
    seed          = cfg.get('seed', 42)  # 随机数种子
    gpu           = cfg.get('gpu', 0)
    output_base   = abs_path(cfg.get('output_dir', 'output/ptq'))

    # 量化方法参数
    observer_name   = cfg.get('observer', 'minmax')
    fake_quant_name = cfg.get('fake_quant', 'adaround')
    adaround_iters  = cfg.get('adaround_iters', 500)
    adaround_lr     = cfg.get('adaround_lr', 3e-3)
    recon_num_batches = cfg.get('recon_num_batches', 8)
    recon_mem_lam = cfg.get('recon_mem_lam', 0.25)
    recon_reg_lam_scale = cfg.get('recon_reg_lam_scale', 1e-4)
    recon_max_code_shift = cfg.get('recon_max_code_shift', 2.0)
    recon_min_iters = cfg.get('recon_min_iters', 50)
    recon_early_stop_patience = cfg.get('recon_early_stop_patience', 0)
    recon_improve_eps = cfg.get('recon_improve_eps', 1e-6)
    recon_grad_clip = cfg.get('recon_grad_clip', 1.0)
    recon_log_interval = cfg.get('recon_log_interval', 100)
    # Scale bridging & 膜电位 observer 参数
    scale_bridge       = cfg.get('scale_bridge', 'weight')
    mem_observer       = cfg.get('mem_observer', '')

    # 模型结构参数
    T             = cfg.get('time_steps', 4)
    num_classes   = cfg.get('num_classes', 10)
    num_heads     = cfg.get('num_heads', 8)
    img_size      = cfg.get('img_size', 32)
    in_channels   = cfg.get('in_channels', 3)
    dim           = cfg.get('dim', 256)
    mlp_ratio     = cfg.get('mlp_ratio', 4)
    depth         = cfg.get('layer', 2)
    pooling_stat  = cfg.get('pooling_stat', '1111')
    use_tet       = cfg.get('TET', False)
    dvs_mode      = dataset_name in ('cifar10-dvs', 'cifar10-dvs-tet', 'gesture')

    # ---------- 设置随机数种子 ----------
    set_seed(seed)

    # ---------- GPU ----------
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---------- 输出目录 ----------
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    ds_short = dataset_name.replace('torch/', '').replace('-', '')
    output_dir = os.path.join(output_base, f'{ts}-sdt-w{weight_bit}-{ds_short}')
    os.makedirs(output_dir, exist_ok=True)

    # 直接复制 config.yml 到输出目录
    shutil.copy2(config_path, os.path.join(output_dir, 'config.yml'))

    # ---------- 日志 ----------
    logger = setup_logging(output_dir)
    log = logger.info

    log('=' * 60)
    log('PTQ for Spike-Driven Transformer')
    log('=' * 60)
    log(f'Config:      {config_path}')
    log(f'Timestamp:   {ts}')
    log(f'Output:      {output_dir}')
    log(f'Checkpoint:  {checkpoint}')
    log(f'Dataset:     {dataset_name}')
    log(f'Weight bits: {weight_bit} (first: {first_layer}, last: {last_layer})')
    log(f'Mem bits:    {mem_bit}' + (' (disabled)' if mem_bit >= 32 else ''))
    if mem_bit < 32:
        log(
            f'  patch_embed.proj_lif mem: {first_mem_bit}-bit'
            + (' (FP)' if first_mem_bit >= 32 else '')
        )
    log(f'Fold BN:     {fold_bn}')
    log(f'FakeQuant:   {fake_quant_name}')
    if fake_quant_name == 'adaround':
        log(f'  AdaRound iters:    {adaround_iters}')
        log(f'  AdaRound lr:       {adaround_lr}')
        log(
            f'  Recon:             batches={recon_num_batches}, '
            f'mem_lam={recon_mem_lam}, reg_scale={recon_reg_lam_scale}, '
            f'max_shift={recon_max_code_shift}'
        )
        log(
            f'                     min_iters={recon_min_iters}, '
            f'patience={recon_early_stop_patience or "auto"}, '
            f'grad_clip={recon_grad_clip}, log_interval={recon_log_interval}'
        )
    if mem_bit < 32:
        log(f'Mem observer: {mem_observer or "(same as weight)"}')
        log(f'Scale bridge: {scale_bridge}')
    log(f'Seed:        {seed}')
    log(f'GPU: {gpu}  Device: {device}  DVS: {dvs_mode}  TET: {use_tet}')
    log('=' * 60)

    # ---------- [1/6] 加载模型 ----------
    log('\n[1/6] Loading model...')
    mdl = create_model(
        'sdt', T=T, num_classes=num_classes, num_heads=num_heads,
        img_size_h=img_size, img_size_w=img_size, patch_size=None,
        embed_dims=dim, mlp_ratios=mlp_ratio, in_channels=in_channels,
        depths=depth, spike_mode='lif', pooling_stat=pooling_stat,
        sr_ratios=1, dvs_mode=dvs_mode, TET=use_tet,
        drop_rate=0.0, drop_path_rate=0.0, qkv_bias=False,
    )
    load_checkpoint(mdl, checkpoint, strict=True)
    mdl = mdl.to(device).eval()
    orig_size = get_model_size(mdl)
    log(f'  Model loaded. Size: {orig_size:.2f} MB')

    # ---------- [2/6] 数据集 ----------
    log('\n[2/6] Creating dataset...')
    if dataset_name == 'cifar10-dvs':
        ds_full = CIFAR10DVS(data_dir, data_type='frame', frames_number=T,
                             split_by='number', transform=dvs_utils.Resize(img_size))
        _, ds_eval = dvs_utils.split_to_train_test_set(0.9, ds_full, 10)
        loader = DataLoader(ds_eval, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
        log(f'  CIFAR10-DVS: {len(ds_eval)} samples')
    else:
        ds_eval = create_dataset(dataset_name, root=data_dir,
                                 split='validation', is_training=False,
                                 batch_size=batch_size)
        stats = DATASET_STATS.get(dataset_name, DATASET_STATS['torch/cifar10'])
        loader = create_loader(ds_eval, input_size=stats['input_size'],
                               batch_size=batch_size, is_training=False,
                               use_prefetcher=False, mean=stats['mean'],
                               std=stats['std'], num_workers=4,
                               crop_pct=crop_pct, pin_memory=True)
        log(f'  {dataset_name}: {len(ds_eval)} samples')

    # ---------- [3/6] 原始精度 ----------
    log('\n[3/6] Testing original model...')
    orig_acc1, orig_acc5 = validate(mdl, loader, device, logger, use_tet)
    log(f'  Original  Acc@1: {orig_acc1:.2f}%  Acc@5: {orig_acc5:.2f}%')

    # ---------- [4/6] Fold BN 并测试 ----------
    folded_acc1, folded_acc5 = orig_acc1, orig_acc5  # 默认值
    if fold_bn:
        log('\n[4/6] Folding BatchNorm into Conv...')
        import copy
        from ptq.quantize import fold_bn as do_fold_bn

        mdl_folded = copy.deepcopy(mdl)
        mdl_folded = do_fold_bn(mdl_folded)
        mdl_folded = mdl_folded.to(device).eval()

        log('  Testing model after BN folding...')
        folded_acc1, folded_acc5 = validate(mdl_folded, loader, device, logger, use_tet)
        log(f'  After fold BN Acc@1: {folded_acc1:.2f}%  Acc@5: {folded_acc5:.2f}%')

        bn_drop = orig_acc1 - folded_acc1
        if abs(bn_drop) < 0.1:
            log(f'  BN folding OK! (diff: {bn_drop:.4f}%)')
        else:
            log(f'  WARNING: BN folding changed accuracy by {bn_drop:.2f}%!')

        del mdl_folded
        torch.cuda.empty_cache()
    else:
        log('\n[4/6] Skipping BN folding (fold_bn=False)')

    # ---------- [5/6] 量化 ----------
    log('\n[5/6] Quantizing...')
    ptq_cfg = PTQConfig(
        weight_bit=weight_bit, first_layer_bit=first_layer,
        last_layer_bit=last_layer, mem_bit=mem_bit,
        first_mem_bit=first_mem_bit, fold_bn=fold_bn,
        observer=observer_name,
        fake_quant=fake_quant_name,
        adaround_iters=adaround_iters, adaround_lr=adaround_lr,
        recon_num_batches=recon_num_batches,
        recon_mem_lam=recon_mem_lam,
        recon_reg_lam_scale=recon_reg_lam_scale,
        recon_max_code_shift=recon_max_code_shift,
        recon_min_iters=recon_min_iters,
        recon_early_stop_patience=recon_early_stop_patience,
        recon_improve_eps=recon_improve_eps,
        recon_grad_clip=recon_grad_clip,
        recon_log_interval=recon_log_interval,
        scale_bridge=scale_bridge,
        use_tet=use_tet,
        mem_observer=mem_observer,
    )

    log(f'  Using {fake_quant_name} PTQ')
    quant_mdl = quantize_model(
        mdl, ptq_cfg, cali_data=loader, device=device, logger=logger,
        output_dir=output_dir,
    )

    quant_size = calculate_quantized_model_size(mdl, ptq_cfg)
    log(f'  Quantized size (theoretical): {quant_size:.2f} MB')

    print_quantized_model(quant_mdl, logger)

    torch.cuda.empty_cache()

    # ---------- [6/6] 量化精度 ----------
    log('\n[6/6] Testing quantized model...')
    quant_acc1, quant_acc5 = validate(quant_mdl, loader, device, logger, use_tet)
    log(f'  Quantized Acc@1: {quant_acc1:.2f}%  Acc@5: {quant_acc5:.2f}%')

    # ---------- 结果 ----------
    drop = orig_acc1 - quant_acc1
    ratio = orig_size / quant_size

    log('\n' + '=' * 60)
    log('PTQ Results')
    log('=' * 60)
    log(f'  Weight bits:       {weight_bit} (first: {first_layer}, last: {last_layer})')
    if mem_bit < 32:
        log(
            f'  Membrane bits:     proj_lif {first_mem_bit}, others {mem_bit}'
        )
    log(f'  Original  Acc@1:   {orig_acc1:.2f}%')
    if fold_bn:
        log(f'  After BN  Acc@1:   {folded_acc1:.2f}%  (diff: {orig_acc1 - folded_acc1:.4f}%)')
    log(f'  Quantized Acc@1:   {quant_acc1:.2f}%')
    log(f'  Accuracy drop:     {drop:.2f}%')
    log(f'  Original  size:    {orig_size:.2f} MB')
    log(f'  Quantized size:    {quant_size:.2f} MB')
    log(f'  Compression:       {ratio:.2f}x')
    log('=' * 60)

    # ---------- 保存 ----------
    torch.save({
        'model': quant_mdl.state_dict(),
        'ptq_config': ptq_cfg.__dict__,
        'orig_acc1': orig_acc1, 'orig_acc5': orig_acc5,
        'folded_acc1': folded_acc1, 'folded_acc5': folded_acc5,
        'quant_acc1': quant_acc1, 'quant_acc5': quant_acc5,
        'acc_drop': drop, 'compression': ratio,
    }, os.path.join(output_dir, 'model_quant.pth'))

    log(f'\nOutput: {output_dir}')
    log(f'  config.yml                    (原始配置副本)')
    log(f'  ptq.log                       (完整日志)')
    log(f'  membrane_scale_details.json   (膜电位 scale 逐通道详情)')
    log(f'  wm_*.png                      (权重 vs 膜电位/2^k 分布图，量化线与 s_w)')
    log(f'  model_quant.pth               (量化模型)')


if __name__ == '__main__':
    main()
