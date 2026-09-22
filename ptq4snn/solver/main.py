import numpy as np  # noqa: F401
import copy
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import argparse
import json
import ptq4snn.utils.utils as utils
from ptq4snn.utils.datasets import get_datasets
from .fold_bn import search_fold_and_remove_bn
from ptq4snn.model.common import LIFNeuron
from ptq4snn.quantization.state import enable_calibration_woquantization, enable_quantization, disable_all
from ptq4snn.quantization.quantized_module import QNeuron, QuantizedModule, QuantizedLayer, QuantizedBlock, Quantizer, specials
from ptq4snn.quantization.fake_quant import QuantizeBase
from ptq4snn.quantization.observer import ObserverBase
from .recon import reconstruction
from .bit_allocation import compute_activity_stats, allocate_bits_per_channel, calculate_weighted_avg_bits, compute_loss_salience
from torch.utils.data import DataLoader


def _is_full_precision_mem_config(m_qconfig):
    try:
        return int(getattr(m_qconfig, "bit", 0)) >= 32
    except Exception:
        return False


def _flowq_scale(x, bit, eps=1e-7, max_iters=1000):
    """PPT FlowTune independent scale fixed-point iteration."""
    x = x.detach().float().reshape(-1)
    if x.numel() == 0:
        return torch.tensor(1.0, device=x.device)
    qmin, qmax = -2 ** (bit - 1), 2 ** (bit - 1) - 1
    scale = x.abs().max().clamp_min(1e-8) / float(2 ** (bit - 1))
    for _ in range(max_iters):
        q = torch.clamp(torch.round(x / scale), qmin, qmax)
        denom = q.square().sum()
        if denom <= 0:
            break
        new_scale = ((x * q).sum() / denom).abs().clamp_min(1e-8)
        if torch.abs(new_scale - scale) <= eps:
            scale = new_scale
            break
        scale = new_scale
    return scale


@torch.no_grad()
def _flowq_calibrate(model, cali_data, logger):
    """Layer-wise, reconstruction-free FlowQ calibration for conventional SNNs."""
    parents = dict(model.named_modules())
    neurons = {}

    for name, module in model.named_modules():
        if not isinstance(module, QNeuron):
            continue
        neurons[name] = module
        module._flowq_records = {}
        module._flowq_record_enabled = True

    disable_all(model)
    model.eval()
    batch_size = min(32, cali_data.size(0))
    try:
        for start in range(0, cali_data.size(0), batch_size):
            model(cali_data[start:start + batch_size].cuda())
    finally:
        for module in neurons.values():
            module._flowq_record_enabled = False

    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase) and "weight_fake_quant" in name:
            scale = _flowq_scale(module.observer.min_val.new_tensor(
                [0.0]), module.bit)
            # Reconstruct the associated weight tensor from its parent QConv.
            parent_name = name.rsplit(".", 1)[0]
            parent = parents.get(parent_name)
            if parent is not None and hasattr(parent, "weight"):
                scale = _flowq_scale(parent.weight, module.bit)
            module.scale.resize_(1)
            module.scale.fill_(float(scale))
            module.zero_point.resize_(1)
            module.zero_point.zero_()

    for name, neuron in neurons.items():
        records_by_timestep = neuron._flowq_records
        if not records_by_timestep:
            continue
        parent_name = name.rsplit(".", 1)[0]
        parent = parents.get(parent_name)
        if parent is None or not hasattr(parent, "module"):
            continue
        weight_fq = getattr(parent.module, "weight_fake_quant", None)
        if weight_fq is None:
            continue
        s_w = weight_fq.scale.detach().reshape(-1)[0]
        bits_value = int(torch.as_tensor(neuron.n_bits).reshape(-1)[0].item())
        # FlowTune independently optimizes U[t] over all calibration samples,
        # then selects the maximum scale across timesteps.
        timestep_scales = [
            _flowq_scale(torch.cat(records), bits_value)
            for _, records in sorted(records_by_timestep.items())
            if records
        ]
        if not timestep_scales:
            continue
        s_u = torch.stack(timestep_scales).max()
        k = int(torch.round(torch.log2(s_u / s_w)).item())
        # QNeuron applies x * factor / scale before rounding. The inverse
        # exponent realizes the PPT fractional precision shift.
        neuron.factor_exp.data.resize_(1)
        neuron.factor_exp.data.fill_(-float(k))
        neuron.factor = 2 ** neuron.factor_exp
        neuron.factor_should_init = False
        logger.info("[FlowQ] %s: layerwise s_w=%.6g s_u=%.6g k=%d",
                    name, float(s_w), float(s_w * (2.0 ** k)), k)
        neuron._flowq_records = {}




def quantize_model(model, config_quant, logger, cali_data, save_dir=None):

    def replace_module(module, w_qconfig, m_qconfig):
        childs = list(iter(module.named_children()))
        st, ed = 0, len(childs)
        prev_quantmodule = None
        while(st < ed):
            name, child_module = childs[st][0], childs[st][1]
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, w_qconfig, m_qconfig))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantizedLayer(child_module, None, w_qconfig, m_qconfig))
                prev_quantmodule = getattr(module, name)
            elif isinstance(child_module, LIFNeuron):
                if prev_quantmodule is not None:
                    prev_quantmodule.neuron = Quantizer(child_module, m_qconfig)
                else:
                    pass
            elif isinstance(child_module, nn.Identity):
                pass
            else:
                replace_module(child_module, w_qconfig, m_qconfig)
            st += 1

    replace_module(model, config_quant.w_qconfig, config_quant.m_qconfig)

    module_to_del_dict = {

    }
    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            childrens = list(iter(module.children()))
            to_del_idx_list = []
            for i, child_module in enumerate(childrens):
                if isinstance(child_module, LIFNeuron) and not isinstance(child_module, QNeuron):
                    to_del_idx_list.append(i)
                if isinstance(child_module, nn.Identity):
                    to_del_idx_list.append(i)
            module_to_del_dict[name] = module, to_del_idx_list

    # del redundancy LIFNeuron
    for name, (module, to_del_idx_list) in module_to_del_dict.items():
        childrens = list(iter(module.children()))
        for idx in sorted(to_del_idx_list, reverse=True):
            del childrens[idx]
        setattr(model, name, nn.Sequential(*childrens))

    if _is_full_precision_mem_config(config_quant.m_qconfig):
        w_list, n_list = [], []
        for name, module in model.named_modules():
            if isinstance(module, QuantizeBase) and 'weight' in name:
                w_list.append(module)
            if isinstance(module, LIFNeuron):
                n_list.append(module)

        if len(w_list) > 0:
            w_list[0].set_bit(8)

        for neuron in n_list:
            neuron.set_bit(32)
            if hasattr(neuron, "set_factor_init_mask"):
                neuron.set_factor_init_mask(False)
            else:
                neuron.factor_init_by_observe = False

        logger.info("[MemQuant] m_qconfig.bit >= 32; membrane quantization disabled for W4/M32 baseline.")
        logger.info('finish quantize model:\n{}'.format(str(model)))
        return model



    # 只保留基于膜电位 & 脉冲发放率的按通道分配策略
    activity_map = compute_activity_stats(model, cali_data, logger, config_quant)
    logger.info("[BitAlloc] Using CHANNEL-level activity-based bit allocation.")

    # 检查是否需要计算 loss salience（MixLLM 方法）
    salience_map = None
    loss_salience_weight = float(getattr(config_quant, "loss_salience_weight", 0))
    if loss_salience_weight > 0:
        logger.info("[BitAlloc] Computing loss salience (MixLLM-inspired method)...")
        salience_map = compute_loss_salience(model, cali_data, logger, config_quant)
        if len(salience_map) == 0:
            logger.warning("[BitAlloc] Loss salience computation returned empty map, falling back to activity-based method.")
            salience_map = None


    # 可视化activity统计
    if save_dir is None:
        # 如果没有指定保存目录，尝试从logger获取
        log_file = getattr(logger, 'handlers', [None])[0]
        if log_file and hasattr(log_file, 'baseFilename'):
            save_dir = os.path.dirname(log_file.baseFilename)
        else:
            save_dir = './'
    # visualize_activity_stats(activity_map, save_dir, logger)

    bit_allocation = allocate_bits_per_channel(activity_map, logger, config_quant, model=model, cali_data=cali_data, salience_map=salience_map)

    allocation_output_path = getattr(config_quant, 'allocation_output_path', None)
    if allocation_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(allocation_output_path)), exist_ok=True)
        with open(allocation_output_path, 'w') as f:
            json.dump({name: np.asarray(bits).astype(int).tolist() for name, bits in bit_allocation.items()}, f, indent=2)
        logger.info(f"[BitAlloc] Saved allocation map to {allocation_output_path}")

    # Uniform precision uses the configured membrane bit for every non-stem channel.
    if hasattr(config_quant, 'mix_precise') and not config_quant.mix_precise:
        uniform_bit = int(config_quant.m_qconfig.bit)
        logger.info(f"[BitAlloc] mix_precise is False, setting all channels to {uniform_bit} bits")
        for neuron_name in bit_allocation:
            bits_arr = bit_allocation[neuron_name]
            if isinstance(bits_arr, np.ndarray):
                bit_allocation[neuron_name] = np.full_like(bits_arr, uniform_bit, dtype=bits_arr.dtype)
            elif isinstance(bits_arr, torch.Tensor):
                bit_allocation[neuron_name] = torch.full_like(bits_arr, uniform_bit, dtype=bits_arr.dtype)
            else:
                bit_allocation[neuron_name] = np.full(len(bits_arr), uniform_bit, dtype=np.int32)

    # 计算加权平均量化比特数
    weighted_avg_bits = calculate_weighted_avg_bits(bit_allocation, activity_map, model, cali_data, logger, config_quant)

    w_list, n_list = [], []
    #收集名称
    neuron_names = []
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase) and 'weight' in name:
            w_list.append(module)
        if isinstance(module, LIFNeuron):
            n_list.append(module)
            neuron_names.append(name) # 同时收集名称

    # 设置第一层权重量化
    if len(w_list) > 0:
        w_list[0].set_bit(8)



    # 打印一下膜电位量化位宽分配结果
    logger.info("Neuron bit allocation summary (per-channel):")
    for neuron_name, bits in bit_allocation.items():
        unique, counts = np.unique(np.asarray(bits), return_counts=True)
        logger.info(f"{neuron_name}: {dict(zip(unique.astype(int).tolist(), counts.astype(int).tolist()))}")


    for i, (neuron, neuron_name) in enumerate(zip(n_list, neuron_names)):
        if neuron_name in bit_allocation:
            allocated_bits = bit_allocation[neuron_name]
            neuron.set_bit(allocated_bits)

            bits_arr = np.asarray(allocated_bits)
            mask = np.ones_like(bits_arr, dtype=bool)
            mask[bits_arr == 8] = False  # 8bit 通道关闭 factor_init_by_observe，其余打开
            if hasattr(neuron, "set_factor_init_mask"):
                neuron.set_factor_init_mask(mask)
            else:
                neuron.factor_init_by_observe = bool(np.all(mask))

            off_cnt = int((~mask).sum()) if mask.dtype == bool else 0
            logger.info(
                f"Set {neuron_name} to per-channel bits; 8bit_channels={off_cnt}, "
                f"factor_init_by_observe={'per-channel' if hasattr(neuron, 'set_factor_init_mask') else neuron.factor_init_by_observe}"
            )
        else:
            default_bit = int(config_quant.m_qconfig.bit)
            neuron.set_bit(default_bit)
            if hasattr(neuron, "set_factor_init_mask"):
                neuron.set_factor_init_mask(True)
            else:
                neuron.factor_init_by_observe = True
            logger.info(f"Set {neuron_name} to default {default_bit} bits (factor_init_by_observe=on)")


    # 特殊处理第一层：高精度 & 关闭 factor_init_by_observe
    if len(n_list) > 0:
        n_list[0].set_bit(16)
        if hasattr(n_list[0], "set_factor_init_mask"):
            n_list[0].set_factor_init_mask(False)
        else:
            n_list[0].factor_init_by_observe = False

    logger.info('finish quantize model:\n{}'.format(str(model)))

    return model


def get_cali_data(train_loader, num_samples):
    cali_data = []
    for batch in train_loader:
        cali_data.append(batch[0])
        if len(cali_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(cali_data, dim=0)[:num_samples]


def run_observer_calibration(model, cali_data, batch_size=None):
    if batch_size is None or batch_size <= 0:
        batch_size = cali_data.size(0)

    for start in range(0, cali_data.size(0), batch_size):
        end = min(start + batch_size, cali_data.size(0))
        model(cali_data[start:end].cuda())


def get_observer_calibration_batch_size(config):
    if hasattr(config.quant, 'calibration_batch_size'):
        return int(config.quant.calibration_batch_size)
    if hasattr(config.quant, 'recon') and hasattr(config.quant.recon, 'batch_size'):
        return int(config.quant.recon.batch_size)
    return int(config.batch_size)


def collect_membrane_metrics(model, fp_model, data_loader, max_samples=256):
    q_neurons = {name: module for name, module in model.named_modules() if isinstance(module, QNeuron)}
    fp_neurons = {name: module for name, module in fp_model.named_modules() if isinstance(module, QNeuron)}
    common = sorted(set(q_neurons) & set(fp_neurons))
    sq_error = np.zeros(4, dtype=np.float64)
    sq_target = np.zeros(4, dtype=np.float64)
    mismatches = np.zeros(4, dtype=np.int64)
    spike_values = np.zeros(4, dtype=np.int64)
    samples = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for images, _ in data_loader:
            if samples >= max_samples:
                break
            images = images[: max_samples - samples].to(device)
            model(images)
            q_vpre = {name: q_neurons[name].v_pre_seq.detach().float() for name in common}
            q_spikes = {
                name: (
                    q_vpre[name]
                    >= q_neurons[name].thresh
                    * q_neurons[name].factor.to(device).view(1, 1, -1, 1, 1)
                ).detach()
                for name in common
            }
            fp_model(images)
            for name in common:
                target = fp_neurons[name].v_pre_seq.detach().float()
                pred = q_vpre[name]
                teacher_spikes = target >= fp_neurons[name].thresh
                time_steps = min(4, pred.shape[0], target.shape[0])
                for t in range(time_steps):
                    delta = pred[t] - target[t]
                    sq_error[t] += float(delta.pow(2).sum().item())
                    sq_target[t] += float(target[t].pow(2).sum().item())
                    mismatches[t] += int((q_spikes[name][t] != teacher_spikes[t]).sum().item())
                    spike_values[t] += int(teacher_spikes[t].numel())
            samples += images.size(0)
    nrmse = np.sqrt(sq_error / np.maximum(sq_target, 1e-12))
    mismatch = mismatches / np.maximum(spike_values, 1)
    return {
        'vpre_nrmse_by_t': nrmse.tolist(),
        'spike_mismatch_by_t': mismatch.tolist(),
        'vpre_nrmse': float(np.sqrt(sq_error.sum() / max(sq_target.sum(), 1e-12))),
        'spike_mismatch': float(mismatches.sum() / max(spike_values.sum(), 1)),
        'metric_samples': samples,
    }


def collect_state_storage_metrics(model, sample):
    q_neurons = [(name, module) for name, module in model.named_modules() if isinstance(module, QNeuron)]
    with torch.no_grad():
        model(sample[:1].to(next(model.parameters()).device))
    total_bits = 0.0
    total_values = 0
    budget_bits = 0.0
    budget_values = 0
    for name, neuron in q_neurons:
        if not isinstance(neuron.mem, torch.Tensor):
            continue
        values_per_channel = neuron.mem[0, 0].numel() if neuron.mem.dim() >= 2 else 1
        bits = np.asarray(neuron.n_bits).reshape(-1)
        if bits.size == 1:
            layer_bits = float(bits[0]) * neuron.mem.numel()
            layer_values = neuron.mem.numel()
        else:
            layer_bits = float(bits.sum()) * values_per_channel
            layer_values = int(bits.size * values_per_channel)
        total_bits += layer_bits
        total_values += layer_values
        if not name.startswith('stem.0.'):
            budget_bits += layer_bits
            budget_values += layer_values
    return {
        'actual_avg_bits': float(budget_bits / budget_values) if budget_values else 0.0,
        'all_state_avg_bits': float(total_bits / total_values) if total_values else 0.0,
        'peak_state_mb': float(total_bits / 8 / (1024 ** 2)),
        'state_values_batch1': total_values,
    }


def main(config_path, log_save_dir=None):
    config = utils.parse_config(config_path)

    exp_dir = os.path.dirname(os.path.abspath(config_path))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    # 如果用户指定了log_save_dir，则使用用户指定的目录，否则使用默认目录
    if log_save_dir is None:
        log_save_dir = os.path.join(exp_dir, 'logs', timestamp)

    log_file_path = os.path.join(log_save_dir, 'log.txt')
    os.makedirs(log_save_dir, exist_ok=True)
    logger = utils.init_logger(log_file_path, 'ptq4snn')

    copied_config_path = os.path.join(log_save_dir, os.path.basename(config_path))
    if os.path.abspath(config_path) != os.path.abspath(copied_config_path):
        shutil.copy2(config_path, copied_config_path)

    # set device (already set in __main__, but keep for compatibility)
    if 'CUDA_VISIBLE_DEVICES' not in os.environ or os.environ.get('CUDA_VISIBLE_DEVICES') != str(config.gpu):
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{config.gpu}'
        logger.info(f"Set CUDA_VISIBLE_DEVICES={config.gpu}")
    utils.set_seed(config.process.seed)
    # cali data
    train_dataset, train4test_dataset, test_dataset, train_sampler, train4test_sampler, test_sampler, nclass, in_channels, collate_fn = get_datasets(**config.data, distributed=False)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, num_workers=config.num_workers, collate_fn=collate_fn, drop_last=True, pin_memory=True)
    train4test_dataloader = DataLoader(train4test_dataset, batch_size=config.batch_size, sampler=train4test_sampler, num_workers=config.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, sampler=test_sampler, num_workers=config.num_workers, pin_memory=True)
    cali_data = get_cali_data(train_dataloader, config.quant.calibrate)
    # 'model'
    model = torch.load(config.model.path, map_location='cpu', weights_only=False)

    search_fold_and_remove_bn(model)



    if hasattr(config, 'quant'):
        model = quantize_model(model, config.quant, logger, cali_data, save_dir=log_save_dir)

    # initial test在量化前，做一次测试
    model.cuda()
    model.eval()
    utils.val(test_dataloader, model, criterion=utils.CriterionWrapper(is_tet=False), device=torch.device('cuda'), logger=logger, log_interval=100, header='Test:')


    # 生成fp副本，并且禁止其量化
    fp_model = copy.deepcopy(model)
    disable_all(fp_model)
    for name, module in model.named_modules():
        if isinstance(module, ObserverBase):
            module.set_name(name)

    # calibrate first
    # 启动observer的校准模式，但不进行量化，只是收集统计信息
    with torch.no_grad():
        st = time.time()
        enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
        # 使用全部校准样本进行observer统计，保证逐通道统计充分
        calibration_batch_size = get_observer_calibration_batch_size(config)
        logger.info(f'run observer calibration with batch_size={calibration_batch_size}')
        run_observer_calibration(model, cali_data, calibration_batch_size)
        ed = time.time()
        logger.info('the calibration time is {}'.format(ed - st))

    if bool(getattr(config.quant, 'flowq', False)):
        logger.info('[FlowQ] reconstruction disabled; running layer-wise FlowTune calibration')
        _flowq_calibrate(model, cali_data, logger)
        enable_quantization(model)



    for name, module in model.named_modules():
        if isinstance(module, QNeuron):
            logger.info(f"factor_exp: {module.factor_exp.data}")



    if hasattr(config.quant, 'recon') and not bool(getattr(config.quant, 'flowq', False)):
        enable_quantization(model)
        recon_scope = getattr(config.quant.recon, 'scope', 'block')
        logger.info(f"Reconstruction scope: {recon_scope}")

        def recon_model(module: nn.Module, fp_module: nn.Module):
            """
            Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
            """
            for name, child_module in module.named_children():
                fp_child_module = getattr(fp_module, name)
                if recon_scope == 'layer' and isinstance(child_module, QuantizedBlock):
                    recon_model(child_module, fp_child_module)
                elif isinstance(child_module, QuantizedLayer) or (
                    recon_scope != 'layer' and isinstance(child_module, QuantizedBlock)
                ):
                    logger.info('begin reconstruction for module:\n{}'.format(str(child_module)))
                    reconstruction(model, fp_model, child_module, fp_child_module, cali_data, config.quant.recon, logger)
                else:
                    recon_model(child_module, fp_child_module)
        # Start reconstruction
        recon_model(model, fp_model)
    enable_quantization(model)
    for module in model.modules():
        if isinstance(module, QNeuron):
            module.saturation_count = 0
            module.quantized_value_count = 0

    loss, acc1, acc5 = utils.val(test_dataloader, model, criterion=utils.CriterionWrapper(is_tet=False), device=torch.device('cuda'), logger=logger, log_interval=100, header='Test:')

    q_neurons = [m for m in model.modules() if isinstance(m, QNeuron)]
    saturated = sum(int(getattr(m, 'saturation_count', 0)) for m in q_neurons)
    quantized_values = sum(int(getattr(m, 'quantized_value_count', 0)) for m in q_neurons)
    bit_counts = {2: 0, 4: 0, 8: 0, 16: 0, 32: 0}
    for neuron in q_neurons:
        bits = np.asarray(neuron.n_bits).reshape(-1)
        for bit in bit_counts:
            bit_counts[bit] += int((bits == bit).sum())
    metric_samples = int(getattr(config.process, 'metric_samples', 256))
    summary = {
        'acc1': float(acc1),
        'acc5': float(acc5),
        'seed': int(config.process.seed),
        'checkpoint': str(config.model.path),
        'saturation_rate': float(saturated / quantized_values) if quantized_values else 0.0,
        'bit_channel_counts': {str(k): v for k, v in bit_counts.items()},
    }
    summary.update(collect_membrane_metrics(model, fp_model, test_dataloader, max_samples=metric_samples))
    summary.update(collect_state_storage_metrics(model, cali_data))
    with open(os.path.join(log_save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    if bool(getattr(config.process, 'save_model', False)):
        logger.info(f'Save quantized model to {log_save_dir}')
        torch.save(model, os.path.join(log_save_dir, f'quantized_model_{acc1:.4f}.pth'))




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='configuration',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--log_save_dir', default=None, type=str, help='Directory to save logs and model. If not specified, will use a timestamped directory under logs/ in the experiment directory.')
    args = parser.parse_args()

    # 在第一次使用cuda之前设置CUDA_VISIBLE_DEVICES
    # 先解析config获取gpu设置（使用utils.parse_config避免重复导入）
    import yaml
    from easydict import EasyDict
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    if 'gpu' in config_dict:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config_dict['gpu'])
        print(f"Set CUDA_VISIBLE_DEVICES={config_dict['gpu']}")

    main(args.config, args.log_save_dir)
