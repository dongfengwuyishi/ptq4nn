"""
模型量化主逻辑

提供:
- PTQConfig: 量化配置
- quantize_model: 量化模型主函数
- fold_bn: BatchNorm 融合
- 模型大小计算等工具
"""
import os
import copy
import json
import random
import time
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from .quantized_module import (
    QuantConv2d, QuantLinear, QuantMultiStepLIFNode,
    fold_bn_into_conv,
)
from .observer import (
    MinMaxObserver, AvgMinMaxObserver, MSEObserver, PercentileObserver,
    OBSERVER_MAP, lsq_unified_scale_optim,
)
from .fake_quant import (
    FixedFakeQuantize, AdaRoundFakeQuantize, FAKEQUANT_MAP,
)
from .state import enable_quantization


LIF_TO_CONV_LEAF = {
    'proj_lif': 'proj_conv', 'proj_lif1': 'proj_conv1',
    'proj_lif2': 'proj_conv2', 'proj_lif3': 'proj_conv3',
    'q_lif': 'q_conv', 'k_lif': 'k_conv', 'v_lif': 'v_conv',
    'fc2_lif': 'fc1_conv',
}


@dataclass
class PTQConfig:
    """
    PTQ 配置

    Args:
        weight_bit: 权重量化位宽 (默认 4)
        first_layer_bit: 第一层位宽 (默认 16，保持高精度)
        last_layer_bit: 最后一层位宽 (默认 8)
        mem_bit: 膜电位量化位宽 (默认 32，即不量化; 设为 4/8 等开启)
        first_mem_bit: patch_embed 第一段 proj_lif 的膜电位位宽 (默认 16)，与 proj_conv 共用
            scale；>=32 表示该层不做膜电位量化。仅当 mem_bit < 32 时生效。
        fold_bn: 是否融合 BatchNorm (默认 True)
        observer: Observer 类型 ('minmax', 'avgminmax', 'mse', 'percentile')
        fake_quant: FakeQuantize 类型 ('fixed', 'adaround')
        adaround_iters: AdaRound 逐层优化迭代数
        adaround_lr: AdaRound 学习率
        scale_bridge: 膜电位 scale 策略
            'weight'  : 直接使用权重 scale (PTQ4SNN style)
            'observer': 使用 observer 独立校准的 scale
            'pot'     : Power-of-2 bridging, s_mem = s_w * 2^k
            'unify'   : 联合 MSE 搜索, 对应通道权重+膜电位放在一起找最优共享 scale
        mem_observer: 膜电位 Observer 类型 (默认与 observer 相同)
    """
    weight_bit: int = 4
    first_layer_bit: int = 16
    last_layer_bit: int = 8
    mem_bit: int = 32
    first_mem_bit: int = 16
    fold_bn: bool = True
    observer: str = 'minmax'
    mem_observer: str = ''
    fake_quant: str = 'adaround'
    adaround_iters: int = 500
    adaround_lr: float = 3e-3
    recon_num_batches: int = 8       # layer-wise reconstruction calibration batches
    recon_mem_lam: float = 0.25      # pre-fire membrane reconstruction weight
    recon_reg_lam_scale: float = 1e-4  # reg_lam = recon_reg_lam_scale * layer ref_max
    recon_max_code_shift: float = 2.0
    recon_min_iters: int = 50
    recon_early_stop_patience: int = 0  # <=0 means auto: min(80, max(20, iters // 10))
    recon_improve_eps: float = 1e-6
    recon_grad_clip: float = 1.0
    recon_log_interval: int = 100
    scale_bridge: str = 'weight'
    use_tet: bool = False


def get_observer_class(name):
    """根据名称获取 Observer 类"""
    if name not in OBSERVER_MAP:
        raise ValueError(f"Unknown observer: {name}. Available: {list(OBSERVER_MAP.keys())}")
    return OBSERVER_MAP[name]


def get_fake_quant_class(name):
    """根据名称获取 FakeQuantize 类"""
    if name not in FAKEQUANT_MAP:
        raise ValueError(f"Unknown fake_quant: {name}. Available: {list(FAKEQUANT_MAP.keys())}")
    return FAKEQUANT_MAP[name]


def find_layers_to_quantize(model, layer_types=(nn.Conv2d, nn.Linear)):
    """
    找到需要量化的层

    Args:
        model: 模型
        layer_types: 需要量化的层类型

    Returns:
        layers: [(name, module, parent, attr_name), ...]
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, layer_types):
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
                attr_name = name
            layers.append((name, module, parent, attr_name))
    return layers


def find_lif_nodes(model):
    """
    找到所有 LIF 神经元

    Returns:
        lif_nodes: [(name, module, parent, attr_name), ...]
    """
    lif_nodes = []
    lif_types = (MultiStepLIFNode, MultiStepParametricLIFNode)
    for name, module in model.named_modules():
        if isinstance(module, lif_types):
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
                attr_name = name
            lif_nodes.append((name, module, parent, attr_name))
    return lif_nodes


def find_bn_after_conv(model):
    """
    找到 Conv + BN 的组合

    Returns:
        pairs: [(conv_name, conv, bn_name, bn, parent, conv_attr), ...]
    """
    pairs = []
    prev_conv = None
    prev_conv_name = None
    prev_parent = None
    prev_attr = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prev_conv = module
            prev_conv_name = name
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                prev_parent = dict(model.named_modules())[parts[0]]
                prev_attr = parts[1]
            else:
                prev_parent = model
                prev_attr = name

        elif isinstance(module, nn.BatchNorm2d) and prev_conv is not None:
            if module.num_features == prev_conv.out_channels:
                pairs.append((prev_conv_name, prev_conv, name, module, prev_parent, prev_attr))
            prev_conv = None
            prev_conv_name = None

    return pairs


def fold_bn(model):
    """
    融合 BatchNorm 到 Conv

    Args:
        model: 原始模型

    Returns:
        model: 融合后的模型
    """
    pairs = find_bn_after_conv(model)

    for conv_name, conv, bn_name, bn, parent, conv_attr in pairs:
        # 融合权重
        w_new, b_new = fold_bn_into_conv(conv, bn)

        # 创建新的 Conv（带 bias）
        new_conv = nn.Conv2d(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            conv.stride, conv.padding, conv.dilation, conv.groups,
            bias=True, padding_mode=conv.padding_mode
        )
        new_conv.weight.data = w_new
        new_conv.bias.data = b_new

        # 替换原来的 Conv
        setattr(parent, conv_attr, new_conv)

        # 将 BN 替换为 Identity
        bn_parts = bn_name.rsplit('.', 1)
        if len(bn_parts) == 2:
            bn_parent = dict(model.named_modules())[bn_parts[0]]
            bn_attr = bn_parts[1]
        else:
            bn_parent = model
            bn_attr = bn_name
        setattr(bn_parent, bn_attr, nn.Identity())

    print(f"Folded {len(pairs)} BN layers into Conv layers")
    return model


def replace_with_quantized(model, config: PTQConfig):
    """
    将模型中的层替换为量化版本

    Args:
        model: 原始模型
        config: PTQ 配置

    Returns:
        model: 量化后的模型
    """
    # 1. 先融合 BN
    if config.fold_bn:
        model = fold_bn(model)

    # 2. 获取 Observer 和 FakeQuantize 类
    observer_cls = get_observer_class(config.observer)
    fake_quant_cls = get_fake_quant_class(config.fake_quant)

    print(f"  Observer:      {observer_cls.__name__}")
    print(f"  FakeQuantize:  {fake_quant_cls.__name__}")

    # 3. 找到所有需要量化的层 (Conv + Linear 统一排序)
    conv_layers = find_layers_to_quantize(model, (nn.Conv2d,))
    linear_layers = find_layers_to_quantize(model, (nn.Linear,))
    all_layers = conv_layers + linear_layers

    total = len(all_layers)

    # 4. 统一替换，只有全局第一层和全局最后一层使用特殊位宽
    for i, (name, module, parent, attr_name) in enumerate(all_layers):
        if i == 0:
            bit = config.first_layer_bit
        elif i == total - 1:
            bit = config.last_layer_bit
        else:
            bit = config.weight_bit

        if isinstance(module, nn.Conv2d):
            q_module = QuantConv2d(
                module, bit=bit,
                observer=observer_cls,
                fake_quant=fake_quant_cls
            )
            setattr(parent, attr_name, q_module)
            print(f"Replaced {name} -> QuantConv2d {bit}-bit ({observer_cls.__name__}, {module.out_channels} scales)")
        else:
            q_module = QuantLinear(
                module, bit=bit,
                observer=observer_cls,
                fake_quant=fake_quant_cls
            )
            setattr(parent, attr_name, q_module)
            print(f"Replaced {name} -> QuantLinear {bit}-bit ({observer_cls.__name__}, {module.out_features} scales)")

    # 5. 膜电位量化：替换 LIF 神经元
    # patch_embed.proj_lif：与首层 proj_conv 共用 scale，位宽单独可配 (first_mem_bit)
    _PATCH_EMBED_FIRST_MEM_LIF = "patch_embed.proj_lif"
    if config.mem_bit < 32:
        mem_obs_name = config.mem_observer if config.mem_observer else config.observer
        mem_observer_cls = get_observer_class(mem_obs_name)
        lif_nodes = find_lif_nodes(model)
        n_replaced = 0
        for name, module, parent, attr_name in lif_nodes:
            if name == _PATCH_EMBED_FIRST_MEM_LIF and config.first_mem_bit >= 32:
                print(
                    f"Skipped {name} -> keep FP membrane "
                    f"(first_mem_bit={config.first_mem_bit})"
                )
                continue
            mem_b = (
                config.first_mem_bit
                if name == _PATCH_EMBED_FIRST_MEM_LIF
                else config.mem_bit
            )
            q_lif = QuantMultiStepLIFNode(
                module, mem_bit=mem_b, observer=mem_observer_cls,
            )
            setattr(parent, attr_name, q_lif)
            print(
                f"Replaced {name} -> QuantMultiStepLIFNode {mem_b}-bit mem "
                f"({mem_observer_cls.__name__})"
            )
            n_replaced += 1
        print(
            f"Replaced {n_replaced} LIF nodes with membrane quantization "
            f"({len(lif_nodes) - n_replaced} skipped)"
        )

    return model


def quantize_model(model, config: PTQConfig, cali_data=None, device='cuda',
                   logger=None, output_dir=None):
    """
    量化模型的主函数

    当 config.fake_quant == 'adaround' 时, 需要提供 cali_data (DataLoader)
    用于逐层 AdaRound 优化.

    Args:
        model: 原始模型
        config: PTQ 配置
        cali_data: 校准数据 (DataLoader). AdaRound 必须提供
        device: 设备
        logger: 日志器
        output_dir: 输出目录 (用于保存 scale 详情 JSON)

    Returns:
        model: 量化后的模型
    """
    log = logger.info if logger else print

    log("=" * 60)
    log("PTQ for Spike-Driven Transformer (Per-channel)")
    log("=" * 60)
    log(f"  First layer:   {config.first_layer_bit}-bit")
    log(f"  Middle layers: {config.weight_bit}-bit")
    log(f"  Last layer:    {config.last_layer_bit}-bit")
    if config.mem_bit >= 32:
        log(f"  Membrane:      {config.mem_bit}-bit (disabled)")
    else:
        fp_first = " (FP)" if config.first_mem_bit >= 32 else ""
        log(
            f"  Membrane:      patch_embed.proj_lif {config.first_mem_bit}-bit{fp_first}, "
            f"others {config.mem_bit}-bit"
        )
    log(f"  BN folding:    {config.fold_bn}")
    log(f"  Observer:      {config.observer}")
    log(f"  FakeQuantize:  {config.fake_quant}")
    if config.fake_quant == 'adaround':
        log(f"  AdaRound iters: {config.adaround_iters}")
        log(f"  AdaRound lr:    {config.adaround_lr}")
        log(
            f"  Recon:          batches={config.recon_num_batches}, "
            f"mem_lam={config.recon_mem_lam}, "
            f"reg_scale={config.recon_reg_lam_scale}, "
            f"max_shift={config.recon_max_code_shift}"
        )
        log(
            f"                  min_iters={config.recon_min_iters}, "
            f"patience={config.recon_early_stop_patience or 'auto'}, "
            f"grad_clip={config.recon_grad_clip}"
        )
    if config.mem_bit < 32:
        mem_obs_name = config.mem_observer if config.mem_observer else config.observer
        log(f"  Mem observer:  {mem_obs_name}")
        log(f"  Scale bridge:  {config.scale_bridge}")
    log("=" * 60)

    model = copy.deepcopy(model)
    model = replace_with_quantized(model, config)

    # 校准权重 scale
    for name, module in model.named_modules():
        if isinstance(module, (QuantConv2d, QuantLinear)):
            module.calibrate()
    log(f"\nCalibrated weight scales")

    # 膜电位 scale 校准
    if config.mem_bit < 32:
        _calibrate_membrane(model, cali_data, device=device, logger=logger,
                            scale_bridge=config.scale_bridge,
                            output_dir=output_dir)

    # AdaRound：逐层优化权重取整方向
    if config.fake_quant == 'adaround':
        if cali_data is None:
            raise ValueError("AdaRound requires cali_data (DataLoader) for optimization")
        adaround_optimize(
            model, cali_data,
            num_batches=config.recon_num_batches,
            num_iters=config.adaround_iters,
            lr=config.adaround_lr,
            mem_lam=config.recon_mem_lam,
            reg_lam_scale=config.recon_reg_lam_scale,
            max_code_shift=config.recon_max_code_shift,
            min_iters=config.recon_min_iters,
            early_stop_patience=config.recon_early_stop_patience,
            improve_eps=config.recon_improve_eps,
            grad_clip=config.recon_grad_clip,
            log_interval=config.recon_log_interval,
            device=device,
            logger=logger,
        )

    enable_quantization(model)

    log("=" * 60)
    log("PTQ completed!")
    log("=" * 60)

    return model


# ============================================================
# 膜电位量化校准（模仿 PTQ4SNN 的元组传递机制）
# ============================================================


def _symmetric_quant_range(bit):
    """Signed symmetric quant integer range (matches observer)."""
    qmin = -2 ** (bit - 1)
    qmax = 2 ** (bit - 1) - 1
    return qmin, qmax


def _draw_quant_grid_on_axis(ax, scale_w, qmin, qmax, color, linestyle,
                             max_inner_lines=24):
    """Draw symmetric quant boundaries; inner ticks only if level count is small."""
    levels = list(range(qmin, qmax + 1))
    n = len(levels)
    if n > max_inner_lines:
        levels = [qmin, qmax]
    for i in levels:
        x = i * scale_w
        is_bdry = i in (qmin, qmax)
        lw = 1.8 if is_bdry else 0.9
        al = 0.95 if is_bdry else 0.45
        ax.axvline(x, color=color, linestyle=linestyle, linewidth=lw, alpha=al)


def _plot_weight_and_scaled_membrane(layer_name, w_per_ch, m_per_ch, k, s_w,
                                     w_bit, m_bit, output_dir, n_samples=3):
    """
    可视化: 每个选中通道上下两排。
    - 上: 权重 + 膜电位 / 2^k（与 s_mem = s_w·2^k 时 m/s_mem = (m/2^k)/s_w 一致），
      膜电位档在 m/2^k 域为 j·s_w。
    - 下: 同一通道权重 + 原始膜电位（不除 2^k），膜电位档在原始域为 j·(s_w·2^k)。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    C = w_per_ch.shape[0]
    n = min(n_samples, C)
    chs = sorted(random.sample(range(C), n))

    qmin_w, qmax_w = _symmetric_quant_range(w_bit)
    qmin_m, qmax_m = _symmetric_quant_range(m_bit)

    fig, axes = plt.subplots(2, n, figsize=(6 * n, 9), sharex=False)
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, c in enumerate(chs):
        ax_top = axes[0, j]
        ax_bot = axes[1, j]
        w_np = w_per_ch[c].float().numpy()
        k_val = k[c].item()
        scale_w = float(s_w[c].item())
        scale_mem = scale_w * (2.0 ** k_val)
        m_scaled = (m_per_ch[c].float() / (2.0 ** k_val)).numpy()
        m_raw = m_per_ch[c].float().numpy()

        # ---- 上: m / 2^k ----
        ax_top.hist(w_np, bins=60, alpha=0.6, density=True,
                    label='Weight', color='#4C72B0')
        ax_top.hist(m_scaled, bins=60, alpha=0.6, density=True,
                    label=rf'Mem / $2^{{{int(k_val)}}}$', color='#DD8452')
        ax_top.set_title(f'Ch {c}  (bridged: m / 2^k)')
        _draw_quant_grid_on_axis(ax_top, scale_w, qmin_w, qmax_w, '#2E5AAC', '--')
        if m_bit <= 8:
            _draw_quant_grid_on_axis(ax_top, scale_w, qmin_m, qmax_m, '#C45C26', ':')
        m_legend_top = (
            rf'm/2^k: [{qmin_m},{qmax_m}]·$s_w$'
            if m_bit <= 8
            else rf'm/2^k: {m_bit}-bit (grid omitted)'
        )
        ax_top.text(
            0.02, 0.98,
            rf'$s_w$={scale_w:.4g}, $k$={int(k_val)}' + '\n'
            rf'w: [{qmin_w},{qmax_w}]·$s_w$' + '\n'
            + m_legend_top,
            transform=ax_top.transAxes, fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85, edgecolor='#888'),
        )
        ax_top.legend(fontsize=8, loc='upper right')

        # ---- 下: 原始 m ----
        ax_bot.hist(w_np, bins=60, alpha=0.6, density=True,
                    label='Weight', color='#4C72B0')
        ax_bot.hist(m_raw, bins=60, alpha=0.6, density=True,
                    label='Mem (raw)', color='#55A868')
        ax_bot.set_title(f'Ch {c}  (raw membrane)')
        _draw_quant_grid_on_axis(ax_bot, scale_w, qmin_w, qmax_w, '#2E5AAC', '--')
        if m_bit <= 8:
            _draw_quant_grid_on_axis(ax_bot, scale_mem, qmin_m, qmax_m, '#2A9D4F', ':')
        m_legend_bot = (
            rf'm: [{qmin_m},{qmax_m}]·$s_w 2^k$'
            if m_bit <= 8
            else rf'm: {m_bit}-bit (grid omitted)'
        )
        ax_bot.text(
            0.02, 0.98,
            rf'$s_w 2^k$={scale_mem:.4g}' + '\n'
            rf'w: [{qmin_w},{qmax_w}]·$s_w$' + '\n'
            + m_legend_bot,
            transform=ax_bot.transAxes, fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85, edgecolor='#888'),
        )
        ax_bot.legend(fontsize=8, loc='upper right')

    safe = layer_name.replace('.', '_')
    fig.suptitle(
        f'{layer_name}  (top: m/2^k vs w; bottom: raw m vs w; '
        f'{w_bit}-bit W / {m_bit}-bit mem)',
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_dir, f'wm_{safe}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def _recalibrate_observer_from_buffer(observer, mem_buf, device, n_per_ch_max=8192):
    """
    用拼好的 per-channel 膜电位数据一次性重跑 observer，覆盖 per-batch 聚合得到
    的 min_val/max_val，再算 (scale, zero_point)。

    背景：MSEObserver / PercentileObserver 在按 batch 调用时，每次只能拿当前
    batch 的"最优裁剪边界"，再用 min/max 与历史值聚合，结果是各 batch 极端值
    的并集，会显著高估实际范围。本函数把所有 batch 的样本拼成 [C, N] 后一次性
    搜索，得到对全量数据真正的最优 scale。

    Args:
        observer: m.mem_fake_quant.observer 实例 (要求 ch_axis=1)
        mem_buf: list of [C, N_i] CPU tensors, 已采样
        device: 计算 device
        n_per_ch_max: per-channel 样本上限，超出再做随机子采样 (控制显存/算力)

    Returns:
        (scale, zero_point) on `device`，失败时返回 (None, None)
    """
    if not mem_buf:
        return None, None
    full = torch.cat(mem_buf, dim=1)              # [C, N]
    if full.numel() == 0:
        return None, None
    if full.shape[1] > n_per_ch_max:
        idx = torch.randperm(full.shape[1])[:n_per_ch_max]
        full = full[:, idx]
    # observer 内部按 ch_axis=1 处理，传 [N, C]
    full = full.t().contiguous().to(device)

    # 重置 observer 状态，让单次 forward 自己写 min_val / max_val
    observer.min_val.data.fill_(float('inf'))
    observer.max_val.data.fill_(float('-inf'))
    if hasattr(observer, 'one_side_dist'):
        observer.one_side_dist = None
    if hasattr(observer, 'cnt'):
        observer.cnt = 0

    observer(full)
    s, zp = observer.calculate_qparams(observer.min_val, observer.max_val)
    return s.to(device), zp.to(device)


def _find_conv_for_lif(model, q_lif_nodes):
    """Map LIF module names to the preceding paired QuantConv2d modules."""
    all_modules = dict(model.named_modules())
    conv_for_lif = {}
    for lif_name, _ in q_lif_nodes:
        parts = lif_name.rsplit('.', 1)
        parent, leaf = (parts[0], parts[1]) if len(parts) == 2 else ('', parts[0])
        conv_leaf = LIF_TO_CONV_LEAF.get(leaf)
        if conv_leaf:
            conv_name = f"{parent}.{conv_leaf}" if parent else conv_leaf
            conv = all_modules.get(conv_name)
            if isinstance(conv, QuantConv2d):
                conv_for_lif[lif_name] = conv
    return conv_for_lif


@torch.no_grad()
def _calibrate_membrane(model, loader, num_batches=8, device='cuda', logger=None,
                        scale_bridge='weight', output_dir=None):
    """
    校准 QuantMultiStepLIFNode 的膜电位量化 scale。

    scale_bridge 模式:
    - 'weight'  : PTQ4SNN 风格，直接使用前置 Conv 的权重 scale
    - 'observer': 使用 observer 独立校准的 scale（精度最高，但硬件不友好）
    - 'pot'     : Power-of-2 bridging, s_mem = s_w * 2^k
                  兼顾精度和硬件友好（只需 bit-shift）
    - 'unify'   : 联合配对通道, 将对应通道的权重值和膜电位值放在一起做 MSE
                  搜索最优共享 scale, 无需 scale 转换, 硬件最友好

    对于没有前置 QuantConv2d 的 LIF 节点（如 shortcut_lif, attn_lif），
    所有模式下都使用 observer 校准 scale。
    """
    log = logger.info if logger else print
    log(f"\n[Membrane] Calibrating membrane potential (scale_bridge={scale_bridge})...")

    q_lif_nodes = [(n, m) for n, m in model.named_modules()
                   if isinstance(m, QuantMultiStepLIFNode)]

    if not q_lif_nodes:
        log("  No QuantMultiStepLIFNode found, skipping")
        return

    use_pot = scale_bridge in ('pot', 'observer', 'unify')

    # 'unify' 模式需要找到每个 LIF 配对的 QuantConv2d，以便同步更新权重 scale
    conv_for_lif = {}
    if scale_bridge in ('pot', 'unify'):
        conv_for_lif = _find_conv_for_lif(model, q_lif_nodes)

    for _, m in q_lif_nodes:
        m.mem_fake_quant.to(device)
        m.mem_fake_quant.enable_observer()
        m.mem_fake_quant.disable_fake_quant()
        m.has_weight_scale = False
        m._stored_weight_scale = None
        m._pot_calibrating = use_pot

    # 'unify' / 'pot' / 'observer' 模式: 用 hook 收集每个 LIF 的原始膜电位
    # 数据 (per-channel)。observer 模式下，最后用拼好的 buffer 一次性跑 observer，
    # 避免 MSEObserver/PercentileObserver 等按 batch 聚合 (min-of-mins / max-of-maxes)
    # 把 scale 撑大的问题。
    _unify_hooks = []
    if scale_bridge in ('pot', 'unify', 'observer'):
        for _, m in q_lif_nodes:
            m._unify_mem_buf = []

            def _make_hook(node, max_per_call=1024):
                def hook(mod, inp):
                    x = inp[0]
                    if x.dim() < 2:
                        return
                    ch = x.shape[1]
                    x_flat = x.transpose(0, 1).reshape(ch, -1)
                    n = x_flat.shape[1]
                    if n > max_per_call:
                        idx = torch.randperm(n, device=x_flat.device)[:max_per_call]
                        x_flat = x_flat[:, idx]
                    node._unify_mem_buf.append(x_flat.detach().cpu())
                return hook

            _unify_hooks.append(
                m.mem_fake_quant.register_forward_pre_hook(_make_hook(m))
            )

    def _reset_snn_only(net):
        from spikingjelly.clock_driven.neuron import BaseNode
        for mod in net.modules():
            if isinstance(mod, BaseNode):
                mod.reset()

    model.eval()
    for i, (images, _) in enumerate(loader):
        if i >= num_batches:
            break
        images = images.float().to(device)
        model(images)
        _reset_snn_only(model)

    for h in _unify_hooks:
        h.remove()

    def _t2list(t):
        """Tensor -> list of Python floats (rounded to 8 decimal places)."""
        return [round(v, 8) for v in t.detach().cpu().flatten().tolist()]

    scale_details = {"scale_bridge": scale_bridge, "layers": {}}

    for name, m in q_lif_nodes:
        m._pot_calibrating = False
        m.mem_fake_quant.disable_observer()

        obs = m.mem_fake_quant.observer
        s_obs, zp_obs = obs.calculate_qparams(obs.min_val, obs.max_val)
        s_obs = s_obs.to(device)
        zp_obs = zp_obs.to(device)
        n_ch = int(s_obs.numel())

        if m.has_weight_scale and m._stored_weight_scale is not None and scale_bridge == 'pot':
            s_w = m._stored_weight_scale.to(device)
            ratio = s_obs / (s_w + 1e-10)
            k = torch.round(torch.log2(ratio.clamp(min=1e-10))).clamp(-8, 8)
            s_bridge = s_w * (2.0 ** k)

            m._pot_k = k

            if m.mem_fake_quant.scale.shape != s_bridge.shape:
                m.mem_fake_quant.scale.resize_(s_bridge.shape)
                m.mem_fake_quant.zero_point.resize_(s_bridge.shape)
            m.mem_fake_quant.scale.data.copy_(s_bridge)
            m.mem_fake_quant.zero_point.data.zero_()

            layer_info = {
                "method": "pot_bridge", "num_channels": n_ch,
                "k": _t2list(k), "s_w": _t2list(s_w),
                "s_obs": _t2list(s_obs), "s_pot": _t2list(s_bridge),
            }
            scale_details["layers"][name] = layer_info
            log(f"  {name}: PoT bridge ({n_ch} ch), "
                f"k=[{k.min().item():.0f},{k.max().item():.0f}], "
                f"s_w=[{s_w.min().item():.6f},{s_w.max().item():.6f}], "
                f"s_obs=[{s_obs.min().item():.6f},{s_obs.max().item():.6f}], "
                f"s_pot=[{s_bridge.min().item():.6f},{s_bridge.max().item():.6f}]")

            mem_buf = getattr(m, '_unify_mem_buf', [])
            if output_dir and name in conv_for_lif and mem_buf:
                conv_mod = conv_for_lif[name]
                w_ch = conv_mod.weight.data.view(
                    conv_mod.weight.shape[0], -1).cpu()
                m_ch = torch.cat(mem_buf, dim=1)
                _plot_weight_and_scaled_membrane(
                    name, w_ch, m_ch, k.cpu(), s_w.cpu(),
                    conv_mod.bit, m.mem_bit, output_dir)
                log(f"    plot saved: wm_{name.replace('.', '_')}.png")

        elif m.has_weight_scale and m._stored_weight_scale is not None and scale_bridge == 'unify':
            mem_buf = getattr(m, '_unify_mem_buf', [])
            has_conv = name in conv_for_lif
            if mem_buf and has_conv:
                conv_mod = conv_for_lif[name]
                w_per_ch = conv_mod.weight.data.view(
                    conv_mod.weight.shape[0], -1).cpu()           # [C, K]
                m_per_ch = torch.cat(mem_buf, dim=1)              # [C, N]

                # Step 1: PoT — 计算 k (固定)
                s_w = m._stored_weight_scale.cpu()
                ratio = s_obs.cpu() / (s_w + 1e-10)
                k_pot = torch.round(torch.log2(ratio.clamp(min=1e-10))).clamp(-8, 8)

                # Step 2: LSQ — 固定 k, 梯度优化 s (weight loss + spike loss)
                log(f"  {name}: LSQ optimizing scale ({n_ch} ch, k from PoT, "
                    f"k=[{k_pot.min().item():.0f},{k_pot.max().item():.0f}])...")
                s_opt = lsq_unified_scale_optim(
                    w_per_ch, m_per_ch,
                    s_init=s_w, k=k_pot,
                    w_bit=conv_mod.bit, m_bit=m.mem_bit,
                    log_fn=log,
                ).to(device)

                k_pot = k_pot.to(device)
                s_mem = s_opt * (2.0 ** k_pot)

                # 膜电位 scale = s * 2^k
                if m.mem_fake_quant.scale.shape != s_mem.shape:
                    m.mem_fake_quant.scale.resize_(s_mem.shape)
                    m.mem_fake_quant.zero_point.resize_(s_mem.shape)
                m.mem_fake_quant.scale.data.copy_(s_mem)
                m.mem_fake_quant.zero_point.data.zero_()
                m._observer_calibrated = True
                m._pot_k = k_pot

                if output_dir:
                    _plot_weight_and_scaled_membrane(
                        name, w_per_ch, m_per_ch, k_pot.cpu(), s_opt.cpu(),
                        conv_mod.bit, m.mem_bit, output_dir)
                    log(f"    plot saved: wm_{name.replace('.', '_')}.png")

                # 同步 Conv 权重 scale = s
                conv_fq = conv_mod.weight_fake_quant
                if conv_fq.scale.shape != s_opt.shape:
                    conv_fq.scale.resize_(s_opt.shape)
                conv_fq.scale.data.copy_(s_opt)

                layer_info = {
                    "method": "lsq_unify", "num_channels": n_ch,
                    "k": _t2list(k_pot), "s_w": _t2list(s_opt),
                    "s_mem": _t2list(s_mem), "s_w_init": _t2list(s_w),
                    "s_obs": _t2list(s_obs),
                }
                scale_details["layers"][name] = layer_info
                log(f"    done ({n_ch} ch): "
                    f"s_w=[{s_opt.min().item():.6f},{s_opt.max().item():.6f}], "
                    f"s_mem=[{s_mem.min().item():.6f},{s_mem.max().item():.6f}], "
                    f"s_w_init=[{s_w.min().item():.6f},{s_w.max().item():.6f}], "
                    f"s_obs=[{s_obs.min().item():.6f},{s_obs.max().item():.6f}]")
            else:
                if m.mem_fake_quant.scale.shape != s_obs.shape:
                    m.mem_fake_quant.scale.resize_(s_obs.shape)
                    m.mem_fake_quant.zero_point.resize_(s_obs.shape)
                m.mem_fake_quant.scale.data.copy_(s_obs)
                m.mem_fake_quant.zero_point.data.zero_()
                m._observer_calibrated = True

                layer_info = {
                    "method": "unify_fallback_observer", "num_channels": n_ch,
                    "scale": _t2list(s_obs),
                }
                scale_details["layers"][name] = layer_info
                log(f"  {name}: unified fallback (observer, {n_ch} ch), "
                    f"scale=[{s_obs.min().item():.6f},{s_obs.max().item():.6f}]")

            if mem_buf:
                del m._unify_mem_buf

        elif m.has_weight_scale and scale_bridge == 'weight':
            scale = m.mem_fake_quant.scale.data
            layer_info = {
                "method": "weight_scale", "num_channels": n_ch,
                "scale": _t2list(scale),
            }
            scale_details["layers"][name] = layer_info
            log(f"  {name}: weight_scale ({n_ch} ch), "
                f"scale=[{scale.min().item():.6f}, {scale.max().item():.6f}]")

        else:
            # 如果 hook 收集到了完整 per-channel 膜电位 buffer (observer 模式 /
            # pot/unify 模式中没有匹配 conv 的 LIF), 用整段数据重跑一次 observer,
            # 取代逐 batch 聚合得到的 s_obs。
            mem_buf = getattr(m, '_unify_mem_buf', [])
            method_label = "observer" if scale_bridge != 'weight' else "observer (no conv)"
            s_use = s_obs
            s_full = None
            if mem_buf:
                s_full, _ = _recalibrate_observer_from_buffer(
                    m.mem_fake_quant.observer, mem_buf, device,
                )
                if s_full is not None and s_full.numel() == n_ch:
                    s_use = s_full
                    method_label = f"{method_label} (full-buffer)"

            if m.mem_fake_quant.scale.shape != s_use.shape:
                m.mem_fake_quant.scale.resize_(s_use.shape)
                m.mem_fake_quant.zero_point.resize_(s_use.shape)
            m.mem_fake_quant.scale.data.copy_(s_use)
            m.mem_fake_quant.zero_point.data.zero_()
            m._observer_calibrated = True

            layer_info = {
                "method": method_label, "num_channels": n_ch,
                "scale": _t2list(s_use),
            }
            if s_full is not None and s_use is s_full:
                layer_info["scale_per_batch_obs"] = _t2list(s_obs)
            scale_details["layers"][name] = layer_info
            log(f"  {name}: {method_label} ({n_ch} ch), "
                f"scale=[{s_use.min().item():.6f}, {s_use.max().item():.6f}]"
                + (f"  (per-batch s_obs=[{s_obs.min().item():.6f},"
                   f"{s_obs.max().item():.6f}])" if s_use is s_full else ""))

        if hasattr(m, '_unify_mem_buf'):
            del m._unify_mem_buf

    # 写 JSON 详情文件
    if output_dir:
        json_path = os.path.join(output_dir, 'membrane_scale_details.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(scale_details, f, indent=2, ensure_ascii=False)
        log(f"  Per-channel scale details saved to {json_path}")

    log(f"  Calibrated {len(q_lif_nodes)} LIF nodes (bridge={scale_bridge})")


# ============================================================
# Layer-wise weight distillation (uses the AdaRound config path)
# ============================================================

@torch.no_grad()
def _collect_layer_io(model, loader, num_batches=8, device='cuda', logger=None):
    """
    用 forward hook 收集每个量化层的输入/输出 (原始权重, 未启用 fake_quant)

    Returns:
        OrderedDict: {layer_name: {'inputs': [...], 'outputs': [...]}}
    """
    from collections import OrderedDict
    from spikingjelly.clock_driven import functional

    log = logger.info if logger else print

    data = OrderedDict()
    hooks = []
    current_batch_size = {"value": None}

    def _make_hook(name):
        def hook_fn(module, inp, out):
            if name not in data:
                data[name] = {'inputs': [], 'outputs': [], 'batch_sizes': []}
            if len(data[name]['inputs']) < num_batches:
                data[name]['inputs'].append(inp[0].detach().cpu())
                out_tensor = out[0] if isinstance(out, tuple) else out
                data[name]['outputs'].append(out_tensor.detach().cpu())
                data[name]['batch_sizes'].append(current_batch_size["value"])
        return hook_fn

    for name, m in model.named_modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            hooks.append(m.register_forward_hook(_make_hook(name)))

    model.eval()
    for i, (images, _) in enumerate(loader):
        if i >= num_batches:
            break
        images = images.float().to(device)
        current_batch_size["value"] = int(images.shape[0])
        model(images)
        functional.reset_net(model)

    for h in hooks:
        h.remove()

    log(f"  Collected {num_batches} batches for {len(data)} layers")
    return data


def _weight_qparams_view(module, weight):
    """Broadcast this module's fixed weight quantization params to ``weight``."""
    fq = module.weight_fake_quant
    if fq.ch_axis != -1:
        shape = [1] * weight.dim()
        shape[fq.ch_axis] = -1
        scale = fq.scale.detach().to(device=weight.device, dtype=weight.dtype).reshape(shape)
        zero_point = fq.zero_point.detach().to(
            device=weight.device, dtype=weight.dtype).reshape(shape)
    else:
        scale = fq.scale.detach().to(device=weight.device, dtype=weight.dtype)
        zero_point = fq.zero_point.detach().to(device=weight.device, dtype=weight.dtype)
    return scale.clamp(min=1e-12), zero_point


def _fake_quant_scaled_weight_ste(module, scaled_weight, scale, zero_point):
    """Quantize a trainable integer-domain weight coordinate with STE."""
    fq = module.weight_fake_quant
    w_int = (scaled_weight.round() - scaled_weight).detach() + scaled_weight
    w_int = torch.clamp(w_int + zero_point, fq.quant_min, fq.quant_max)
    return (w_int - zero_point) * scale


def _forward_layer_with_weight(module, x, weight):
    import torch.nn.functional as Fn
    bias = module.bias.detach() if module.bias is not None else None
    if isinstance(module, QuantConv2d):
        return Fn.conv2d(
            x, weight, bias,
            module.stride, module.padding,
            module.dilation, module.groups,
        )
    return Fn.linear(x, weight, bias)


def _find_paired_lif_for_layer(model, layer_name, module):
    """Return the LIF that immediately consumes this quant layer's output."""
    if not isinstance(module, QuantConv2d):
        return None
    all_modules = dict(model.named_modules())
    for lif_name, conv in _find_conv_for_lif(
        model,
        [(n, m) for n, m in model.named_modules()
         if isinstance(m, QuantMultiStepLIFNode)],
    ).items():
        if conv is module:
            # In MS_SPS, proj_conv3 can be followed by maxpool3 before proj_lif3
            # depending on pooling_stat. Skip its local membrane term rather than
            # distilling against a path that is not the model's real LIF input.
            if lif_name == 'patch_embed.proj_lif3':
                return None
            return lif_name, all_modules[lif_name]
    return None


def _tensor_mse(a, b):
    return ((a - b) ** 2).mean()


def _reshape_flat_tb_to_tbc(x, batch_size):
    if x.dim() != 4 or not batch_size or x.shape[0] % batch_size != 0:
        return None
    time_steps = x.shape[0] // batch_size
    if time_steps <= 0:
        return None
    return x.reshape(time_steps, batch_size, x.shape[1], x.shape[2], x.shape[3]).contiguous()


def _set_lif_fake_quant_state(lif, enabled):
    if enabled:
        lif.enable_fake_quant()
    else:
        lif.disable_fake_quant()


def _distill_layer(module, cached_inps, cached_outs, cached_batch_sizes=None,
                   paired_lif=None, num_iters=500, lr=3e-3,
                   mem_lam=0.25, reg_lam_scale=1e-4,
                   max_code_shift=2.0, min_iters=50,
                   early_stop_patience=0, improve_eps=1e-6,
                   grad_clip=1.0, log_interval=100,
                   device='cuda', logger=None):
    """
    Optimize the layer weight itself with STE quantization.

    The output term preserves the old AdaRound reconstruction objective. When
    this layer feeds a QuantMultiStepLIFNode directly, an additional term aligns
    the LIF pre-fire membrane potential after weight and membrane quantization.
    """

    log = logger.info if logger else print

    w_ref = module.weight.data.detach().clone()
    w_ref_dev = w_ref.to(device)
    scale_view, zero_point_view = _weight_qparams_view(module, w_ref_dev)
    z_ref = w_ref_dev / scale_view
    z_opt = nn.Parameter(z_ref.clone())

    # Keep the public AdaRound lr usable: optimize in quantization-step units
    # instead of raw weight units. This makes lr=0.003 a small fraction of one
    # integer bin for every channel, rather than an absolute 0.003 weight jump.
    optimizer = torch.optim.Adam([z_opt], lr=lr)

    ref_max = max(y.abs().max().item() for y in cached_outs) + 1e-8
    reg_lam = reg_lam_scale * ref_max
    min_iters = min(int(min_iters), num_iters)
    if early_stop_patience and early_stop_patience > 0:
        early_stop_patience = int(early_stop_patience)
    else:
        early_stop_patience = min(80, max(20, num_iters // 10))
    log_interval = max(1, int(log_interval))
    use_mem = paired_lif is not None and len(cached_inps) > 0

    best_loss = float('inf')
    best_weight = w_ref.clone()
    best_iter = -1
    best_stats = {}
    lif_fq_state = None
    if paired_lif is not None:
        lif_fq_state = bool(getattr(paired_lif, 'fake_quant_enabled', False))

    for it in range(num_iters):
        w_float = z_opt * scale_view
        w_q = _fake_quant_scaled_weight_ste(
            module, z_opt, scale_view, zero_point_view)

        recon_loss = torch.tensor(0.0, device=device)
        mem_loss = torch.tensor(0.0, device=device)

        for idx, (x_cpu, y_cpu) in enumerate(zip(cached_inps, cached_outs)):
            x = x_cpu.to(device)
            y_ref = y_cpu.to(device)

            y_q = _forward_layer_with_weight(module, x, w_q)

            recon_loss = recon_loss + _tensor_mse(y_q, y_ref)

            if use_mem:
                lif_in_ref = y_ref
                lif_in_q = y_q
                if lif_in_ref.dim() == 4 and lif_in_q.dim() == 4:
                    batch_size = None
                    if cached_batch_sizes is not None and idx < len(cached_batch_sizes):
                        batch_size = cached_batch_sizes[idx]
                    try:
                        scale = module.weight_fake_quant.scale.data.to(device)
                        lif_in_ref = _reshape_flat_tb_to_tbc(lif_in_ref, batch_size)
                        lif_in_q = _reshape_flat_tb_to_tbc(lif_in_q, batch_size)
                        if lif_in_ref is None or lif_in_q is None:
                            continue
                        _set_lif_fake_quant_state(paired_lif, False)
                        paired_lif.start_recording_firing(
                            record_v_pre=True, record_spike=False,
                            detach=True, to_cpu=False,
                        )
                        paired_lif((lif_in_ref, scale))
                        v_ref = paired_lif.get_last_v_pre()
                        paired_lif.stop_recording_firing()
                        paired_lif.original_lif.reset()

                        _set_lif_fake_quant_state(paired_lif, True)
                        paired_lif.start_recording_firing(
                            record_v_pre=True, record_spike=False,
                            detach=False, to_cpu=False,
                        )
                        paired_lif((lif_in_q, scale))
                        v_q = paired_lif.get_last_v_pre()
                        paired_lif.stop_recording_firing()
                        paired_lif.original_lif.reset()
                        if v_ref is not None and v_q is not None and v_ref.shape == v_q.shape:
                            mem_loss = mem_loss + _tensor_mse(v_q, v_ref)
                    finally:
                        paired_lif.stop_recording_firing()
                        paired_lif.original_lif.reset()
                        if lif_fq_state is not None:
                            _set_lif_fake_quant_state(paired_lif, lif_fq_state)

        recon_loss = recon_loss / len(cached_inps)
        if use_mem:
            mem_loss = mem_loss / len(cached_inps)

        reg = (z_opt - z_ref) ** 2
        reg = reg.mean()
        total_loss = recon_loss + reg_lam * reg + mem_lam * mem_loss

        loss_val = total_loss.item()
        if loss_val < best_loss - improve_eps:
            best_loss = loss_val
            best_iter = it
            # Save the exact weights that produced this loss. Saving after
            # optimizer.step() would pair a pre-step loss with post-step weights.
            best_weight = w_float.detach().clone().cpu()
            code_shift = (z_opt.detach() - z_ref).abs()
            best_stats = {
                "out": recon_loss.item(),
                "mem": mem_loss.item(),
                "reg": reg.item(),
                "d_code_mean": code_shift.mean().item(),
                "d_code_max": code_shift.max().item(),
            }

        optimizer.zero_grad()
        total_loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([z_opt], max_norm=grad_clip)
        optimizer.step()
        if max_code_shift and max_code_shift > 0:
            with torch.no_grad():
                z_opt.clamp_(z_ref - max_code_shift, z_ref + max_code_shift)

        if it % log_interval == 0 or it == num_iters - 1:
            code_shift = (z_opt.detach() - z_ref).abs()
            log(
                f"    iter {it:4d}/{num_iters}: loss={loss_val:.6f} "
                f"(best={best_loss:.6f}), out={recon_loss.item():.6f}, "
                f"mem={mem_loss.item():.6f}, reg={reg.item():.6f}, "
                f"|d_code|={code_shift.mean().item():.4f}/"
                f"{code_shift.max().item():.4f}"
            )

        if it + 1 >= min_iters and it - best_iter >= early_stop_patience:
            log(
                f"    early stop at iter {it}: no best improvement for "
                f"{early_stop_patience} iters (best at {best_iter})"
            )
            break

    module.weight.data.copy_(best_weight.to(module.weight.device))
    if best_stats:
        log(
            f"    Final: best_loss={best_loss:.6f} (iter {best_iter}), "
            f"best_out={best_stats['out']:.6f}, "
            f"best_mem={best_stats['mem']:.6f}, "
            f"best_reg={best_stats['reg']:.6f}, "
            f"best|d_code|={best_stats['d_code_mean']:.4f}/"
            f"{best_stats['d_code_max']:.4f}"
        )
    else:
        log(f"    Final: best_loss={best_loss:.6f} (iter {best_iter})")


def adaround_optimize(model, loader, num_batches=8, num_iters=500, lr=3e-3,
                      mem_lam=0.25, reg_lam_scale=1e-4,
                      max_code_shift=2.0, min_iters=50,
                      early_stop_patience=0, improve_eps=1e-6,
                      grad_clip=1.0, log_interval=100,
                      device='cuda', logger=None):
    """
    Layer-wise weight distillation entry point.

    It intentionally keeps the historical ``adaround`` config path so existing
    configs do not need to change.

    流程:
    1. 收集每层输入/输出 (原始权重下)
    2. 逐层优化该层权重，经 STE fake-quant 后对齐原始层输出
    3. 若该层直接接 LIF，同时对齐 LIF pre-fire 膜电位
    """
    log = logger.info if logger else print

    log("\n[Distill] Collecting calibration data...")
    layer_data = _collect_layer_io(model, loader, num_batches, device, logger)

    quant_layers = [(n, m) for n, m in model.named_modules()
                    if isinstance(m, (QuantConv2d, QuantLinear))]

    log(f"\n[Distill] Optimizing {len(quant_layers)} layers (output + membrane loss)...")
    for idx, (name, module) in enumerate(quant_layers):
        tag = f"[{idx+1}/{len(quant_layers)}]"
        if name not in layer_data:
            log(f"  {tag} {name}: no data, skipping")
            continue

        log(f"  {tag} {name} ({module.bit}-bit)")

        if module.bit >= 12:
            log(f"    Skipped (bit={module.bit} >= 12)")
            continue

        t0 = time.time()
        ld = layer_data[name]
        lif_pair = _find_paired_lif_for_layer(model, name, module)
        paired_lif = lif_pair[1] if lif_pair is not None else None
        if lif_pair is not None:
            log(f"    membrane distill target: {lif_pair[0]}")

        _distill_layer(
            module, ld['inputs'], ld['outputs'],
            cached_batch_sizes=ld.get('batch_sizes'),
            paired_lif=paired_lif,
            num_iters=num_iters, lr=lr,
            mem_lam=mem_lam,
            reg_lam_scale=reg_lam_scale,
            max_code_shift=max_code_shift,
            min_iters=min_iters,
            early_stop_patience=early_stop_patience,
            improve_eps=improve_eps,
            grad_clip=grad_clip,
            log_interval=log_interval,
            device=device, logger=logger,
        )

        elapsed = time.time() - t0
        log(f"    Done ({elapsed:.1f}s)")

    del layer_data
    torch.cuda.empty_cache()
    log("[Distill] Optimization completed.")


def get_model_size(model, unit='MB'):
    """
    获取模型大小

    Args:
        model: 模型
        unit: 单位 ('MB', 'KB', 'B')

    Returns:
        模型大小
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size

    if unit == 'MB':
        return total_size / 1024 / 1024
    elif unit == 'KB':
        return total_size / 1024
    return total_size


def calculate_quantized_model_size(model, config: PTQConfig, unit='MB'):
    """
    计算量化后的模型大小（理论值）

    Args:
        model: 模型
        config: PTQ 配置
        unit: 单位

    Returns:
        理论模型大小
    """
    total_bits = 0

    for name, param in model.named_parameters():
        if 'weight' in name:
            total_bits += param.numel() * config.weight_bit
        else:
            total_bits += param.numel() * 32  # bias 等使用 32 bit

    total_bytes = total_bits / 8

    if unit == 'MB':
        return total_bytes / 1024 / 1024
    elif unit == 'KB':
        return total_bytes / 1024
    return total_bytes


def print_quantized_model(model, logger=None):
    """
    打印量化模型的完整结构

    Args:
        model: 量化后的模型
        logger: 日志器
    """
    log = logger.info if logger else print

    log("\n" + "=" * 70)
    log("Quantized Model Structure:")
    log("=" * 70)

    model_str = repr(model)
    for line in model_str.split('\n'):
        log(line)

    log("=" * 70)
