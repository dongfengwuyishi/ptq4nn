"""
状态管理模块

提供统一的接口来控制模型的量化状态:
- enable_quantization / disable_quantization: 控制伪量化开关
- enable_observer / disable_observer: 控制 observer 开关
- calibrate_model: 校准模型
"""
import torch
import torch.nn as nn
from .quantized_module import QuantConv2d, QuantLinear, QuantMultiStepLIFNode


def enable_quantization(model):
    """
    启用模型中所有量化模块的伪量化

    Args:
        model: 包含量化模块的模型
    """
    for module in model.modules():
        if hasattr(module, 'enable_fake_quant'):
            module.enable_fake_quant()


def disable_quantization(model):
    """
    禁用模型中所有量化模块的伪量化

    Args:
        model: 包含量化模块的模型
    """
    for module in model.modules():
        if hasattr(module, 'disable_fake_quant'):
            module.disable_fake_quant()


def enable_observer(model):
    """
    启用模型中所有量化模块的 observer (校准模式)

    Args:
        model: 包含量化模块的模型
    """
    for module in model.modules():
        if hasattr(module, 'enable_observer'):
            module.enable_observer()


def disable_observer(model):
    """
    禁用模型中所有量化模块的 observer

    Args:
        model: 包含量化模块的模型
    """
    for module in model.modules():
        if hasattr(module, 'disable_observer'):
            module.disable_observer()


def calibrate_model(model, cali_data=None, device='cuda'):
    """
    校准模型中的所有量化模块

    对于权重量化:
    - 直接根据权重计算 scale (不需要校准数据)

    对于激活量化 (如果启用):
    - 需要校准数据来统计激活分布

    Args:
        model: 包含量化模块的模型
        cali_data: 校准数据 (可选，权重量化不需要)
        device: 设备

    Returns:
        model: 校准后的模型
    """
    model.eval()
    model.to(device)

    # 收集所有量化层
    quant_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (QuantConv2d, QuantLinear)):
            quant_layers.append((name, module))

    # 校准权重 (直接根据权重计算 scale)
    for name, module in quant_layers:
        module.calibrate()
        print(f"Calibrated {name}")

    return model


def get_quant_state(model):
    """
    获取模型的量化状态

    Returns:
        dict: 包含每个量化层的状态信息
    """
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, (QuantConv2d, QuantLinear)):
            state[name] = {
                'bit': module.bit,
                'fake_quant_enabled': module.fake_quant_enabled,
                'calibrated': module.calibrated,
                'scale_min': module.scale.min().item(),
                'scale_max': module.scale.max().item(),
                'scale_mean': module.scale.mean().item(),
            }
    return state


def print_quant_state(model, logger=None):
    """
    打印模型的量化状态

    Args:
        model: 包含量化模块的模型
        logger: 日志器 (可选)
    """
    log = logger.info if logger else print
    state = get_quant_state(model)

    log("\n" + "=" * 70)
    log("Quantization State")
    log("=" * 70)
    log(f"{'Layer':<40} {'Bit':>4} {'Enabled':>8} {'Calibrated':>10} {'Scale Range':>20}")
    log("-" * 70)

    for name, info in state.items():
        scale_range = f"[{info['scale_min']:.2e}, {info['scale_max']:.2e}]"
        log(f"{name:<40} {info['bit']:>4} {info['fake_quant_enabled']:>8} {info['calibrated']:>10} {scale_range:>20}")

    log("=" * 70)


def set_bit(model, bit, layer_names=None):
    """
    设置模型中量化层的位宽

    Args:
        model: 包含量化模块的模型
        bit: 新的位宽
        layer_names: 要修改的层名称列表 (None 表示所有层)
    """
    for name, module in model.named_modules():
        if isinstance(module, (QuantConv2d, QuantLinear)):
            if layer_names is None or name in layer_names:
                module.bit = bit
                # 重新计算 quant_min/quant_max
                print(f"Set {name} to {bit}-bit")


def freeze_quantization(model):
    """
    冻结量化参数 (用于 QAT 后的推理)

    将 scale 和 zero_point 转换为固定值，不再更新
    """
    for module in model.modules():
        if hasattr(module, 'disable_observer'):
            module.disable_observer()
        if hasattr(module, 'weight_fake_quant'):
            module.weight_fake_quant.disable_observer()


def get_quantized_layers(model):
    """
    获取模型中所有量化层

    Returns:
        list: [(name, module), ...]
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (QuantConv2d, QuantLinear, QuantMultiStepLIFNode)):
            layers.append((name, module))
    return layers


def count_quantized_params(model):
    """
    统计量化参数数量

    Returns:
        dict: 包含各类参数的统计
    """
    stats = {
        'total_params': 0,
        'quantized_weight_params': 0,
        'full_precision_params': 0,
        'layers': {},
    }

    for name, module in model.named_modules():
        if isinstance(module, (QuantConv2d, QuantLinear)):
            weight_params = module.weight.numel()
            bias_params = module.bias.numel() if module.bias is not None else 0

            stats['total_params'] += weight_params + bias_params
            stats['quantized_weight_params'] += weight_params
            stats['full_precision_params'] += bias_params  # bias 通常不量化

            stats['layers'][name] = {
                'weight_params': weight_params,
                'bias_params': bias_params,
                'bit': module.bit,
            }

    return stats
