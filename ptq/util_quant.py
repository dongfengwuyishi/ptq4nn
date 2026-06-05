"""
量化工具函数

包含:
- round_ste: Straight-Through Estimator 取整
- fake_quantize_per_tensor_affine: Per-tensor 伪量化
- fake_quantize_per_channel_affine: Per-channel 伪量化
"""
import torch


def round_ste(x):
    """
    Straight-Through Estimator (STE) 取整

    前向: round(x)
    反向: 梯度直通 (gradient = 1)
    """
    return (x.round() - x).detach() + x


def fake_quantize_per_tensor_affine(x, scale, zero_point, quant_min, quant_max):
    """
    Per-tensor 伪量化

    Args:
        x: 输入张量
        scale: 量化缩放因子 (标量)
        zero_point: 零点偏移 (标量)
        quant_min: 量化最小值
        quant_max: 量化最大值

    Returns:
        伪量化后的张量 (与输入相同形状和精度)
    """
    x_int = round_ste(x / scale) + zero_point
    x_int = torch.clamp(x_int, quant_min, quant_max)
    x_quant = (x_int - zero_point) * scale
    return x_quant


def fake_quantize_per_channel_affine(x, scale, zero_point, ch_axis, quant_min, quant_max):
    """
    Per-channel 伪量化

    Args:
        x: 输入张量
        scale: 每通道的量化缩放因子 [num_channels]
        zero_point: 每通道的零点偏移 [num_channels]
        ch_axis: 通道轴
        quant_min: 量化最小值
        quant_max: 量化最大值

    Returns:
        伪量化后的张量
    """
    # 扩展 scale 和 zero_point 的维度以便广播
    shape = [1] * x.dim()
    shape[ch_axis] = -1
    scale_view = scale.view(*shape)
    zp_view = zero_point.view(*shape).float()

    x_int = round_ste(x / scale_view) + zp_view
    x_int = torch.clamp(x_int, quant_min, quant_max)
    x_quant = (x_int - zp_view) * scale_view
    return x_quant
