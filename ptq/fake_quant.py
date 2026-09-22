"""
FakeQuantize 模块：在前向传播时模拟量化效果

支持的 FakeQuantize:
- FixedFakeQuantize: 固定 scale，round-to-nearest (标准 PTQ)
- AdaRoundFakeQuantize: 学习最优取整方向 (AdaRound, ICML 2020)
"""
import torch
import torch.nn as nn
from .observer import ObserverBase, MinMaxObserver
from .util_quant import (
    fake_quantize_per_tensor_affine,
    fake_quantize_per_channel_affine,
)


class FakeQuantizeBase(nn.Module):
    """
    FakeQuantize 基类

    工作流程:
    1. observer_enabled=1 时: 调用 observer 统计数据分布，更新 scale/zero_point
    2. fake_quant_enabled=1 时: 执行伪量化 (量化 -> 反量化)
    """

    def __init__(self, observer=MinMaxObserver, bit=8, symmetric=True, ch_axis=-1):
        super().__init__()
        self.observer = observer(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.bit = bit
        self.symmetric = symmetric
        self.ch_axis = ch_axis

        # 控制开关
        self.observer_enabled = 0
        self.fake_quant_enabled = 0

        # 从 observer 获取量化范围
        self.quant_min = self.observer.quant_min
        self.quant_max = self.observer.quant_max

    def set_bit(self, bit):
        """动态修改位宽"""
        self.observer.set_bit(bit)
        self.bit = bit
        self.quant_min = self.observer.quant_min
        self.quant_max = self.observer.quant_max

    def set_name(self, name):
        """设置名称（用于调试）"""
        self.name = name

    @torch.jit.export
    def enable_observer(self):
        """启用 observer (校准模式)"""
        self.observer_enabled = 1

    @torch.jit.export
    def disable_observer(self):
        """禁用 observer"""
        self.observer_enabled = 0

    @torch.jit.export
    def enable_fake_quant(self):
        """启用伪量化"""
        self.fake_quant_enabled = 1

    @torch.jit.export
    def disable_fake_quant(self):
        """禁用伪量化"""
        self.fake_quant_enabled = 0

    @torch.jit.export
    def extra_repr(self):
        return (
            f"fake_quant_enabled={self.fake_quant_enabled}, "
            f"observer_enabled={self.observer_enabled}, "
            f"symmetric={self.symmetric}, bit={self.bit}, "
            f"ch_axis={self.ch_axis}, "
            f"quant_min={self.quant_min}, quant_max={self.quant_max}"
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """保存 scale 和 zero_point 到 state_dict"""
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "scale"] = self.scale
        destination[prefix + "zero_point"] = self.zero_point

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs
    ):
        """从 state_dict 加载 scale 和 zero_point"""
        local_state = ["scale", "zero_point"]
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                if name == "scale":
                    if isinstance(self.scale, nn.Parameter):
                        self.scale.data = torch.ones_like(val.to(self.scale.device))
                    else:
                        self.scale.resize_(val.shape)
                else:
                    if isinstance(self.zero_point, nn.Parameter):
                        self.zero_point.data = torch.ones_like(val.to(self.zero_point.device))
                    else:
                        self.zero_point.resize_(val.shape)

                if torch.jit.is_scripting():
                    if name == "scale":
                        self.scale.copy_(val)
                    else:
                        self.zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )


class FixedFakeQuantize(FakeQuantizeBase):
    """
    固定 scale 的 FakeQuantize (标准 PTQ)

    校准阶段:
    - observer_enabled=1: 调用 observer 统计数据，计算 scale

    推理阶段:
    - fake_quant_enabled=1: 使用固定 scale 进行伪量化
    """

    def __init__(self, observer=MinMaxObserver, bit=8, symmetric=True, ch_axis=-1):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.register_buffer("scale", torch.tensor([1.0], dtype=torch.float))
        self.register_buffer("zero_point", torch.tensor([0], dtype=torch.int))

    def forward(self, X):
        # 校准: 更新 scale
        if self.observer_enabled == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.observer.calculate_qparams(
                self.observer.min_val, self.observer.max_val
            )
            _scale = _scale.to(self.scale.device)
            _zero_point = _zero_point.to(self.zero_point.device)

            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        # 伪量化
        if self.fake_quant_enabled == 1:
            if self.ch_axis != -1:
                X = fake_quantize_per_channel_affine(
                    X, self.scale.data, self.zero_point.data.int(),
                    self.ch_axis, self.quant_min, self.quant_max
                )
            else:
                X = fake_quantize_per_tensor_affine(
                    X, self.scale.item(), self.zero_point.item(),
                    self.quant_min, self.quant_max
                )
        return X


class AdaRoundFakeQuantize(FakeQuantizeBase):
    """
    AdaRound FakeQuantize: 学习最优取整方向 (up or down)

    两种工作模式:
    - adaround=False: 等价于 FixedFakeQuantize (round-to-nearest)
    - adaround=True:  使用学习到的 alpha 参数决定取整方向

    Reference: Nagel et al., "Up or Down? Adaptive Rounding for
    Post-Training Quantization", ICML 2020
    """

    def __init__(self, observer=MinMaxObserver, bit=8, symmetric=True, ch_axis=-1):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.register_buffer("scale", torch.tensor([1.0], dtype=torch.float))
        self.register_buffer("zero_point", torch.tensor([0], dtype=torch.int))
        self.adaround = False
        self.hard_value = False
        self.gamma, self.zeta = -0.1, 1.1

    def init(self, weight_tensor: torch.Tensor, round_mode='learned_hard_sigmoid'):
        """初始化 AdaRound, 从 weight_tensor 计算 alpha 参数"""
        self.adaround = True
        self.hard_value = False
        self.round_mode = round_mode
        self._init_alpha(x=weight_tensor.data.clone().detach())

    def _init_alpha(self, x: torch.Tensor):
        """
        根据权重小数部分初始化 alpha, 使 rectified_sigmoid(alpha) ≈ remainder
        从而初始状态近似 round-to-nearest
        """
        if self.ch_axis != -1:
            new_shape = [1] * len(x.shape)
            new_shape[self.ch_axis] = x.shape[self.ch_axis]
            scale = self.scale.data.reshape(new_shape)
        else:
            scale = self.scale.data
        x_floor = torch.floor(x / scale)
        if self.round_mode == 'learned_hard_sigmoid':
            rest = (x / scale) - x_floor  # [0, 1)
            # h(alpha) = clamp(sigmoid(alpha) * (zeta - gamma) + gamma, 0, 1) = rest
            # => sigmoid(alpha) = (rest - gamma) / (zeta - gamma)
            alpha = -torch.log(
                (self.zeta - self.gamma) / (rest - self.gamma + 1e-6) - 1
            )
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError(f"Unknown round_mode: {self.round_mode}")

    def rectified_sigmoid(self):
        """h(alpha): 连续松弛取整掩码, 值域 [0, 1]"""
        return (
            (self.zeta - self.gamma) * torch.sigmoid(self.alpha) + self.gamma
        ).clamp(0, 1)

    def adaround_forward(self, X, hard_value=False):
        """
        AdaRound 前向: floor(X/s) + h(alpha) 然后 clamp & 反量化

        Args:
            hard_value: True 时使用硬取整 (alpha >= 0 → 1, 否则 → 0)
        """
        if self.ch_axis != -1:
            new_shape = [1] * len(X.shape)
            new_shape[self.ch_axis] = X.shape[self.ch_axis]
            scale = self.scale.data.reshape(new_shape)
            zero_point = self.zero_point.data.int().reshape(new_shape)
        else:
            scale = self.scale.item()
            zero_point = self.zero_point.item()
        X = torch.floor(X / scale)
        if hard_value:
            X += (self.alpha >= 0).float()
        else:
            X += self.rectified_sigmoid()
        X += zero_point
        X = torch.clamp(X, self.quant_min, self.quant_max)
        X = (X - zero_point) * scale
        return X

    def get_hard_value(self, X):
        """优化完成后, 用硬取整获取最终量化权重"""
        return self.adaround_forward(X, hard_value=True)

    def forward(self, X):
        if self.observer_enabled == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.observer.calculate_qparams(
                self.observer.min_val, self.observer.max_val
            )
            _scale = _scale.to(self.scale.device)
            _zero_point = _zero_point.to(self.zero_point.device)

            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled == 1:
            if not self.adaround:
                if self.ch_axis != -1:
                    X = fake_quantize_per_channel_affine(
                        X, self.scale.data, self.zero_point.data.int(),
                        self.ch_axis, self.quant_min, self.quant_max,
                    )
                else:
                    X = fake_quantize_per_tensor_affine(
                        X, self.scale.item(), self.zero_point.item(),
                        self.quant_min, self.quant_max,
                    )
            else:
                if not hasattr(self, 'alpha'):
                    raise RuntimeError("AdaRound enabled but alpha not initialized. Call init() first.")
                if self.round_mode == 'learned_hard_sigmoid':
                    X = self.adaround_forward(X, hard_value=self.hard_value)
                else:
                    raise NotImplementedError(f"Unknown round_mode: {self.round_mode}")
        return X

# FakeQuantize 映射表
FAKEQUANT_MAP = {
    'fixed': FixedFakeQuantize,
    'adaround': AdaRoundFakeQuantize,
    # GPTQ uses fixed round-to-nearest fake quant after weight compensation.
    'gptq': FixedFakeQuantize,
    # BRECQ/QDrop-style baselines optimize weights around a fixed quant grid.
    'brecq': FixedFakeQuantize,
    'qdrop': FixedFakeQuantize,
}


def get_fake_quantize(name='fixed', **kwargs):
    """
    根据名称获取 FakeQuantize 类

    Args:
        name: fake_quantize 名称 ('fixed', 'lsq')
        **kwargs: 传递给 FakeQuantize 的参数

    Returns:
        FakeQuantize 实例
    """
    if name not in FAKEQUANT_MAP:
        raise ValueError(f"Unknown fake_quantize: {name}. Available: {list(FAKEQUANT_MAP.keys())}")
    return FAKEQUANT_MAP[name](**kwargs)
