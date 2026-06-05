"""
量化模块：替换原始模块的量化版本

支持的量化模块:
- QuantConv2d: 量化 Conv2d (支持 per-channel)
- QuantLinear: 量化 Linear (支持 per-channel)
- QuantBatchNorm2d: 量化 BatchNorm2d

量化方式: Per-channel 对称量化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .observer import MinMaxObserver, ObserverBase
from .fake_quant import FixedFakeQuantize


def unpack_quant(x):
    """解包 QuantConv2d 的输出: (tensor, scale) → tensor, scale; tensor → tensor, None"""
    if isinstance(x, tuple):
        return x[0], x[1]
    return x, None


def repack_quant(x, scale):
    """为 QuantMultiStepLIFNode 打包输入: 有 scale 时返回元组，否则返回原 tensor"""
    if scale is not None:
        return (x, scale)
    return x


class QuantConv2d(nn.Module):
    """
    量化 Conv2d 模块

    权重量化通过 weight_fake_quant (FakeQuantize 实例) 实现,
    支持 FixedFakeQuantize (RTN) 和 AdaRoundFakeQuantize 等.

    Args:
        original_conv: 原始 Conv2d 模块
        bit: 量化位宽
        observer: Observer 类
        fake_quant: FakeQuantize 类
        quant_act: 是否量化激活
    """

    def __init__(
        self,
        original_conv,
        bit=4,
        observer=MinMaxObserver,
        fake_quant=FixedFakeQuantize,
        quant_act=False,
    ):
        super().__init__()

        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups
        self.padding_mode = original_conv.padding_mode

        self.weight = nn.Parameter(original_conv.weight.clone())
        if original_conv.bias is not None:
            self.bias = nn.Parameter(original_conv.bias.clone())
        else:
            self.register_parameter('bias', None)

        self.bit = bit
        self.fake_quant_enabled = False
        self.calibrated = False
        self.observer_cls = observer

        # 权重 FakeQuantize (per-channel, ch_axis=0)
        self.weight_fake_quant = fake_quant(
            observer=observer, bit=bit, symmetric=True, ch_axis=0
        )

        self.quant_act = quant_act
        if quant_act:
            self.act_fake_quant = fake_quant(
                observer=observer, bit=bit, symmetric=True, ch_axis=-1
            )

    @property
    def scale(self):
        return self.weight_fake_quant.scale

    @property
    def zero_point(self):
        return self.weight_fake_quant.zero_point

    @property
    def quant_min(self):
        return self.weight_fake_quant.quant_min

    @property
    def quant_max(self):
        return self.weight_fake_quant.quant_max

    def calibrate(self):
        """校准：通过 weight_fake_quant 的 observer 计算 scale"""
        with torch.no_grad():
            self.weight_fake_quant.to(self.weight.device)
            self.weight_fake_quant.enable_observer()
            self.weight_fake_quant(self.weight)
            self.weight_fake_quant.disable_observer()
            self.calibrated = True

    def forward(self, x):
        if self.quant_act and hasattr(self, 'act_fake_quant'):
            x = self.act_fake_quant(x)

        if self.fake_quant_enabled and self.calibrated:
            w_quant = self.weight_fake_quant(self.weight)
            out = F.conv2d(
                x, w_quant, self.bias,
                self.stride, self.padding,
                self.dilation, self.groups
            )
        else:
            out = F.conv2d(
                x, self.weight, self.bias,
                self.stride, self.padding,
                self.dilation, self.groups
            )
        return (out, self.weight_fake_quant.scale.data)

    def enable_fake_quant(self):
        self.fake_quant_enabled = True
        self.weight_fake_quant.enable_fake_quant()
        if self.quant_act and hasattr(self, 'act_fake_quant'):
            self.act_fake_quant.enable_fake_quant()

    def disable_fake_quant(self):
        self.fake_quant_enabled = False
        self.weight_fake_quant.disable_fake_quant()
        if self.quant_act and hasattr(self, 'act_fake_quant'):
            self.act_fake_quant.disable_fake_quant()

    def enable_observer(self):
        if self.quant_act and hasattr(self, 'act_fake_quant'):
            self.act_fake_quant.enable_observer()

    def disable_observer(self):
        if self.quant_act and hasattr(self, 'act_fake_quant'):
            self.act_fake_quant.disable_observer()

    def extra_repr(self):
        return f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}'

    def __repr__(self):
        fq_name = type(self.weight_fake_quant).__name__
        main_str = f'QuantConv2d(\n'
        main_str += f'  {self.in_channels}, {self.out_channels}, '
        main_str += f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}\n'
        main_str += f'  (weight_fake_quant): {fq_name}(\n'
        main_str += f'    bit={self.bit}, fake_quant_enabled={1 if self.fake_quant_enabled else 0}, '
        main_str += f'calibrated={1 if self.calibrated else 0}, '
        main_str += f'quant_min={self.quant_min}, quant_max={self.quant_max}\n'
        main_str += f'    num_scales={self.out_channels}\n'
        main_str += f'  )\n'
        main_str += f')'
        return main_str


class QuantLinear(nn.Module):
    """
    量化 Linear 模块

    权重量化通过 weight_fake_quant (FakeQuantize 实例) 实现,
    支持 FixedFakeQuantize (RTN) 和 AdaRoundFakeQuantize 等.

    Args:
        original_linear: 原始 Linear 模块
        bit: 量化位宽
        observer: Observer 类
        fake_quant: FakeQuantize 类
        quant_act: 是否量化激活
    """

    def __init__(
        self,
        original_linear,
        bit=4,
        observer=MinMaxObserver,
        fake_quant=FixedFakeQuantize,
        quant_act=False,
    ):
        super().__init__()

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        self.weight = nn.Parameter(original_linear.weight.clone())
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.clone())
        else:
            self.register_parameter('bias', None)

        self.bit = bit
        self.fake_quant_enabled = False
        self.calibrated = False
        self.observer_cls = observer

        # 权重 FakeQuantize (per-channel, ch_axis=0)
        self.weight_fake_quant = fake_quant(
            observer=observer, bit=bit, symmetric=True, ch_axis=0
        )

        self.quant_act = quant_act
        if quant_act:
            self.act_fake_quant = fake_quant(
                observer=observer, bit=bit, symmetric=True, ch_axis=-1
            )

    @property
    def scale(self):
        return self.weight_fake_quant.scale

    @property
    def zero_point(self):
        return self.weight_fake_quant.zero_point

    @property
    def quant_min(self):
        return self.weight_fake_quant.quant_min

    @property
    def quant_max(self):
        return self.weight_fake_quant.quant_max

    def calibrate(self):
        """校准：通过 weight_fake_quant 的 observer 计算 scale"""
        with torch.no_grad():
            self.weight_fake_quant.to(self.weight.device)
            self.weight_fake_quant.enable_observer()
            self.weight_fake_quant(self.weight)
            self.weight_fake_quant.disable_observer()
            self.calibrated = True

    def forward(self, x):
        if self.quant_act and hasattr(self, 'act_fake_quant'):
            x = self.act_fake_quant(x)

        if self.fake_quant_enabled and self.calibrated:
            w_quant = self.weight_fake_quant(self.weight)
            return F.linear(x, w_quant, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)

    def enable_fake_quant(self):
        self.fake_quant_enabled = True
        self.weight_fake_quant.enable_fake_quant()
        if self.quant_act and hasattr(self, 'act_fake_quant'):
            self.act_fake_quant.enable_fake_quant()

    def disable_fake_quant(self):
        self.fake_quant_enabled = False
        self.weight_fake_quant.disable_fake_quant()
        if self.quant_act and hasattr(self, 'act_fake_quant'):
            self.act_fake_quant.disable_fake_quant()

    def enable_observer(self):
        if self.quant_act and hasattr(self, 'act_fake_quant'):
            self.act_fake_quant.enable_observer()

    def disable_observer(self):
        if self.quant_act and hasattr(self, 'act_fake_quant'):
            self.act_fake_quant.disable_observer()

    def extra_repr(self):
        return f'{self.in_features}, {self.out_features}'

    def __repr__(self):
        fq_name = type(self.weight_fake_quant).__name__
        main_str = f'QuantLinear(\n'
        main_str += f'  {self.in_features}, {self.out_features}\n'
        main_str += f'  (weight_fake_quant): {fq_name}(\n'
        main_str += f'    bit={self.bit}, fake_quant_enabled={1 if self.fake_quant_enabled else 0}, '
        main_str += f'calibrated={1 if self.calibrated else 0}, '
        main_str += f'quant_min={self.quant_min}, quant_max={self.quant_max}\n'
        main_str += f'    num_scales={self.out_features}\n'
        main_str += f'  )\n'
        main_str += f')'
        return main_str


class QuantMultiStepLIFNode(nn.Module):
    """
    量化版 MultiStepLIFNode — 逐时间步执行 LIF，每步之后量化膜电位

    原始 MultiStepLIFNode 的 cupy 后端把 T 步融合到一个 kernel 里，
    无法在步间插入量化。因此本模块用纯 Python 循环代替：
        for t in T:
            charge → fire → reset  (BaseNode.forward)
            v = fake_quant(v)      ← 量化膜电位

    量化位置与 ptq4snn 一致：charge → fire → reset → **quantize v**
    使用 FixedFakeQuantize (per-channel, symmetric) 做 round-to-nearest。
    ch_axis=1 对应膜电位 v 的通道维 ([B, C, ...])。

    膜电位 scale 来源（模仿 PTQ4SNN 的 QNeuron）:
    - 若输入是 (tensor, weight_scale) 元组: 直接使用 weight_scale，
      与 PTQ4SNN 的 ``input, scale = input; self.scale = scale`` 一致。
    - 若输入是普通 tensor: 使用自身 Observer 校准的 scale。

    Args:
        original_lif: 原始 MultiStepLIFNode / MultiStepParametricLIFNode
        mem_bit: 膜电位量化位宽
        observer: Observer 类
    """

    def __init__(self, original_lif, mem_bit=4, observer=MinMaxObserver):
        super().__init__()
        self.original_lif = original_lif
        self.mem_bit = mem_bit
        self.mem_fake_quant = FixedFakeQuantize(
            observer=observer, bit=mem_bit, symmetric=True, ch_axis=1
        )
        self.fake_quant_enabled = False
        self.has_weight_scale = False
        self._pot_calibrating = False
        self._stored_weight_scale = None
        self._pot_k = None
        self._observer_calibrated = False
        self._record_membrane = False
        self._mem_records = None
        # Membrane-aware reconstruction records v_pre before the LIF fire step.
        self._record_v_pre = False
        self._record_detach = True
        self._record_to_cpu = True
        self._last_v_pre = None

    def _apply_weight_scale(self, scale):
        """将权重 scale 写入 mem_fake_quant，同时关闭 observer（不再需要）。

        四种行为:
        1. _pot_calibrating=True: 仅存储 weight scale，保持 observer 运行（校准阶段）
        2. _observer_calibrated=True: observer 模式已校准，保留 observer scale 不覆盖
        3. _pot_k is not None: 应用 PoT bridging: s_mem = s_w * 2^k（推理/优化阶段）
        4. 默认: 直接使用 weight scale（PTQ4SNN 原始行为）
        """
        if self._pot_calibrating:
            self._stored_weight_scale = scale.clone()
            self.has_weight_scale = True
            return

        if self._observer_calibrated:
            return

        if self._pot_k is not None:
            scale = scale * (2.0 ** self._pot_k.to(scale.device))

        tgt = self.mem_fake_quant
        if tgt.scale.shape != scale.shape:
            tgt.scale.resize_(scale.shape)
            tgt.zero_point.resize_(scale.shape)
        tgt.scale.data.copy_(scale)
        tgt.zero_point.data.zero_()
        tgt.disable_observer()
        self.has_weight_scale = True

    def _single_step_forward(self, x_t):
        """分解 BaseNode.forward: charge → fire → reset，方便抓取 pre-fire 膜电位。

        等价于 spikingjelly.BaseNode.forward，但在 fire 之前可选地把 v 克隆到
        ``self._last_v_pre`` 用于膜电位感知重构。
        """
        lif = self.original_lif
        lif.neuronal_charge(x_t)
        v_pre_clone = None
        if self._record_v_pre:
            v_pre_clone = lif.v.detach() if self._record_detach else lif.v

        spike = lif.neuronal_fire()
        lif.neuronal_reset(spike)
        return spike, v_pre_clone

    def forward(self, x_seq):
        # ---- 模仿 PTQ4SNN: input, scale = input ----
        if isinstance(x_seq, tuple):
            x_seq, weight_scale = x_seq
            self._apply_weight_scale(weight_scale)

        assert x_seq.dim() > 1
        lif = self.original_lif

        if isinstance(lif.v, float):
            v_init = lif.v
            lif.v = torch.zeros_like(x_seq[0].data)
            if v_init != 0.:
                lif.v.fill_(v_init)

        spike_seq = []
        v_seq = []
        v_pre_rec = [] if self._record_v_pre else None

        for t in range(x_seq.shape[0]):
            spike_t, v_pre_t = self._single_step_forward(x_seq[t])

            if v_pre_rec is not None and v_pre_t is not None:
                v_pre_rec.append(v_pre_t.cpu() if self._record_to_cpu else v_pre_t)

            if lif.v is not None:
                self.mem_fake_quant.to(lif.v.device)
                if self._record_membrane:
                    v_raw = lif.v.detach()
                lif.v = self.mem_fake_quant(lif.v)
                if self._record_membrane:
                    self._accumulate_record(v_raw, lif.v.detach())

            spike_seq.append(spike_t.unsqueeze(0))
            v_seq.append(lif.v.unsqueeze(0))

        if v_pre_rec:
            self._last_v_pre = torch.stack(v_pre_rec, 0)

        spike_seq = torch.cat(spike_seq, 0)
        lif.v_seq = torch.cat(v_seq, 0)

        return spike_seq

    def start_recording_firing(self, record_v_pre=True, record_spike=True,
                               detach=True, to_cpu=True):
        """Enable capturing pre-fire membrane in the next forward(s)."""
        self._record_v_pre = record_v_pre
        self._record_detach = detach
        self._record_to_cpu = to_cpu
        self._last_v_pre = None

    def stop_recording_firing(self):
        self._record_v_pre = False
        self._record_detach = True
        self._record_to_cpu = True
        self._last_v_pre = None

    def get_last_v_pre(self):
        return self._last_v_pre

    def _accumulate_record(self, v_raw, v_quant):
        """累积 per-channel 统计量（running sum / sum_sq / min / max / count）"""
        if v_raw.dim() < 2:
            return
        ch = v_raw.shape[1]
        raw_flat = v_raw.transpose(0, 1).reshape(ch, -1)
        q_flat = v_quant.transpose(0, 1).reshape(ch, -1)
        n = raw_flat.shape[1]

        if self._mem_records is None:
            self._mem_records = {
                "raw_sum": torch.zeros(ch, device="cpu", dtype=torch.float64),
                "raw_sq":  torch.zeros(ch, device="cpu", dtype=torch.float64),
                "raw_min": torch.full((ch,), float("inf"), device="cpu", dtype=torch.float64),
                "raw_max": torch.full((ch,), float("-inf"), device="cpu", dtype=torch.float64),
                "q_sum":   torch.zeros(ch, device="cpu", dtype=torch.float64),
                "q_sq":    torch.zeros(ch, device="cpu", dtype=torch.float64),
                "q_min":   torch.full((ch,), float("inf"), device="cpu", dtype=torch.float64),
                "q_max":   torch.full((ch,), float("-inf"), device="cpu", dtype=torch.float64),
                "err_sq":  torch.zeros(ch, device="cpu", dtype=torch.float64),
                "err_max": torch.zeros(ch, device="cpu", dtype=torch.float64),
                "n_changed": torch.zeros(ch, device="cpu", dtype=torch.int64),
                "count":   0,
            }
        r = self._mem_records
        raw_c = raw_flat.cpu().double()
        q_c = q_flat.cpu().double()
        diff = (raw_c - q_c)

        r["raw_sum"] += raw_c.sum(1)
        r["raw_sq"]  += (raw_c ** 2).sum(1)
        r["raw_min"] = torch.min(r["raw_min"], raw_c.min(1).values)
        r["raw_max"] = torch.max(r["raw_max"], raw_c.max(1).values)
        r["q_sum"]   += q_c.sum(1)
        r["q_sq"]    += (q_c ** 2).sum(1)
        r["q_min"]   = torch.min(r["q_min"], q_c.min(1).values)
        r["q_max"]   = torch.max(r["q_max"], q_c.max(1).values)
        r["err_sq"]  += (diff ** 2).sum(1)
        r["err_max"] = torch.max(r["err_max"], diff.abs().max(1).values)
        r["n_changed"] += (diff.abs() > 1e-10).sum(1)
        r["count"]   += n

    def start_recording(self):
        self._record_membrane = True
        self._mem_records = None

    def stop_recording(self):
        self._record_membrane = False

    def get_record_summary(self):
        """返回 dict，可直接 json.dump"""
        r = self._mem_records
        if r is None or r["count"] == 0:
            return None
        n = r["count"]
        raw_mean = r["raw_sum"] / n
        q_mean   = r["q_sum"] / n
        mse      = r["err_sq"] / n
        frac_changed = r["n_changed"].float() / n

        def _l(t):
            return [round(v, 8) for v in t.tolist()]

        return {
            "mem_bit": self.mem_bit,
            "fake_quant_enabled": int(self.mem_fake_quant.fake_quant_enabled),
            "scale": _l(self.mem_fake_quant.scale.data.cpu().float()),
            "num_channels": int(raw_mean.numel()),
            "num_samples_per_ch": n,
            "raw": {
                "mean": _l(raw_mean), "min": _l(r["raw_min"]),
                "max": _l(r["raw_max"]),
            },
            "quant": {
                "mean": _l(q_mean), "min": _l(r["q_min"]),
                "max": _l(r["q_max"]),
            },
            "error": {
                "mse": _l(mse), "max_abs": _l(r["err_max"]),
                "frac_changed": _l(frac_changed),
            },
        }

    def enable_fake_quant(self):
        self.fake_quant_enabled = True
        self.mem_fake_quant.enable_fake_quant()

    def disable_fake_quant(self):
        self.fake_quant_enabled = False
        self.mem_fake_quant.disable_fake_quant()

    def set_bit(self, bit):
        self.mem_bit = bit
        self.mem_fake_quant.set_bit(bit)


class QuantBatchNorm2d(nn.Module):
    """
    量化版 BatchNorm2d

    注意: 通常 BN 会与 Conv 融合，此类用于未融合的情况
    """

    def __init__(self, original_bn):
        super().__init__()

        self.num_features = original_bn.num_features
        self.eps = original_bn.eps
        self.momentum = original_bn.momentum
        self.affine = original_bn.affine
        self.track_running_stats = original_bn.track_running_stats

        if self.affine:
            self.weight = nn.Parameter(original_bn.weight.clone())
            self.bias = nn.Parameter(original_bn.bias.clone())
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', original_bn.running_mean.clone())
            self.register_buffer('running_var', original_bn.running_var.clone())
            self.register_buffer('num_batches_tracked', original_bn.num_batches_tracked.clone())
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, x):
        return F.batch_norm(
            x, self.running_mean, self.running_var,
            self.weight, self.bias,
            False, self.momentum, self.eps
        )


def fold_bn_into_conv(conv, bn):
    """
    将 BatchNorm 融合到 Conv 中

    公式:
        y = BN(Conv(x))
        y = gamma * (Conv(x) - mean) / sqrt(var + eps) + beta
        y = gamma/sqrt(var+eps) * Conv(x) + (beta - gamma*mean/sqrt(var+eps))

        令 scale = gamma / sqrt(var + eps)
        新权重: w_new = w * scale
        新偏置: b_new = (b - mean) * scale + beta

    Args:
        conv: Conv2d 模块
        bn: BatchNorm2d 模块

    Returns:
        w_new: 融合后的权重
        b_new: 融合后的偏置
    """
    # 获取 BN 参数
    gamma = bn.weight if bn.weight is not None else torch.ones(bn.num_features, device=conv.weight.device)
    beta = bn.bias if bn.bias is not None else torch.zeros(bn.num_features, device=conv.weight.device)
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    # 计算融合系数
    std = torch.sqrt(var + eps)
    scale = gamma / std

    # 融合权重
    w_shape = [1] * len(conv.weight.shape)
    w_shape[0] = -1
    w_new = conv.weight * scale.view(*w_shape)

    # 融合偏置
    if conv.bias is not None:
        b_new = (conv.bias - mean) * scale + beta
    else:
        b_new = -mean * scale + beta

    return w_new, b_new


# 向后兼容的别名
QConv2d = QuantConv2d
QLinear = QuantLinear
QMultiStepLIFNode = QuantMultiStepLIFNode
QBatchNorm2d = QuantBatchNorm2d
