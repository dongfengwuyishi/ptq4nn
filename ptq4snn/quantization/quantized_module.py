from torch import nn
from spikingjelly.activation_based import layer
from .observer import (
    MSEFastObserver,
    MinMaxObserver,
    AvgMinMaxObserver,
    MSEObserver,
    AvgMSEObserver,
    AvgMSEFastObserver,
)
from .fake_quant import (
    AdaRoundFakeQuantize,
    FixedFakeQuantize,
    LSQFakeQuantize,
    LSQPlusFakeQuantize,
)
import torch.nn.functional as F
import torch
from ptq4snn.model.common import LIFNeuron, atan
from ptq4snn.model.resnet import BasicBlock, Bottleneck
from ptq4snn.quantization.util_quant import round_ste


ObserverDict = {
    "MinMaxObserver": MinMaxObserver,  # noqa: E241
    "AvgMinMaxObserver": AvgMinMaxObserver,  # noqa: E241
    "MSEObserver": MSEObserver,  # noqa: E241
    "AvgMSEObserver": AvgMSEObserver,  # noqa: E241
    "MSEFastObserver": MSEFastObserver,  # noqa: E241
    "AvgMSEFastObserver": AvgMSEFastObserver,  # noqa: E241
}

FakeQuantizeDict = {
    "FixedFakeQuantize": FixedFakeQuantize,  # noqa: E241
    "LSQFakeQuantize": LSQFakeQuantize,  # noqa: E241
    "LSQPlusFakeQuantize": LSQPlusFakeQuantize,  # noqa: E241
    "AdaRoundFakeQuantize": AdaRoundFakeQuantize,  # noqa: E241
}


def WeightQuantizer(qconfig):
    return FakeQuantizeDict[qconfig.quantizer](
        ObserverDict[qconfig.observer],
        bit=qconfig.bit,
        symmetric=qconfig.symmetric,
        ch_axis=qconfig.ch_axis,
    )


class QuantizedOperator:
    pass


class QConv2d(QuantizedOperator, layer.Conv2d):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode,
        step_mode,
        qconfig,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            step_mode=step_mode,
        )
        self.weight_fake_quant = WeightQuantizer(qconfig)

    def forward(self, x: torch.Tensor):
        w_quant = self.weight_fake_quant(self.weight)
        b_quant = self.bias
        if self.weight_fake_quant.fake_quant_enabled:
            b_quant = (
                round_ste(self.bias / self.weight_fake_quant.scale)
                * self.weight_fake_quant.scale
            )

        # b_quant = self.bias
        if self.step_mode == "s":
            x = F.conv2d(
                x,
                w_quant,
                b_quant,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        elif self.step_mode == "m":
            if x.dim() != 5:
                raise ValueError(
                    f"expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!"
                )
            T, N, C, H, W = x.shape
            x = x.reshape(T * N, C, H, W)
            x = F.conv2d(
                x,
                w_quant,
                b_quant,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            TN, C2, H2, W2 = x.shape
            x = x.reshape(T, N, C2, H2, W2)
        return (x, self.weight_fake_quant.scale.data)


class QLinear(QuantizedOperator, layer.Linear):

    def __init__(self, in_features, out_features, bias, step_mode, qconfig):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            step_mode=step_mode,
        )
        self.weight_fake_quant = WeightQuantizer(qconfig)

    def forward(self, input):
        w_quant = self.weight_fake_quant(self.weight)
        if self.step_mode == "s":
            return F.linear(input, w_quant, self.bias)
        elif self.step_mode == "m":
            if input.dim() != 3:
                raise ValueError(
                    f"expected input with shape [T, N, C], but got input with shape {input.shape}!"
                )
            T, N, C = input.shape
            input = input.reshape(T * N, C)
            output = F.linear(input, w_quant, self.bias)
            TN, C2 = output.shape
            output = output.reshape(T, N, C2)
            return output


class QNeuron(QuantizedOperator, LIFNeuron):
    def __init__(
        self, tau=2.0, v_threshold=1.0, step_mode="s", is_if=False, qconfig={}
    ):
        super().__init__(
            tau=tau, v_threshold=v_threshold, step_mode=step_mode, is_if=is_if
        )

        self.fake_quant_enabled = 0
        self.observer_enabled = 0

        try:
            self.factor_init_by_observe = qconfig.factor_init_by_observe
        except:
            self.factor_init_by_observe = False

        try:
            self.limit_mem_range = qconfig.limit_mem_range
        except:
            self.limit_mem_range = True

        try:
            self.idea1 = qconfig.idea1
        except:
            self.idea1 = True  # 默认启用 idea1

        self.scale_mode = getattr(qconfig, "scale_mode", "bridge")

        self.n_bits = qconfig.bit
        # 支持逐通道bit：初始化为标量，set_bit中可设置为tensor[C]
        self.max_val = 2 ** (self.n_bits - 1) - 1
        self.min_val = -(2 ** (self.n_bits - 1))
        self.scale = 1.0

        self.use_learnable_factor = qconfig.learnable_factor
        self.factor_should_init = True
        self.ch_axis = qconfig.ch_axis
        self.factor_exp = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.factor = torch.ones(1, dtype=torch.float32)
        self.saturation_count = 0
        self.quantized_value_count = 0
        # 逐通道控制是否用 observer 来初始化 factor：None 表示全局开关
        self.factor_init_mask = None
        # FlowTune calibrates the recurrent membrane state U[t].  These
        # transient fields are enabled only during FlowQ calibration.
        self._flowq_record_enabled = False
        self._flowq_records = {}

    def _record_flowq_membrane(self, timestep=0):
        if not self._flowq_record_enabled or not torch.is_tensor(self.mem):
            return
        flat = self.mem.detach().reshape(-1)
        if flat.numel() > 8192:
            indices = torch.randperm(flat.numel(), device=flat.device)[:8192]
            flat = flat[indices]
        self._flowq_records.setdefault(int(timestep), []).append(flat.cpu())

    def _membrane_quant_scale(self):
        if self.scale_mode == "flowq":
            # FlowQ represents U with alpha_u = alpha_w * 2^k.  factor is
            # 2^-k in this implementation, hence alpha_u = alpha_w / factor.
            return self.scale / self.factor.to(self.scale.device)
        return self.scale

    def set_bit(self, n_bits):
        # n_bits 可为:
        #   - 标量 int（全通道同一bit）
        #   - list/tuple/np.ndarray/torch.Tensor（一维逐通道bit）
        import torch as _torch
        try:
            import numpy as _np  # noqa: F401
        except Exception:
            _np = None

        # 判断“类似数组”的情况：有 __len__ 并且不是单个标量 int/bool
        is_array_like = not isinstance(n_bits, (int, bool)) and hasattr(n_bits, "__len__")

        if isinstance(n_bits, _torch.Tensor) or is_array_like:
            # 统一转成 tensor，支持 list/tuple/np.ndarray 等
            n_bits_t = _torch.as_tensor(n_bits, dtype=_torch.int32, device=self.factor_exp.device)
            self.n_bits = n_bits_t
            # 元素级计算范围：每个通道一个 min/max
            self.max_val = (_torch.pow(2, (n_bits_t - 1)) - 1).to(_torch.float32)
            self.min_val = (-( _torch.pow(2, (n_bits_t - 1)))).to(_torch.float32)
        else:
            # 标量bit
            self.n_bits = int(n_bits)
            self.max_val = 2 ** (self.n_bits - 1) - 1
            self.min_val = -(2 ** (self.n_bits - 1))

    def set_factor_init_mask(self, mask):
        """
        设置逐通道的 factor_init_by_observe 开关。
        mask 可以是标量 bool，或一维张量/数组（长度=C）。
        True 表示该通道使用 observer 计算 factor，False 则禁用。
        """
        import torch as _torch
        is_array_like = not isinstance(mask, (int, bool)) and hasattr(mask, "__len__")
        if isinstance(mask, _torch.Tensor) or is_array_like:
            mask_t = _torch.as_tensor(mask, dtype=_torch.bool, device=self.factor_exp.device)
            self.factor_init_mask = mask_t
        else:
            self.factor_init_mask = bool(mask)

    @torch.jit.export
    def disable_fake_quant(self):
        self.fake_quant_enabled = 0

    @torch.jit.export
    def enable_fake_quant(self):
        self.fake_quant_enabled = 1

    @torch.jit.export
    def disable_observer(self):
        self.observer_enabled = 0

    @torch.jit.export
    def enable_observer(self):
        self.observer_enabled = 1

    @torch.jit.export
    def disable_limit_mem_range(self):
        self.limit_mem_range = 0

    @torch.jit.export
    def enable_limit_mem_range(self):
        self.limit_mem_range = 1

    def neuronal_charge(self, x):
        # 如果 idea1 为 False，直接设置 factor_exp = 0（即 factor = 1），不进行任何判断
        if not getattr(self, "idea1", True):
            # 设置 factor_exp 为全 0，确保形状正确
            device_target = x.device
            if torch.is_tensor(self.max_val):
                factor_exp_shape = self.max_val.shape
            else:
                factor_exp_shape = (1,)

            # 设置 factor_exp 为全 0
            if self.factor_exp.shape != factor_exp_shape:
                self.factor_exp.data = torch.zeros(factor_exp_shape, dtype=torch.float32, device=device_target)
            else:
                self.factor_exp.data.zero_()
            self.factor = 2**self.factor_exp  # factor = 1
        elif self.observer_enabled:
            # 逐通道或全局的 factor_init_by_observe 控制
            device_scale = self.scale.device if torch.is_tensor(self.scale) else x.device
            scale_t = self.scale if torch.is_tensor(self.scale) else torch.tensor(self.scale, dtype=torch.float32, device=device_scale)
            thresh_integer = self.thresh / scale_t
            max_val_to_save = (thresh_integer * self.leak)
            # self.max_val may be scalar or tensor; move to same device
            if torch.is_tensor(self.max_val):
                max_v = self.max_val.to(device_scale)
            else:
                max_v = torch.tensor(self.max_val, dtype=torch.float32, device=device_scale)

            use_mask = getattr(self, "factor_init_mask", None)
            if isinstance(use_mask, torch.Tensor):
                mask = use_mask.to(max_v.device).bool()
            else:
                mask = use_mask  # bool 或 None

            # 先计算所有通道的 tmp 值
            tmp = torch.log2(max_v / max_val_to_save).float()
            if self.scale_mode == "reuse":
                tmp = torch.zeros_like(tmp)
            elif self.scale_mode == "bridge":
                tmp = tmp.round()
            elif self.scale_mode == "flowq":
                # FlowQ fractional precision is calibrated explicitly after
                # weight calibration; do not replace it with a threshold rule.
                tmp = self.factor_exp.detach().to(tmp.device)
            elif self.scale_mode != "observer":
                raise ValueError(f"Unsupported membrane scale_mode: {self.scale_mode}")

            if (isinstance(mask, torch.Tensor) and mask.any()) or (isinstance(mask, bool) and mask):
                # 如果是逐通道 mask，True 的通道使用计算值，False 的通道根据规则处理
                if isinstance(mask, torch.Tensor):
                    # 对齐形状
                    if mask.numel() == tmp.numel():
                        mask_view = mask.view_as(tmp)
                    else:
                        mask_view = mask
                    factor_exp_new = torch.zeros_like(tmp)
                    # True 的通道：直接使用计算值
                    factor_exp_new[mask_view] = tmp[mask_view]
                    # False 的通道：如果 tmp > 0 则置 0，如果 tmp <= 0 则保持 tmp
                    factor_exp_new[~mask_view] = torch.where(
                        tmp[~mask_view] > 0,
                        torch.zeros_like(tmp[~mask_view]),
                        tmp[~mask_view]
                    )
                else:
                    # 全局 mask 为 True：直接使用计算值
                    factor_exp_new = tmp
                self.factor_exp.data = factor_exp_new.to(self.factor_exp.device)
                self.factor = 2**self.factor_exp

                # print(f"factor: {self.factor.data}")
                # print(f"factor_exp: {self.factor_exp.data.item()}")

            else:
                # 全部关闭 observer 初始化：如果 tmp > 0 则置 0，如果 tmp <= 0 则保持 tmp
                factor_exp_new = torch.where(
                    tmp > 0,
                    torch.zeros_like(tmp),
                    tmp
                )
                self.factor_exp.data = factor_exp_new.to(self.factor_exp.device)
                self.factor = 2**self.factor_exp

        if self.fake_quant_enabled:
            if self.use_learnable_factor:
                # update factor
                self.factor = 2 ** round_ste(self.factor_exp)
            self.factor = self.factor.to(x.device)
            if self.scale_mode == "flowq":
                # Keep the weighted input at alpha_w precision.  At integer
                # inference FlowQ left-shifts U^q into the alpha_w accumulator;
                # right-shifting the weighted input would discard information.
                x = round_ste(
                    x / self.scale.view(1, -1, 1, 1)
                ) * self.scale.view(1, -1, 1, 1)
            else:
                x = round_ste(
                    x * self.factor.view(1, -1, 1, 1) / self.scale.view(1, -1, 1, 1)
                ) * self.scale.view(1, -1, 1, 1)

        super().neuronal_charge(x)

    def neuronal_fire(self):
        if self.fake_quant_enabled:
            if self.scale_mode == "flowq":
                return atan.apply(self.mem - self.thresh)
            return atan.apply(self.mem - self.thresh * self.factor.view(1, -1, 1, 1))  #
        else:
            return super().neuronal_fire()

    def single_step_forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        self.v_pre_seq = self.mem.unsqueeze(0)
        spike = self.neuronal_fire()
        if self.is_if:
            self.neuronal_only_reset(spike)
        else:
            self.neuronal_reset_and_leak(spike)

        self._record_flowq_membrane(timestep=0)
        if self.fake_quant_enabled:
            mem_scale = self._membrane_quant_scale()
            # get quantized mem (should be interger in infer)
            self.mem = round_ste(
                self.mem / mem_scale.view(1, -1, 1, 1)
            ) * mem_scale.view(1, -1, 1, 1)

            # save mem in limited range (n bits)
            if self.limit_mem_range:
                # self.min_val/self.max_val 可为标量或[C]
                device_target = self.mem.device
                min_v = self.min_val
                max_v = self.max_val
                if not torch.is_tensor(min_v):
                    min_v = torch.tensor(min_v, dtype=torch.float32)
                if not torch.is_tensor(max_v):
                    max_v = torch.tensor(max_v, dtype=torch.float32)
                min_v = min_v.to(device_target)
                max_v = max_v.to(device_target)

                lower = min_v.view(1, -1, 1, 1) * mem_scale.view(1, -1, 1, 1)
                upper = max_v.view(1, -1, 1, 1) * mem_scale.view(1, -1, 1, 1)
                with torch.no_grad():
                    self.saturation_count += int(((self.mem < lower) | (self.mem > upper)).sum().item())
                    self.quantized_value_count += int(self.mem.numel())

                self.mem = (
                    torch.clamp(
                        self.mem,
                        lower,
                        upper,
                    ).detach()
                    + self.mem
                    - self.mem.detach()
                )
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        y_seq = []
        v_pre_seq = []
        for t in range(x_seq.shape[0]):
            self.neuronal_charge(x_seq[t])
            v_pre_seq.append(self.mem)
            spike = self.neuronal_fire()
            if self.is_if:
                self.neuronal_only_reset(spike)
            else:
                self.neuronal_reset_and_leak(spike)

            self._record_flowq_membrane(timestep=t)
            if self.fake_quant_enabled:
                mem_scale = self._membrane_quant_scale()
                self.mem = round_ste(self.mem / mem_scale.view(1, -1, 1, 1)) * mem_scale.view(1, -1, 1, 1)
                if self.limit_mem_range:
                    device_target = self.mem.device
                    min_v = self.min_val if torch.is_tensor(self.min_val) else torch.tensor(self.min_val, dtype=torch.float32)
                    max_v = self.max_val if torch.is_tensor(self.max_val) else torch.tensor(self.max_val, dtype=torch.float32)
                    min_v = min_v.to(device_target)
                    max_v = max_v.to(device_target)
                    lower = min_v.view(1, -1, 1, 1) * mem_scale.view(1, -1, 1, 1)
                    upper = max_v.view(1, -1, 1, 1) * mem_scale.view(1, -1, 1, 1)
                    with torch.no_grad():
                        self.saturation_count += int(((self.mem < lower) | (self.mem > upper)).sum().item())
                        self.quantized_value_count += int(self.mem.numel())
                    self.mem = torch.clamp(self.mem, lower, upper).detach() + self.mem - self.mem.detach()
            y_seq.append(spike)
        self.v_pre_seq = torch.stack(v_pre_seq)
        return torch.stack(y_seq)

    def forward(self, input):
        input, scale = input
        # use the same scale for weight quant and activation quant, following MINT
        # scale.dim() ==  [c], input.dim() == [t, b, c, h, w] or [b, c, h, w]
        # input is not quantized, as input = spike * w_q + b, b is not quantized
        self.scale = scale
        if self.factor_should_init and self.ch_axis != -1:
            assert (
                self.ch_axis == 0
            ), "only support ch_axis == 0, i.e. per channel quantization"
            if self.factor_exp.shape[0] != input.shape[-3]:
                self.factor_exp.data = torch.zeros(input.shape[-3])
                self.factor_should_init = False
        return super().forward(input)

    def extra_repr(self):
        # 展示 factor_init_by_observe / factor_init_mask 的实际状态
        if isinstance(self.factor_init_mask, torch.Tensor):
            on_cnt = int(self.factor_init_mask.sum().item())
            off_cnt = int(self.factor_init_mask.numel() - on_cnt)
            # if self.factor_init_mask.numel() <= 16:
            #     mask_repr = f"mask_on={on_cnt},off={off_cnt},vals={self.factor_init_mask.tolist()}"
            # else:
            #     mask_repr = f"mask_on={on_cnt},off={off_cnt}"
            mask_repr = f"mask_on={on_cnt},off={off_cnt},vals={self.factor_init_mask.tolist()}"
        else:
            mask_repr = f"{self.factor_init_mask}"

        return (
            super().extra_repr()
            + f", fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, "
            + f"ch_axis={self.ch_axis}, bit={self.n_bits}, quant_min={self.min_val}, quant_max={self.max_val}, "
            + f"use_learnable_factor={self.use_learnable_factor}, limit_mem_range={self.limit_mem_range}, "
            + f"idea1={getattr(self, 'idea1', True)}, factor_init_by_observe={self.factor_init_by_observe}, factor_init_mask={mask_repr}"
            + f", scale_mode={self.scale_mode}"
        )


module_type_to_quant_weight = {
    # nn.Linear: QLinear,
    layer.Linear: QLinear,
    # nn.Conv2d: QConv2d,
    layer.Conv2d: QConv2d,
    LIFNeuron: QNeuron,
}


def get_module_args(module):
    if isinstance(module, layer.Linear):
        return dict(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            step_mode=module.step_mode,
        )
    elif isinstance(module, layer.Conv2d):
        return dict(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            step_mode=module.step_mode,
        )
    elif isinstance(module, LIFNeuron):
        return dict(
            tau=1 / module.leak,
            v_threshold=module.thresh,
            step_mode=module.step_mode,
            is_if=module.is_if,
        )
    else:
        raise NotImplementedError


def Quantizer(module, config):
    module_type = type(module)
    if module_type in module_type_to_quant_weight:
        kwargs = get_module_args(module)
        qmodule = module_type_to_quant_weight[module_type](**kwargs, qconfig=config)
        if getattr(module, "weight", None) is not None:
            qmodule.weight.data = module.weight.data.clone()
        if getattr(module, "bias", None) is not None:
            qmodule.bias.data = module.bias.data.clone()
        return qmodule
    return module


class QuantizedModule(nn.Module):
    def __init__(self):
        super().__init__()


class QuantizedLayer(QuantizedModule):
    def __init__(self, module, neuron, w_qconfig, m_qconfig):
        super().__init__()
        self.module = Quantizer(module, w_qconfig)
        if neuron is not None:
            self.neuron = Quantizer(neuron, m_qconfig)
        else:
            self.neuron = None

    def forward(self, x):
        x = self.module(x)
        if self.neuron is not None:
            x = self.neuron(x)
        return x


class QuantizedBlock(QuantizedModule):
    def __init__(self):
        super().__init__()


class QuantBasicBlock(QuantizedBlock):
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    """

    def __init__(self, org_module: BasicBlock, w_qconfig, m_qconfig):
        super().__init__()
        self.is_sew = org_module.is_sew
        self.conv_lif1 = QuantizedLayer(
            org_module.conv1, org_module.lif1, w_qconfig, m_qconfig
        )
        self.conv_lif2 = QuantizedLayer(
            org_module.conv2,
            org_module.lif2 if self.is_sew else None,
            w_qconfig,
            m_qconfig,
        )
        self.post_add_neuron = (
            None if self.is_sew else Quantizer(org_module.lif2, m_qconfig)
        )
        if org_module.downsample is None or isinstance(
            org_module.downsample, nn.Identity
        ):
            self.downsample = nn.Identity()
        else:
            if self.is_sew:
                self.downsample = QuantizedLayer(
                    org_module.downsample[0],
                    org_module.downsample[2],
                    w_qconfig,
                    m_qconfig,
                )
            else:
                self.downsample = QuantizedLayer(
                    org_module.downsample[0], None, w_qconfig, m_qconfig
                )

    def forward(self, x):
        out = self.conv_lif1(x)
        out = self.conv_lif2(out)
        if self.is_sew:
            residual = self.downsample(x)
            out = out + residual
        else:
            out, scale = out
            residual = self.downsample(x)
            if isinstance(residual, tuple):
                residual = residual[0]
            out = self.post_add_neuron((out + residual, scale))
        return out

    def __repr__(self):
        if self.is_sew:
            return "Sew" + super().__repr__()
        else:
            return super().__repr__()


class QuantBottleneck(QuantizedBlock):
    """
    Implementation of Quantized Bottleneck Block used in ResNet-50, -101 and -152.
    """

    def __init__(self, org_module: Bottleneck, w_qconfig, m_qconfig):
        super().__init__()
        self.is_sew = org_module.is_sew
        self.conv_lif1 = QuantizedLayer(
            org_module.conv1, org_module.lif1, w_qconfig, m_qconfig
        )
        self.conv_lif2 = QuantizedLayer(
            org_module.conv2, org_module.lif2, w_qconfig, m_qconfig
        )
        self.conv_lif3 = QuantizedLayer(
            org_module.conv3, org_module.lif3, w_qconfig, m_qconfig
        )
        if org_module.downsample is None or isinstance(
            org_module.downsample, nn.Identity
        ):
            self.downsample = nn.Identity()
        else:
            if self.is_sew:
                self.downsample = QuantizedLayer(
                    org_module.downsample[0],
                    org_module.downsample[2],
                    w_qconfig,
                    m_qconfig,
                )
            else:
                raise NotImplementedError("Not supported for non-SEW Bottleneck.")

    def forward(self, x):
        out = self.conv_lif1(x)
        out = self.conv_lif2(out)
        out = self.conv_lif3(out)
        if self.is_sew:
            residual = self.downsample(x)
            out = out + residual
        else:
            raise NotImplementedError("Not supported for non-SEW Bottleneck.")
        return out

    def __repr__(self):
        if self.is_sew:
            return "Sew" + super().__repr__()
        else:
            return super().__repr__()


specials = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
}
