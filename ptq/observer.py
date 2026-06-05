"""
Observer 模块：用于校准阶段统计数据分布，计算量化参数 (scale, zero_point)

支持的 Observer:
- MinMaxObserver: 取全局 min/max
- AvgMinMaxObserver: 多 batch 平均 min/max
- MSEObserver: 网格搜索最小化 MSE
- PercentileObserver: 使用百分位数裁剪异常值
"""
import torch
import torch.nn as nn
import numpy as np


def _transform_to_ch_axis(x, ch_axis):
    """将张量变换为 (channel, -1) 形状，便于 per-channel 操作"""
    if ch_axis == -1:
        return x
    x_dim = x.size()
    new_axis_list = [i for i in range(len(x_dim))]
    new_axis_list[ch_axis] = 0
    new_axis_list[0] = ch_axis
    x_channel = x.permute(new_axis_list)
    y = torch.flatten(x_channel, start_dim=1)
    return y


class ObserverBase(nn.Module):
    """
    Observer 基类

    Args:
        bit: 量化位宽
        symmetric: 是否对称量化
        ch_axis: per-channel 量化的通道轴，-1 表示 per-tensor
    """

    def __init__(self, bit=8, symmetric=True, ch_axis=-1):
        super().__init__()
        self.bit = bit
        self.symmetric = symmetric
        self.ch_axis = ch_axis
        self.eps = torch.tensor(1e-8, dtype=torch.float32)

        # 量化范围
        if self.symmetric:
            self.quant_min = -2 ** (self.bit - 1)
            self.quant_max = 2 ** (self.bit - 1) - 1
        else:
            self.quant_min = 0
            self.quant_max = 2 ** self.bit - 1

        # 统计值
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def set_bit(self, bit):
        """动态修改位宽"""
        self.bit = bit
        if self.symmetric:
            self.quant_min = -2 ** (self.bit - 1)
            self.quant_max = 2 ** (self.bit - 1) - 1
        else:
            self.quant_min = 0
            self.quant_max = 2 ** self.bit - 1

    def set_name(self, name):
        """设置 observer 名称（用于调试）"""
        self.name = name

    @torch.jit.export
    def calculate_qparams(self, min_val=None, max_val=None):
        """
        根据 min/max 计算量化参数 scale 和 zero_point

        Returns:
            scale: 量化缩放因子
            zero_point: 零点偏移
        """
        if min_val is None:
            min_val = self.min_val
        if max_val is None:
            max_val = self.max_val

        quant_min, quant_max = self.quant_min, self.quant_max
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int, device=device)

        if self.symmetric:
            # 对称量化: scale = max(|min|, |max|) / (qmax - qmin) / 2
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale = torch.max(scale, self.eps.to(device))
        else:
            # 非对称量化
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps.to(device))
            zero_point = quant_min - torch.round(min_val_neg / scale)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)

        return scale, zero_point

    def reset(self):
        """重置统计值"""
        self.min_val.fill_(float("inf"))
        self.max_val.fill_(float("-inf"))


class MinMaxObserver(ObserverBase):
    """
    MinMax Observer: 取整个校准数据集的全局 min/max

    特点: 简单快速，但对异常值敏感
    """

    def __init__(self, bit=8, symmetric=True, ch_axis=-1):
        super().__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)

    def forward(self, x_orig):
        """观测数据，更新 min/max"""
        if x_orig.numel() == 0:
            return x_orig

        x = x_orig.clone().detach().to(self.min_val.dtype)

        if self.ch_axis == -1:
            # per-tensor
            min_val_cur, max_val_cur = torch.aminmax(x)
        else:
            # per-channel
            y = _transform_to_ch_axis(x, self.ch_axis)
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)

        # 更新全局 min/max
        self.min_val = torch.min(self.min_val, min_val_cur)
        self.max_val = torch.max(self.max_val, max_val_cur)


class AvgMinMaxObserver(ObserverBase):
    """
    Average MinMax Observer: 多 batch 平均 min/max

    特点: 比 MinMax 更稳定，减少异常值影响
    """

    def __init__(self, bit=8, symmetric=True, ch_axis=-1):
        super().__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.cnt = 0
        assert self.ch_axis == -1, "AvgMinMaxObserver only supports per-tensor"

    def forward(self, x_orig):
        """观测数据，累计平均 min/max"""
        if x_orig.numel() == 0:
            return x_orig

        x = x_orig.clone().detach().to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch.aminmax(x)

        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = self.min_val * self.cnt + min_val_cur
            self.max_val = self.max_val * self.cnt + max_val_cur

        self.cnt += 1
        self.min_val /= self.cnt
        self.max_val /= self.cnt


class PercentileObserver(ObserverBase):
    """
    Percentile Observer: 使用百分位数裁剪异常值

    Args:
        percentile: 裁剪百分比，例如 0.01 表示裁掉最大/最小的 1%
    """

    def __init__(self, bit=8, symmetric=True, ch_axis=-1, percentile=0.01):
        super().__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.percentile = percentile

    def forward(self, x_orig):
        """观测数据，使用百分位数计算 min/max"""
        if x_orig.numel() == 0:
            return x_orig

        x = x_orig.clone().detach().to(self.min_val.dtype)

        if self.ch_axis == -1:
            # per-tensor
            min_val_cur = torch.quantile(x.flatten(), self.percentile)
            max_val_cur = torch.quantile(x.flatten(), 1 - self.percentile)
        else:
            # per-channel
            y = _transform_to_ch_axis(x, self.ch_axis)
            min_val_cur = torch.quantile(y, self.percentile, dim=1)
            max_val_cur = torch.quantile(y, 1 - self.percentile, dim=1)

        if self.min_val.numel() <= 1 and self.min_val.isinf():
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = torch.min(self.min_val, min_val_cur)
            self.max_val = torch.max(self.max_val, max_val_cur)


class MSEObserver(ObserverBase):
    """
    MSE Observer: 网格搜索最小化量化误差 (MSE)

    特点: 更精确，但比 MinMax 慢
    """

    def __init__(self, bit=8, symmetric=True, ch_axis=-1):
        super().__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.p = 2.4  # Lp 范数的 p 值
        self.num = 100  # 搜索候选数量
        self.one_side_dist = None  # 'pos', 'neg', 'no'

    def lp_loss(self, pred, tgt, p=2.0):
        """计算 Lp 损失"""
        x = (pred - tgt).abs().pow(p)
        if self.ch_axis == -1:
            return x.mean()
        else:
            y = _transform_to_ch_axis(x, self.ch_axis)
            return y.mean(1)

    def _fake_quantize(self, x, scale, zero_point):
        """伪量化（支持 per-channel）"""
        if self.ch_axis != -1 and scale.dim() >= 1 and scale.numel() > 1:
            new_shape = [1] * x.dim()
            new_shape[self.ch_axis] = x.shape[self.ch_axis]
            scale = scale.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        x_int = torch.round(x / scale) + zero_point
        x_int = torch.clamp(x_int, self.quant_min, self.quant_max)
        x_q = (x_int - zero_point) * scale
        return x_q

    def loss_fx(self, x, new_min, new_max):
        """计算给定量化范围的损失"""
        scale, zero_point = self.calculate_qparams(new_min, new_max)
        x_q = self._fake_quantize(x, scale, zero_point)
        score = self.lp_loss(x_q, x, p=self.p)
        return score

    def perform_1D_search(self, x):
        """对称量化的一维搜索"""
        if self.ch_axis != -1:
            y = _transform_to_ch_axis(x, self.ch_axis)
            x_min, x_max = torch.aminmax(y, dim=1)
        else:
            x_min, x_max = torch.aminmax(x)

        xrange = torch.max(x_min.abs(), x_max)
        best_score = torch.zeros_like(x_min) + 1e10
        best_min = x_min.clone()
        best_max = x_max.clone()

        for i in range(1, self.num + 1):
            thres = xrange / self.num * i
            new_min = torch.zeros_like(x_min) if self.one_side_dist == 'pos' else -thres
            new_max = torch.zeros_like(x_max) if self.one_side_dist == 'neg' else thres
            score = self.loss_fx(x, new_min, new_max)
            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(score, best_score)

        return best_min, best_max

    def forward(self, x_orig):
        """观测数据，使用 MSE 搜索最优量化范围"""
        if x_orig.numel() == 0:
            return x_orig

        x = x_orig.clone().detach().to(self.min_val.dtype)

        if self.one_side_dist is None:
            self.one_side_dist = 'pos' if x.min() >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'

        best_min, best_max = self.perform_1D_search(x)

        self.min_val = torch.min(self.min_val, best_min)
        self.max_val = torch.max(self.max_val, best_max)


# Observer 映射表，便于通过名称创建
OBSERVER_MAP = {
    'minmax': MinMaxObserver,
    'avgminmax': AvgMinMaxObserver,
    'percentile': PercentileObserver,
    'mse': MSEObserver,
}


def lsq_unified_scale_optim(w_per_ch, m_per_ch, s_init, k, w_bit, m_bit,
                            num_iters=300, lr=4e-4, log_fn=None):
    """
    LSQ-style gradient optimization of per-channel scale for unified
    weight-membrane quantization.

    Two-step approach (following PTQ4SNN + LSQ):
      Step 1 (caller): compute k via PoT  —  k = round(log2(s_obs / s_w))
      Step 2 (here):   fix k, learn s by Adam, jointly minimizing
          L = L_weight(s) + L_spike(s * 2^k)

    where:
        L_weight = MSE( Q(W, s, w_bit),  W )       — weight reconstruction
        L_spike  = MSE( Q(M, s·2^k, m_bit),  M )   — membrane quant error
                                                       (proxy for spike loss)

    The gradient flows through the STE (Straight-Through Estimator) of
    the round operation, allowing Adam to find a more precise scale than
    discrete grid search.

    Final:
        weight  scale = s
        membrane scale = s * 2^k   (hardware: bit-shift by k)

    Args:
        w_per_ch: [C, Kw] per-channel weight data (flattened)
        m_per_ch: [C, Km] per-channel membrane potential data (flattened)
        s_init:   [C] initial weight scale (from observer calibration)
        k:        [C] PoT exponent (integer tensor, fixed throughout)
        w_bit, m_bit: quantization bit widths
        num_iters: Adam iterations (default 300)
        lr: learning rate (default 4e-4)
        log_fn: optional logging function (e.g. logger.info)

    Returns:
        s_opt: [C] optimized weight scale
    """
    def _ste_round(x):
        return (x.round() - x).detach() + x

    w_qmin = -2 ** (w_bit - 1)
    w_qmax = 2 ** (w_bit - 1) - 1
    m_qmin = -2 ** (m_bit - 1)
    m_qmax = 2 ** (m_bit - 1) - 1

    factor = (2.0 ** k.float()).detach()            # [C], fixed

    s = torch.nn.Parameter(s_init.clone().detach().abs().clamp(min=1e-8))
    optimizer = torch.optim.Adam([s], lr=lr)

    best_loss = float('inf')
    best_s = s.data.clone()

    w_energy = (w_per_ch ** 2).mean().detach().clamp(min=1e-10)
    m_energy = (m_per_ch ** 2).mean().detach().clamp(min=1e-10)

    with torch.enable_grad():
        for it in range(num_iters):
            s.data.abs_().clamp_(min=1e-8)
            optimizer.zero_grad()

            # --- weight reconstruction loss (relative MSE) ---
            sw = s.unsqueeze(1)                         # [C, 1]
            w_q = torch.clamp(_ste_round(w_per_ch / sw), w_qmin, w_qmax) * sw
            L_w = ((w_q - w_per_ch) ** 2).mean() / w_energy

            # --- membrane / spike loss (relative MSE) ---
            sm = (s * factor).unsqueeze(1)              # [C, 1]
            m_q = torch.clamp(_ste_round(m_per_ch / sm), m_qmin, m_qmax) * sm
            L_m = ((m_q - m_per_ch) ** 2).mean() / m_energy

            loss = L_w + L_m
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_s = s.data.abs().clamp(min=1e-8).clone()

            if log_fn and (it % 50 == 0 or it == num_iters - 1):
                log_fn(f"      LSQ iter {it:4d}/{num_iters}: "
                       f"L_w={L_w.item():.6f}  L_m={L_m.item():.6f}  "
                       f"total={loss_val:.6f} (best={best_loss:.6f})")

    return best_s


def get_observer(name='minmax', **kwargs):
    """
    根据名称获取 Observer 类

    Args:
        name: observer 名称 ('minmax', 'avgminmax', 'percentile', 'mse')
        **kwargs: 传递给 Observer 的参数

    Returns:
        Observer 实例
    """
    if name not in OBSERVER_MAP:
        raise ValueError(f"Unknown observer: {name}. Available: {list(OBSERVER_MAP.keys())}")
    return OBSERVER_MAP[name](**kwargs)
