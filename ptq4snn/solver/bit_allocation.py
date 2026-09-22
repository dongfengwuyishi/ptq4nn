"""
Activity-based and loss-based bit allocation for SNN quantization.

This module provides functions for:
1. Collecting activity statistics (membrane potential and spike rate) per channel
2. Computing loss-based salience scores (inspired by MixLLM) - global importance of output features
3. Allocating quantization bits per channel based on activity scores or loss salience
4. Calculating weighted average bits

Current method:
- Activity-based: score = w_mem * normalized(avg_membrane_potential) + w_rate * normalized(avg_spike_rate)
- Loss-based (MixLLM-inspired): salience = gradient of loss w.r.t. output features (global view)
"""

import numpy as np
import json
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset

from ptq4snn.model.common import LIFNeuron, reset_net
from ptq4snn.quantization.state import disable_all


def _is_stem_layer(neuron_name):
    """
    检查是否是 stem.0. 层（第一层），需要被排除
    """
    return neuron_name.startswith("stem.0.") or neuron_name == "stem.0"


def compute_loss_salience(model, cali_tensor, logger, cfg):
    """
    计算每个 LIFNeuron 输出通道的损失贡献度（salience），按照论文公式：
    S_c = (1/|D|) Σ_{d∈D} |g_dᵀ(c_q - c_0) + ½(g_dᵀ(c_q - c_0))²|

    其中：
    - g_d: 损失关于通道的梯度（对数据样本d）
    - c_q: 量化后的通道输出
    - c_0: 全精度通道输出
    - (c_q - c_0): 量化误差

    Args:
        model: 模型
        cali_tensor: 校准数据 [N, C, H, W]
        logger: logger对象
        cfg: 配置对象

    Returns:
        salience_map: {neuron_name: salience_tensor[C]}，每个通道的salience分数
    """
    if cali_tensor.size(0) == 0:
        logger.warning("No calibration data provided; skip loss salience computation.")
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orig_device = next(model.parameters()).device
    if orig_device != device:
        model.to(device)
    # 需要梯度，所以使用 train 模式（但禁用 dropout 等）
    model.train()
    # 禁用 dropout 和 batch norm 的训练行为
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.BatchNorm2d)):
            m.eval()
    disable_all(model)

    batch_size = int(getattr(cfg, "salience_batch_size", 16))
    max_batches = getattr(cfg, "salience_batches", 10)
    dataset = TensorDataset(cali_tensor.cpu())
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        drop_last=False,
    )
    target_batches = max_batches if max_batches is not None else len(loader)
    target_batches = max(1, min(target_batches, len(loader)))

    module_to_name = {module: name for name, module in model.named_modules()}

    # 存储每个神经元的salience
    salience_map = {}
    counts = {}

    # 存储全精度输出和量化输出
    neuron_outputs_fp = {}  # 全精度输出 c_0
    neuron_outputs_q = {}   # 量化输出 c_q

    def _make_output_hook_fp(neuron_name):
        def _hook(module, inputs, output):
            # 保存全精度输出
            if isinstance(output, torch.Tensor):
                neuron_outputs_fp[neuron_name] = output.detach().clone()
            return output
        return _hook

    def _make_output_hook_q(neuron_name):
        def _hook(module, inputs, output):
            # 保存量化输出，并保留梯度
            if isinstance(output, torch.Tensor):
                neuron_outputs_q[neuron_name] = output
                if output.requires_grad:
                    output.retain_grad()
            return output
        return _hook

    # 获取模型的输出层，用于计算损失
    def _get_model_output(model, x):
        """获取模型输出，用于计算损失"""
        output = model(x)
        # 如果输出是tuple，取第一个
        if isinstance(output, tuple):
            output = output[0]
        # 如果输出是 [T, N, C] 格式，取最后一个时间步
        if output.dim() == 3 and output.size(0) > 1:
            output = output[-1]  # 取最后一个时间步
        elif output.dim() == 3:
            output = output[0]
        return output

    # 使用交叉熵损失
    criterion = nn.CrossEntropyLoss()

    # 使用4bit量化来计算量化误差（用于计算salience）
    from ptq4snn.quantization.state import enable_quantization, enable_calibration_woquantization
    quant_bit = 4  # 用于计算salience的量化bit

    # 遍历校准数据计算salience
    for batch_idx, (inputs,) in enumerate(loader):
        if batch_idx >= target_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        reset_net(model)

        # 步骤1: 全精度前向传播，保存全精度输出 c_0
        neuron_outputs_fp.clear()
        handles_fp = []
        for module, name in module_to_name.items():
            if isinstance(module, LIFNeuron) and not _is_stem_layer(name):
                handles_fp.append(module.register_forward_hook(_make_output_hook_fp(name)))

        with torch.no_grad():
            _ = model(inputs)
            # 获取伪标签
            model_output = _get_model_output(model, inputs)
            if model_output.dim() == 2 and model_output.size(1) > 1:
                pseudo_labels = model_output.argmax(dim=1)
            else:
                continue

        # 移除全精度hook
        for h in handles_fp:
            h.remove()

        # 步骤2: 启用量化，再次前向传播，保存量化输出 c_q
        # 临时设置所有neuron为quant_bit
        temp_bit_settings = {}
        temp_idea1_settings = {}  # 保存原始的idea1设置
        for module, name in module_to_name.items():
            if isinstance(module, LIFNeuron) and not _is_stem_layer(name):
                if hasattr(module, 'n_bits'):
                    temp_bit_settings[name] = module.n_bits
                    module.set_bit(quant_bit)
                # 统一设置idea1为True，确保salience计算时量化方式一致
                # 这样idea1的不同不会影响salience计算，只影响最终推理时的量化
                if hasattr(module, 'idea1'):
                    temp_idea1_settings[name] = module.idea1
                    module.idea1 = True  # 统一使用idea1=True来计算salience

        # 先启用observer来重新计算scale（因为bit改变了，scale需要重新计算）
        enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
        # 进行一次前向传播来更新scale
        with torch.no_grad():
            _ = model(inputs)
        # 然后禁用observer并启用fake_quant
        enable_quantization(model)
        model.zero_grad()
        neuron_outputs_q.clear()
        inputs.requires_grad_(True)

        # 注册量化hook
        handles_q = []
        for module, name in module_to_name.items():
            if isinstance(module, LIFNeuron) and not _is_stem_layer(name):
                handles_q.append(module.register_forward_hook(_make_output_hook_q(name)))

        # 量化前向传播
        final_output = _get_model_output(model, inputs)
        if not final_output.requires_grad:
            final_output = final_output.requires_grad_(True)

        # 计算损失
        loss = criterion(final_output, pseudo_labels)

        # 反向传播计算梯度
        loss.backward(retain_graph=False)

        # 移除量化hook
        for h in handles_q:
            h.remove()

        # 恢复原始bit设置和idea1设置
        disable_all(model)
        for module, name in module_to_name.items():
            if isinstance(module, LIFNeuron) and name in temp_bit_settings:
                module.set_bit(temp_bit_settings[name])
            if isinstance(module, LIFNeuron) and name in temp_idea1_settings:
                module.idea1 = temp_idea1_settings[name]

        # 步骤3: 按照论文公式计算salience
        # S_c = |g_dᵀ(c_q - c_0) + ½(g_dᵀ(c_q - c_0))²|
        for neuron_name in neuron_outputs_fp.keys():
            if neuron_name not in neuron_outputs_q:
                continue

            fp_output = neuron_outputs_fp[neuron_name]  # c_0: 全精度输出
            q_output = neuron_outputs_q[neuron_name]    # c_q: 量化输出


            # 计算量化误差
            quantization_error = q_output - fp_output  # (c_q - c_0)

            # 获取梯度 g_d
            if not hasattr(q_output, 'grad') or q_output.grad is None:
                continue
            grad = q_output.grad  # g_d

            # 确定通道维度
            if grad.dim() == 5:  # [T, N, C, H, W]
                ch_dim = 2
            elif grad.dim() == 4:  # [N, C, H, W]
                ch_dim = 1
            elif grad.dim() == 3:  # [T, N, C] 或 [N, C, H]
                ch_dim = 2 if grad.size(0) > 1 else 1
            elif grad.dim() == 2:  # [N, C]
                ch_dim = 1
            else:
                ch_dim = max(grad.dim() - 1, 0)

            # 按通道计算salience: 对每个通道，计算 g_dᵀ(c_q - c_0)
            # 需要将grad和quantization_error reshape为 [..., C, ...] 的形式
            dims_to_reduce = [d for d in range(grad.dim()) if d != ch_dim]

            # 将grad和quantization_error reshape为 [C, N_elements_per_channel]
            grad_flat = grad.transpose(ch_dim, 0).contiguous()
            grad_flat = grad_flat.view(grad.shape[ch_dim], -1)  # [C, ...]

            qe_flat = quantization_error.transpose(ch_dim, 0).contiguous()
            qe_flat = qe_flat.view(quantization_error.shape[ch_dim], -1)  # [C, ...]

            # 确保形状匹配
            min_elements = min(grad_flat.shape[1], qe_flat.shape[1])
            grad_flat = grad_flat[:, :min_elements]
            qe_flat = qe_flat[:, :min_elements]

            # 对每个通道计算: g_dᵀ(c_q - c_0) = sum(grad_channel * qe_channel)
            first_order_per_channel = (grad_flat * qe_flat).sum(dim=1)  # [C]

            # ½(g_dᵀ(c_q - c_0))²
            second_order_per_channel = 0.5 * (first_order_per_channel ** 2)

            # S_c = |first_order + second_order|，按通道
            salience_vec = (first_order_per_channel + second_order_per_channel).abs()

            salience_vec = salience_vec.detach().cpu()

            if neuron_name not in salience_map:
                salience_map[neuron_name] = salience_vec.clone()
                counts[neuron_name] = 1
            else:
                stored = salience_map[neuron_name]
                if stored.shape == salience_vec.shape:
                    salience_map[neuron_name] = stored + salience_vec
                else:
                    # 形状不匹配，取较小长度
                    min_len = min(stored.numel(), salience_vec.numel())
                    stored.view(-1)[:min_len] += salience_vec.view(-1)[:min_len]
                counts[neuron_name] += 1

    # 归一化为平均值（排除 stem.0. 层）
    stem_keys = [name for name in salience_map.keys() if _is_stem_layer(name)]
    for key in stem_keys:
        del salience_map[key]
        if key in counts:
            del counts[key]

    for name, c in counts.items():
        if c <= 0 or _is_stem_layer(name):
            continue
        if name in salience_map:
            salience_map[name] = salience_map[name] / float(c)

    if orig_device != device:
        model.to(orig_device)

    logger.info(
        f"[LossSalience] Computed salience for {len(salience_map)} neurons "
        f"across {target_batches} batches."
    )
    return salience_map


def compute_activity_stats(model, cali_tensor, logger, cfg):
    """
    按通道统计每个 LIFNeuron 的平均膜电位和平均脉冲发放率。
    统计在未量化（disable_all）的模型上进行，只做前向传播。
    返回:
        activity_map: {neuron_name: {'mem': tensor[C], 'rate': tensor[C]}}
    """
    if cali_tensor.size(0) == 0:
        logger.warning("No calibration data provided; skip activity stats collection.")
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orig_device = next(model.parameters()).device
    if orig_device != device:
        model.to(device)
    model.eval()
    disable_all(model)

    batch_size = int(getattr(cfg, "activity_batch_size", 32))
    max_batches = getattr(cfg, "activity_batches", None)
    dataset = TensorDataset(cali_tensor.cpu())
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        drop_last=False,
    )
    target_batches = max_batches if max_batches is not None else len(loader)
    target_batches = max(1, target_batches)

    module_to_name = {module: name for name, module in model.named_modules()}

    activity_map = {}
    counts = {}
    handles = []

    def _reduce_per_channel(x: torch.Tensor, channel_dim: int):
        # 统一按通道求均值：剩余维度全部平均
        if x is None:
            return None
        if not isinstance(x, torch.Tensor):
            return None
        dims = [d for d in range(x.dim()) if d != channel_dim]
        if len(dims) == 0:
            return x.detach().float()
        return x.detach().float().mean(dim=dims)

    def _make_forward_hook(neuron_name):
        def _hook(module, inputs, output):
            with torch.no_grad():
                spikes = output
                # 推断通道维度
                if isinstance(spikes, torch.Tensor):
                    if spikes.dim() == 5:  # [T, N, C, H, W]
                        ch_dim_spike = 2
                    elif spikes.dim() == 4:  # [N, C, H, W]
                        ch_dim_spike = 1
                    elif spikes.dim() == 3:  # [T, N, C] 或 [N, C, H]
                        ch_dim_spike = 2
                    elif spikes.dim() == 2:  # [N, C]
                        ch_dim_spike = 1
                    else:
                        ch_dim_spike = max(spikes.dim() - 1, 0)
                    rate_vec = _reduce_per_channel(spikes, ch_dim_spike)
                else:
                    rate_vec = None

                mem = getattr(module, "mem", None)
                if isinstance(mem, torch.Tensor):
                    if mem.dim() == 5:
                        ch_dim_mem = 2
                    elif mem.dim() == 4:
                        ch_dim_mem = 1
                    elif mem.dim() == 3:
                        ch_dim_mem = 1
                    elif mem.dim() == 2:
                        ch_dim_mem = 1
                    else:
                        ch_dim_mem = max(mem.dim() - 1, 0)
                    mem_vec = _reduce_per_channel(mem, ch_dim_mem)
                else:
                    mem_vec = None

                if mem_vec is None and rate_vec is None:
                    return

                if neuron_name not in activity_map:
                    activity_map[neuron_name] = {}
                    counts[neuron_name] = 0

                if mem_vec is not None:
                    mem_vec = mem_vec.cpu()
                    if "mem" not in activity_map[neuron_name]:
                        activity_map[neuron_name]["mem"] = mem_vec.clone()
                    else:
                        stored = activity_map[neuron_name]["mem"]
                        if stored.shape != mem_vec.shape:
                            min_len = min(stored.numel(), mem_vec.numel())
                            stored.view(-1)[:min_len] += mem_vec.view(-1)[:min_len]
                        else:
                            stored += mem_vec

                if rate_vec is not None:
                    rate_vec = rate_vec.cpu()
                    if "rate" not in activity_map[neuron_name]:
                        activity_map[neuron_name]["rate"] = rate_vec.clone()
                    else:
                        stored = activity_map[neuron_name]["rate"]
                        if stored.shape != rate_vec.shape:
                            min_len = min(stored.numel(), rate_vec.numel())
                            stored.view(-1)[:min_len] += rate_vec.view(-1)[:min_len]
                        else:
                            stored += rate_vec

                counts[neuron_name] += 1

        return _hook

    # 注册 hook（排除 stem.0. 层）
    for module, name in module_to_name.items():
        if isinstance(module, LIFNeuron) and not _is_stem_layer(name):
            handles.append(module.register_forward_hook(_make_forward_hook(name)))

    # 前向遍历校准数据
    for batch_idx, (inputs,) in enumerate(loader):
        if batch_idx >= target_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        reset_net(model)
        with torch.no_grad():
            _ = model(inputs)

    # 移除 hook
    for h in handles:
        h.remove()

    # 归一化为平均值（排除 stem.0. 层）
    for name, c in counts.items():
        if c <= 0 or _is_stem_layer(name):
            continue
        stats = activity_map.get(name, {})
        if "mem" in stats:
            stats["mem"] = stats["mem"] / float(c)
        if "rate" in stats:
            stats["rate"] = stats["rate"] / float(c)

    # 从 activity_map 中移除 stem.0. 层的数据
    stem_keys = [name for name in activity_map.keys() if _is_stem_layer(name)]
    for key in stem_keys:
        del activity_map[key]
        if key in counts:
            del counts[key]

    if orig_device != device:
        model.to(orig_device)

    logger.info(
        f"[ActivityStats] Collected activity for {len(activity_map)} neurons "
        f"across {target_batches} batches."
    )
    return activity_map


def allocate_bits_per_channel(activity_map, logger, cfg, model=None, cali_data=None, salience_map=None):
    """
    按通道分配bit：对每个通道计算综合score

    方法1（Activity-based，默认）:
        score = w_mem * 归一化(平均膜电位) + w_rate * 归一化(平均脉冲发放率)

    方法2（Loss-based，借鉴 MixLLM，可选）:
        score = w_salience * 归一化(损失贡献度)
        其中损失贡献度通过计算每个输出通道对最终损失的梯度贡献得到（全局视角）

    方法3（混合，可选）:
        score = w_mem * 归一化(平均膜电位) + w_rate * 归一化(平均脉冲发放率) + w_salience * 归一化(损失贡献度)

    然后在所有通道上根据 score 的分位数分配 2/4/8bit：
        <= p50 -> 2bit
        p50~p75 -> 4bit
        > p75 -> 8bit

    如果提供了model和cali_data，会迭代调整p50阈值，使得加权平均量化比特接近4bit。
    p75阈值固定（4bit和8bit的分界），p50阈值可调整（2bit和4bit的分界，可以是小数）。

    Args:
        activity_map: {neuron_name: {'mem': tensor[C], 'rate': tensor[C]}}
        logger: logger对象
        cfg: 配置对象
        model: 模型（可选，用于计算加权平均）
        cali_data: 校准数据（可选，用于计算加权平均）
        salience_map: {neuron_name: salience_tensor[C]}，损失贡献度（可选，借鉴 MixLLM）
    """
    bit_allocation = {}
    if len(activity_map) == 0:
        logger.warning("[ChannelAlloc] No activity stats; fallback to empty allocation.")
        return bit_allocation

    w_mem = float(getattr(cfg, "activity_mem_weight", 0))
    w_rate = float(getattr(cfg, "activity_rate_weight", 1))
    w_salience = float(getattr(cfg, "loss_salience_weight", 0))  # MixLLM-inspired loss salience weight

    # 如果提供了salience_map，默认启用loss salience（如果权重为0且没有其他权重）
    use_salience = (salience_map is not None and len(salience_map) > 0)
    if use_salience and w_salience == 0 and (w_mem + w_rate) == 0:
        w_salience = 1.0
        logger.info("[ChannelAlloc] Using loss salience (MixLLM method) as primary metric.")

    if (w_mem + w_rate + w_salience) == 0:
        w_mem, w_rate, w_salience = 0.33, 0.33, 0.34

    mem_list = []
    rate_list = []
    salience_list = []

    for neuron_name, stats in activity_map.items():
        # 排除 stem.0. 层
        if _is_stem_layer(neuron_name):
            continue
        if "mem" in stats:
            mem_list.append(stats["mem"].reshape(-1))
        if "rate" in stats:
            rate_list.append(stats["rate"].reshape(-1))

    if use_salience:
        for neuron_name, salience_vec in salience_map.items():
            # 排除 stem.0. 层
            if _is_stem_layer(neuron_name):
                continue
            if isinstance(salience_vec, torch.Tensor):
                salience_list.append(salience_vec.reshape(-1))
            else:
                salience_list.append(torch.tensor(salience_vec).reshape(-1))

    if len(mem_list) == 0 and len(rate_list) == 0 and len(salience_list) == 0:
        logger.warning("[ChannelAlloc] Empty mem/rate/salience stats; fallback to empty allocation.")
        return bit_allocation

    eps = 1e-8
    mem_all = torch.cat(mem_list) if len(mem_list) > 0 else None
    rate_all = torch.cat(rate_list) if len(rate_list) > 0 else None
    salience_all = torch.cat(salience_list) if len(salience_list) > 0 else None

    if mem_all is not None:
        mem_min, mem_max = mem_all.min().item(), mem_all.max().item()
    else:
        mem_min, mem_max = 0.0, 1.0
    if rate_all is not None:
        rate_min, rate_max = rate_all.min().item(), rate_all.max().item()
    else:
        rate_min, rate_max = 0.0, 1.0
    if salience_all is not None:
        salience_min, salience_max = salience_all.min().item(), salience_all.max().item()
    else:
        salience_min, salience_max = 0.0, 1.0

    def _norm(x, xmin, xmax):
        if x is None:
            return None
        if xmax - xmin < eps:
            return torch.zeros_like(x)
        return (x - xmin) / (xmax - xmin + eps)

    # 收集所有通道的最终score用于分位数（排除 stem.0. 层）
    all_scores = []

    # 确定要遍历的神经元集合
    if use_salience and (w_mem + w_rate) == 0:
        # 只使用 loss salience，从 salience_map 遍历
        neuron_names_to_process = set(salience_map.keys())
        logger.info(f"[ChannelAlloc] Using salience_map with {len(neuron_names_to_process)} neurons")
    else:
        # 使用 activity 或混合模式，从 activity_map 遍历
        neuron_names_to_process = set(activity_map.keys())
        if use_salience:
            # 如果也使用 salience，合并 salience_map 中的神经元
            neuron_names_to_process.update(salience_map.keys())

    for neuron_name in neuron_names_to_process:
        # 排除 stem.0. 层
        if _is_stem_layer(neuron_name):
            continue

        # 从 activity_map 获取 mem 和 rate
        mem = None
        rate = None
        if neuron_name in activity_map:
            stats = activity_map[neuron_name]
            mem = stats.get("mem", None)
            rate = stats.get("rate", None)
        mem_n = _norm(mem, mem_min, mem_max)
        rate_n = _norm(rate, rate_min, rate_max)

        # 获取对应的salience（如果可用）
        salience = None
        if use_salience and neuron_name in salience_map:
            salience = salience_map[neuron_name]
            if not isinstance(salience, torch.Tensor):
                salience = torch.tensor(salience)
            salience = salience.reshape(-1)
        salience_n = _norm(salience, salience_min, salience_max)

        # 如果只使用 salience 但 salience_n 是 None，跳过这个神经元
        if (w_mem + w_rate) == 0 and w_salience > 0 and salience_n is None:
            logger.warning(f"[ChannelAlloc] Skipping {neuron_name}: salience not found in salience_map")
            continue

        score = 0
        if mem_n is not None:
            score = score + w_mem * mem_n
        if rate_n is not None:
            score = score + w_rate * rate_n
        if salience_n is not None:
            score = score + w_salience * salience_n

        # 只有当至少有一个指标可用时才添加
        if (mem_n is not None) or (rate_n is not None) or (salience_n is not None):
            all_scores.append(score.reshape(-1))

    if len(all_scores) == 0:
        logger.warning("[ChannelAlloc] No valid scores computed; fallback to empty allocation.")
        return bit_allocation

    all_scores = torch.cat(all_scores)
    if all_scores.numel() == 0:
        logger.warning("[ChannelAlloc] No valid scores; fallback to empty allocation.")
        return bit_allocation

    scores_np = all_scores.detach().cpu().numpy().astype("float64")
    # 支持从配置中读取百分位数，默认为[50, 75]
    percentile_low = getattr(cfg, "activity_percentile_low", 50)
    percentile_high = getattr(cfg, "activity_percentile_high", 75)

    # 固定p75（4bit和8bit的分界）
    p75 = np.percentile(scores_np, percentile_high)

    # 初始p50（2bit和4bit的分界）
    p50_initial = np.percentile(scores_np, percentile_low)
    p50 = p50_initial

    # 记录使用的方法
    method_parts = []
    if w_mem > 0:
        method_parts.append(f"mem(w={w_mem:.2f})")
    if w_rate > 0:
        method_parts.append(f"rate(w={w_rate:.2f})")
    if w_salience > 0:
        method_parts.append(f"loss_salience(w={w_salience:.2f}, MixLLM-inspired)")
    method_str = " + ".join(method_parts) if method_parts else "none"

    logger.info(
        f"[ChannelAlloc] Method: score = {method_str}"
    )
    logger.info(
        f"[ChannelAlloc] Initial thresholds: p50={float(p50):.4e}, p75={float(p75):.4e} "
        f"(percentiles=[{percentile_low}, {percentile_high}])"
    )

    min_bits, mid_bits, max_bits = 2, 4, 8
    target_avg_bits = float(getattr(cfg, "target_avg_bits", 4.0))
    tolerance = 0.01  # 允许的误差范围

    # 辅助函数：根据阈值分配bit并计算加权平均
    def _allocate_and_compute_avg(p50_threshold, p75_threshold):
        """根据阈值分配bit，返回bit_allocation和加权平均（如果可能）"""
        bit_alloc = {}
        all_scores_dict = {}  # 保存每个神经元的score，用于后续计算

        # 使用相同的神经元集合逻辑
        if use_salience and (w_mem + w_rate) == 0:
            # 只使用 loss salience，从 salience_map 遍历
            neuron_names_to_process = set(salience_map.keys())
        else:
            # 使用 activity 或混合模式，从 activity_map 遍历
            neuron_names_to_process = set(activity_map.keys())
            if use_salience:
                # 如果也使用 salience，合并 salience_map 中的神经元
                neuron_names_to_process.update(salience_map.keys())

        for neuron_name in neuron_names_to_process:
            # 排除 stem.0. 层
            if _is_stem_layer(neuron_name):
                continue

            # 从 activity_map 获取 mem 和 rate
            mem = None
            rate = None
            if neuron_name in activity_map:
                stats = activity_map[neuron_name]
                mem = stats.get("mem", None)
                rate = stats.get("rate", None)
            mem_n = _norm(mem, mem_min, mem_max)
            rate_n = _norm(rate, rate_min, rate_max)

            # 获取对应的salience（如果可用）
            salience = None
            if use_salience and neuron_name in salience_map:
                salience = salience_map[neuron_name]
                if not isinstance(salience, torch.Tensor):
                    salience = torch.tensor(salience)
                salience = salience.reshape(-1)
            salience_n = _norm(salience, salience_min, salience_max)

            # 如果只使用 salience 但 salience_n 是 None，跳过这个神经元
            if (w_mem + w_rate) == 0 and w_salience > 0 and salience_n is None:
                continue

            score = 0
            if mem_n is not None:
                score = score + w_mem * mem_n
            if rate_n is not None:
                score = score + w_rate * rate_n
            if salience_n is not None:
                score = score + w_salience * salience_n

            # 只有当至少有一个指标可用时才处理
            if (mem_n is not None) or (rate_n is not None) or (salience_n is not None):
                score_np = score.detach().cpu().numpy().astype("float64")
                all_scores_dict[neuron_name] = score_np

                # 按通道分配：p50可以是小数，但最终分配是整数
                # score <= p50 -> 2bit
                # p50 < score <= p75 -> 4bit
                # score > p75 -> 8bit
                bits_vec = np.full_like(score_np, fill_value=min_bits, dtype=np.int32)
                bits_vec[score_np > p50_threshold] = mid_bits
                bits_vec[score_np > p75_threshold] = max_bits

                bit_alloc[neuron_name] = bits_vec

        return bit_alloc, all_scores_dict

    # 如果提供了model和cali_data，迭代调整p50使得加权平均接近4bit
    if model is not None and cali_data is not None:
        logger.info(f"[ChannelAlloc] Adjusting p50 threshold to achieve weighted average of {target_avg_bits:.1f} bits...")

        # 先计算一次加权平均，需要获取每个通道的膜电位数量
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        orig_device = next(model.parameters()).device
        if orig_device != device:
            model.to(device)
        model.eval()
        disable_all(model)

        sample_batch = min(1, cali_data.size(0))
        sample_input = cali_data[:sample_batch].to(device)

        # 获取每个神经元的输出形状
        neuron_output_shapes = {}
        handles = []

        def _make_shape_hook(neuron_name):
            def _hook(module, inputs, output):
                if isinstance(output, torch.Tensor):
                    neuron_output_shapes[neuron_name] = output.shape
            return _hook

        module_to_name = {module: name for name, module in model.named_modules()}
        for module, name in module_to_name.items():
            if isinstance(module, LIFNeuron) and not _is_stem_layer(name):
                handles.append(module.register_forward_hook(_make_shape_hook(name)))

        with torch.no_grad():
            reset_net(model)
            _ = model(sample_input)

        for h in handles:
            h.remove()

        if orig_device != device:
            model.to(orig_device)

        # 计算每个通道的膜电位数量（排除 stem.0. 层）
        channel_mem_counts = {}  # {neuron_name: mem_count_per_channel}
        for neuron_name in activity_map.keys():
            # 排除 stem.0. 层
            if _is_stem_layer(neuron_name) or neuron_name not in neuron_output_shapes:
                continue
            output_shape = neuron_output_shapes[neuron_name]
            if len(output_shape) == 5:  # [T, N, C, H, W]
                mem_count = output_shape[0] * output_shape[1] * output_shape[3] * output_shape[4]
            elif len(output_shape) == 4:  # [N, C, H, W]
                mem_count = output_shape[0] * output_shape[2] * output_shape[3]
            elif len(output_shape) == 3:
                if output_shape[0] == sample_batch:
                    mem_count = output_shape[2] if len(output_shape) > 2 else 1
                else:
                    mem_count = output_shape[0] * output_shape[1]
            elif len(output_shape) == 2:  # [N, C]
                mem_count = output_shape[0]
            else:
                non_channel_dims = [d for i, d in enumerate(output_shape) if i != len(output_shape) - 2]
                mem_count = 1
                for dim in non_channel_dims:
                    mem_count *= dim
                if mem_count == 0:
                    mem_count = 1
            channel_mem_counts[neuron_name] = mem_count

        # 辅助函数：计算加权平均比特数（排除 stem.0. 层）
        def _compute_weighted_avg(bit_alloc):
            total_weighted = 0.0
            total_count = 0
            for neuron_name, bits_vec in bit_alloc.items():
                # 排除 stem.0. 层
                if _is_stem_layer(neuron_name) or neuron_name not in channel_mem_counts:
                    continue
                mem_count = channel_mem_counts[neuron_name]
                for bits in bits_vec:
                    total_weighted += float(bits) * mem_count
                    total_count += mem_count
            return total_weighted / total_count if total_count > 0 else 0.0

        # 迭代调整p50
        max_iter = 30
        score_min = scores_np.min()
        score_max = scores_np.max()

        # 初始化搜索范围：p50在[score_min, p75)范围内
        p50_min = score_min
        p50_max = p75 - 1e-6  # p50必须小于p75

        # 先测试初始p50
        bit_alloc_temp, _ = _allocate_and_compute_avg(p50, p75)
        avg_bits_initial = _compute_weighted_avg(bit_alloc_temp)

        logger.info(
            f"[ChannelAlloc] Initial state: p50={p50:.6f}, "
                f"weighted_avg_bits={avg_bits_initial:.4f}, target={target_avg_bits:.1f}, "
            f"score_range=[{score_min:.6f}, {score_max:.6f}], p75={p75:.6f}"
        )

        # 如果初始加权平均太大，需要增大p50以增加2bit通道
        if avg_bits_initial > target_avg_bits:
            # 需要增大p50，让更多通道变成2bit（因为 score <= p50 -> 2bit）
            # 重新设置搜索范围的下界为当前p50
            p50_min = p50
            logger.info(
                f"[ChannelAlloc] Initial avg_bits too high ({avg_bits_initial:.4f} > {target_avg_bits}), "
                f"need more 2bit channels -> will increase p50 in range [{p50_min:.6f}, {p50_max:.6f}]"
            )
        elif avg_bits_initial < target_avg_bits:
            # 加权平均太小，需要减小p50（让更多通道变成4bit，减少2bit）
            p50_max = p50
            logger.info(
                f"[ChannelAlloc] Initial avg_bits too low ({avg_bits_initial:.4f} < {target_avg_bits}), "
                f"need fewer 2bit channels -> will decrease p50 in range [{p50_min:.6f}, {p50_max:.6f}]"
            )

        for iter_idx in range(max_iter):
            bit_alloc_temp, _ = _allocate_and_compute_avg(p50, p75)
            avg_bits = _compute_weighted_avg(bit_alloc_temp)

            logger.info(
                f"[ChannelAlloc] Iter {iter_idx+1}: p50={p50:.6f}, "
                f"weighted_avg_bits={avg_bits:.4f}, target={target_avg_bits:.1f}, "
                f"search_range=[{p50_min:.6f}, {p50_max:.6f}]"
            )

            if abs(avg_bits - target_avg_bits) < tolerance:
                # 统计2bit通道的百分比
                total_channels = 0
                channels_2bit = 0
                for bits_vec in bit_alloc_temp.values():
                    total_channels += len(bits_vec)
                    channels_2bit += int((bits_vec == 2).sum())
                pct_2bit = (channels_2bit / total_channels * 100) if total_channels > 0 else 0.0

                logger.info(
                    f"[ChannelAlloc] Converged! Final p50={p50:.6f}, weighted_avg_bits={avg_bits:.4f}, "
                    f"2bit_channels={pct_2bit:.2f}% ({channels_2bit}/{total_channels})"
                )
                bit_allocation = bit_alloc_temp
                break

            # 二分搜索调整p50
            if avg_bits < target_avg_bits:
                # 加权平均太小，需要减少2bit通道，增加4bit通道 -> 减小p50
                # (因为 score <= p50 -> 2bit，p50越小，2bit通道越少)
                p50_max = p50
            else:
                # 加权平均太大，需要增加2bit通道，减少4bit通道 -> 增大p50
                # (因为 score <= p50 -> 2bit，p50越大，2bit通道越多)
                p50_min = p50

            p50_new = (p50_min + p50_max) / 2.0
            if abs(p50_new - p50) < 1e-10:
                # 统计2bit通道的百分比
                total_channels = 0
                channels_2bit = 0
                for bits_vec in bit_alloc_temp.values():
                    total_channels += len(bits_vec)
                    channels_2bit += int((bits_vec == 2).sum())
                pct_2bit = (channels_2bit / total_channels * 100) if total_channels > 0 else 0.0

                logger.warning(
                    f"[ChannelAlloc] p50 adjustment too small at iter {iter_idx+1}. "
                    f"Final p50={p50:.6f}, weighted_avg_bits={avg_bits:.4f}, "
                    f"target={target_avg_bits:.1f}, diff={abs(avg_bits - target_avg_bits):.4f}, "
                    f"2bit_channels={pct_2bit:.2f}% ({channels_2bit}/{total_channels})"
                )
                bit_allocation = bit_alloc_temp
                break

            p50 = p50_new
        else:
            # 统计2bit通道的百分比
            total_channels = 0
            channels_2bit = 0
            for bits_vec in bit_alloc_temp.values():
                total_channels += len(bits_vec)
                channels_2bit += int((bits_vec == 2).sum())
            pct_2bit = (channels_2bit / total_channels * 100) if total_channels > 0 else 0.0

            logger.warning(
                f"[ChannelAlloc] Max iterations reached. Final p50={p50:.6f}, weighted_avg_bits={avg_bits:.4f}, "
                f"2bit_channels={pct_2bit:.2f}% ({channels_2bit}/{total_channels})"
            )
            bit_allocation = bit_alloc_temp
    else:
        # 没有提供model和cali_data，使用初始阈值直接分配
        bit_allocation, _ = _allocate_and_compute_avg(p50, p75)
        logger.info(f"[ChannelAlloc] Using initial thresholds (no model provided for weighted average adjustment)")

    # 统计最终分配结果
    bit_counters = defaultdict(int)
    for bits_vec in bit_allocation.values():
        unique, counts_local = np.unique(bits_vec, return_counts=True)
        for b, c in zip(unique.tolist(), counts_local.tolist()):
            bit_counters[int(b)] += int(c)

    total_channels = sum(bit_counters.values())
    logger.info(
        f"[ChannelAlloc] Final allocation: channels={total_channels}, "
        f"{min_bits}bit={bit_counters[min_bits]}, "
        f"{mid_bits}bit={bit_counters[mid_bits]}, "
        f"{max_bits}bit={bit_counters[max_bits]}, "
        f"final_p50={p50:.6f}, fixed_p75={p75:.6f}"
    )

    # 测试模式：强制所有通道都分配4bit
    force_4bit = getattr(cfg, "force_all_channels_4bit", False)
    if force_4bit:
        logger.info("[ChannelAlloc] TEST MODE: Forcing all channels to 4bit for validation.")
        for neuron_name in bit_allocation.keys():
            bits_vec = bit_allocation[neuron_name]
            bit_allocation[neuron_name] = np.full_like(bits_vec, fill_value=4, dtype=np.int32)
        logger.info(f"[ChannelAlloc] All {total_channels} channels forced to 4bit.")

    reference_path = getattr(cfg, "allocation_reference_path", None)
    reference_mode = getattr(cfg, "allocation_reference_mode", None)
    if reference_path and reference_mode:
        with open(reference_path) as f:
            reference = json.load(f)
        rng = np.random.default_rng(int(getattr(cfg, "allocation_random_seed", 0)))
        _, ranking_scores = _allocate_and_compute_avg(p50, p75)
        for neuron_name, bits_vec in list(bit_allocation.items()):
            if neuron_name not in reference:
                raise KeyError(f"Missing {neuron_name} in allocation reference {reference_path}")
            ref = np.asarray(reference[neuron_name], dtype=np.int32)
            if ref.shape != np.asarray(bits_vec).shape:
                raise ValueError(f"Allocation shape mismatch for {neuron_name}: {ref.shape} vs {np.asarray(bits_vec).shape}")
            if reference_mode == "random":
                rng.shuffle(ref)
                bit_allocation[neuron_name] = ref
            elif reference_mode == "ranked":
                order = np.argsort(np.asarray(ranking_scores[neuron_name]), kind="stable")
                assigned = np.empty_like(ref)
                assigned[order] = np.sort(ref)
                bit_allocation[neuron_name] = assigned
            else:
                raise ValueError(f"Unsupported allocation_reference_mode: {reference_mode}")
        logger.info(f"[ChannelAlloc] Applied {reference_mode} assignment with histogram from {reference_path}")

    return bit_allocation


def calculate_weighted_avg_bits(bit_allocation, activity_map, model, cali_data, logger, config_quant):
    """
    计算加权平均量化比特数

    每个通道的膜电位数量 = T × N × H × W（所有非通道维度）
    加权平均 = sum(每个通道的bit数 × 该通道的膜电位数量) / sum(所有通道的膜电位数量)

    Args:
        bit_allocation: {neuron_name: bits_vec[C]}，每个通道的bit分配
        model: 模型
        cali_data: 校准数据，用于获取实际输入形状
        logger: logger对象
        config_quant: 量化配置

    Returns:
        weighted_avg_bits: 加权平均比特数
    """
    if len(bit_allocation) == 0:
        logger.warning("[WeightedBits] No bit allocation provided.")
        return 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orig_device = next(model.parameters()).device
    if orig_device != device:
        model.to(device)
    model.eval()
    disable_all(model)

    # 使用一个小的校准样本获取输出形状
    sample_batch = min(1, cali_data.size(0))
    sample_input = cali_data[:sample_batch].to(device)

    # 记录每个神经元的输出形状
    neuron_output_shapes = {}
    handles = []

    def _make_shape_hook(neuron_name):
        def _hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                neuron_output_shapes[neuron_name] = output.shape
        return _hook

    # 注册hook获取输出形状（排除 stem.0. 层）
    module_to_name = {module: name for name, module in model.named_modules()}
    for module, name in module_to_name.items():
        if isinstance(module, LIFNeuron) and not _is_stem_layer(name):
            handles.append(module.register_forward_hook(_make_shape_hook(name)))

    # 运行一次前向传播获取形状
    with torch.no_grad():
        reset_net(model)
        _ = model(sample_input)

    # 移除hook
    for h in handles:
        h.remove()

    if orig_device != device:
        model.to(orig_device)

    # 计算加权平均
    total_weighted_bits = 0.0
    total_mem_values = 0

    # bit_allocation是一个字典，键是神经元名称，值是每个通道的bit分配向量
    # 排除 stem.0. 层
    for neuron_name, bits_vec in bit_allocation.items():
        # 排除 stem.0. 层
        if _is_stem_layer(neuron_name):
            continue
        if neuron_name not in neuron_output_shapes:
            logger.warning(f"[WeightedBits] No output shape found for {neuron_name}, skip.")
            continue

        output_shape = neuron_output_shapes[neuron_name]
        # output_shape 可能是 [T, N, C, H, W] 或 [N, C, H, W] 或 [T, N, C] 或 [N, C]
        # 需要找到通道维度，然后计算非通道维度的乘积
        # 经过测试，其实就是[T, N, C, H, W]，N代表batch size

        # 从activity_map获取通道数（更可靠）
        if neuron_name in activity_map:
            stats = activity_map[neuron_name]
            mem = stats.get("mem", None)
            if mem is not None:
                num_channels = mem.numel()
            else:
                rate = stats.get("rate", None)
                if rate is not None:
                    num_channels = rate.numel()
                else:
                    logger.warning(f"[WeightedBits] No mem/rate for {neuron_name}, skip.")
                    continue
        else:
            # 如果activity_map中没有，尝试从输出形状推断
            if len(output_shape) >= 2:
                # 假设通道在倒数第二维（常见情况）
                num_channels = output_shape[-2]
            else:
                logger.warning(f"[WeightedBits] Cannot infer channels for {neuron_name}, skip.")
                continue

        if num_channels != len(bits_vec):
            logger.warning(
                f"[WeightedBits] Channel mismatch for {neuron_name}: "
                f"bits_vec has {len(bits_vec)} channels, but activity_map has {num_channels} channels. "
                f"Using bits_vec length."
            )
            num_channels = len(bits_vec)

        # 计算每个通道的膜电位数量
        # 对于SNN，每个时间步、每个batch、每个空间位置都需要一个膜电位
        # 所以：膜电位数量 = T × N × H × W（所有非通道维度）
        if len(output_shape) == 5:  # [T, N, C, H, W]
            mem_values_per_channel = output_shape[0] * output_shape[1] * output_shape[3] * output_shape[4]
        elif len(output_shape) == 4:  # [N, C, H, W]
            mem_values_per_channel = output_shape[0] * output_shape[2] * output_shape[3]
        elif len(output_shape) == 3:  # [T, N, C] 或 [N, C, H]
            # 需要判断，但通常SNN是 [T, N, C]
            if output_shape[0] == sample_batch:  # 可能是 [N, C, H]
                mem_values_per_channel = output_shape[2] if len(output_shape) > 2 else 1
            else:  # 可能是 [T, N, C]
                mem_values_per_channel = output_shape[0] * output_shape[1]
        elif len(output_shape) == 2:  # [N, C]
            mem_values_per_channel = output_shape[0]
        else:
            # 默认：假设每个通道在每个时间步和batch都需要一个膜电位
            # 使用输出形状的所有非通道维度
            non_channel_dims = [d for i, d in enumerate(output_shape) if i != len(output_shape) - 2]
            mem_values_per_channel = 1
            for dim in non_channel_dims:
                mem_values_per_channel *= dim
            if mem_values_per_channel == 0:
                mem_values_per_channel = 1  # 避免除零

        # 累加加权比特数
        for ch_idx, bits in enumerate(bits_vec):
            total_weighted_bits += float(bits) * mem_values_per_channel
            total_mem_values += mem_values_per_channel

    if total_mem_values == 0:
        logger.warning("[WeightedBits] No membrane potential values found.")
        return 0.0

    weighted_avg_bits = total_weighted_bits / total_mem_values

    logger.info(
        f"[WeightedBits] Total membrane potential values: {total_mem_values:,}, "
        f"Weighted average bits: {weighted_avg_bits:.4f}"
    )

    return weighted_avg_bits
