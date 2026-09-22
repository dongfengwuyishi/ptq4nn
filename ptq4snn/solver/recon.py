import torch
import torch.nn as nn
import logging
import warnings
from ptq4snn.model.common import LIFNeuron, reset_net
from ptq4snn.quantization.quantized_module import QNeuron
from ptq4snn.utils.utils import DataSaverHook, StopForwardException, Dim1ToDim0Wrapper


def save_inp_oup_data(
    model,
    module,
    cali_data: list,
    store_inp=False,
    store_oup=False,
    bs: int = 32,
    keep_gpu: bool = True,
):

    device = next(model.parameters()).device
    data_saver = DataSaverHook(
        store_input=store_inp, store_output=store_oup, stop_forward=True
    )
    handle = module.register_forward_hook(data_saver)
    cached = [[], []]
    with torch.no_grad():
        for i in range(int(cali_data.size(0) / bs)):
            try:
                _ = model(cali_data[i * bs : (i + 1) * bs].to(device))
            except StopForwardException:
                pass
            if store_inp:
                if keep_gpu:
                    cached[0].append(data_saver.input_store[0].detach())
                else:
                    cached[0].append(data_saver.input_store[0].detach().cpu())
            if store_oup:
                if keep_gpu:
                    cached[1].append(data_saver.output_store.detach())
                else:
                    cached[1].append(data_saver.output_store.detach().cpu())
    if store_inp:
        cached[0] = torch.cat(
            [x for x in cached[0]], dim=1
        )  # [T, B , C, H, W] -> [T, B * x, C, H, W]
    if store_oup:
        cached[1] = torch.cat([x for x in cached[1]], dim=1)
    handle.remove()
    torch.cuda.empty_cache()
    return cached


def save_vpre_data(model, module, cali_data, bs=32, keep_gpu=True):
    """Cache pre-fire membrane trajectories for every LIF neuron in a module."""
    device = next(model.parameters()).device
    neurons = [m for m in module.modules() if isinstance(m, QNeuron)]
    if not neurons:
        neurons = [m for m in module.modules() if isinstance(m, LIFNeuron)]
    cached = [[] for _ in neurons]
    data_saver = DataSaverHook(store_output=False, stop_forward=True)
    handle = module.register_forward_hook(data_saver)
    with torch.no_grad():
        for start in range(0, cali_data.size(0), bs):
            try:
                _ = model(cali_data[start : start + bs].to(device))
            except StopForwardException:
                pass
            for idx, neuron in enumerate(neurons):
                value = neuron.v_pre_seq.detach()
                cached[idx].append(value if keep_gpu else value.cpu())
    handle.remove()
    return [torch.cat(chunks, dim=1) for chunks in cached]


class LinearTempDecay:
    def __init__(self, t_max=20000, warm_up=0.2, start_b=20, end_b=2):
        self.t_max = t_max
        self.start_decay = warm_up * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


class LossFunction:
    r"""loss function to calculate mse reconstruction loss and relaxation loss
    use some tempdecay to balance the two losses.
    """

    def __init__(
        self,
        module,
        weight: float = 1.0,
        iters: int = 20000,
        b_range: tuple = (20, 2),
        warm_up: float = 0.0,
        p: float = 2.0,
        logger: logging.Logger = None,
    ):

        self.module = module
        self.weight = weight
        self.loss_start = iters * warm_up
        self.p = p

        self.temp_decay = LinearTempDecay(
            iters, warm_up=warm_up, start_b=b_range[0], end_b=b_range[1]
        )
        self.count = 0
        self.logger = logger

    def __call__(self, pred, tgt):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :return: total loss function
        """
        self.count += 1
        rec_loss = lp_loss(pred, tgt, p=self.p)

        b = self.temp_decay(self.count)
        if self.count < self.loss_start:
            round_loss = 0
        else:
            round_loss = 0
            for layer in self.module.modules():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    round_vals = layer.weight_fake_quant.rectified_sigmoid()
                    round_loss += (
                        self.weight * (1 - ((round_vals - 0.5).abs() * 2).pow(b)).sum()
                    )
        total_loss = rec_loss + round_loss
        if self.count % 500 == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.logger.info(
                    "Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}".format(
                        float(total_loss),
                        float(rec_loss),
                        float(round_loss),
                        b,
                        self.count,
                    )
                )
        return total_loss


def lp_loss(pred, tgt, p=2.0):
    """
    loss function
    """
    return (pred - tgt).abs().pow(p).sum(2).mean()  #


def reconstruction(model, fp_model, module, fp_module, cali_data, config, logger):
    device = next(module.parameters()).device
    qdrop_prob = float(getattr(config, "qdrop_prob", 0.0))
    # get data first
    quant_inp, _ = save_inp_oup_data(
        model,
        module,
        cali_data,
        store_inp=True,
        store_oup=False,
        bs=config.batch_size,
        keep_gpu=config.keep_gpu,
    )
    fp_inp, fp_oup = save_inp_oup_data(
        fp_model,
        fp_module,
        cali_data,
        store_inp=True,
        store_oup=True,
        bs=config.batch_size,
        keep_gpu=config.keep_gpu,
    )
    lambda_pre = float(getattr(config, "lambda_pre", 0.0))
    fp_vpre = []
    if lambda_pre > 0:
        fp_vpre = save_vpre_data(
            fp_model,
            fp_module,
            cali_data,
            bs=config.batch_size,
            keep_gpu=config.keep_gpu,
        )
        logger.info(f"PMR enabled: lambda_pre={lambda_pre:g}, trajectories={len(fp_vpre)}")

    quant_inp = Dim1ToDim0Wrapper(quant_inp)
    if qdrop_prob > 0:
        fp_inp = Dim1ToDim0Wrapper(fp_inp)
    fp_oup = Dim1ToDim0Wrapper(fp_oup)

    # prepare for up or down tuning
    w_para = []
    b_para = []
    factor_para = []
    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            weight_quantizer = layer.weight_fake_quant
            weight_quantizer.init(layer.weight.data, config.round_mode)
            w_para += [weight_quantizer.alpha]
            b_para += [layer.bias]
        elif isinstance(layer, QNeuron):
            factor_para += [layer.factor_exp]

    # for para in model.parameters():
    #     if not any(para is w for w in w_para):
    #         w_para += [para]

    w_opt = torch.optim.Adam(w_para)
    # b_opt = torch.optim.Adam(b_para, lr=config.lr * 0.1)
    # w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_opt, config.iters)
    # b_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(b_opt, config.iters)
    if len(factor_para) > 0:
        factor_opt = torch.optim.Adam(factor_para, lr=config.lr)
        factor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            factor_opt, config.iters
        )
    else:
        factor_opt = None
        factor_scheduler = None
    loss_func = LossFunction(
        module=module,
        weight=config.weight,
        iters=config.iters,
        b_range=config.b_range,
        warm_up=config.warm_up,
        logger=logger,
    )

    sz = quant_inp.size(0)
    if qdrop_prob > 0:
        logger.info(f"QDrop input mixing enabled: qdrop_prob={qdrop_prob:.3f}")
    for i in range(config.iters):
        idx = torch.randint(0, sz, (config.batch_size,))
        cur_inp = quant_inp[idx].to(device)
        if qdrop_prob > 0:
            cur_fp_inp = fp_inp[idx].to(device)
            if qdrop_prob >= 1:
                cur_inp = cur_fp_inp
            else:
                mask_shape = [1] * cur_inp.dim()
                if cur_inp.dim() >= 2:
                    mask_shape[1] = cur_inp.shape[1]
                else:
                    mask_shape[0] = cur_inp.shape[0]
                use_fp = (torch.rand(mask_shape, device=device) < qdrop_prob).to(cur_inp.dtype)
                cur_inp = cur_inp * (1 - use_fp) + cur_fp_inp * use_fp
        cur_fp_oup = fp_oup[idx].to(device)
        w_opt.zero_grad()
        if factor_opt:
            factor_opt.zero_grad()
        reset_net(module)
        cur_quant_oup = module(cur_inp)
        err = loss_func(cur_quant_oup, cur_fp_oup)
        if lambda_pre > 0 and fp_vpre:
            q_neurons = [m for m in module.modules() if isinstance(m, QNeuron)]
            pre_loss = cur_quant_oup.new_zeros(())
            matched = 0
            for q_neuron, teacher_vpre in zip(q_neurons, fp_vpre):
                if q_neuron.v_pre_seq is None:
                    continue
                target = teacher_vpre[:, idx].to(device)
                pred = q_neuron.v_pre_seq
                denom = target.detach().pow(2).mean().clamp_min(1e-8)
                pre_loss = pre_loss + (pred - target).pow(2).mean() / denom
                matched += 1
            if matched:
                err = err + lambda_pre * pre_loss / matched
        err.backward()
        w_opt.step()
        # b_opt.step()
        # w_scheduler.step()
        # b_scheduler.step()
        if factor_opt:
            factor_opt.step()
            factor_scheduler.step()
    torch.cuda.empty_cache()
    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            weight_quantizer = layer.weight_fake_quant
            layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
            weight_quantizer.adaround = False
