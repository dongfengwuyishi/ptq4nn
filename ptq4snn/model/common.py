from spikingjelly.activation_based import layer
import spikingjelly.activation_based.functional as sjf
import torch
import torch.nn as nn
import math
import logging

def reset_net(net: nn.Module):
    for m in net.modules():
        if hasattr(m, 'reset'):
            m.reset()
            # if not isinstance(m, LIFNeuron):
            #     logging.warning(f'Trying to call `reset()` of {m}, which is not LIFNeuron')

def set_step_mode(net: nn.Module, step_mode: str):
    for m in net.modules():
        if hasattr(m, 'step_mode'):
            m.step_mode = step_mode

def conv3x3(in_planes, out_planes, stride=1):
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    return layer.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, bias=False)

def conv7x7(in_planes, out_planes, stride=1):
    return layer.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)

def conv1x1(in_plane, out_plane, stride=1):
    return layer.Conv2d(in_plane, out_plane,
                     kernel_size=1, stride=stride, padding=0, bias=False)



@torch.jit.script
def heaviside(x: torch.Tensor):
    return (x >= 0).to(x)

@torch.jit.script
def atan_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float = 2.0):
    return alpha / 2 / (1 + (math.pi / 2 * alpha * x).pow_(2)) * grad_output, None, None


class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return atan_backward(grad_output, x)

class LIFNeuron(nn.Module):
    def __init__(self, tau=2.0, v_threshold=1.0, step_mode='s', is_if=False):
        super().__init__()
        self.mem = 0
        self.leak = 1/tau
        self.thresh = v_threshold
        self.step_mode = step_mode
        self.is_if = is_if
        self.v_pre_seq = None

    def reset(self):
        self.mem = 0
        self.v_pre_seq = None

    def forward(self, input):
        if self.step_mode =='m':
            return self.multi_step_forward(input)
        else:
            return self.single_step_forward(input)

    def neuronal_charge(self, x):
        self.mem = self.mem + x

    def neuronal_fire(self):
        spike = atan.apply(self.mem - self.thresh)
        return spike

    def neuronal_reset_and_leak(self, spike):
        self.mem = self.leak * ((1. - spike) * self.mem)

    def neuronal_only_reset(self, spike):
        self.mem = ((1. - spike) * self.mem)

    def single_step_forward(self, x: torch.Tensor):
        """
            :param x: 输入到神经元的电压增量
            :type x: torch.Tensor
            :return: 神经元的输出脉冲
            :rtype: torch.Tensor

            按照充电、放电、重置的顺序进行前向传播。
        """
        self.neuronal_charge(x)
        self.v_pre_seq = self.mem.unsqueeze(0)
        spike = self.neuronal_fire()
        if self.is_if:
            self.neuronal_only_reset(spike)
        else:
            self.neuronal_reset_and_leak(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        v_pre_seq = []
        for t in range(T):
            self.neuronal_charge(x_seq[t])
            v_pre_seq.append(self.mem)
            y = self.neuronal_fire()
            if self.is_if:
                self.neuronal_only_reset(y)
            else:
                self.neuronal_reset_and_leak(y)
            y_seq.append(y)
        self.v_pre_seq = torch.stack(v_pre_seq)
        return torch.stack(y_seq)

    def extra_repr(self):
        return f"tau={1/self.leak}, v_threshold={self.thresh}, step_mode={self.step_mode}, is_if={self.is_if}"



def create_neuron(is_if=False):
    return LIFNeuron(tau=2.0, v_threshold=1.0, is_if=is_if)
    # from spikingjelly.activation_based import neuron, surrogate
    # return neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, surrogate_function=surrogate.ATan(), detach_reset=True, store_v_seq=True, backend='cupy', step_mode='m')
    # return neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, surrogate_function=surrogate.ATan(), detach_reset=True, store_v_seq=True)


if __name__ == '__main__':
    neuron = LIFNeuron(tau=2.0, v_threshold=1.0, step_mode='m')
    T, B, C, H, W = 4, 2, 1, 3, 3
    input = torch.rand(T, B, C, H, W)
    output = neuron(input)
    print(output)
