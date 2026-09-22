import torch.nn as nn
from ptq4snn.model.common import conv3x3, create_neuron, reset_net, set_step_mode
from spikingjelly.activation_based import layer
import torch
from ptq4snn.utils.utils import timeExpand2d


class VGG(nn.Module):
    def __init__(self,
                 cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
                 num_classes = 10,
                 in_channels = 3,
                 dvs = False):
        super(VGG, self).__init__()
        self.conv_fc = nn.Sequential()
        self.is_dvs = dvs
        i = 0
        for c in cfg:
            if c == 'M':
                self.conv_fc.add_module(f'pool{i - 1}', layer.MaxPool2d(2, 2))
            elif c == 'A':
                self.conv_fc.add_module(f'pool{i - 1}', layer.AvgPool2d(2, 2))
            elif isinstance(c, int):
                out_channels = c
                self.conv_fc.add_module(f'conv{i}', conv3x3(in_channels, out_channels))
                self.conv_fc.add_module(f'bn{i}', layer.BatchNorm2d(out_channels))
                self.conv_fc.add_module(f'lif{i}', create_neuron())
                i += 1
                in_channels = out_channels
            else:
                raise ValueError(f'Invalid layer type {c}')

        self.conv_fc.add_module(f'adaptive_avg_pool', layer.AdaptiveAvgPool2d((1,1)))
        self.conv_fc.add_module(f'flatten', layer.Flatten())
        self.conv_fc.add_module(f'fc', layer.Linear(out_channels, num_classes))

        set_step_mode(self, 'm')

    def get_model_cfg(self):
        cfg = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                cfg.append(m.out_channels)
            elif isinstance(m, nn.MaxPool2d):
                cfg.append('M')
            elif isinstance(m, nn.AvgPool2d):
                cfg.append('A')
        return cfg

    def forward(self, x: torch.Tensor, reset_state=True):
        if self.is_dvs: # from [B, T, C, H, W] to [T, B, C, H, W]
            x = x.transpose(0, 1)
        x = timeExpand2d(x)
        if reset_state:
            reset_net(self)
        return self.conv_fc(x)

def vgg16cifar(num_classes=10, in_channels=3):
    return VGG(cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
               num_classes=num_classes, in_channels=in_channels, dvs=False)

def vgg4dvs(num_classes=10, in_channels=2):
    return VGG(cfg = [64, 'M', 128, 'M', 128, 'M', 256, 'M', 256],
               num_classes=num_classes, in_channels=in_channels, dvs=True)
