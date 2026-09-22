import torch.nn as nn
from ptq4snn.model.common import conv3x3, conv1x1, conv7x7, create_neuron, reset_net, set_step_mode
from spikingjelly.activation_based import layer
import torch
from ptq4snn.utils.utils import timeExpand2d

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None, is_if=False, is_sew=False):
        super(BasicBlock, self).__init__()
        self.is_sew = is_sew
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = layer.BatchNorm2d(planes)
        self.lif1 = create_neuron(is_if=is_if)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = layer.BatchNorm2d(planes)
        self.lif2 = create_neuron(is_if=is_if)
        if downsample is not None:
            self.downsample = downsample
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if not self.is_sew:
            residual = self.downsample(x)
            out += residual
        out = self.lif2(out)
        if self.is_sew:
            residual = self.downsample(x)
            out = out + residual
        return out

    def __repr__(self):
        if self.is_sew:
            return 'Sew' + super().__repr__()
        else:
            return super().__repr__()

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None, is_if=False, is_sew=False):
        super(Bottleneck, self).__init__()
        self.is_sew = is_sew
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = layer.BatchNorm2d(planes)
        self.lif1 = create_neuron(is_if=is_if)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = layer.BatchNorm2d(planes)
        self.lif2 = create_neuron(is_if=is_if)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = layer.BatchNorm2d(planes * self.expansion)
        self.lif3 = create_neuron(is_if=is_if)
        if downsample is not None:
            self.downsample = downsample
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lif2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if not self.is_sew:
            residual = self.downsample(x)
            out += residual
        out = self.lif3(out)
        if self.is_sew:
            residual = self.downsample(x)
            out = out + residual
        return out

    def __repr__(self):
        if self.is_sew:
            return 'Sew' + super().__repr__()
        else:
            return super().__repr__()

def zero_init_blocks(net: nn.Module):
    for m in net.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.conv3.weight, 0)
        elif isinstance(m, BasicBlock):
            nn.init.constant_(m.conv2.weight, 0)

class ResNet(nn.Module):
    def __init__(self, block = BasicBlock, layers = [2, 2, 2, 2], num_classes=10, in_channels = 3, stem_stride = 1, dvs = False, is_sew = False, is_if= False, zero_init_residual=True):
        super().__init__()
        self.is_dvs = dvs
        self.is_if = is_if
        self.is_sew = is_sew
        self.in_planes = 64
        if stem_stride == 4:
            self.stem = nn.Sequential(
                conv7x7(in_channels, 64, stride=2),
                layer.BatchNorm2d(64),
                create_neuron(),
                layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        elif stem_stride == 1:
            self.stem = nn.Sequential(
                conv3x3(in_channels, 64, stride=1),
                layer.BatchNorm2d(64),
                create_neuron(),
            )
        else:
            raise ValueError("stem_stride should be 1 or 4, 1 for cifar(32), 4 for imagenet(224)")
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.flatten = layer.Flatten()
        self.fc = layer.Linear(512 * block.expansion, num_classes)
        set_step_mode(self, 'm')

        if zero_init_residual:
            zero_init_blocks(self)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv3x3(self.in_planes, planes * block.expansion, stride),
                layer.BatchNorm2d(planes * block.expansion),
                *([create_neuron(is_if=self.is_if)] if self.is_sew else [])
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, self.is_if, self.is_sew))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, is_if=self.is_if, is_sew=self.is_sew))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, reset_state=True):
        if self.is_dvs: # from [B, T, C, H, W] to [T, B, C, H, W]
            x = x.transpose(0, 1)
        x = timeExpand2d(x)
        if reset_state:
            reset_net(self)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def resnet18(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
