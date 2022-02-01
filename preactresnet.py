'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from cayley_ortho_conv import Cayley
from block_ortho_conv import BCOP
from skew_ortho_conv import SOC

from custom_activations import *
from utils import conv_mapping, activation_mapping

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, conv_layer, activation_name, stride=1):
        super(PreActBlock, self).__init__()
        self.activation1 = activation_mapping(activation_name, channels=in_planes)
        self.activation2 = activation_mapping(activation_name, channels=planes)
        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv_layer(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.activation1(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.activation2(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, conv_name, activation_name, stride=1):
        super(PreActBottleneck, self).__init__()
        self.activation1 = activation_mapping(activation_name, channels=in_planes)
        self.activation2 = activation_mapping(activation_name, channels=planes)
        self.activation3 = activation_mapping(activation_name, channels=planes)

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv_layer(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = conv_layer(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv_layer(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )


    def forward(self, x):
        out = self.activation1(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.activation2(self.bn2(out)))
        out = self.conv3(self.activation3(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, conv_name, activation_name, num_classes=10):
        super(PreActResNet, self).__init__()
        conv_layer = conv_mapping[conv_name]
        self.in_planes = 64

        self.conv1 = conv_layer(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], conv_layer, activation_name, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], conv_layer, activation_name, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], conv_layer, activation_name, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], conv_layer, activation_name, stride=2)
        
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.activation = activation_mapping(activation_name, channels=512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, num_classes)            

    def _make_layer(self, block, planes, num_blocks, conv_layer, activation_name, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, conv_layer, activation_name, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.activation(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18(conv_name='standard', activation_name='relu', num_classes=10):
    return PreActResNet(PreActBlock, [2,2,2,2], conv_name, activation_name, num_classes)

def PreActResNet34(conv_name='standard', activation_name='relu', num_classes=10):
    return PreActResNet(PreActBlock, [3,4,6,3], conv_name, activation_name, num_classes)

def PreActResNet50(conv_name='standard', activation_name='relu', num_classes=10):
    return PreActResNet(PreActBottleneck, [3,4,6,3], conv_name, activation_name, num_classes)

def PreActResNet101(conv_name='standard', activation_name='relu', num_classes=10):
    return PreActResNet(PreActBottleneck, [3,4,23,3], conv_name, activation_name, num_classes)

def PreActResNet152(conv_name='standard', activation_name='relu', num_classes=10):
    return PreActResNet(PreActBottleneck, [3,8,36,3], conv_name, activation_name, num_classes)


resnet_mapping = {
    'resnet18': PreActResNet18,
    'resnet34': PreActResNet34,
    'resnet50': PreActResNet50,
    'resnet101': PreActResNet101,
    'resnet152': PreActResNet152
}
