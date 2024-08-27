import torch
import torch.nn as nn
from keras.src.utils.module_utils import torchvision


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.ReLU()  # default activation
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard convolution layer with optional batch normalization and activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        return self.act(self.bn(self.conv(x)))
class Layer(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, s=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        s = s if isinstance(s, list) else [s]
        if len(s) == 1:
            self.m = nn.Sequential(*(BasicBlock(c1, c2, s[-1]) for _ in range(n)))
        else:
            self.m = nn.Sequential(BasicBlock(c1, c2, s[0]),
                                  *(BasicBlock(c2, c2, s[-1]) for _ in range(n-1)))
    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.m(x)
# 定义一个简单的ResNet Block类
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class Linear(nn.Module):
    def __init__(self, in_channels=512, out_channels=1000, stride=1):
        super(Linear, self).__init__()
        self.fc = nn.Linear(in_channels,out_channels)
    def forward(self,X):
        return self.fc(X.flatten(start_dim=1))

class Classify(nn.Module):
    # YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(
        self, c1, c2
    ):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
        """Initializes YOLOv5 classification head with convolution, pooling, and dropout layers for input to output
        channel transformation.
        """
        super().__init__()
        c_ = 256  # efficientnet_b0 size
        self.fc1 = nn.Linear(c1, c_)  # to x(c1,c_)
        self.fc2 = nn.Linear(c_, c2)  # to x(c_,c2)

    def forward(self, x):
        """Processes input through conv, pool, drop, and linear layers; supports list concatenation input."""
        return self.fc2(self.fc1(x))

