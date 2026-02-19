import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# noinspection SpellCheckingInspection
class ResNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2)
        self.bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.resblock1 = ResBlock(64, 64, 1)
        self.resblock2 = ResBlock(64, 128, 2)
        self.resblock3 = ResBlock(128, 256, 2)
        self.resblock4 = ResBlock(256, 512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))     # GlobalAvgPool resulting in a 1x1 value for each of the 512 feature maps
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(512, 2)


    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        x = self.maxpool(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        #x = x.view(x.size(0), -1)   # flatten dynamically (onnx had a problem with nn.Flatten)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class ResBlock(nn.Module):
    """
    Residual block composed of two convolution layers followed by BN and ReLU.
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride)  # to compensate possible striding effects of first conv
        self.bn3 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        """
        Computes the residual output y = F(x) + x with F(x).
        F(x) is two convolutions followed by BN and ReLU.
        """
        Fx = self.conv1(x)
        Fx = F.relu(self.bn1(Fx))
        Fx = self.conv2(Fx)
        Fx = self.bn2(Fx)

        x = self.conv1x1(x)
        x = self.bn3(x)

        y = F.relu(Fx + x)
        return y