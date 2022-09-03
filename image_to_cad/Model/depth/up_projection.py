#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F

class UpProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=5, stride=1, padding=2
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=5, stride=1, padding=2
        )
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, size):
        if x.shape[-2:] != size:
            x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        x_conv1 = self.relu1(self.conv1(x))

        bran1 = self.conv1_2(x_conv1)
        bran2 = self.conv2(x)
        return self.relu2(bran1 + bran2)

