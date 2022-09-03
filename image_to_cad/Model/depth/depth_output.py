#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F

class DepthOutput(nn.Module):
    def __init__(self,
                 in_channels, up_ratio,
                 num_hiddens=2, hidden_channels=128):
        super().__init__()

        assert up_ratio == int(up_ratio)
        up_ratio = int(up_ratio)

        convs = []
        for _ in range(num_hiddens):
            convs.append(nn.Conv2d(
                in_channels, hidden_channels, kernel_size=5, padding=2
            ))
            convs.append(nn.ReLU(True))
            in_channels = hidden_channels
        self.convs = nn.Sequential(*convs)

        if up_ratio > 1:
            # import pdb; pdb.set_trace()
            self.output = nn.Sequential(
                nn.Conv2d(hidden_channels, up_ratio**2, kernel_size=1),
                nn.PixelShuffle(up_ratio)
            )
        else:
            self.output = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, features, output_size=None):
        features = self.convs(features)
        depth = self.output(features)
        if output_size is not None and output_size != depth.shape[-2:]:
            # import pdb; pdb.set_trace()
            depth = F.interpolate(
                depth, output_size, mode='bilinear', align_corners=True
            )
        return depth

