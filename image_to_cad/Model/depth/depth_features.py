#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from image_to_cad.Model.depth.up_projection import UpProjection

class DepthFeatures(nn.Module):
    def __init__(self,
                 size=(120, 160),
                 num_levels=4,
                 in_channels=256,
                 out_channels_per_level=32):
        super().__init__()
        self.size = size
        self.ups = nn.ModuleList([
            UpProjection(in_channels, out_channels_per_level)
            for _ in range(num_levels)
        ])
        self.out_channels = out_channels_per_level * num_levels

    def forward(self, features):
        return torch.cat([
            up(x, self.size) for x, up in zip(features, self.ups)
        ], dim=1)

