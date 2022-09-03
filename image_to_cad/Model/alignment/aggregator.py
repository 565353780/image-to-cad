#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

class Aggregator(nn.Module):
    def __init__(self,
                 shared_net=nn.Identity(),
                 global_net=nn.Identity()):
        super().__init__()
        self.shared_net = shared_net
        self.global_pool = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(1)
        )
        self.global_net = global_net

    def forward(self, features, mask):
        features = self.shared_net(features)
        if features.numel() > 0:
            features = self.global_pool(features * mask)
        else:
            features = features.view(0, features.size(1))
        return self.global_net(features)

