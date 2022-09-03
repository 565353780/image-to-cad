#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class STN3d(nn.Module):
    def __init__(self, k=3):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k**2)

        self.k = k
        return

    def forward(self, x, mask=None):
        x = F.relu_(self.conv1(x))
        x = F.relu_(self.conv2(x))
        x = F.relu_(self.conv3(x))

        if mask is not None:
            x = x * mask

        x = torch.max(x, 2).values
        x = F.relu_(self.fc1(x))
        x = F.relu_(self.fc2(x))
        x = self.fc3(x)

        iden = torch.eye(self.k, device=x.device).reshape(1, -1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

