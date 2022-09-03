#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from image_to_cad.Model.retrieval.retrieval_network import RetrievalNetwork
from image_to_cad.Model.retrieval.stn3d import STN3d

from image_to_cad.Method.retrieval_ops import mask_point_features

class PointNet(RetrievalNetwork):
    def __init__(self, relu_out=True, feat_trs=True, ret_trs=False):
        super().__init__()

        self.relu_out = relu_out
        self.feat_trs = feat_trs
        self.ret_trs = ret_trs

        self.stn = STN3d()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
    
        if self.feat_trs:
            self.fstn = STN3d(k=64)
        return

    @property
    def embedding_dim(self):
        return self.conv3.out_channels

    def forward(self, x, mask=None):
        if x.numel() == 0:
            return self._empty_output(x)

        x = x.flatten(2)  # points are channels
        if mask is not None:
            mask = mask.flatten(2)  # num-points is the spatial dim

        trs = self.stn(x, mask)
        x = trs @ x

        x = F.relu_(self.conv1(x))
        if self.feat_trs:
            ftrs = self.fstn(x, mask)
            x = ftrs @ x
        else:
            ftrs = None
        
        x = F.relu_(self.conv2(x))
        x = self.conv3(x)

        x = mask_point_features(x, mask)
        x = torch.max(x, 2).values

        if self.relu_out:
            x = F.relu_(x)

        return (x, trs, ftrs) if self.ret_trs else x

    def _empty_output(self, x):
        with torch.device(x.device):
            x = torch.zeros(0, 1024)
            trs = torch.zeros(0, 3, 3)
            ftrs = torch.zeros(0, 64, 64) if self.feat_trs else None
        return (x, trs, ftrs) if self.ret_trs else x

