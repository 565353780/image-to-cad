#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from image_to_cad.Model.depth.depth_features import DepthFeatures
from image_to_cad.Model.depth.depth_output import DepthOutput

from image_to_cad.Loss.loss_functions import inverse_huber_loss

from image_to_cad.Metric.logging_metrics import depth_metrics

class DepthHead(nn.Module):
    def __init__(self, cfg, in_features):
        super().__init__()
        self.device = torch.device("cuda")
        self.in_features = in_features

        # FIXME: This calculation must change if resizing!
        # depth_width = cfg.INPUT.DEPTH_RES[-1]
        # up_ratio = depth_width / 160
        up_ratio = 4
        feat_size = tuple(d // up_ratio for d in cfg.INPUT.DEPTH_RES)

        self.fpn_depth_features = DepthFeatures(size=feat_size)
        self.fpn_depth_output = DepthOutput(
            self.fpn_depth_features.out_channels,
            up_ratio
        )

        self.use_batch_average = cfg.MODEL.DEPTH_BATCH_AVERAGE
        return

    @property
    def out_channels(self):
        return self.fpn_depth_features.out_channels

    def forward(self, data):
        if self.training:
            image_depths = []
            for batched_input in data['inputs']['batched_inputs']:
                image_depths.append(batched_input.pop('image_depth'))
            data['inputs']['image_depths'] = torch.cat(image_depths, dim=0).to(self.device)
            assert data['inputs']['image_depths'] is not None
        else:
            data['inputs']['image_depths'] = None

        if self.training:
            mask = data['inputs']['image_depths'] > 1e-5
            flt = mask.flatten(1).any(1)

            if not flt.any():
                depth_features = torch.zeros(
                    data['inputs']['image_depths'].size(0),
                    self.fpn_depth_features.out_channels,
                    *self.fpn_depth_features.size,
                    device=data['inputs']['image_depths'].device
                )
                depth_pred = torch.zeros_like(data['inputs']['image_depths'])
                data['predictions']['depths'] = depth_pred
                data['predictions']['depth_features'] = depth_features
                return data

        features = [data['predictions']['features'][f] for f in self.in_features]

        depth_features = self.fpn_depth_features(features)
        depth_pred = self.fpn_depth_output(depth_features)
        data['predictions']['depths'] = depth_pred
        data['predictions']['depth_features'] = depth_features

        if self.training:
            data = self.depth_loss(data)
        return data

    def depth_loss(self, data):
        assert data['inputs']['image_depths'] is not None

        mask = data['inputs']['image_depths'] > 1e-5
        flt = mask.flatten(1).any(1)

        if not flt.any():
            zero_loss = torch.tensor(0.0, device=mask.device)
            data['losses']['loss_image_depth'] = zero_loss
            return data

        mask = mask[flt]
        depth_pred = data['predictions']['depths'][flt] * mask
        depth_gt = data['inputs']['image_depths'][flt] * mask

        # Directly compare the depths
        data['losses']['loss_image_depth'] = inverse_huber_loss(
            depth_pred, depth_gt,
            mask, mask_inputs=False,
            instance_average=self.use_batch_average)

        depth_metric_dict = depth_metrics(
            depth_pred, depth_gt,
            mask, mask_inputs=False,
            pref='depth/image_')
        data['logs'].update(depth_metric_dict)
        return data

