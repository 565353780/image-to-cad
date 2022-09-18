#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from image_to_cad.Loss.loss_functions import \
    cosine_distance, inverse_huber_loss

from image_to_cad.Metric.logging_metrics import depth_metrics

from image_to_cad.Model.depth.depth_features import DepthFeatures
from image_to_cad.Model.depth.depth_output import DepthOutput
from image_to_cad.Model.depth.sobel import Sobel

class DepthHead(nn.Module):
    def __init__(self, cfg, in_features):
        super().__init__()
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

        if cfg.MODEL.DEPTH_GRAD_LOSSES:
            self.sobel = Sobel()

        self.use_grad_losses = cfg.MODEL.DEPTH_GRAD_LOSSES
        self.use_batch_average = cfg.MODEL.DEPTH_BATCH_AVERAGE
        return

    @property
    def out_channels(self):
        return self.fpn_depth_features.out_channels

    def forward(self, data):
        if self.training:
            assert data['inputs']['image_depths'] is not None

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
            data = self.loss(data)
        return data

    def loss(self, data):
        if data['inputs']['image_depths'] is None:
            print("[WARN][DepthHead::loss]")
            print("\t data['inputs']['image_depths'] not exist!")
            return data

        mask = data['inputs']['image_depths'] > 1e-5
        flt = mask.flatten(1).any(1)

        if not flt.any():
            zero_loss = torch.tensor(0.0, device=mask.device)
            data['losses']['loss_image_depth'] = zero_loss
            if self.use_grad_losses:
                data['losses'].update({
                    'loss_grad_x': zero_loss.clone(),
                    'loss_grad_y': zero_loss.clone(),
                    'loss_normal': zero_loss.clone()
                })
            return data

        mask = mask[flt]
        depth_pred = data['predictions']['depths'][flt] * mask
        depth_gt = data['inputs']['image_depths'][flt] * mask

        # Directly compare the depths
        data['losses']['loss_image_depth'] = inverse_huber_loss(
            depth_pred, depth_gt,
            mask, mask_inputs=False,
            instance_average=self.use_batch_average)

        # Grad loss
        if self.use_grad_losses:
            gradx_pred, grady_pred = self.sobel(depth_pred).chunk(2, dim=1)
            gradx_gt, grady_gt = self.sobel(depth_gt).chunk(2, dim=1)
            data['losses']['loss_grad_x'] = inverse_huber_loss(
                gradx_pred, gradx_gt,
                mask, mask_inputs=False)
            data['losses']['loss_grad_y'] = inverse_huber_loss(
                grady_pred, grady_gt,
                mask, mask_inputs=False)

            # Normal consistency loss
            # https://github.com/JunjH/Revisiting_Single_Depth_Estimation/blob/master/train.py
            ones = torch.ones_like(gradx_pred)
            normal_pred = torch.cat([-gradx_pred, -grady_pred, ones], 1)
            normal_gt = torch.cat([-gradx_gt, -grady_gt, ones], 1)
            data['losses']['loss_normal'] = 5 * cosine_distance(
                normal_pred, normal_gt, mask)

        #FIXME: Log depth metrics to tensorboard
        #  metric_dict = depth_metrics(
            #  depth_pred, depth_gt,
            #  mask, mask_inputs=False,
            #  pref='depth/image_')
        return data

