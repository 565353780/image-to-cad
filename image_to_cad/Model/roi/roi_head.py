#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from detectron2.modeling import StandardROIHeads

from image_to_cad.Model.roi.weighted_fast_rcnn_output_layers import WeightedFastRCNNOutputLayers

class ROIHead(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.device = torch.device("cuda")

        self.set_verbose(False)

        self.init_class_weights(cfg)

        self.box_predictor = WeightedFastRCNNOutputLayers(cfg, self.box_head.output_shape)
        self.mask_head.predictor = nn.Conv2d(
            self.mask_head.deconv.out_channels, self.num_classes + 1, kernel_size=(1, 1))

        self.output_grid_size = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION * 2

        self.test_min_score = cfg.MODEL.ROI_HEADS.CONFIDENCE_THRESH_TEST
        return

    def init_class_weights(self, cfg):
        class_weights = cfg.MODEL.CLASS_SCALES
        class_weight_tensor = torch.zeros(1 + len(class_weights))
        for i, scale in class_weights:
            class_weight_tensor[i + 1] = scale
        class_weight_tensor[0] = torch.max(class_weight_tensor[1:])
        self.register_buffer('class_weights', class_weight_tensor)
        return

    def set_verbose(self, verbose=True):
        self.verbose = verbose
        return

