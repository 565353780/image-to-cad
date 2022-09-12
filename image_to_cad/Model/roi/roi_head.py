#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import detectron2.layers as L
from detectron2.modeling import StandardROIHeads
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.structures import Boxes

from image_to_cad.Data.masks.masks import Masks

from image_to_cad.Loss.loss_functions import \
    binary_cross_entropy_with_logits, mask_iou_loss

from image_to_cad.Metric.logging_metrics import mask_metrics

from image_to_cad.Model.depth.depth_head import DepthHead
from image_to_cad.Model.roi.weighted_fast_rcnn_output_layers import WeightedFastRCNNOutputLayers
from image_to_cad.Model.alignment.alignment_head import AlignmentHead

from image_to_cad.Method.common_ops import create_xy_grids, select_classes

class ROCAROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.set_verbose(False)

        self.init_class_weights(cfg)

        self.box_predictor = WeightedFastRCNNOutputLayers(cfg, self.box_head.output_shape)
        self.depth_head = DepthHead(cfg, self.in_features)
        self.alignment_head = AlignmentHead(cfg, self.num_classes, self.depth_head.out_channels)
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

    @property
    def has_cads(self):
        return self.alignment_head.has_cads

    def set_verbose(self, verbose=True):
        self.verbose = verbose
        return

    def forward(self, images, features, proposals,
                targets=None, gt_depths=None, scenes=None):
        image_size = images[0].shape[-2:]  # Assume single image size!

        losses = {}

        if self.training:
            assert targets
            assert gt_depths is not None
            proposals = self.label_and_sample_proposals(proposals, targets)

            box_losses = self._forward_box(features, proposals)
            losses.update(box_losses)

            depths, depth_features = self.depth_head(features, gt_depths)
            depth_losses = self.depth_head.loss(depths, gt_depths)
            losses.update(depth_losses)

            _, _, alignment_losses = self.forward_alignment(
                features,
                proposals,
                image_size,
                depths,
                depth_features,
                None,
                gt_depths,
                None
            )
            losses.update(alignment_losses)
            return proposals, losses

        inference_args = targets  # Extra arguments for inference

        pred_instances = self._forward_box(features, proposals)

        pred_depths, depth_features = self.depth_head(features)
        extra_outputs = {'pred_image_depths': pred_depths}

        pred_instances, alignment_outputs, _ = self.forward_alignment(
            features,
            pred_instances,
            image_size,
            pred_depths,
            depth_features,
            inference_args,
            None,
            scenes
        )
        extra_outputs.update(alignment_outputs)

        return pred_instances, extra_outputs

    def _forward_box(self, *args, **kwargs):
        self.box_predictor.set_class_weights(self.class_weights)
        return super()._forward_box(*args, **kwargs)

    def forward_alignment(
        self,
        features,
        instances,
        image_size,
        depths,
        depth_features,
        inference_args=None,
        gt_depths=None,
        scenes=None
    ):
        features = [features[f] for f in self.in_features]

        losses = {}

        if self.training:
            # Declare some useful variables
            instances, _ = select_foreground_proposals(
                instances, self.num_classes
            )
        else:
            score_flt = [p.scores >= self.test_min_score for p in instances]
            instances = [p[flt] for p, flt in zip(instances, score_flt)]

        mask_classes = None
        xy_grid = None
        xy_grid_n = None
        class_weights = None
        gt_classes = None
        if self.training:
            proposal_boxes = [x.proposal_boxes for x in instances]
            if self.train_on_pred_boxes:
                for pb in proposal_boxes:
                    pb.clip(image_size)
            features = self.mask_pooler(features, proposal_boxes)
            boxes = Boxes.cat(proposal_boxes)
            batch_size = features.size(0)
            gt_classes = L.cat([p.gt_classes for p in instances])
            mask_classes = gt_classes

            # Get class weight for losses
            class_weights = self.class_weights[gt_classes + 1]

            # Create xy-grids for back-projection and cropping, respectively
            xy_grid, xy_grid_n = create_xy_grids(
                boxes,
                image_size,
                batch_size,
                self.output_grid_size
            )
        else:
            pred_classes = [x.pred_classes for x in instances]
            mask_classes = L.cat(pred_classes)
            pred_boxes = [x.pred_boxes for x in instances]
            features = self.mask_pooler(features, pred_boxes)

        instance_sizes = [len(x) for x in instances]

        # Mask
        mask_probs, mask_pred, mask_losses, mask_gt = self.forward_mask(
            features,
            mask_classes,
            instances,
            xy_grid_n,
            class_weights
        )

        losses.update(mask_losses)

        predictions, alignment_losses, extra_outputs = self.alignment_head(
            instances, depth_features, depths,
            image_size, mask_probs, mask_pred,
            inference_args, scenes, gt_depths,
            gt_classes, class_weights, xy_grid,
            xy_grid_n, mask_gt
        )
        losses.update(alignment_losses)

        predictions['pred_masks'] = mask_probs

        if not self.training:
            # Fill the instances
            for name, preds in predictions.items():
                for instance, pred in zip(instances, preds.split(instance_sizes)):
                    setattr(instance, name, pred)
        return instances, extra_outputs, losses

    def forward_mask(
        self,
        features,
        classes,
        instances=None,
        xy_grid_n=None,
        class_weights=None
    ):
        mask_logits = self.mask_head.layers(features)
        mask_logits = select_classes(mask_logits, self.num_classes + 1, classes)

        mask_probs = torch.sigmoid(mask_logits)

        mask_pred = None
        if self.training:
            mask_pred = mask_probs > 0.5
        else:
            mask_pred = mask_probs > 0.7

        mask_gt = None
        if self.training:
            assert instances is not None
            assert xy_grid_n is not None
            mask_gt = Masks\
                .cat([p.gt_masks for p in instances])\
                .crop_and_resize_with_grid(xy_grid_n, self.output_grid_size)

        losses = self.mask_loss(
            mask_logits,
            mask_probs,
            mask_pred,
            mask_gt,
            class_weights)

        mask_pred = mask_pred.to(mask_probs.dtype)

        return mask_probs, mask_pred, losses, mask_gt

    def mask_loss(
        self,
        mask_logits,
        mask_probs,
        mask_pred,
        mask_gt=None,
        class_weights=None
    ):
        losses = {}

        if mask_gt is None:
            return losses

        losses['loss_mask'] = binary_cross_entropy_with_logits(
            mask_logits, mask_gt, class_weights
        )
        losses['loss_mask_iou'] = mask_iou_loss(
            mask_probs, mask_gt, class_weights
        )

        #FIXME: log this loss
        # Log the mask performance and then convert mask_pred to float
        metric_dict = mask_metrics(mask_pred, mask_gt.bool())
        print("[INFO][ROCAROIHeads::mask_loss]")
        print("\t mask metric is")
        print(metric_dict)

        return losses

