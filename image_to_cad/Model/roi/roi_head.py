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

        if self.training:
            assert targets
            assert gt_depths is not None
            proposals = self.label_and_sample_proposals(proposals, targets)

            losses = self._forward_box(features, proposals)

            depth_losses, depths, depth_features = self.depth_head(features, gt_depths)
            losses.update(depth_losses)

            losses.update(self._forward_alignment(
                features,
                proposals,
                image_size,
                depths,
                depth_features,
                gt_depths=gt_depths
            ))
            return proposals, losses

        inference_args = targets  # Extra arguments for inference

        pred_instances = self._forward_box(features, proposals)

        pred_depths, depth_features = self.depth_head(features, None)
        extra_outputs = {'pred_image_depths': pred_depths}

        pred_instances, alignment_outputs = self._forward_alignment(
            features,
            pred_instances,
            image_size,
            pred_depths,
            depth_features,
            inference_args=inference_args,
            scenes=scenes
        )
        extra_outputs.update(alignment_outputs)

        return pred_instances, extra_outputs

    def _forward_box(self, *args, **kwargs):
        self.box_predictor.set_class_weights(self.class_weights)
        return super()._forward_box(*args, **kwargs)

    def _forward_alignment(self, features, instances, image_size,
                           depths,depth_features, inference_args=None,
                           gt_depths=None, scenes=None):
        features = [features[f] for f in self.in_features]

        if self.training:
            return self._forward_alignment_train(
                features,
                instances,
                image_size,
                depths,
                depth_features,
                gt_depths)

        return self._forward_alignment_inference(
            features,
            instances,
            image_size,
            depths,
            depth_features,
            inference_args,
            scenes)

    def _forward_alignment_train(self, features, instances, image_size,
                                 depths, depth_features, gt_depths):
        losses = {}

        # Declare some useful variables
        instances, _ = select_foreground_proposals(
            instances, self.num_classes
        )
        proposal_boxes = [x.proposal_boxes for x in instances]
        if self.train_on_pred_boxes:
            for pb in proposal_boxes:
                pb.clip(image_size)
        features = self.mask_pooler(features, proposal_boxes)
        boxes = Boxes.cat(proposal_boxes)
        batch_size = features.size(0)
        gt_classes = L.cat([p.gt_classes for p in instances])

        # Get class weight for losses
        class_weights = self.class_weights[gt_classes + 1]

        # Create xy-grids for back-projection and cropping, respectively
        xy_grid, xy_grid_n = create_xy_grids(
            boxes,
            image_size,
            batch_size,
            self.output_grid_size
        )

        # Mask
        mask_losses, mask_probs, mask_pred, mask_gt = self._forward_mask(
            features,
            gt_classes,
            instances,
            xy_grid_n=xy_grid_n,
            class_weights=class_weights
        )
        losses.update(mask_losses)

        inference_args = None
        scenes = None
        predictions, alignment_losses, extra_outputs = self.alignment_head.forward_new(
            instances, depth_features, depths,
            image_size, mask_probs, mask_pred,
            inference_args, scenes, gt_depths,
            gt_classes, class_weights, xy_grid,
            xy_grid_n, mask_gt
        )

        losses.update(alignment_losses)
        return losses

    def _forward_alignment_inference(self, features, instances, image_size,
                                     depths, depth_features, inference_args,
                                     scenes=None):
        score_flt = [p.scores >= self.test_min_score for p in instances]
        instances = [p[flt] for p, flt in zip(instances, score_flt)]

        pred_classes = [x.pred_classes for x in instances]
        pred_boxes = [x.pred_boxes for x in instances]
        instance_sizes = [len(x) for x in instances]
        features = self.mask_pooler(features, pred_boxes)

        # Predict the mask
        pred_classes = L.cat(pred_classes)
        pred_mask_probs, pred_masks = self._forward_mask(
            features, pred_classes
        )

        # Predict alignments
        old_predictions, old_extra_outputs = self.alignment_head(
            instances, depth_features, depths,
            image_size, pred_mask_probs, pred_masks,
            inference_args=inference_args,
            scenes=scenes
        )

        gt_depths = None
        gt_classes = None
        class_weights = None
        xy_grid = None
        xy_grid_n = None
        mask_gt = None
        predictions, alignment_losses, extra_outputs = self.alignment_head.forward_new(
            instances, depth_features, depths,
            image_size, pred_mask_probs, pred_masks,
            inference_args, scenes, gt_depths,
            gt_classes, class_weights, xy_grid,
            xy_grid_n, mask_gt
        )

        print("=========================")
        print(old_predictions)
        print(predictions)
        print("=========================")
        print(old_extra_outputs)
        print(extra_outputs)
        exit()

        predictions['pred_masks'] = pred_mask_probs

        # Fill the instances
        for name, preds in predictions.items():
            for instance, pred in zip(instances, preds.split(instance_sizes)):
                setattr(instance, name, pred)
        return instances, extra_outputs

    def _forward_mask(self, features, classes, instances=None,
                      xy_grid_n=None, class_weights=None):
        mask_logits = self.mask_head.layers(features)
        mask_logits = select_classes(mask_logits, self.num_classes + 1, classes)

        if self.training:
            assert instances is not None
            assert xy_grid_n is not None

            losses = {}

            mask_probs = torch.sigmoid(mask_logits)
            mask_pred = mask_probs > 0.5

            mask_gt = Masks\
                .cat([p.gt_masks for p in instances])\
                .crop_and_resize_with_grid(xy_grid_n, self.output_grid_size)

            losses['loss_mask'] = binary_cross_entropy_with_logits(
                mask_logits, mask_gt, class_weights
            )
            losses['loss_mask_iou'] = mask_iou_loss(
                mask_probs, mask_gt, class_weights
            )

            # Log the mask performance and then convert mask_pred to float
            metric_dict = mask_metrics(mask_pred, mask_gt.bool())
            print("[INFO][ROCAROIHeads::_forward_mask]")
            print("\t mask_metrics")
            print(metric_dict)

            mask_pred = mask_pred.to(mask_gt.dtype)

            return losses, mask_probs, mask_pred, mask_gt

        mask_probs = torch.sigmoid_(mask_logits)
        mask_pred = (mask_probs > 0.7).to(mask_probs.dtype)
        return mask_probs, mask_pred

