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
from image_to_cad.Model.retrieval.retrieval_head import RetrievalHead

from image_to_cad.Method.common_ops import create_xy_grids, select_classes

class ROCAROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.set_verbose(False)

        self.init_class_weights(cfg)

        self.box_predictor = WeightedFastRCNNOutputLayers(cfg, self.box_head.output_shape)
        self.depth_head = DepthHead(cfg, self.in_features)
        self.alignment_head = AlignmentHead(cfg, self.num_classes, self.depth_head.out_channels)
        self.retrieval_head = RetrievalHead(cfg)
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

    def forward(self, inputs, predictions):
        if self.training:
            assert inputs['targets']
            assert inputs['image_depths'] is not None

        losses = {}
        extra_outputs = {}

        if self.training:
            predictions['label_and_sample_proposals'] = self.label_and_sample_proposals(
                predictions['proposals'],
                inputs['targets']
            )
        else:
            predictions['label_and_sample_proposals'] = None

        predictions, box_losses = self.forward_box(predictions)
        losses.update(box_losses)

        if self.training:
            predictions['instances'] = predictions['label_and_sample_proposals']
        if not self.training:
            predictions['instances'] = predictions['box_instances']

        predictions, depth_losses = self.depth_head(predictions)
        losses.update(depth_losses)
        extra_outputs['pred_image_depths'] = predictions['depths']

        pred_instances, alignment_outputs, alignment_losses = self.forward_alignment(inputs, predictions)
        losses.update(alignment_losses)
        extra_outputs.update(alignment_outputs)

        return pred_instances, extra_outputs, losses

    def forward_box(self, predictions):
        self.box_predictor.set_class_weights(self.class_weights)
        losses = {}
        pred_instances = None
        if self.training:
            losses = self._forward_box(
                predictions['features'],
                predictions['proposals']
            )
        else:
            pred_instances = self._forward_box(
                predictions['features'],
                predictions['proposals']
            )
            predictions['box_instances'] = pred_instances
        return predictions, losses

    def forward_alignment(self, inputs, predictions):
        losses = {}
        extra_outputs = {}

        features = predictions['features']
        instances = predictions['instances']
        image_size = inputs['image_size']
        depths = predictions['depths']
        depth_features = predictions['depth_features']
        if self.training:
            inference_args = None
        else:
            inference_args = inputs['targets']
        gt_depths = predictions['depths']
        scenes = inputs['scenes']

        if self.training:
            instances, _ = select_foreground_proposals(
                instances, self.num_classes
            )
        else:
            score_flt = [p.scores >= self.test_min_score for p in instances]
            instances = [p[flt] for p, flt in zip(instances, score_flt)]

        pool_boxes = None
        if self.training:
            pool_boxes = [x.proposal_boxes for x in instances]
        else:
            pool_boxes = [x.pred_boxes for x in instances]

        features = [features[f] for f in self.in_features]
        features = self.mask_pooler(features, pool_boxes)

        gt_classes = None
        if self.training:
            gt_classes = L.cat([p.gt_classes for p in instances])

        mask_classes = gt_classes
        if not self.training:
            pred_classes = [x.pred_classes for x in instances]
            mask_classes = L.cat(pred_classes)

        class_weights = None
        if self.training:
            class_weights = self.class_weights[gt_classes + 1]

        # Create xy-grids for back-projection and cropping, respectively
        xy_grid, xy_grid_n = create_xy_grids(
            Boxes.cat(pool_boxes),
            image_size,
            features.size(0),
            self.output_grid_size
        )

        mask_probs, mask_pred, mask_gt, mask_losses = self.forward_mask(
            features,
            mask_classes,
            instances,
            xy_grid_n,
            class_weights
        )
        losses.update(mask_losses)
        predictions['pred_masks'] = mask_probs

        alignment_classes = gt_classes
        if not self.training:
            pred_classes = [x.pred_classes for x in instances]
            alignment_classes = L.cat(pred_classes)

        alignment_predictions, alignment_losses = self.alignment_head(
            instances,
            depth_features,
            depths,
            image_size,
            mask_probs,
            mask_pred,
            xy_grid,
            xy_grid_n,
            alignment_classes,
            inference_args,
            gt_depths,
            class_weights,
            mask_gt
        )
        losses.update(alignment_losses)
        predictions.update(alignment_predictions)

        instance_sizes = [len(x) for x in instances]

        pred_indices, cad_ids, retrieval_losses = self.retrieval_head(
            alignment_classes,
            mask_pred,
            predictions['nocs'],
            predictions['shape_code'],
            instance_sizes,
            predictions['has_alignment'],
            scenes,
            instances,
            predictions['pred_scales'],
            predictions['pred_translations'],
            predictions['pred_rotations'],
            predictions['depth_points'],
            predictions['raw_nocs'],
            mask_pred
        )
        losses.update(retrieval_losses)
        predictions['pred_indices'] = pred_indices
        extra_outputs['cad_ids'] = cad_ids

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
        if self.training:
            assert instances is not None
            assert xy_grid_n is not None

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

        return mask_probs, mask_pred, mask_gt, losses

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

