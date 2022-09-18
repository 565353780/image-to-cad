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
            predictions['label_and_sample_proposals'] = predictions['proposals']

        predictions, box_losses = self.forward_box(predictions)
        losses.update(box_losses)

        if self.training:
            predictions['instances'] = predictions['label_and_sample_proposals']
        else:
            predictions['instances'] = predictions['box_instances']

        predictions, depth_losses = self.depth_head(inputs, predictions)
        losses.update(depth_losses)
        extra_outputs['pred_image_depths'] = predictions['depths']

        predictions, alignment_outputs, alignment_losses = self.forward_alignment(inputs, predictions)
        losses.update(alignment_losses)
        extra_outputs.update(alignment_outputs)

        return predictions['alignment_instances'], extra_outputs, losses

    def forward_box(self, predictions):
        losses = {}

        self.box_predictor.set_class_weights(self.class_weights)

        pred_instances = None
        if self.training:
            losses = self._forward_box(
                predictions['features'],
                predictions['label_and_sample_proposals']
            )
        else:
            pred_instances = self._forward_box(
                predictions['features'],
                predictions['label_and_sample_proposals']
            )

        predictions['box_instances'] = pred_instances
        return predictions, losses

    def forward_alignment(self, inputs, predictions):
        losses = {}
        extra_outputs = {}

        if self.training:
            inputs['inference_args'] = None
        else:
            inputs['inference_args'] = inputs['targets']

        if self.training:
            predictions['alignment_instances'], _ = select_foreground_proposals(
                predictions['instances'],
                self.num_classes
            )
        else:
            score_flt = [p.scores >= self.test_min_score for p in predictions['instances']]
            predictions['alignment_instances'] = [p[flt] for p, flt in zip(predictions['instances'], score_flt)]

        pool_boxes = None
        if self.training:
            pool_boxes = [x.proposal_boxes for x in predictions['alignment_instances']]
        else:
            pool_boxes = [x.pred_boxes for x in predictions['alignment_instances']]
        predictions['pool_boxes'] = pool_boxes

        features = [predictions['features'][f] for f in self.in_features]
        features = self.mask_pooler(features, pool_boxes)
        predictions['alignment_features'] = features

        gt_classes = None
        if self.training:
            gt_classes = L.cat([p.gt_classes for p in predictions['alignment_instances']])

        mask_classes = gt_classes
        if not self.training:
            pred_classes = [x.pred_classes for x in predictions['alignment_instances']]
            mask_classes = L.cat(pred_classes)
        predictions['mask_classes'] = mask_classes

        class_weights = None
        if self.training:
            class_weights = self.class_weights[gt_classes + 1]
        predictions['class_weights'] = class_weights

        # Create xy-grids for back-projection and cropping, respectively
        xy_grid, xy_grid_n = create_xy_grids(
            Boxes.cat(pool_boxes),
            inputs['image_size'],
            predictions['alignment_features'].size(0),
            self.output_grid_size
        )
        predictions['xy_grid'] = xy_grid
        predictions['xy_grid_n'] = xy_grid_n

        predictions, mask_losses = self.forward_mask(predictions)
        losses.update(mask_losses)

        alignment_classes = gt_classes
        if not self.training:
            pred_classes = [x.pred_classes for x in predictions['alignment_instances']]
            alignment_classes = L.cat(pred_classes)
        predictions['alignment_classes'] = alignment_classes

        predictions['alignment_instance_sizes'] = [len(x) for x in predictions['alignment_instances']]

        predictions, alignment_losses = self.alignment_head(inputs, predictions)
        losses.update(alignment_losses)

        predictions, retrieval_losses = self.retrieval_head(inputs, predictions)
        losses.update(retrieval_losses)

        extra_outputs['cad_ids'] = predictions['cad_ids']

        if not self.training:
            # Fill the instances
            for name, preds in predictions.items():
                pred_list = None
                try:
                    pred_list = preds.split(predictions['alignment_instance_sizes'])
                except:
                    continue
                for instance, pred in zip(predictions['alignment_instances'], pred_list):
                    setattr(instance, name, pred)
        return predictions, extra_outputs, losses

    def forward_mask(self, predictions):
        if self.training:
            assert predictions['alignment_instances'] is not None
            assert predictions['xy_grid_n'] is not None

        mask_logits = self.mask_head.layers(predictions['alignment_features'])
        mask_logits = select_classes(mask_logits, self.num_classes + 1, predictions['mask_classes'])
        predictions['mask_logits'] = mask_logits

        predictions['mask_probs'] = torch.sigmoid(mask_logits)

        if self.training:
            predictions['mask_pred'] = predictions['mask_probs'] > 0.5
        else:
            predictions['mask_pred'] = predictions['mask_probs'] > 0.7

        predictions['mask_gt'] = None
        if self.training:
            predictions['mask_gt'] = Masks\
                .cat([p.gt_masks for p in predictions['alignment_instances']])\
                .crop_and_resize_with_grid(predictions['xy_grid_n'], self.output_grid_size)

        losses = self.mask_loss(predictions)

        predictions['mask_pred'] = predictions['mask_pred'].to(predictions['mask_probs'].dtype)

        return predictions, losses

    def mask_loss(self, predictions):
        losses = {}

        if predictions['mask_gt'] is None:
            return losses

        losses['loss_mask'] = binary_cross_entropy_with_logits(
            predictions['mask_logits'],
            predictions['mask_gt'],
            predictions['class_weights']
        )
        losses['loss_mask_iou'] = mask_iou_loss(
            predictions['mask_probs'],
            predictions['mask_gt'],
            predictions['class_weights']
        )

        #FIXME: log this loss
        # Log the mask performance and then convert mask_pred to float
        metric_dict = mask_metrics(predictions['mask_pred'], predictions['mask_gt'].bool())
        print("[INFO][ROCAROIHeads::mask_loss]")
        print("\t mask metric is")
        print(metric_dict)

        return losses

