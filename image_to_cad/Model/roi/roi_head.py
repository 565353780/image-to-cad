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
        self.device = torch.device("cuda")

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

    def forward_box(self, data):
        self.box_predictor.set_class_weights(self.class_weights)

        if self.training:
            data['predictions']['instances'] = data['predictions']['label_and_sample_proposals']

            box_losses = self._forward_box(
                data['predictions']['features'],
                data['predictions']['label_and_sample_proposals']
            )
            data['losses'].update(box_losses)
        else:
            data['predictions']['instances'] = self._forward_box(
                data['predictions']['features'],
                data['predictions']['label_and_sample_proposals']
            )
        return data

    def forward_alignment(self, data):
        if self.training:
            data['inputs']['inference_args'] = None
        else:
            data['inputs']['inference_args'] = data['inputs']['targets']

        if self.training:
            data['predictions']['alignment_instances'], _ = select_foreground_proposals(
                data['predictions']['instances'],
                self.num_classes
            )
        else:
            score_flt = [p.scores >= self.test_min_score for p in data['predictions']['instances']]
            data['predictions']['alignment_instances'] = \
                [p[flt] for p, flt in zip(data['predictions']['instances'], score_flt)]

        if self.training:
            data['predictions']['pool_boxes'] = \
                [x.proposal_boxes for x in data['predictions']['alignment_instances']]
        else:
            data['predictions']['pool_boxes'] = \
                [x.pred_boxes for x in data['predictions']['alignment_instances']]

        features = [data['predictions']['features'][f] for f in self.in_features]
        data['predictions']['alignment_features'] = self.mask_pooler(
            features,
            data['predictions']['pool_boxes']
        )

        if self.training:
            data['predictions']['gt_classes'] = \
                L.cat([p.gt_classes for p in data['predictions']['alignment_instances']])
        else:
            data['predictions']['gt_classes'] = None

        if self.training:
            data['predictions']['mask_classes'] = data['predictions']['gt_classes']
        else:
            pred_classes = [x.pred_classes for x in data['predictions']['alignment_instances']]
            data['predictions']['mask_classes'] = L.cat(pred_classes)

        if self.training:
            data['predictions']['class_weights'] = self.class_weights[
                data['predictions']['gt_classes'] + 1]
        else:
            data['predictions']['class_weights'] = None

        # Create xy-grids for back-projection and cropping, respectively
        data['predictions']['xy_grid'], data['predictions']['xy_grid_n'] = create_xy_grids(
            Boxes.cat(data['predictions']['pool_boxes']),
            data['inputs']['image_size'],
            data['predictions']['alignment_features'].size(0),
            self.output_grid_size
        )

        data = self.forward_mask(data)

        if self.training:
            data['predictions']['alignment_classes'] = data['predictions']['gt_classes']
        else:
            pred_classes = [x.pred_classes for x in data['predictions']['alignment_instances']]
            data['predictions']['alignment_classes'] = L.cat(pred_classes)

        data['predictions']['alignment_instance_sizes'] = \
            [len(x) for x in data['predictions']['alignment_instances']]

        data = self.alignment_head(data)
        return data

    def forward(self, data):
        if self.training:
            data['inputs']['targets'] = data['inputs']['gt_instances']
        else:
            data['inputs']['targets'] = [
                {'intrinsics': input['intrinsics'].to(self.device)}
                for input in data['inputs']['batched_inputs']]

        if self.training:
            image_depths = []
            for batched_input in data['inputs']['batched_inputs']:
                image_depths.append(batched_input.pop('image_depth'))
            data['inputs']['image_depths'] = torch.cat(image_depths, dim=0).to(self.device)
        else:
            data['inputs']['image_depths'] = None

        if self.training:
            data['inputs']['scenes'] = None
        else:
            data['inputs']['scenes'] = [batched_input['scene'] for batched_input in data['inputs']['batched_inputs']]

        #==============================
        #  data = self.prepareData(data)
        if self.training:
            assert data['inputs']['targets']
            assert data['inputs']['image_depths'] is not None

        if self.training:
            data['predictions']['label_and_sample_proposals'] = self.label_and_sample_proposals(
                data['predictions']['proposals'],
                data['inputs']['targets']
            )
        else:
            data['predictions']['label_and_sample_proposals'] = data['predictions']['proposals']

        data = self.forward_box(data)

        data = self.depth_head(data)

        data = self.forward_alignment(data)

        data = self.retrieval_head(data)

        if not self.training:
            # Fill the instances
            for name, preds in data['predictions'].items():
                pred_list = None
                try:
                    pred_list = preds.split(data['predictions']['alignment_instance_sizes'])
                except:
                    continue
                for instance, pred in zip(data['predictions']['alignment_instances'], pred_list):
                    setattr(instance, name, pred)
        return data

    def forward_mask(self, data):
        if self.training:
            assert data['predictions']['alignment_instances'] is not None
            assert data['predictions']['xy_grid_n'] is not None

        mask_logits = self.mask_head.layers(data['predictions']['alignment_features'])
        data['predictions']['mask_logits'] = select_classes(
            mask_logits,
            self.num_classes + 1,
            data['predictions']['mask_classes']
        )

        data['predictions']['mask_probs'] = torch.sigmoid(data['predictions']['mask_logits'])

        if self.training:
            data['predictions']['mask_pred'] = data['predictions']['mask_probs'] > 0.5
        else:
            data['predictions']['mask_pred'] = data['predictions']['mask_probs'] > 0.7

        if self.training:
            data['predictions']['mask_gt'] = Masks\
                .cat([p.gt_masks for p in data['predictions']['alignment_instances']])\
                .crop_and_resize_with_grid(data['predictions']['xy_grid_n'], self.output_grid_size)
        else:
            data['predictions']['mask_gt'] = None

        if self.training:
            data = self.mask_loss(data)

        data['predictions']['mask_pred'] = data['predictions']['mask_pred'].to(
            data['predictions']['mask_probs'].dtype)

        return data

    def mask_loss(self, data):
        if data['predictions']['mask_gt'] is None:
            print("[WARN][ROCAROIHead::mask_loss]")
            print("\t data['predictions']['mask_gt'] not exist!")
            return data

        data['losses']['loss_mask'] = binary_cross_entropy_with_logits(
            data['predictions']['mask_logits'],
            data['predictions']['mask_gt'],
            data['predictions']['class_weights']
        )
        data['losses']['loss_mask_iou'] = mask_iou_loss(
            data['predictions']['mask_probs'],
            data['predictions']['mask_gt'],
            data['predictions']['class_weights']
        )

        #FIXME: log this loss
        # Log the mask performance and then convert mask_pred to float
        #  metric_dict = mask_metrics(
            #  data['predictions']['mask_pred'],
            #  data['predictions']['mask_gt'].bool()
        #  )
        return data

