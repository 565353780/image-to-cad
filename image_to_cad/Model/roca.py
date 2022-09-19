#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import detectron2.layers as L

from detectron2.layers import ShapeSpec
from detectron2.structures import ImageList, Boxes
from detectron2.modeling.proposal_generator.rpn import RPN
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.modeling.postprocessing import detector_postprocess

from image_to_cad.Config.roca.constants import VOXEL_RES

from image_to_cad.Data.masks.masks import Masks

from image_to_cad.Method.common_ops import create_xy_grids, select_classes
from image_to_cad.Method.misc import make_dense_volume

from image_to_cad.Model.roi.roi_head import ROIHead
from image_to_cad.Model.depth.depth_head import DepthHead
from image_to_cad.Model.alignment.alignment_head import AlignmentHead
from image_to_cad.Model.retrieval.retrieval_head import RetrievalHead

from image_to_cad.Loss.loss_functions import \
    binary_cross_entropy_with_logits, mask_iou_loss

from image_to_cad.Metric.logging_metrics import mask_metrics

class ROCA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device("cuda")

        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

        self.backbone = build_resnet_fpn_backbone(cfg, input_shape)
        self.proposal_generator = RPN(cfg, self.backbone.output_shape())
        self.roi_head = ROIHead(cfg, self.backbone.output_shape())
        self.depth_head = DepthHead(cfg, self.roi_head.in_features)
        self.alignment_head = AlignmentHead(cfg, self.roi_head.num_classes, self.depth_head.out_channels)
        self.retrieval_head = RetrievalHead()

        self.output_grid_size = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION * 2
        self.test_min_score = cfg.MODEL.ROI_HEADS.CONFIDENCE_THRESH_TEST

        pixel_mean = cfg.MODEL.PIXEL_MEAN
        pixel_std = cfg.MODEL.PIXEL_STD
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        return

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def postprocess(instances, batched_inputs, image_sizes):
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def forward_backbone(self, data):
        data['inputs']['images'] = self.preprocess_image(data['inputs']['batched_inputs'])

        data['predictions']['features'] = self.backbone(data['inputs']['images'].tensor)
        return data

    def forward_proposals(self, data):
        if self.training:
            data['inputs']['gt_instances'] = [
                x['instances'].to(self.device) for x in data['inputs']['batched_inputs']]
            assert data['inputs']['gt_instances']
        else:
            data['inputs']['gt_instances'] = None

        data['predictions']['proposals'], proposal_losses = self.proposal_generator(
            data['inputs']['images'],
            data['predictions']['features'],
            data['inputs']['gt_instances']
        )
        data['losses'].update(proposal_losses)
        return data

    def forward_box(self, data):
        self.roi_head.box_predictor.set_class_weights(self.roi_head.class_weights)

        if self.training:
            data['predictions']['instances'] = self.roi_head.label_and_sample_proposals(
                data['predictions']['proposals'],
                data['inputs']['gt_instances']
            )

            box_losses = self.roi_head._forward_box(
                data['predictions']['features'],
                data['predictions']['instances']
            )
            data['losses'].update(box_losses)
        else:
            data['predictions']['instances'] = self.roi_head._forward_box(
                data['predictions']['features'],
                data['predictions']['proposals']
            )
        return data

    def forward_mask(self, data):
        if self.training:
            data['predictions']['alignment_instances'], _ = select_foreground_proposals(
                data['predictions']['instances'],
                self.roi_head.num_classes
            )
            assert data['predictions']['alignment_instances'] is not None
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

        features = [data['predictions']['features'][f] for f in self.roi_head.in_features]
        data['predictions']['alignment_features'] = self.roi_head.mask_pooler(
            features,
            data['predictions']['pool_boxes']
        )

        mask_logits = self.roi_head.mask_head.layers(data['predictions']['alignment_features'])

        image_size = data['inputs']['images'][0].shape[-2:]

        # Create xy-grids for back-projection and cropping, respectively
        data['predictions']['xy_grid'], data['predictions']['xy_grid_n'] = create_xy_grids(
            Boxes.cat(data['predictions']['pool_boxes']),
            image_size,
            data['predictions']['alignment_features'].size(0),
            self.output_grid_size
        )

        if self.training:
            assert data['predictions']['xy_grid_n'] is not None

        if self.training:
            data['predictions']['gt_classes'] = \
                L.cat([p.gt_classes for p in data['predictions']['alignment_instances']])

        if self.training:
            mask_classes = data['predictions']['gt_classes']
        else:
            pred_classes = [x.pred_classes for x in data['predictions']['alignment_instances']]
            mask_classes = L.cat(pred_classes)

        data['predictions']['mask_logits'] = select_classes(
            mask_logits,
            self.roi_head.num_classes + 1,
            mask_classes
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
            assert data['predictions']['mask_gt'] is not None

        if self.training:
            data['predictions']['class_weights'] = self.roi_head.class_weights[
                data['predictions']['gt_classes'] + 1]
        else:
            data['predictions']['class_weights'] = None

        if self.training:
            data = self.mask_loss(data)
        return data

    def mask_loss(self, data):
        if data['predictions']['mask_gt'] is None:
            print("[WARN][losses::recordMaskLoss]")
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

        #  Log the mask performance and then convert mask_pred to float
        mask_metric_dict = mask_metrics(
            data['predictions']['mask_pred'],
            data['predictions']['mask_gt'].bool()
        )
        data['logs'].update(mask_metric_dict)
        return data

    def postProcess(self, data):
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

        if self.training:
            data['predictions']['post_results'] = data['predictions']['alignment_instances']
        else:
            data['predictions']['post_results'] = self.postprocess(
                data['predictions']['alignment_instances'],
                data['inputs']['batched_inputs'],
                data['inputs']['images'].image_sizes
            )

            # Attach image depths
            if 'depths' in data['predictions']:
                pred_image_depths = data['predictions']['depths'].unbind(0)
                for depth, result in zip(pred_image_depths, data['predictions']['post_results']):
                    result['pred_image_depth'] = depth

            # Attach CAD ids
            if 'cad_ids' in data['predictions']:
                # indices are global, so all instances should have all CAD ids
                for result in data['predictions']['post_results']:
                    result['cad_ids'] = data['predictions']['cad_ids']
        return data

    def forward(self, batched_inputs):
        data = {
            'inputs': {},
            'predictions': {},
            'losses': {},
            'logs': {}
        }

        data['inputs']['batched_inputs'] = batched_inputs

        data = self.forward_backbone(data)

        data = self.forward_proposals(data)

        data = self.forward_box(data)

        data = self.depth_head(data)

        data = self.forward_mask(data)

        data = self.alignment_head(data)

        data = self.retrieval_head(data)

        data = self.postProcess(data)
        return data['predictions']['post_results'], data['losses']

    def set_train_cads(self, points, ids):
        self.retrieval_head.wild_points_by_class = points
        self.retrieval_head.wild_ids_by_class = ids
        return

    def unset_train_cads(self):
        self.retrieval_head.wild_points_by_class = None
        self.retrieval_head.wild_ids_by_class = None
        return

    def set_cad_models(self, points, ids, scene_data):
        self.retrieval_head.inject_cad_models(
            points=points,
            ids=ids,
            scene_data=scene_data,
            device='cpu' #FIXME: why use cpu?
        )
        return

    def unset_cad_models(self):
        self.retrieval_head.eject_cad_models()
        return

    @torch.no_grad()
    def embed_cads(self, batch_size=16):
        assert self.retrieval_head.has_cads, \
            'Call `set_cad_models` before embedding cads'
        points_by_class = self.retrieval_head.points_by_class

        for cat, points in points_by_class.items():
            embeds = []
            total_size = len(points)
            for i in range(0, total_size, batch_size):
                points_i = points[i:min(i + batch_size, total_size)]
                points_i = torch.stack([
                    make_dense_volume(p, VOXEL_RES) for p in points_i
                ])
                embeds.append(
                    self.retrieval_head.cad_net(points_i.to(self.device).float()).cpu()
                )

            points_by_class[cat] = torch.cat(embeds).to(self.device)
            del embeds
        return

    def __getattr__(self, k):
        # Data dependency injections
        if 'inject' in k or 'eject' in k or k == 'set_verbose':
            return getattr(self.roi_head, k)
        return super().__getattr__(k)

