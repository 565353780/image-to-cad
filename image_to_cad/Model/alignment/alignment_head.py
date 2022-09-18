#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import detectron2.layers as L
from detectron2.structures import Boxes

from image_to_cad.Data.coordinates.depths import Depths
from image_to_cad.Data.camera.intrinsics import Intrinsics
from image_to_cad.Data.alignment.rotations import Rotations
from image_to_cad.Data.alignment.scales import Scales
from image_to_cad.Data.alignment.translations import Translations

from image_to_cad.Model.alignment.aggregator import Aggregator
from image_to_cad.Model.alignment.mlp import MLP
from image_to_cad.Model.alignment.shared_mlp import SharedMLP

from image_to_cad.Loss.loss_functions import \
    l1_loss, l2_loss, masked_l1_loss, smooth_l1_loss

from image_to_cad.Metric.logging_metrics import depth_metrics

from image_to_cad.Method.common_ops import select_classes
from image_to_cad.Method.alignment_ops import \
    back_project, depth_bbox, depth_bbox_center, inverse_transform, \
    irls, make_new, point_count, point_mean, transform

class AlignmentHead(nn.Module):
    def __init__(self, cfg, num_classes, input_channels):
        super().__init__()

        self.num_classes = num_classes
        self.output_grid_size = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION * 2

        # Init the shape feature aggregator
        input_size = input_channels
        shape_code_size = 512

        shared_net = SharedMLP(
            input_size,
            hidden_size=shape_code_size,
            num_hiddens=2)

        global_net = MLP(
            shared_net.out_channels,
            hidden_size=shape_code_size,
            num_hiddens=2)

        self.shape_encoder = Aggregator(shared_net, global_net)
        self.shape_code_drop = nn.Dropout(0.3)

        # Init the scale head
        self.scale_head = MLP(
            shape_code_size,
            3 * self.num_classes,
            num_hiddens=2)

        self.trans_head = MLP(
            3 + shape_code_size,
            3 * self.num_classes,
            num_hiddens=1)

        self.min_nocs = cfg.MODEL.ROI_HEADS.NOC_MIN

        # code + depth_points + scale + embedding
        noc_code_size = shape_code_size + 3 + 3 + 0
        noc_output_size = 3

        self.noc_head = SharedMLP(
            noc_code_size,
            num_hiddens=2,
            output_size=noc_output_size
        )

        # use_noc_weight_head:
        self.noc_weight_head = SharedMLP(
            # noc code + noc + mask prob
            noc_code_size + 3 + 1,
            num_hiddens=1,
            # hidden_size=512,
            output_size=self.num_classes,
            output_activation=nn.Sigmoid)
        return

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data):
        num_instances = sum(data['predictions']['alignment_instance_sizes'])
        data['predictions']['alignment_sizes'] = torch.tensor(
            data['predictions']['alignment_instance_sizes'], device=self.device)

        if not self.training and num_instances == 0:
            data['predictions'].update(self.identity())
            return data

        data = self.encode_shape(data)

        if self.training:
            data['predictions']['intrinsics'] = Intrinsics.cat(
                [p.gt_intrinsics for p in data['predictions']['alignment_instances']]
            ).tensor.inverse()
        else:
            data['predictions']['intrinsics'] = Intrinsics.cat(
                [arg['intrinsics'] for arg in data['inputs']['inference_args']]
            ).tensor.inverse()
            if data['predictions']['intrinsics'].size(0) > 1:
                data['predictions']['intrinsics'] = data['predictions']['intrinsics'].repeat_interleave(
                    data['predictions']['alignment_sizes'], dim=0
                )

        data = self.forward_roi_depth(data)

        data = self.forward_scale(data)

        data = self.forward_trans(data)

        if self.training:
            data['predictions']['depth_points'] = data['predictions']['roi_mask_depth_points']
        else:
            data['predictions']['depth_points'] = data['predictions']['roi_depth_points'].clone()

        if self.training:
            data['predictions']['rot_gt'] = Rotations.cat(
                [p.gt_rotations for p in data['predictions']['alignment_instances']]).tensor
        else:
            data['predictions']['rot_gt'] = None

        data = self.forward_proc(data)

        if data['predictions']['raw_nocs'] is not None:
            data['predictions']['raw_nocs'] *= (data['predictions']['mask_probs'] > 0.5)  # Keep all foreground NOCs!

        data['predictions']['has_alignment'] = torch.ones(num_instances, dtype=torch.bool)
        return data

    def identity(self):
        device = next(self.parameters()).device
        predictions = {
            'scales_pred': Scales.new_empty(0, device).tensor,
            'trans_pred': Translations.new_empty(0, device).tensor,
            'pred_rotations': Rotations.new_empty(0, device).tensor,
            'has_alignment': torch.zeros(0, device=device).bool(),
            'pred_indices': torch.zeros(0, device=device).long(),
            'cad_ids': [],
        }
        return predictions

    def encode_shape(self, data):
        scaled_boxes = []
        for b in data['predictions']['pool_boxes']:
            b = Boxes(b.tensor.detach().clone())
            b.scale(
                data['predictions']['depth_features'].shape[-1] / data['inputs']['image_size'][-1],
                data['predictions']['depth_features'].shape[-2] / data['inputs']['image_size'][-2]
            )
            scaled_boxes.append(b.tensor)

        shape_features = L.roi_align(
            data['predictions']['depth_features'],
            scaled_boxes,
            self.output_grid_size
        )
        data['predictions']['shape_features'] = shape_features

        shape_code = self.shape_encoder(shape_features, data['predictions']['mask_pred'])
        shape_code = self.shape_code_drop(shape_code)
        data['predictions']['shape_code'] = shape_code
        return data

    def forward_scale(self, data):
        if self.training:
            assert data['predictions']['alignment_instances'] is not None

        scales = select_classes(
            self.scale_head(data['predictions']['shape_code']),
            self.num_classes,
            data['predictions']['alignment_classes']
        )
        data['predictions']['scales_pred'] = scales

        if self.training:
            data['predictions']['scales_gt'] = \
                Scales.cat([p.gt_scales for p in data['predictions']['alignment_instances']]).tensor
            data['losses']['loss_scale'] = l1_loss(
                scales,
                data['predictions']['scales_gt'],
                weights=data['predictions']['class_weights']
            )
        else:
            data['predictions']['scales_gt'] = None

        return data

    def forward_roi_depth(self, data):
        if self.training:
            assert data['predictions']['mask_gt'] is not None
            assert data['inputs']['image_depths'] is not None

        data['predictions']['roi_depths'], data['predictions']['roi_depth_points'] = self.crop_and_project_depth(
            data['predictions']['xy_grid'],
            data['predictions']['depths'],
            data['predictions']['intrinsics'],
            data['predictions']['xy_grid_n'],
            data['predictions']['alignment_sizes']
        )

        if self.training:
            data['predictions']['roi_gt_depths'], data['predictions']['roi_gt_depth_points'] = self.crop_and_project_depth(
                data['predictions']['xy_grid'],
                data['inputs']['image_depths'],
                data['predictions']['intrinsics'],
                data['predictions']['xy_grid_n'],
                data['predictions']['alignment_sizes'],
            )

            data['losses']['loss_roi_depth'] = masked_l1_loss(
                data['predictions']['roi_depths'],
                data['predictions']['roi_gt_depths'],
                data['predictions']['mask_gt'],
                weights=data['predictions']['class_weights']
            )

            #FIXME: log this metric
            # Log metrics to tensorboard
            #  metric_dict = depth_metrics(
                #  data['predictions']['roi_depths'],
                #  data['predictions']['roi_gt_depths'],
                #  data['predictions']['mask_gt'],
                #  pref='depth/roi_'
            #  )
        else:
            data['predictions']['roi_gt_depths'] = None
            data['predictions']['roi_gt_depth_points'] = None

        data['predictions']['roi_mask_depths'] = \
            data['predictions']['roi_depths'] * data['predictions']['mask_pred']
        data['predictions']['roi_mask_depth_points'] = \
            data['predictions']['roi_depth_points'] * data['predictions']['mask_pred']

        if self.training:
            data['predictions']['roi_mask_gt_depths'] = \
                data['predictions']['roi_gt_depths'] * data['predictions']['mask_gt']
            data['predictions']['roi_mask_gt_depth_points'] = \
                data['predictions']['roi_gt_depth_points'] * data['predictions']['mask_gt']
        else:
            data['predictions']['roi_mask_gt_depths'] = None
            data['predictions']['roi_mask_gt_depth_points'] = None

        if self.training:
            # Penalize depth means
            # TODO: make this loss optional
            roi_mask_depth_mean_pred = point_mean(
                data['predictions']['roi_mask_depths'],
                point_count(data['predictions']['mask_pred'])
            )

            roi_mask_depth_mean_gt = point_mean(
                data['predictions']['roi_mask_gt_depths'],
                point_count(data['predictions']['mask_gt'])
            )

            data['losses']['loss_mean_depth'] = l1_loss(
                roi_mask_depth_mean_pred,
                roi_mask_depth_mean_gt,
                weights=data['predictions']['class_weights']
            )
        return data

    def crop_and_project_depth(
        self,
        xy_grid,
        depths,
        intrinsics_inv,
        xy_grid_n,
        alignment_sizes
    ):
        depths = Depths(depths, alignment_sizes)\
            .crop_and_resize(
                xy_grid_n,
                crop_size=self.output_grid_size,
                use_grid=True
            )
        depth_points = back_project(
            xy_grid,
            depths,
            intrinsics_inv,
            invert_intr=False
        )
        return depths, depth_points

    def forward_trans(self, data):
        if self.training:
            assert data['predictions']['alignment_instances'] is not None
            assert data['predictions']['roi_mask_gt_depth_points'] is not None
            assert data['predictions']['roi_mask_gt_depths'] is not None

        depth_center, depth_min, depth_max = depth_bbox_center(
            data['predictions']['roi_mask_depth_points'],
            data['predictions']['roi_mask_depths']
        )

        trans_code = L.cat(
            [(depth_max - depth_min).detach(), data['predictions']['shape_code']],
            dim=-1
        )

        trans_offset = self.trans_head(trans_code)

        # per_category_trans:
        trans_offset = select_classes(
            trans_offset,
            self.num_classes,
            data['predictions']['alignment_classes']
        )

        data['predictions']['trans_pred'] = depth_center + trans_offset

        if self.training:
            data['predictions']['trans_gt'] = Translations.cat(
                [p.gt_translations for p in data['predictions']['alignment_instances']]).tensor
        else:
            data['predictions']['trans_gt'] = None

        if self.training:
            depth_min_gt, depth_max_gt = depth_bbox(
                data['predictions']['roi_mask_gt_depth_points'],
                data['predictions']['roi_mask_gt_depths'])

            data['losses']['loss_depth_min'] = smooth_l1_loss(
                depth_min,
                depth_min_gt,
                weights=data['predictions']['class_weights']
            )

            data['losses']['loss_depth_max'] = smooth_l1_loss(
                depth_max,
                depth_max_gt,
                weights=data['predictions']['class_weights']
            )

            data['losses']['loss_trans'] = l2_loss(
                data['predictions']['trans_pred'],
                data['predictions']['trans_gt'],
                weights=data['predictions']['class_weights']
            )
        return data

    def forward_proc(self, data):
        if self.training:
            assert data['predictions']['roi_mask_gt_depth_points'] is not None
            assert data['predictions']['mask_gt'] is not None
            assert data['predictions']['trans_gt'] is not None
            assert data['predictions']['rot_gt'] is not None
            assert data['predictions']['scales_gt'] is not None

        # Untranslate depth using trans
        #  depth_points = inverse_transform(
            #  depth_points,
            #  data['predictions']['mask_pred'],
            #  trans=data['predictions']['trans_pred']
        #  )

        depth_points = inverse_transform(
            data['predictions']['depth_points'],
            trans=data['predictions']['trans_pred']
        )

        # Compute the nocs
        noc_codes = self.encode_shape_grid(
            data['predictions']['shape_code'],
            depth_points,
            data['predictions']['scales_pred'],
        )

        data['predictions']['raw_nocs'] = self.noc_head(noc_codes)
        data['predictions']['nocs'] = \
            data['predictions']['mask_pred'] * data['predictions']['raw_nocs']

        # Perform procrustes steps to sufficiently large regions
        has_enough = data['predictions']['mask_pred'].flatten(1).sum(-1) >= self.min_nocs
        do_proc = has_enough.any()

        rot, trs = None, None
        if do_proc:
            rot, trs = self.solve_proc(
                data['predictions']['nocs'],
                depth_points,
                noc_codes,
                data['predictions']['alignment_classes'],
                has_enough,
                data['predictions']['scales_pred'],
                data['predictions']['mask_pred'],
                data['predictions']['mask_probs']
            )

        if self.training:
            gt_rot = Rotations(data['predictions']['rot_gt']).as_rotation_matrices()
            gt_nocs = inverse_transform(
                data['predictions']['roi_mask_gt_depth_points'],
                data['predictions']['mask_gt'],
                data['predictions']['scales_gt'],
                gt_rot.mats,
                data['predictions']['trans_gt']
            )

            data['losses']['loss_noc'] = 3 * masked_l1_loss(
                data['predictions']['nocs'],
                gt_nocs,
                torch.logical_and(data['predictions']['mask_pred'], data['predictions']['mask_gt']),
                weights=data['predictions']['class_weights']
            )

            if do_proc:
                if data['predictions']['class_weights'] is not None:
                    data['predictions']['proc_class_weights'] = data['predictions']['class_weights'][has_enough]

                data['losses']['loss_proc'] = 2 * l1_loss(
                    rot.flatten(1),
                    gt_rot.tensor[has_enough],
                    weights=data['predictions']['proc_class_weights']
                )
                data['losses']['loss_trans_proc'] = l2_loss(
                    trs + data['predictions']['trans_pred'].detach()[has_enough],
                    data['predictions']['trans_gt'][has_enough],
                    weights=data['predictions']['proc_class_weights']
                )
        else:
            if do_proc:
                data['predictions']['trans_pred'][has_enough] += trs
                rot = Rotations.from_rotation_matrices(rot).tensor
                rot = make_new(Rotations, has_enough, rot)
            else:
                device = data['predictions']['nocs'].device
                batch_size = has_enough.numel()
                rot = Rotations.new_empty(batch_size, device=device).tensor
        data['predictions']['rot_pred'] = rot
        return data

    def encode_shape_grid(self, shape_code, depth_points, scale):
        shape_code_grid = shape_code\
            .view(*shape_code.size(), 1, 1)\
            .expand(*shape_code.size(), *depth_points.size()[-2:])
        scale_grid = scale.view(-1, 3, 1, 1).expand_as(depth_points)

        return L.cat([
            shape_code_grid,
            scale_grid.detach(),
            depth_points.detach()
        ], dim=1)

    def solve_proc(
        self,
        nocs,
        depth_points,
        noc_codes,
        alignment_classes,
        has_enough,
        scale,
        masks,
        mask_probs
    ):
        proc_masks = masks[has_enough]
        s_nocs = transform(nocs[has_enough], scale=scale[has_enough])
        mask_probs = self.prep_mask_probs(
            mask_probs,
            noc_codes,
            nocs,
            alignment_classes,
            has_enough
        )
        return irls(
            s_nocs,
            depth_points[has_enough],
            proc_masks,
            weights=mask_probs,
            num_iter=1)

    def prep_mask_probs(
        self,
        mask_probs,
        noc_codes,
        nocs,
        alignment_classes,
        has_enough=None
    ):
        # use_noc_weights:
        assert mask_probs is not None

        if has_enough is not None:
            mask_probs = mask_probs[has_enough]
            noc_codes = noc_codes[has_enough]
            nocs = nocs[has_enough]
            alignment_classes = alignment_classes[has_enough]

        # use_noc_weight_head:
        weight_inputs = [noc_codes, nocs.detach(), mask_probs.detach()]

        new_probs = select_classes(
            self.noc_weight_head(L.cat(weight_inputs, dim=1)),
            self.num_classes,
            alignment_classes
        )

        mask_probs = new_probs
        return mask_probs

