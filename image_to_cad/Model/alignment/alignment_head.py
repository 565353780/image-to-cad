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

from image_to_cad.Method.common_ops import create_xy_grids, select_classes
from image_to_cad.Method.alignment_ops import \
    back_project, depth_bbox, depth_bbox_center, inverse_transform, \
    irls, make_new, point_count, point_mean, transform

from image_to_cad.Model.retrieval.retrieval_head import RetrievalHead

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

        # Initialize the retrieval head
        self.retrieval_head = RetrievalHead(cfg, shape_code_size)
        return

    @property
    def has_cads(self):
        return self.retrieval_head.has_cads

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        instances,
        depth_features,
        depths,
        image_size,
        mask_probs,
        mask_pred,
        inference_args=None,
        scenes=None,
        gt_depths=None,
        gt_classes=None,
        class_weights=None,
        xy_grid=None,
        xy_grid_n=None,
        mask_gt=None
    ):
        losses = {}
        predictions = {}
        extra_outputs = {}

        instance_sizes = [len(x) for x in instances]
        num_instances = sum(instance_sizes)
        alignment_sizes = torch.tensor(instance_sizes, device=self.device)

        if not self.training:
            if num_instances == 0:
                return self.identity()

        pred_boxes = None
        if self.training:
            pred_boxes = [x.proposal_boxes for x in instances]
        else:
            pred_boxes = [x.pred_boxes for x in instances]

        shape_code = self._encode_shape(
            pred_boxes,
            mask_pred,
            depth_features,
            image_size
        )

        intrinsics = None
        if self.training:
            intrinsics = Intrinsics.cat(
                [p.gt_intrinsics for p in instances]
            ).tensor.inverse()
        else:
            intrinsics = Intrinsics.cat(
                [arg['intrinsics'] for arg in inference_args]
            ).tensor.inverse()
            if intrinsics.size(0) > 1:
                intrinsics = intrinsics.repeat_interleave(
                    alignment_sizes, dim=0
                )

        if not self.training:
            xy_grid, xy_grid_n = create_xy_grids(
                Boxes.cat(pred_boxes),
                image_size,
                num_instances,
                self.output_grid_size
            )

        depths, depth_points, raw_depth_points, depth_losses, gt_depths, gt_depth_points = \
            self.forward_roi_depth(
                xy_grid,
                depths,
                mask_pred,
                intrinsics,
                xy_grid_n,
                alignment_sizes,
                mask_gt,
                gt_depths,
                class_weights
            )
        losses.update(depth_losses)

        alignment_classes = gt_classes
        if not self.training:
            pred_classes = [x.pred_classes for x in instances]
            alignment_classes = L.cat(pred_classes)

        scale_pred, scale_losses, scale_gt = self.forward_scale(
            shape_code,
            alignment_classes,
            instances,
            class_weights
        )
        losses.update(scale_losses)
        predictions['pred_scales'] = scale_pred

        trans_pred, trans_losses, trans_gt = self.forward_trans(
            shape_code,
            depths,
            depth_points,
            alignment_classes,
            instances,
            gt_depth_points,
            gt_depths,
            class_weights
        )
        predictions['pred_translations'] = trans_pred

        proc_depth_points = raw_depth_points
        if self.training:
            proc_depth_points = depth_points
            losses.update(trans_losses)

        rot_gt = None
        if self.training:
            rot_gt = Rotations.cat([p.gt_rotations for p in instances]).tensor

        rot, proc_losses, nocs, raw_nocs = self._forward_proc(
            shape_code,
            proc_depth_points,
            mask_pred,
            trans_pred,
            scale_pred,
            alignment_classes,
            mask_probs=mask_probs,
            gt_depth_points=gt_depth_points,
            gt_masks=mask_gt,
            gt_trans=trans_gt,
            gt_rot=rot_gt,
            gt_scale=scale_gt,
            class_weights=class_weights
        )
        losses.update(proc_losses)
        predictions['pred_rotations'] = rot

        if raw_nocs is not None:
            raw_nocs *= (mask_probs > 0.5)  # Keep all foreground NOCs!

        # Do the retrieval
        has_alignment = torch.ones(sum(instance_sizes), dtype=torch.bool)
        predictions['has_alignment'] = has_alignment

        predictions, retrieval_losses, extra_outputs = self.forward_retrieval(
            alignment_classes,
            mask_pred,
            nocs,
            shape_code,
            instance_sizes,
            has_alignment,
            scenes,
            instances,
            predictions,
            extra_outputs,
            scale_pred,
            trans_pred,
            rot,
            depth_points,
            raw_nocs,
            mask_pred
        )
        losses.update(retrieval_losses)
        return predictions, extra_outputs, losses

    def identity(self):
        losses = {}

        device = next(self.parameters()).device
        predictions = {
            'pred_scales': Scales.new_empty(0, device).tensor,
            'pred_translations': Translations.new_empty(0, device).tensor,
            'pred_rotations': Rotations.new_empty(0, device).tensor,
            'has_alignment': torch.zeros(0, device=device).bool(),
            'pred_indices': torch.zeros(0, device=device).long()
        }

        extra_outputs = {'cad_ids': []}
        return predictions, losses, extra_outputs

    def _encode_shape(
        self,
        pred_boxes,
        pred_masks,
        depth_features,
        image_size
    ):
        scaled_boxes = []
        for b in pred_boxes:
            b = Boxes(b.tensor.detach().clone())
            b.scale(
                depth_features.shape[-1] / image_size[-1],
                depth_features.shape[-2] / image_size[-2]
            )
            scaled_boxes.append(b.tensor)

        shape_features = L.roi_align(
            depth_features,
            scaled_boxes,
            self.output_grid_size
        )
        shape_code = self.shape_encoder(shape_features, pred_masks)
        shape_code = self.shape_code_drop(shape_code)
        return shape_code

    def forward_scale(
        self,
        shape_code,
        alignment_classes,
        instances=None,
        class_weights=None
    ):
        if self.training:
            assert instances is not None

        losses = {}

        scales = select_classes(
            self.scale_head(shape_code),
            self.num_classes,
            alignment_classes
        )

        gt_scales = None
        if self.training:
            gt_scales = Scales.cat([p.gt_scales for p in instances]).tensor
            losses['loss_scale'] = l1_loss(
                scales,
                gt_scales,
                weights=class_weights
            )

        return scales, losses, gt_scales

    def forward_roi_depth(
        self,
        xy_grid,
        depths,
        pred_masks,
        intrinsics_inv,
        xy_grid_n,
        alignment_sizes,
        gt_masks=None,
        gt_depths=None,
        class_weights=None
    ):
        if self.training:
            assert gt_masks is not None
            assert gt_depths is not None

        losses = {}

        depths, depth_points = self.crop_and_project_depth(
            xy_grid,
            depths,
            intrinsics_inv,
            xy_grid_n,
            alignment_sizes
        )

        gt_depth_points = None
        if self.training:
            gt_depths, gt_depth_points = self.crop_and_project_depth(
                xy_grid,
                gt_depths,
                intrinsics_inv,
                xy_grid_n,
                alignment_sizes
            )

            losses['loss_roi_depth'] = masked_l1_loss(
                depths,
                gt_depths,
                gt_masks,
                weights=class_weights
            )

            #FIXME: log this loss
            # Log metrics to tensorboard
            metric_dict = depth_metrics(depths, gt_depths, gt_masks,
                                        pref='depth/roi_')
            print("[INFO][AlignmentHead::forward_roi_depth]")
            print("\t depth_metrics")
            print(metric_dict)

        raw_depth_points = depth_points.clone()
        depths = depths * pred_masks
        depth_points = depth_points * pred_masks

        if self.training:
            gt_depths = gt_depths * gt_masks
            gt_depth_points = gt_depth_points * gt_masks

            # Penalize depth means
            # TODO: make this loss optional
            depth_mean_pred = point_mean(depths, point_count(pred_masks))
            depth_mean_gt = point_mean(gt_depths, point_count(gt_masks))
            losses['loss_mean_depth'] = l1_loss(
                depth_mean_pred,
                depth_mean_gt,
                weights=class_weights
            )

        return depths, depth_points, raw_depth_points, losses, gt_depths, gt_depth_points

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

    def forward_trans(
        self,
        shape_code,
        depths,
        depth_points,
        alignment_classes,
        instances=None,
        gt_depth_points=None,
        gt_depths=None,
        class_weights=None
    ):
        if self.training:
            assert instances is not None
            assert gt_depth_points is not None
            assert gt_depths is not None

        losses = {}

        depth_center, depth_min, depth_max = depth_bbox_center(
            depth_points, depths
        )
        trans_code = L.cat(
            [(depth_max - depth_min).detach(), shape_code],
            dim=-1
        )
        trans_offset = self.trans_head(trans_code)

        # per_category_trans:
        trans_offset = select_classes(
            trans_offset,
            self.num_classes,
            alignment_classes
        )

        trans = depth_center + trans_offset

        trans_gt = None
        if self.training:
            depth_min_gt, depth_max_gt = depth_bbox(gt_depth_points, gt_depths)
            losses['loss_depth_min'] = smooth_l1_loss(
                depth_min,
                depth_min_gt,
                weights=class_weights
            )
            losses['loss_depth_max'] = smooth_l1_loss(
                depth_max,
                depth_max_gt,
                weights=class_weights
            )

            trans_gt = Translations.cat([p.gt_translations for p in instances]).tensor
            losses['loss_trans'] = l2_loss(
                trans,
                trans_gt,
                weights=class_weights
            )

        return trans, losses, trans_gt

    def _forward_proc(
        self,
        shape_code,
        depth_points,
        masks,
        trans,
        scale,
        alignment_classes,
        mask_probs=None,
        gt_depth_points=None,
        gt_masks=None,
        gt_trans=None,
        gt_rot=None,
        gt_scale=None,
        class_weights=None
    ):
        if self.training:
            assert gt_depth_points is not None
            assert gt_masks is not None
            assert gt_trans is not None
            assert gt_rot is not None
            assert gt_scale is not None

        losses = {}

        # Untranslate depth using trans
        # depth_points = inverse_transform(depth_points, masks, trans=trans)
        depth_points = inverse_transform(depth_points, trans=trans)

        # Compute the nocs
        noc_codes = self.encode_shape_grid(
            shape_code,
            depth_points,
            scale,
            alignment_classes
        )

        nocs = self.noc_head(noc_codes)
        raw_nocs = nocs
        nocs = masks * nocs

        # Perform procrustes steps to sufficiently large regions
        has_enough = masks.flatten(1).sum(-1) >= self.min_nocs
        do_proc = has_enough.any()
        rot, trs = None, None
        if do_proc:
            rot, trs = self.solve_proc(
                nocs,
                depth_points,
                noc_codes,
                alignment_classes,
                has_enough,
                scale,
                masks,
                mask_probs
            )

        if self.training:
            gt_rot = Rotations(gt_rot).as_rotation_matrices()
            gt_nocs = inverse_transform(
                gt_depth_points,
                gt_masks,
                gt_scale,
                gt_rot.mats,
                gt_trans
            )

            losses['loss_noc'] = 3 * masked_l1_loss(
                nocs,
                gt_nocs,
                torch.logical_and(masks, gt_masks),
                weights=class_weights
            )

            if do_proc:
                if class_weights is not None:
                    class_weights = class_weights[has_enough]

                losses['loss_proc'] = 2 * l1_loss(
                    rot.flatten(1),
                    gt_rot.tensor[has_enough],
                    weights=class_weights
                )
                losses['loss_trans_proc'] = l2_loss(
                    trs + trans.detach()[has_enough],
                    gt_trans[has_enough],
                    weights=class_weights
                )

        if not self.training:
            if do_proc:
                trans[has_enough] += trs
                rot = Rotations.from_rotation_matrices(rot).tensor
                rot = make_new(Rotations, has_enough, rot)
            else:
                device = nocs.device
                batch_size = has_enough.numel()
                rot = Rotations.new_empty(batch_size, device=device).tensor

        return rot, losses, nocs, raw_nocs

    def encode_shape_grid(self, shape_code, depth_points, scale, classes):
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

    def forward_retrieval(
        self,
        pred_classes=None,
        masks=None,
        nocs=None,
        shape_code=None,
        instance_sizes=None,
        has_alignment=None,
        scenes=None,
        instances=None, 
        predictions=None,
        extra_outputs=None,
        pred_scales=None,
        pred_transes=None,
        pred_rots=None,
        depth_points=None,
        pred_nocs=None,
        pred_masks=None,
    ):
        losses = {}

        noc_points = None
        pos_cads = None
        neg_cads = None
        if self.training:
            pos_cads = L.cat([p.gt_pos_cads for p in instances])
            neg_cads = L.cat([p.gt_neg_cads for p in instances])
            # TODO: make this configurable
            sample = torch.randperm(pos_cads.size(0))[:32]
            masks = masks[sample]
            noc_points = nocs[sample]
            shape_code = shape_code[sample]
            has_alignment = None
            pos_cads = pos_cads[sample]
            neg_cads = neg_cads[sample]
        elif self.has_cads:
            assert scenes is not None
            if pred_nocs is not None:
                noc_points = pred_nocs
            else:
                rotation_mats = Rotations(pred_rots)\
                    .as_rotation_matrices()\
                    .mats
                noc_points = inverse_transform(
                    depth_points,
                    pred_masks,
                    pred_scales,
                    rotation_mats,
                    pred_transes
                )

        cad_ids, pred_indices = None, None
        if self.training:
            cad_ids, pred_indices, retrieval_losses = self.retrieval_head(
                pred_classes,
                masks,
                noc_points,
                shape_code,
                instance_sizes,
                has_alignment,
                scenes,
                pos_cads,
                neg_cads
            )
            losses.update(retrieval_losses)
        elif self.has_cads:
            cad_ids, pred_indices, _ = self.retrieval_head(
                pred_classes,
                pred_masks,
                noc_points,
                shape_code,
                instance_sizes,
                has_alignment,
                scenes,
                pos_cads,
                neg_cads
            )
            extra_outputs['cad_ids'] = cad_ids
            predictions['pred_indices'] = pred_indices

        return predictions, losses, extra_outputs

