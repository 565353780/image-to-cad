#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from detectron2.layers import ShapeSpec
from detectron2.structures import ImageList
from detectron2.modeling.proposal_generator.rpn import RPN
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from detectron2.modeling.postprocessing import detector_postprocess

from image_to_cad.Config.roca.constants import VOXEL_RES

from image_to_cad.Method.misc import make_dense_volume

from image_to_cad.Model.roi.roi_head import ROCAROIHeads

class ROCA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        self.backbone = build_resnet_fpn_backbone(cfg, input_shape)
        self.proposal_generator = RPN(cfg, self.backbone.output_shape())
        self.roi_heads = ROCAROIHeads(cfg, self.backbone.output_shape())
        self.input_format = cfg.INPUT.FORMAT
        self.vis_period = cfg.VIS_PERIOD
        pixel_mean = cfg.MODEL.PIXEL_MEAN
        pixel_std = cfg.MODEL.PIXEL_STD

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        return

    @property
    def device(self):
        return self.pixel_mean.device

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

    def forward(self, batched_inputs):
        losses = {}

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        gt_instances = None
        if self.training:
            gt_instances = [
                x['instances'].to(self.device) for x in batched_inputs]

        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        losses.update(proposal_losses)

        targets = gt_instances
        if not self.training:
            targets = [
                {'intrinsics': input['intrinsics'].to(self.device)}
                for input in batched_inputs]

        image_depths = None
        if self.training:
            image_depths = []
            for input in batched_inputs:
                image_depths.append(input.pop('image_depth'))
            image_depths = torch.cat(image_depths, dim=0).to(self.device)

        scenes = None
        if not self.training:
            scenes = [input['scene'] for input in batched_inputs]

        results, extra_outputs, detector_losses = self.roi_heads(
            images,
            features,
            proposals,
            targets,
            image_depths,
            scenes
        )
        losses.update(detector_losses)

        if not self.training:
            results = self.postprocess(
                results,
                batched_inputs,
                images.image_sizes
            )

            # Attach image depths
            if 'pred_image_depths' in extra_outputs:
                pred_image_depths = extra_outputs['pred_image_depths'].unbind(0)
                for depth, result in zip(pred_image_depths, results):
                    result['pred_image_depth'] = depth

            # Attach CAD ids
            if 'cad_ids' in extra_outputs:
                # indices are global, so all instances should have all CAD ids
                for result in results:
                    result['cad_ids'] = extra_outputs['cad_ids']
        return results, losses

    @property
    def retrieval_head(self):
        return self.roi_heads.alignment_head.retrieval_head

    def set_train_cads(self, points, ids):
        retrieval_head = self.retrieval_head

        retrieval_head.wild_points_by_class = points
        retrieval_head.wild_ids_by_class = ids

        self.train_cads_embedded = False

    def unset_train_cads(self):
        retrieval_head = self.retrieval_head
        retrieval_head.wild_points_by_class = None
        retrieval_head.wild_ids_by_class = None
        self.train_cads_embedded = False

    def embed_train_cads(self, batch_size: int = 16):
        return self._embed_cads(wild=True, batch_size=batch_size)

    def set_cad_models(self, points, ids, scene_data):
        self.retrieval_head.inject_cad_models(
            points=points,
            ids=ids,
            scene_data=scene_data,
            device='cpu' #FIXME: why use cpu?
        )
        self.val_cads_embedded = False

    def unset_cad_models(self):
        self.retrieval_head.eject_cad_models()
        self.val_cads_embedded = False

    def embed_cad_models(self, batch_size: int = 16):
        return self._embed_cads(wild=False, batch_size=batch_size)

    @torch.no_grad()
    def _embed_cads(self, wild: bool = True, batch_size: int = 16):
        retrieval_head = self.retrieval_head
        if wild:
            assert retrieval_head.has_wild_cads, \
                'Call `set_train_cads` before embedding cads'
            points_by_class = retrieval_head.wild_points_by_class
        else:
            assert retrieval_head.has_cads, \
                'Call `set_cad_models` before embedding cads'
            points_by_class = retrieval_head.points_by_class

        # Below makes this function callable twice!
        if wild and self.train_cads_embedded:
            return
        if not wild and self.val_cads_embedded:
            return

        is_voxel = self.retrieval_head.is_voxel
        for cat, points in points_by_class.items():
            embeds = []
            total_size = points.size(0) if not is_voxel else len(points)
            for i in range(0, total_size, batch_size):
                points_i = points[i:min(i + batch_size, total_size)]
                if is_voxel:
                    points_i = torch.stack([
                        make_dense_volume(p, VOXEL_RES) for p in points_i
                    ])
                embeds.append(
                    retrieval_head.embed_cads(points_i.to(self.device)).cpu()
                )

            points_by_class[cat] = torch.cat(embeds).to(self.device)
            del embeds

        if wild:
            self.train_cads_embedded = True
        else:
            self.val_cads_embedded = True

    def __getattr__(self, k):
        # Data dependency injections
        if 'inject' in k or 'eject' in k or k == 'set_verbose':
            return getattr(self.roi_heads, k)
        return super().__getattr__(k)

