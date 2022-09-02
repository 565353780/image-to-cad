#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from detectron2.modeling import GeneralizedRCNN
from detectron2.utils.events import get_event_storage

from network.roca.data.constants import VOXEL_RES
from network.roca.utils.misc import make_dense_volume

class ROCA(GeneralizedRCNN):
    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        gt_instances = None
        if 'instances' in batched_inputs[0] and self.training:
            gt_instances = [
                x['instances'].to(self.device) for x in batched_inputs]

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert 'proposals' in batched_inputs[0]
            proposals = [
                x['proposals'].to(self.device)
                for x in batched_inputs]
            proposal_losses = {}

        if not self.training:
            targets = [
                {'intrinsics': input['intrinsics'].to(self.device)}
                for input in batched_inputs]
            scenes = [input['scene'] for input in batched_inputs]

            results, extra_outputs = self.roi_heads(
                images, features, proposals,
                targets=targets, scenes=scenes)

            results = self.__class__._postprocess(
                results, batched_inputs, images.image_sizes)

            # Attach image depths
            if 'pred_image_depths' in extra_outputs:
                pred_image_depths = extra_outputs['pred_image_depths'].unbind(0)
                for depth, result in zip(pred_image_depths, results):
                    result['pred_image_depth'] = depth
            
            # Attach CAD ids
            for cad_ids in ('cad_ids', 'wild_cad_ids'):
                if cad_ids in extra_outputs:
                    # indices are global, so all instances should have all CAD ids
                    for result in results:
                        result[cad_ids] = extra_outputs[cad_ids]
            return results

        # Extract the image depth
        image_depths = []
        for input in batched_inputs:
            image_depths.append(input.pop('image_depth'))
        image_depths = torch.cat(image_depths, dim=0).to(self.device)

        _, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances, image_depths
        )

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    @property
    def retrieval_head(self):
        return self.roi_heads.alignment_head.retrieval_head

    def set_train_cads(self, points, ids):
        retrieval_head = self.retrieval_head

        retrieval_head.wild_points_by_class = (
            {k: p.to(self.device) for k, p in points.items()}
            if retrieval_head.baseline
            else points
        )
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
            device=self.device if self.retrieval_head.baseline else 'cpu'
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
        if retrieval_head.baseline:
            return

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

