#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import detectron2.layers as L
from itertools import chain
from collections import defaultdict

from image_to_cad.Data.alignment.rotations import Rotations

#  from image_to_cad.Model.retrieval.pointnet import PointNet
from image_to_cad.Model.retrieval.resnet_decoder import ResNetDecoder
from image_to_cad.Model.retrieval.resnet_encoder import ResNetEncoder

from image_to_cad.Method.retrieval_ops import \
    embedding_lookup, grid_to_point_list, voxelize_nocs

from image_to_cad.Method.alignment_ops import inverse_transform

class RetrievalHead(nn.Module):
    def __init__(self, shape_code_size=512, margin=0.5):
        super().__init__()
        self.has_cads = False
        self.shape_code_size = shape_code_size

        # NOTE: Make them embeddings for the learned model
        self.wild_points_by_class = None
        self.wild_ids_by_class = None

        self.loss = nn.TripletMarginLoss(margin=margin)

        self.cad_net = ResNetEncoder()

        resnet = ResNetEncoder()
        self.noc_net = nn.ModuleDict({
            'resnet': resnet,
            'image': self.make_image_mlp(),
            'comp': ResNetDecoder(relu_in=True, feats=resnet.feats)
        })
        self.comp_loss = nn.BCELoss()
        return

    def make_image_mlp(self, relu_out=True):
        return nn.Sequential(
            nn.Linear(self.shape_code_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.cad_net.embedding_dim),
            nn.ReLU(True) if relu_out else nn.Identity()
        )

    @property
    def has_wild_cads(self) -> bool:
        return self.wild_points_by_class is not None

    def inject_cad_models(self, points, ids, scene_data, device='cpu'):
        self.device = device
        self.has_cads = True
        self.points_by_class = points
        self.cad_ids_by_class = ids
        # self.dummy_mesh = ico_sphere()

        # Parse scene data
        classes = list(self.cad_ids_by_class.keys())
        scene_by_ids = defaultdict(lambda: [])
        for scene, models in scene_data.items():
            for model in models:
                model_id = (model['catid_cad'], model['id_cad'])
                scene_by_ids[model_id].append(scene)

        self.indices_by_scene = {
            scene: {k: [] for k in classes}
            for scene in scene_data.keys()
        }
        for k in classes:
            for i, cad_id in enumerate(self.cad_ids_by_class[k]):
                scenes = scene_by_ids[cad_id]
                for scene in scenes:
                    self.indices_by_scene[scene][k].append(i)
        return

    def eject_cad_models(self):
        if self.has_cads:
            del self.points_by_class
            del self.cad_ids_by_class
            del self.indices_by_scene
            self.has_cads = False
        return

    def forward(self, data):
        if not self.training and not self.has_cads:
            print("[WARN][RetrievalHead::forward]")
            print("\t self.has_cads is False!")
            return data

        if self.training:
            pos_cads = L.cat([p.gt_pos_cads for p in data['predictions']['alignment_instances']])
            neg_cads = L.cat([p.gt_neg_cads for p in data['predictions']['alignment_instances']])
            assert pos_cads is not None
            assert neg_cads is not None
            # TODO: make this configurable
            sample = torch.randperm(pos_cads.size(0))[:32]

        if self.training:
            data['predictions']['retrieval_masks'] = data['predictions']['mask_pred'][sample]
            data['predictions']['retrieval_shape_code'] = data['predictions']['shape_code'][sample]
            data['predictions']['retrieval_noc_points'] = data['predictions']['nocs'][sample]
            assert data['predictions']['retrieval_masks'] is not None
            assert data['predictions']['retrieval_shape_code'] is not None
            assert data['predictions']['retrieval_noc_points'] is not None
        elif self.has_cads:
            if data['predictions']['raw_nocs'] is not None:
                # Keep all foreground NOCs!
                data['predictions']['retrieval_noc_points'] = \
                    data['predictions']['raw_nocs'] * (data['predictions']['mask_probs'] > 0.5)
            else:
                rotation_mats = Rotations(data['predictions']['rot_pred'])\
                    .as_rotation_matrices()\
                    .mats
                data['predictions']['retrieval_noc_points'] = inverse_transform(
                    data['predictions']['roi_mask_depth_points'],
                    data['predictions']['mask_pred'],
                    data['predictions']['scales_pred'],
                    rotation_mats,
                    data['predictions']['trans_pred']
                )

        if not self.training:
            data['inputs']['scenes'] = [batched_input['scene'] for batched_input in data['inputs']['batched_inputs']]
            if self.has_cads:
                assert data['inputs']['scenes'] is not None

        if not self.training:
            scenes = list(chain(*(
                [scene] * isize
                for scene, isize in zip(data['inputs']['scenes'], data['predictions']['alignment_instance_sizes'])
            )))

            num_instances = sum(data['predictions']['alignment_instance_sizes'])
            has_alignment = torch.ones(num_instances, dtype=torch.bool)

            data['predictions']['cad_ids'], data['predictions']['pred_indices'] = self._embedding_lookup(
                has_alignment,
                data['predictions']['alignment_classes'],
                data['predictions']['mask_pred'],
                scenes,
                data['predictions']['retrieval_noc_points'],
                shape_code=data['predictions']['shape_code'],
            )

        if self.training:
            data['predictions']['retrieval_pos_cads'] = pos_cads[sample]
            data['predictions']['retrieval_neg_cads'] = neg_cads[sample]

        if self.training:
            data = self.retrieval_loss(data)
        return data

    def retrieval_loss(self, data):
        noc_embed = self.embed_nocs(
            shape_code=data['predictions']['retrieval_shape_code'],
            noc_points=data['predictions']['retrieval_noc_points'],
            mask=data['predictions']['retrieval_masks']
        )
        if isinstance(noc_embed, tuple):
            noc_embed, noc_comp = noc_embed
            data['losses']['loss_noc_comp'] = self.comp_loss(
                noc_comp, data['predictions']['retrieval_pos_cads'].to(dtype=noc_comp.dtype)
            )

        cad_embeds = self.cad_net(torch.cat([
            data['predictions']['retrieval_pos_cads'],
            data['predictions']['retrieval_neg_cads']
        ]).float())
        pos_embed, neg_embed = torch.chunk(cad_embeds, 2)
        data['losses']['loss_triplet'] = self.loss(noc_embed, pos_embed, neg_embed)
        return data

    def embed_nocs(self, shape_code=None, noc_points=None, mask=None):
        noc_points = voxelize_nocs(grid_to_point_list(noc_points, mask))

        noc_embed = self.noc_net['resnet'](noc_points)
        result = noc_embed + self.noc_net['image'](shape_code)
        if self.training:
            comp = self.noc_net['comp'](noc_embed)
            # TODO: here can use result as input
            #  comp = self.noc_net['comp'](result)
            return result, comp.sigmoid()
        else:
            return result

    def _embedding_lookup(self, has_alignment, pred_classes, pred_masks,
                          scenes, noc_points, shape_code):
        noc_embeds = self.embed_nocs(shape_code, noc_points, pred_masks)

        assert scenes is not None
        assert has_alignment is not None

        cad_ids = [None for _ in scenes]
        for scene in set(scenes):
            scene_mask = [scene_ == scene for scene_ in scenes]
            scene_noc_embeds = noc_embeds[scene_mask]
            scene_classes = pred_classes[scene_mask]

            indices = self.indices_by_scene[scene]
            points_by_class = {}
            ids_by_class = {}
            for c in scene_classes.tolist():
                ind = indices[c]
                if not len(ind):
                    continue
                points_by_class[c] = self.points_by_class[c][ind]
                ids_by_class[c] = \
                    [self.cad_ids_by_class[c][i] for i in ind]

            cad_ids_scene = embedding_lookup(
                scene_classes,
                scene_noc_embeds,
                points_by_class,
                ids_by_class
            )
            cad_ids_scene.reverse()
            for i, m in enumerate(scene_mask):
                if m:
                    cad_ids[i] = cad_ids_scene.pop()
        has_alignment[[id is None for id in cad_ids]] = False

        pred_indices = torch.arange(pred_classes.numel(), dtype=torch.long)
        return cad_ids, pred_indices

