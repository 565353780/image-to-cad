#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import detectron2.layers as L
from itertools import chain
from collections import defaultdict

from image_to_cad.Data.alignment.rotations import Rotations

from image_to_cad.Model.retrieval.pointnet import PointNet
from image_to_cad.Model.retrieval.resnet_decoder import ResNetDecoder
from image_to_cad.Model.retrieval.resnet_encoder import ResNetEncoder

from image_to_cad.Method.retrieval_ops import \
    embedding_lookup, grid_to_point_list, \
    nearest_points_retrieval, random_retrieval, voxelize_nocs

from image_to_cad.Method.alignment_ops import inverse_transform

class RetrievalHead(nn.Module):
    def __init__(self, cfg, shape_code_size=512, margin=0.5):
        super().__init__()
        self.has_cads = False
        self.mode = cfg.MODEL.RETRIEVAL_MODE
        self.shape_code_size = shape_code_size
        self.is_voxel = cfg.INPUT.CAD_TYPE == 'voxel'

        # NOTE: Make them embeddings for the learned model
        self.wild_points_by_class = None
        self.wild_ids_by_class = None

        self.loss = nn.TripletMarginLoss(margin=margin)

        if '_' in self.mode:
            self.cad_mode, self.noc_mode = self.mode.split('_')
        else:
            self.cad_mode = self.noc_mode = self.mode

        if self.cad_mode == 'pointnet':
            assert not self.is_voxel, 'Inconsistent CAD modality'
            self.cad_net = PointNet()
        elif self.cad_mode == 'resnet':
            assert self.is_voxel, 'Inconsistent CAD modality'
            self.cad_net = ResNetEncoder()
        else:
            raise ValueError(
                'Unknown CAD network type {}'.format(self.cad_mode)
            )

        if self.noc_mode == 'pointnet':
            self.noc_net = PointNet()
        elif self.noc_mode == 'image':
            self.noc_net = self.make_image_mlp()
        elif self.noc_mode == 'pointnet+image':
            self.noc_net = nn.ModuleDict({
                'pointnet': PointNet(),
                'image': self.make_image_mlp()
            })
        elif self.noc_mode == 'resnet':
            self.noc_net = ResNetEncoder()
        elif self.noc_mode == 'resnet+image':
            self.noc_net = nn.ModuleDict({
                'resnet': ResNetEncoder(),
                'image': self.make_image_mlp()
            })
        elif self.noc_mode in ('resnet+image+comp', 'resnet+image+fullcomp'):
            resnet = ResNetEncoder()
            self.noc_net = nn.ModuleDict({
                'resnet': resnet,
                'image': self.make_image_mlp(),
                'comp': ResNetDecoder(relu_in=True, feats=resnet.feats)
            })
            self.comp_loss = nn.BCELoss()
        else:
            raise ValueError('Unknown noc mode {}'.format(self.noc_mode))
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
        if self.is_voxel:
            self.points_by_class = points
        else:
            self.points_by_class = {k: v.to(device) for k, v in points.items()}
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
            # TODO: make this configurable
            sample = torch.randperm(pos_cads.size(0))[:32]

        if self.training:
            data['predictions']['retrieval_masks'] = data['predictions']['mask_pred'][sample]
            data['predictions']['retrieval_shape_code'] = data['predictions']['shape_code'][sample]
            data['predictions']['retrieval_noc_points'] = data['predictions']['nocs'][sample]
        else:
            data['predictions']['retrieval_masks'] = data['predictions']['mask_pred']
            data['predictions']['retrieval_shape_code'] = data['predictions']['shape_code']

            if self.has_cads:
                if data['predictions']['raw_nocs'] is not None:
                    # Keep all foreground NOCs!
                    data['predictions']['raw_nocs'] *= (data['predictions']['mask_probs'] > 0.5)
                    data['predictions']['retrieval_noc_points'] = data['predictions']['raw_nocs']
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

        if self.training:
            data['inputs']['scenes'] = None
        else:
            data['inputs']['scenes'] = [batched_input['scene'] for batched_input in data['inputs']['batched_inputs']]
            if self.has_cads:
                assert data['inputs']['scenes'] is not None

        num_instances = sum(data['predictions']['alignment_instance_sizes'])
        data['predictions']['has_alignment'] = torch.ones(num_instances, dtype=torch.bool)

        if self.training:
            data['predictions']['cad_ids'] = None
            data['predictions']['pred_indices'] = None
        else:
            scenes = list(chain(*(
                [scene] * isize
                for scene, isize in zip(data['inputs']['scenes'], data['predictions']['alignment_instance_sizes'])
            )))

            cad_ids, pred_indices = self._embedding_lookup(
                data['predictions']['has_alignment'],
                data['predictions']['alignment_classes'],
                data['predictions']['retrieval_masks'],
                scenes,
                data['predictions']['retrieval_noc_points'],
                shape_code=data['predictions']['retrieval_shape_code'],
            )
            data['predictions']['cad_ids'] = cad_ids
            data['predictions']['pred_indices'] = pred_indices

        if self.training:
            data['predictions']['retrieval_pos_cads'] = pos_cads[sample]
            data['predictions']['retrieval_neg_cads'] = neg_cads[sample]
            assert data['predictions']['retrieval_pos_cads'] is not None
            assert data['predictions']['retrieval_neg_cads'] is not None
        else:
            data['predictions']['retrieval_pos_cads'] = None
            data['predictions']['retrieval_neg_cads'] = None

        if self.training:
            noc_embed = self.embed_nocs(
                shape_code=data['predictions']['retrieval_shape_code'],
                noc_points=data['predictions']['retrieval_noc_points'],
                mask=data['predictions']['retrieval_masks']
            )
            if isinstance(noc_embed, tuple):  # Completion
                noc_embed, noc_comp = noc_embed
                data['losses']['loss_noc_comp'] = self.comp_loss(
                    noc_comp, data['predictions']['retrieval_pos_cads'].to(dtype=noc_comp.dtype)
                )

            cad_embeds = self.embed_cads(torch.cat([
                data['predictions']['retrieval_pos_cads'],
                data['predictions']['retrieval_neg_cads']
            ]))
            pos_embed, neg_embed = torch.chunk(cad_embeds, 2)
            data['losses']['loss_triplet'] = self.loss(noc_embed, pos_embed, neg_embed)
        return data

    def embed_nocs(self, shape_code=None, noc_points=None, mask=None):
        # Assertions
        if 'image' in self.noc_mode:
            assert shape_code is not None
        if self.noc_mode != 'image':
            assert noc_points is not None
            assert mask is not None

        if self.is_voxel:
            noc_points = voxelize_nocs(grid_to_point_list(noc_points, mask))

        if self.noc_mode == 'image':
            return self.noc_net(shape_code)
        elif self.noc_mode == 'pointnet':
            return self.noc_net(noc_points, mask)
        elif self.noc_mode == 'pointnet+image':
            return (
                self.noc_net['pointnet'](noc_points, mask)
                + self.noc_net['image'](shape_code)
            )
        elif self.noc_mode == 'resnet':
            return self.noc_net(noc_points)
        elif self.noc_mode == 'resnet+image':
            return (
                self.noc_net['resnet'](noc_points)
                + self.noc_net['image'](shape_code)
            )
        elif self.noc_mode in ('resnet+image+comp', 'resnet+image+fullcomp'):
            noc_embed = self.noc_net['resnet'](noc_points)
            result = noc_embed + self.noc_net['image'](shape_code)
            if self.training:
                if self.noc_mode == 'resnet+image+comp':
                    comp = self.noc_net['comp'](noc_embed)
                else:  # full comp
                    comp = self.noc_net['comp'](result)
                return result, comp.sigmoid_()
            else:
                return result
        else:
            raise ValueError('Unknown noc embedding type {}'
                             .format(self.noc_mode))

    def embed_cads(self, cad_points):
        if self.is_voxel:
            return self.cad_net(cad_points.float())
        else:  # Point clouds
            return self.cad_net(cad_points.transpose(-2, -1))

    def _perform_baseline(self, has_alignment, pred_classes,
                          pred_masks, scenes,
                          noc_points=None):
        num_instances = pred_classes.numel()
        if has_alignment is None:
            has_alignment = torch.ones(num_instances, dtype=torch.bool)

        if self.mode == 'nearest':
            function = nearest_points_retrieval
        elif self.mode == 'random':
            function = random_retrieval
        elif self.mode == 'first':
            function = 'first'
        else:
            raise ValueError('Unknown retrieval mode: {}'.format(self.mode))

        # meshes = []
        ids = []
        j = -1
        for i, scene in enumerate(scenes):
            if not has_alignment[i].item():
                # meshes.append(self.dummy_mesh)
                ids.append(None)
                continue
            j += 1

            pred_class = pred_classes[j].item()

            points_by_class = self.points_by_class[pred_class]
            point_indices = self.indices_by_scene[scene][pred_class]
            if len(point_indices) == 0:
                # meshes.append(self.dummy_mesh)
                ids.append(None)
                has_alignment[i] = False  # No CAD -> No Alignment
                continue
            points_by_class = points_by_class[point_indices]
            cad_ids_by_class = self.cad_ids_by_class[pred_class]

            if function is nearest_points_retrieval:
                assert noc_points is not None
                index, _ = function(
                    noc_points[j],
                    pred_masks[j],
                    points_by_class,
                    use_median=False,  # True
                    # mask_probs=mask_probs[j]
                )
            elif isinstance(function, str) and function == 'first':
                index = torch.zeros(1).int()
            elif function is random_retrieval:
                index, _ = random_retrieval(points_by_class)
            else:
                raise ValueError('Unknown baseline {}'.format(function))

            index = index.item()
            index = point_indices[index]
            ids.append(cad_ids_by_class[index])

        # Model ids
        cad_ids = ids
        # To handle sorting and filtering of instances
        pred_indices = torch.arange(num_instances, dtype=torch.long)
        return cad_ids, pred_indices

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

