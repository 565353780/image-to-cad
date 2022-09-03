#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import quaternion
import numpy as np
from pytorch3d.transforms import \
    quaternion_to_matrix, matrix_to_quaternion, standardize_quaternion

from image_to_cad.Data.alignment.alignment_base import AlignmentBase
from image_to_cad.Data.alignment.rotation_mats import RotationMats

class Rotations(AlignmentBase):
    ndim = 4
    identity = [1, 0, 0, 0]

    def __init__(self, tensor):
        # if not torch.allclose(tensor, torch.zeros_like(tensor)):
        super().__init__(tensor, Rotations.ndim)
        self.tensor = self.tensor / torch.norm(tensor, dim=1, keepdim=True)
        self.tensor = standardize_quaternion(self.tensor)
        return

    @staticmethod
    def from_rotation_matrices(rotation_mats, use_pt3d=True):
        if isinstance(rotation_mats, RotationMats):
            rotation_mats = rotation_mats.tensor.view(-1, 3, 3)
        else:
            assert isinstance(rotation_mats, torch.Tensor), \
                'rotation_mats must be a tensor or RotationsMats object'

        assert rotation_mats.ndim == 3 and rotation_mats.shape[-2:] == (3, 3)
        if use_pt3d:
            return Rotations(tensor=matrix_to_quaternion(rotation_mats))
        else:
            device = rotation_mats.device
            rotation_mats = rotation_mats.detach().cpu()
            quats = [
                quaternion.from_rotation_matrix(rm.numpy())
                for rm in rotation_mats.unbind(0)
            ]
            res = Rotations(tensor=torch.stack([
                torch.from_numpy(quaternion.as_float_array(q)) for q in quats
            ]))
            res = res.to(device)
            return res

    @torch.no_grad()
    def as_quaternions(self):
        tensor = self.tensor.cpu()
        quats = []
        for q in tensor.unbind(0):
            quats.append(np.quaternion(*q.tolist()))
        return quats

    def as_rotation_matrices(self, keep_device=True, use_pt3d=True):
        if use_pt3d:
            mats = quaternion_to_matrix(self.tensor)
        else:
            quats = self.tensor.cpu()
            mats = []
            for q in quats.unbind(0):
                mats.append(torch.from_numpy(
                    quaternion.as_rotation_matrix(np.quaternion(*q.tolist()))
                ))
            mats = torch.stack(mats, dim=0).flatten(1)
            if keep_device:
                mats = mats.to(self.device)
        return RotationMats(tensor=mats)

