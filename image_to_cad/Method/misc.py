#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

def make_dense_volume(ind, voxel_res):
    if isinstance(voxel_res, int):
        voxel_res = (voxel_res, voxel_res, voxel_res)

    grid = torch.zeros(voxel_res, dtype=torch.bool)
    grid[ind[:, 0], ind[:, 1], ind[:, 2]] = True
    return grid.unsqueeze(0)

