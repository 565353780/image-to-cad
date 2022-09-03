#!/usr/bin/env python
# -*- coding: utf-8 -*-

from image_to_cad.Data.alignment.alignment_base import AlignmentBase

class RotationMats(AlignmentBase):
    ndim = 9
    identity = [1, 0, 0,
                0, 1, 0,
                0, 0, 1]

    def __init__(self, tensor):
        tensor = tensor.contiguous().flatten(1)
        super().__init__(tensor, RotationMats.ndim)
        return

    @property
    def mats(self):
        return self.tensor.view(-1, 3, 3)

