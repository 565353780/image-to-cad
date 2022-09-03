#!/usr/bin/env python
# -*- coding: utf-8 -*-

from image_to_cad.Data.alignment.alignment_base import AlignmentBase

class Scales(AlignmentBase):
    ndim = 3
    identity = [1, 1, 1]

    def __init__(self, tensor):
        super().__init__(tensor, Scales.ndim)
        return

