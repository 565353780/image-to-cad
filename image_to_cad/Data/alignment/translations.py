#!/usr/bin/env python
# -*- coding: utf-8 -*-

from image_to_cad.Data.alignment.alignment_base import AlignmentBase

class Translations(AlignmentBase):
    ndim = 3
    identity = [0, 0, 0]

    def __init__(self, tensor):
        super().__init__(tensor, Translations.ndim)
        return

