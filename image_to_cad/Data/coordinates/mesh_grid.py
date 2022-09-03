#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from image_to_cad.Data.coordinates.coordinates_base import CoordinatesBase

class MeshGrid(CoordinatesBase):
    def __init__(self, image_size=None, batch_size=1,
                 device=None, tensor=None, **kwargs):
        if tensor is not None:
            super().__init__(tensor=tensor, **kwargs)

        elif batch_size == 0:
            tensor = torch.tensor([], device=device).view(0, 2, *image_size)
            super().__init__(tensor=tensor, **kwargs)

        else:
            # import pdb; pdb.set_trace()
            assert image_size is not None
            h, w = image_size
            y, x = torch.meshgrid(
                torch.linspace(0, w-1, w, device=device),
                torch.linspace(0, h-1, h, device=device)
            )
            tensor = torch.stack([x, y], dim=0).unsqueeze(0)
            if batch_size > 1:
                tensor = tensor.expand(batch_size, *tensor.shape[-3:])
            super().__init__(tensor=tensor, **kwargs)

    def crop_and_resize_with_norm(self, *args, **kwargs):
        res = self.crop_and_resize(*args, **kwargs)
        h, w = self.tensor.shape[-2:]
        res_n = 2 * res / torch.tensor(
            [w-1, h-1], device=self.device, dtype=torch.float32
        ).view(1, 2, 1, 1) - 1
        return res, res_n

