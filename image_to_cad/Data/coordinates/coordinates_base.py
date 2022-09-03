#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import detectron2.layers as L
import numpy as np

class CoordinatesBase(object):
    def __init__(self, tensor, repeats=None, eps=1e-5):
        # See detectron2.structures.BitMasks
        if not torch.is_tensor(tensor):
            device = torch.device('cpu')
            tensor = torch.as_tensor(tensor, device=device)

        if repeats is not None:
            # import pdb; pdb.set_trace()
            if tensor.size(0) > 1:
                tensor = tensor.repeat_interleave(repeats, dim=0)
            else:
                if not hasattr(repeats, '__len__'):
                    repeats = [repeats]
                else:
                    assert len(repeats) == 1
                tensor = tensor.expand(repeats[0], *tensor.shape[1:])

        self.tensor = tensor
        self.eps = eps
        return

    def to(self, device):
        return type(self)(tensor=self.tensor.to(device))

    @property
    def image_size(self):
        return self.tensor.shape[1:]

    @property
    def device(self):
        return self.tensor.device

    def clone(self):
        obj = type(self)(tensor=self.tensor.clone())
        obj.ndim = self.ndim
        return obj

    def __getitem__(self, item):
        return type(self)(tensor=self.tensor[item])

    def __iter__(self):
        yield from self.tensor

    def __repr__(self):
        return self.__class__.__name__ + \
            "(tensor={}, eps={})".\
            format(str(self.tensor), str(self.eps))

    def __len__(self):
        return self.tensor.shape[0]

    def nonempty(self):
        return (self.tensor >= self.eps).flatten(1).any(dim=1)

    def crop_and_resize(self, boxes_or_grids, crop_size,
                        use_interpolate=False, use_grid=False):
        if use_grid:
            grid = boxes_or_grids.permute(0, 2, 3, 1)
            crops = F.grid_sample(
                self.tensor,
                grid,
                'nearest',
                align_corners=False
            )
            return crops

        boxes = boxes_or_grids
        assert len(boxes) == len(self), \
            "{} != {}".format(len(boxes), len(self))
        device = self.tensor.device

        if use_interpolate:
            boxes = boxes.detach().cpu()
            tensor = self.tensor.detach().cpu()
            crops = []
            for i in range(boxes.size(0)):
                [xs, ys, xe, ye] = boxes[i].round().int().tolist()
                # print(i, [xs, ys, xe, ye], tensor.shape)
                if xs >= xe or ys >= ye:
                    crop = torch.zeros(
                        tensor.size(1), crop_size, crop_size
                    )
                else:
                    crop = F.interpolate(
                        tensor[i, :, ys:ye, xs:xe].unsqueeze(0),
                        size=(crop_size, crop_size),
                        mode='nearest'
                    ).squeeze(0)
                crops.append(crop)
            return torch.stack(crops).to(device)

        batch_inds = torch.arange(len(boxes), device=device).to(
            dtype=boxes.dtype
        )[:, None]
        rois = torch.cat([batch_inds, boxes], dim=1)  # Nx5
        rois = rois.to(device=device)
        output = L.ROIAlign((crop_size, crop_size), 1.0, 1, aligned=True)\
            .forward(self.tensor, rois)
        return output

    @classmethod
    def cat(cls, coords_list):
        assert isinstance(coords_list, (list, tuple))
        assert len(coords_list) > 0
        assert all(isinstance(coords, cls) for coords in coords_list)

        result = type(coords_list[0])(
            tensor=L.cat([c.tensor for c in coords_list], dim=0)
        )
        return result

    @classmethod
    def decode(cls, image, scale, offset=None,
               device='cpu', eps=1e-5):
        assert image.ndim in (2, 3), 'Unsupported image encoding'
        if image.ndim == 3:  # Make channel first
            image = image.transpose(2, 0, 1)

        tensor = torch.tensor(image.astype(np.float32), device=device)
        tensor /= scale
        if offset is not None:
            tensor -= offset
        tensor[torch.from_numpy(image == 0)] = 0

        if tensor.ndim == 3:  # RGB encoding
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 2:  # Single channel encoding
            tensor = tensor.view(1, 1, *tensor.size())

        tensor = tensor.to(device)

        return cls(tensor=tensor, eps=eps)

