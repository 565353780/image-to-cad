#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from detectron2.layers import cat

class AlignmentBase(object):
    def __init__(self, tensor, ndim=3):
        '''
        Args:
            tensor: Nxndim matrix
        '''
        # See Boxes in detectron2.structures

        device = tensor.device \
            if isinstance(tensor, torch.Tensor) else torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = tensor.reshape((0, ndim)).to(
                dtype=torch.float32,
                device=device
            )
        assert tensor.dim() == 2 and tensor.size(-1) == ndim, tensor.size()

        self.tensor = tensor.contiguous()
        return

    @classmethod
    def new_empty(cls, batch_size=1, device=None):
        tensor = torch.zeros(batch_size, cls.ndim, device=device)
        if hasattr(cls, 'identity'):
            tensor[:, :] = torch.tensor(cls.identity)
        return cls(tensor=tensor)

    def empty(self):
        empty = torch.zeros_like(self.tensor)
        cls = self.__class__
        if hasattr(cls, 'identity'):
            empty[:, :] = torch.tensor(cls.identity)
        return torch.all(torch.isclose(self.tensor, empty), dim=-1)

    def __len__(self):
        return self.tensor.shape[0]

    @property
    def device(self):
        return self.tensor.device

    def to(self, device):
        obj = type(self)(self.tensor.to(device))
        return obj

    def clone(self):
        obj = type(self)(self.tensor.clone())
        return obj

    def __getitem__(self, idx):
        obj = type(self)(self.tensor[idx])
        return obj

    def __iter__(self):
        yield from self.tensor

    def __repr__(self):
        return '{}(tensor={})'.format(type(self).__name__, str(self.tensor))

    def split(self, sizes):
        tensors = self.tensor.split(sizes)
        return [self.__class__(tensor=tensor) for tensor in tensors]

    @staticmethod
    def cat(mat_list):
        assert isinstance(mat_list, (list, tuple))
        assert len(mat_list) > 0
        dtype = type(mat_list[0])
        assert all(isinstance(mat, dtype) for mat in mat_list)

        cat_mats = dtype(tensor=cat([b.tensor for b in mat_list], dim=0))
        return cat_mats

