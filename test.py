#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

a = torch.tensor([1., 2.], device=torch.device("cuda"))

collection = {
    'test': a,
}

b = collection['test']

print(a)
print(collection['test'])
print(b)

