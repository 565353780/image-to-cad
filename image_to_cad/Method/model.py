#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from collections import OrderedDict

def changeModelName(model_path, new_model_path):
    backup = torch.load(model_path)

    new_state_dict = OrderedDict()

    for key, item in backup['model'].items():
        new_name = key
        if key in ["pixel_mean", "pixel_std"]:
            continue
        new_state_dict[new_name] = item
    backup['model'] = new_state_dict

    torch.save(backup, new_model_path)
    return True

