#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from collections import OrderedDict

def changeModelName(model_path, new_model_path):
    backup = torch.load(model_path)

    new_state_dict = OrderedDict()

    for key, item in backup['model'].items():
        new_name = key
        if "roi_heads" in key:
            new_name = key.replace("roi_heads",
                                   "roi_head")
        new_state_dict[new_name] = item
        backup['model'] = new_state_dict

    torch.save(backup, new_model_path)
    return True

