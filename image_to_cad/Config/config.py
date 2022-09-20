#!/usr/bin/env python
# -*- coding: utf-8 -*-

TRAIN_CONFIG = {
    "data_dir": "./Dataset/Dataset/",
    "image_root": "./Dataset/Images/",
    "rendering_root": "./Dataset/Rendering/",
    "full_annot": "./Dataset/full_annotations.json",

    "freq_scale": "image", # ['none', 'image', 'cad]

    #  "enable_nocs": 1,

    #  "checkpoint": "./output/last_checkpoint",
    "checkpoint": "",
    "resume": 0,

}

