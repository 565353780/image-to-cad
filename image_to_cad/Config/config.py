#!/usr/bin/env python
# -*- coding: utf-8 -*-

TRAIN_CONFIG = {
    "data_dir": "./Dataset/Dataset/",
    "image_root": "./Dataset/Images/",
    "rendering_root": "./Dataset/Rendering/",
    "full_annot": "./Dataset/full_annotations.json",

    "output_dir": "./output/",

    "lr": 1e-3,
    "max_iter": 80000,
    "batch_size": 4,
    "num_proposals": 128,
    "eval_period": 100,
    "freq_scale": "image", # ['none', 'image', 'cad]

    "steps": [60000],
    "workers": 0,
    "eval_step": 0,
    "gclip": 10.0,
    "augment": 1,

    "pooler_size": 16,
    "batch_average": 0,
    "depth_grad_losses": 0,
    "per_category_mask": 1,
    "enable_nocs": 1,
    "per_category_noc": 0,
    "noc_embed": 0,
    "noc_weights": 1,
    "per_category_trans": 1,
    "custom_noc_weights": 1,
    "noc_weight_skip": 0,
    "noc_rot_init": 0,
    "zero_center": 0,
    "irls_iters": 1,
    "retrieval_mode": "resnet_resnet+image+comp",
    "wild_retrieval": 0,
    "confidence_thresh_test": 0.5,
    "e2e": 1,

    #  "checkpoint": "./output/last_checkpoint",
    "checkpoint": "",
    "resume": 0,

    "seed": 2021,
}

