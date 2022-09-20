#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import log

from image_to_cad.Config.roca.constants import IMAGE_SIZE
from image_to_cad.Config.roca.maskrcnn_config import maskrcnn_config

def roca_config(
    num_classes=17,
    class_freqs=None,
):
    cfg = maskrcnn_config(
        train_data='Scan2CAD_train',
        test_data='Scan2CAD_val',
        batch_size=4,
        num_classes=num_classes,
        max_iter=100000,
        lr=1e-3,
        num_workers=2,
        eval_period=100,
        output_dir='./output/',
        custom_mask=True,
        disable_flip=True,
        min_anchor_size=64
    )

    cfg.INPUT.MIN_SIZE_TRAIN = min(IMAGE_SIZE)
    cfg.INPUT.MIN_SIZE_TEST = min(IMAGE_SIZE)
    cfg.INPUT.MAX_SIZE_TRAIN = max(IMAGE_SIZE)
    cfg.INPUT.MAX_SIZE_TEST = max(IMAGE_SIZE)

    cfg.INPUT.NOC_SCALE = 10000
    cfg.INPUT.NOC_OFFSET = 1
    cfg.INPUT.DEPTH_SCALE = 1000
    cfg.INPUT.AUGMENT = True

    #FIXME: can not remove this
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 16

    cfg.SOLVER.STEPS = (60000,) # (60000, 80000)
    cfg.SOLVER.WORKERS = 2 # can speed up!
    cfg.SOLVER.CHECKPOINT_PERIOD = 100
    cfg.SOLVER.EVAL_STEP = False
    
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_VALUE = 10.0

    if not class_freqs:
        class_scales = []
    else:
        class_scales = sorted((k, 1 / log(v)) for k, v in class_freqs.items())
        # class_scales = sorted((k, 1 / v) for k, v in class_freqs.items())
        ratio = 1 / max(v for _, v in class_scales)
        class_scales = [(k, v * ratio) for k, v in class_scales]
    cfg.MODEL.CLASS_SCALES = class_scales

    cfg.INPUT.CUSTOM_FLIP = False
    cfg.INPUT.CUSTOM_JITTER = False

    cfg.SEED = 2022
    return cfg

