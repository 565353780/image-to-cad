#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("./network/")

import json
import argparse
import os.path as path
from os import makedirs

from network.roca.config import roca_config
from network.roca.data import CategoryCatalog
from network.roca.data.datasets import register_scan2cad
from network.roca.engine import Trainer

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
    "eval_period": 2500,
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

    "eval_only": 0,
    "checkpoint": "./output/last_checkpoint",
    "resume": 1,

    "seed": 2021,
}

def register_data(config):
    data_dir = config["data_dir"]
    train_name = 'Scan2CAD_train'
    val_name = 'Scan2CAD_val'

    register_scan2cad(
        name=train_name,
        split='train',
        data_dir=data_dir,
        metadata={'scenes': path.abspath('../metadata/scannetv2_train.txt')},
        image_root=config["image_root"],
        rendering_root=config["rendering_root"],
        full_annot=config["full_annot"],
        class_freq_method=config["freq_scale"]
    )
    register_scan2cad(
        name=val_name,
        split='val',
        data_dir=data_dir,
        metadata={'scenes': path.abspath('../metadata/scannetv2_val.txt')},
        image_root=config["image_root"],
        rendering_root=config["rendering_root"],
        full_annot=config["full_annot"]
    )

    return train_name, val_name

def make_config(config):
    train_name, val_name = register_data(config)

    cfg = roca_config(
        train_data=train_name,
        test_data=val_name,
        batch_size=config["batch_size"],
        num_proposals=config["num_proposals"],
        num_classes=CategoryCatalog.get(train_name).num_classes,
        max_iter=config["max_iter"],
        lr=config["lr"],
        output_dir=config["output_dir"],
        eval_period=config["eval_period"],
        eval_step=bool(config["eval_step"]),
        workers=config["workers"],
        class_freqs=CategoryCatalog.get(train_name).freqs,
        steps=config["steps"],
        pooler_size=config["pooler_size"],
        batch_average=bool(config["batch_average"]),
        depth_grad_losses=bool(config["depth_grad_losses"]),
        per_category_mask=bool(config["per_category_mask"]),
        per_category_noc=bool(config["per_category_noc"]),
        noc_embed=bool(config["noc_embed"]),
        noc_weights=bool(config["noc_weights"]),
        per_category_trans=bool(config["per_category_trans"]),
        noc_weight_head=bool(config["custom_noc_weights"]),
        noc_weight_skip=bool(config["noc_weight_skip"]),
        noc_rot_init=bool(config["noc_rot_init"]),
        seed=config["seed"],
        gclip=config["gclip"],
        augment=bool(config["augment"]),
        zero_center=bool(config["zero_center"]),
        irls_iters=config["irls_iters"],
        retrieval_mode=config["retrieval_mode"],
        wild_retrieval=bool(config["wild_retrieval"]),
        confidence_thresh_test=config["confidence_thresh_test"],
        e2e=bool(config["e2e"])
    )

    # NOTE: Training state will be reset in this case!
    if config["checkpoint"].lower() not in ('', 'none'):
        cfg.MODEL.WEIGHTS = config["checkpoint"]

    return cfg

def setup_output_dir(config, cfg):
    output_dir = config["output_dir"]
    makedirs(output_dir, exist_ok=True)

    if not config["eval_only"] and not config["resume"]:
        # Save command line arguments
        with open(path.join(config["output_dir"], 'args.json'), 'w') as f:
            json.dump(config, f)
        # Save the config as yaml
        with open(path.join(output_dir, 'config.yaml'), 'w') as f:
            cfg.dump(stream=f)

def train_or_eval(config, cfg):
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=config["resume"])
    if config["eval_only"]:
        trainer.test(cfg, trainer.model)
    elif config["resume"]:
        trainer.test(cfg, trainer.model)
        trainer.train()
    else:
        trainer.train()

class ROCATrainer(object):
    def __init__(self):
        return

def demo():
    cfg = make_config(TRAIN_CONFIG)
    setup_output_dir(TRAIN_CONFIG, cfg)
    train_or_eval(TRAIN_CONFIG, cfg)
    return True

