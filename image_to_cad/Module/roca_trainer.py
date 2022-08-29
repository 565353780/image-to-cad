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

cfg = {
    "data_dir": "./Dataset/Dataset/",
    "image_root": "./Dataset/Images/",
    "rendering_root": "./Dataset/Rendering/",
    "full_annot": "./Dataset/full_annotations.json",

    "output_dir": "./output/",
    "override_output": 0,

    "lr": 1e-3,
    "max_iter": 80000,
    "batch_size": 4,
    "num_proposals": 128,
    "eval_period": 2500,
    "freq_scale": "image", # ['none', 'image', 'cad]

    "steps": 60000,
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
    "checkpoint": "",
    "resume": 0,

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

def make_config(train_name, val_name, args):
    cfg = roca_config(
        train_data=train_name,
        test_data=val_name,
        batch_size=args.batch_size,
        num_proposals=args.num_proposals,
        num_classes=CategoryCatalog.get(train_name).num_classes,
        max_iter=args.max_iter,
        lr=args.lr,
        output_dir=args.output_dir,
        eval_period=args.eval_period,
        eval_step=bool(args.eval_step),
        workers=args.workers,
        class_freqs=CategoryCatalog.get(train_name).freqs,
        steps=args.steps,
        pooler_size=args.pooler_size,
        batch_average=bool(args.batch_average),
        depth_grad_losses=bool(args.depth_grad_losses),
        per_category_mask=bool(args.per_category_mask),
        per_category_noc=bool(args.per_category_noc),
        noc_embed=bool(args.noc_embed),
        noc_weights=bool(args.noc_weights),
        per_category_trans=bool(args.per_category_trans),
        noc_weight_head=bool(args.custom_noc_weights),
        noc_weight_skip=bool(args.noc_weight_skip),
        noc_rot_init=bool(args.noc_rot_init),
        seed=args.seed,
        gclip=args.gclip,
        augment=bool(args.augment),
        zero_center=bool(args.zero_center),
        irls_iters=args.irls_iters,
        retrieval_mode=args.retrieval_mode,
        wild_retrieval=bool(args.wild_retrieval),
        confidence_thresh_test=args.confidence_thresh_test,
        e2e=bool(args.e2e)
    )

    # NOTE: Training state will be reset in this case!
    if args.checkpoint.lower() not in ('', 'none'):
        cfg.MODEL.WEIGHTS = args.checkpoint

    return cfg

def setup_output_dir(args, cfg):
    output_dir = args.output_dir
    assert not args.resume or path.isdir(args.output_dir), \
        'No backup found in {}'.format(args.output_dir)
    makedirs(output_dir, exist_ok=args.override_output or args.resume)

    if not args.eval_only and not args.resume:
        # Save command line arguments
        with open(path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f)
        # Save the config as yaml
        with open(path.join(output_dir, 'config.yaml'), 'w') as f:
            cfg.dump(stream=f)


def train_or_eval(args, cfg):
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if args.eval_only:
        trainer.test(cfg, trainer.model)
    elif args.resume:
        trainer.test(cfg, trainer.model)
        trainer.train()
    else:
        trainer.train()

class ROCATrainer(object):
    def __init__(self):
        return

def demo():
    train_name, val_name = register_data(cfg)
    cfgs = make_config(train_name, val_name, cfg)
    setup_output_dir(cfg, cfgs)
    train_or_eval(cfg, cfgs)
    return True

