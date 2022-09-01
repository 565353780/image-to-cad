#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("./network/")

import os
import json
import torch
import random
import numpy as np
import os.path as path
from os import makedirs
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.data import \
    build_detection_train_loader, get_detection_dataset_dicts
from detectron2.utils.events import EventStorage

from network.roca.config import roca_config
from network.roca.data import CategoryCatalog, Mapper
from network.roca.data.datasets import register_scan2cad

from image_to_cad.Config.config import TRAIN_CONFIG

from image_to_cad.Model.roca import ROCA

from image_to_cad.Method.time import getCurrentTimeStr

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
    return

def make_config(config):
    train_name = 'Scan2CAD_train'
    val_name = 'Scan2CAD_val'

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

    if config["resume"]:
        return

    # Save command line arguments
    with open(path.join(config["output_dir"], 'args.json'), 'w') as f:
        json.dump(config, f)
    # Save the config as yaml
    with open(path.join(output_dir, 'config.yaml'), 'w') as f:
        cfg.dump(stream=f)
    return

def build_train_loader(cfg):
    datasets = cfg.DATASETS.TRAIN
    assert len(datasets) == 1
    workers = cfg.SOLVER.WORKERS
    mapper = Mapper(cfg, True, datasets[0])
    seed = cfg.SEED
    if seed > 0:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    return build_detection_train_loader(cfg,
                                        mapper=mapper,
                                        num_workers=workers)

class ROCATrainer(object):
    def __init__(self, config):
        self.cfg = make_config(config)
        setup_output_dir(config, self.cfg)
        self.model = ROCA(self.cfg)
        self.model.to(torch.device("cuda"))
        self.optimizer = build_optimizer(self.cfg, self.model)
        self.data_loader = build_train_loader(self.cfg)
        self.scheduler = build_lr_scheduler(self.cfg, self.optimizer)

        self.output_folder_path = config["output_dir"] + getCurrentTimeStr() + "/"
        self.writter = SummaryWriter(self.output_folder_path)

        self._data_loader_iter = iter(self.data_loader)

        self.iter = self.start_iter = 0
        self.max_iter = config["steps"][0]

        self.do_val_step = True

        self.model.requires_grad_()
        self._init_val_step()

        self.loadModel(config["checkpoint"])
        return

    def loadModel(self, model_path):
        if not os.path.exists(model_path):
            print("[ERROR][ROCATrainer::loadModel]")
            print("\t model not exist!")
            return False

        if "last_checkpoint" in model_path:
            model_folder_path = model_path.split("/last_checkpoint")[0] + "/"
            with open(model_path, "r") as f:
                lines = f.readlines()
                model_file_name = lines[0].split("\n")[0]
                model_path = model_folder_path + model_file_name

        if not os.path.exists(model_path):
            print("[ERROR][ROCATrainer::loadModel]")
            print("\t model not exist!")
            return False

        backup = torch.load(model_path)
        self.model.load_state_dict(backup['model'])
        return True

    def _init_val_step(self):
        if self.do_val_step:
            test_datasets = self.cfg.DATASETS.TEST
            assert len(test_datasets) == 1, \
                'multiple test datasets not supported'

            dataset = get_detection_dataset_dicts(self.cfg.DATASETS.TEST)
            num_workers = self.cfg.SOLVER.WORKERS
            mapper = Mapper(self.cfg, True, test_datasets[0])
            self._sample_val_data = build_detection_train_loader(
                mapper=mapper,
                dataset=dataset,
                total_batch_size=self.cfg.SOLVER.IMS_PER_BATCH,
                num_workers=num_workers
            )
            self._sample_val_iter = iter(self._sample_val_data)
        return True

    def train_step(self):
        data = next(self._data_loader_iter)

        with EventStorage(self.iter) as self.storage:
            loss_dict = self.model(data)

        for key, item in loss_dict.items():
            self.writer.add_scalar("train/" + key, item, self.iter)
        train_total_loss = sum(loss_dict.values())
        self.writer.add_scalar("train/total_loss", train_total_loss, self.iter)

        losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        losses.backward()

        self.optimizer.step()
        return True

    def eval_step(self):
        with torch.no_grad():
            if self.do_val_step:
                data = next(self._sample_val_iter)
                with EventStorage(self.iter) as self.storage:
                    val_loss_dict = self.model(data)
                val_loss_dict['total_loss'] = sum(val_loss_dict.values())

                for key, item in val_loss_dict.items():
                    self.writer.add_scalar("val/" + key, item, self.iter)
        return True

    def train(self):
        for self.iter in tqdm(range(self.start_iter, self.max_iter)):
            self.train_step()
            self.eval_step()
        return True

def demo():
    register_data(TRAIN_CONFIG)

    roca_trainer = ROCATrainer(TRAIN_CONFIG)
    roca_trainer.train()
    return True

