#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
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

from image_to_cad.Config.config import TRAIN_CONFIG
from image_to_cad.Config.roca.roca_config import roca_config

from image_to_cad.Data.roca.category_manager import CategoryCatalog
from image_to_cad.Data.roca.mapper import Mapper
from image_to_cad.Data.roca.datasets import register_scan2cad

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
    cfg = roca_config(
        num_classes=CategoryCatalog.get('Scan2CAD_train').num_classes,
        class_freqs=CategoryCatalog.get('Scan2CAD_train').freqs,
    )

    # NOTE: Training state will be reset in this case!
    if config["checkpoint"].lower() not in ('', 'none'):
        cfg.MODEL.WEIGHTS = config["checkpoint"]

    return cfg

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
        self.model = ROCA(self.cfg)
        self.model.to(torch.device("cuda"))
        self.optimizer = build_optimizer(self.cfg, self.model)
        self.data_loader = build_train_loader(self.cfg)
        self.scheduler = build_lr_scheduler(self.cfg, self.optimizer)

        output_folder_path = "./output/logs/" + getCurrentTimeStr() + "/"
        makedirs(output_folder_path, exist_ok=True)
        os.makedirs(output_folder_path, exist_ok=True)
        self.writer = SummaryWriter(output_folder_path)

        self._data_loader_iter = iter(self.data_loader)

        self.iter = self.start_iter = 0
        self.max_iter = 100000
        self.eval_period = 100

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
            data = self.model(data)

        for key, item in data['losses'].items():
            self.writer.add_scalar("train/" + key, item, self.iter)
        for key, item in data['logs'].items():
            self.writer.add_scalar(key, item, self.iter)
        train_total_loss = sum(data['losses'].values())
        self.writer.add_scalar("train/total_loss", train_total_loss, self.iter)

        losses = sum(data['losses'].values())

        self.optimizer.zero_grad()
        losses.backward()

        self.optimizer.step()
        return True

    def eval_step(self):
        with torch.no_grad():
            if self.do_val_step:
                data = next(self._sample_val_iter)
                with EventStorage(self.iter) as self.storage:
                    data = self.model(data)
                data['losses']['total_loss'] = sum(data['losses'].values())

                for key, item in data['losses'].items():
                    self.writer.add_scalar("val/" + key, item, self.iter)
        return True

    def train(self):
        for self.iter in tqdm(range(self.start_iter, self.max_iter)):
            self.train_step()
            if self.iter % self.eval_period == 0:
                self.eval_step()
        return True

def demo():
    register_data(TRAIN_CONFIG)

    roca_trainer = ROCATrainer(TRAIN_CONFIG)
    roca_trainer.train()
    return True

