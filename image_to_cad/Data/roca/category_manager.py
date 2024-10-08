#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from collections import Counter

from detectron2.data import DatasetCatalog, MetadataCatalog

from image_to_cad.Config.roca.constants import BENCHMARK_CLASSES

class CategoryManager:
    def __init__(self, category_json, dataset_name):
        with open(category_json) as f:
            self._categories = set(json.load(f))
        metadata = MetadataCatalog.get(dataset_name)
        self._index_to_id = {
            v: k
            for k, v in metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._id_to_name = metadata.thing_classes
        self._metadata = metadata
        self._name_to_id = {v: k for k, v in enumerate(self._id_to_name)}
        assert len(self._name_to_id) == len(self._id_to_name)

        self.dataset_name = dataset_name
        self.num_classes = len(self._id_to_name)
        self.freqs = None

    def is_alignment_class(self, index_or_name) -> bool:
        if not isinstance(index_or_name, str):
            index_or_name = self.get_name(index_or_name)
        return index_or_name in self._categories

    def is_benchmark_class(self, index_or_name) -> bool:
        if not isinstance(index_or_name, str):
            index_or_name = self.get_name(index_or_name)
        return index_or_name in BENCHMARK_CLASSES

    def get_name(self, index) -> str:
        idx = self._index_to_id[index]
        name = self._id_to_name[idx]
        return name

    def get_id(self, name) -> str:
        return self._name_to_id[name]

    def set_freqs(self, class_freq_method='none'):
        if class_freq_method == 'none':
            return
        self.freqs = Counter()
        data = DatasetCatalog.get(self.dataset_name)
        for record in data:
            if 'annotations' in record:
                for annot in record['annotations']:
                    idx = self._index_to_id[annot['category_id']]
                    self.freqs[idx] += 1
        return

class CategoryCatalog:
    _managers = {}

    @classmethod
    def register(cls, dataset_name: str, category_json: str):
        if dataset_name in cls._managers:
            raise RuntimeError(
                'Categories for {} are already registered'.format(dataset_name)
            )
        cls._managers[dataset_name] = CategoryManager(
            category_json, dataset_name
        )

    @classmethod
    def deregister(cls, name):
        del cls._managers[name]

    @classmethod
    def get(cls, name: str) -> CategoryManager:
        return cls._managers[name]


def register_categories(
    dataset_name: str,
    category_json: str,
    class_freq_method: str = 'none'
):
    CategoryCatalog.register(dataset_name, category_json)
    CategoryCatalog.get(dataset_name).set_freqs(class_freq_method)


def deregister_categories(dataset_name: str):
    CategoryCatalog.deregister(dataset_name)
