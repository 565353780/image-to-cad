#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import open3d as o3d
from PIL import Image
from trimesh.exchange.export import export_mesh
from trimesh.util import concatenate as stack_meshes
from multiprocessing import Process

import sys
sys.path.append("./network/")
#  from roca.engine import Predictor
from image_to_cad.Module.roca_predictor import Predictor

class ROCADetector(object):
    def __init__(self):
        self.predictor = None
        self.to_file = None

        self.image = None
        self.scene_name = None
        self.instances = None
        self.cad_ids = None
        self.meshes = None
        self.masked_image = None
        self.save_idx = 0
        return

    def reset(self):
        self.predictor = None
        self.to_file = None

        self.image = None
        self.scene_name = None
        self.instances = None
        self.cad_ids = None
        self.meshes = None
        self.masked_image = None
        self.save_idx = 0
        return True

    def loadSettings(self, roca_settings):
        self.predictor = Predictor(
            data_dir=roca_settings["data_dir"],
            model_path=roca_settings["model_path"],
            config_path=roca_settings["config_path"],
            wild=roca_settings["wild"])

        self.to_file = roca_settings["output_dir"] != "none"
        return True

    def updateSceneName(self, scene_name):
        if scene_name is None:
            return True
        self.scene_name = scene_name
        return True

    def detect(self):
        if self.image is None:
            print("[ERROR][ROCADetector::detect]")
            print("\t image is None!")
            return False
        if self.scene_name is None:
            print("[ERROR][ROCADetector::detect]")
            print("\t scene_name is None!")
            return False

        self.image = np.asarray(self.image)

        self.instances, self.cad_ids = self.predictor(
            self.image, scene=self.scene_name)

        self.meshes = self.predictor.output_to_mesh(
            self.instances, self.cad_ids,
            excluded_classes={'table'} if self.predictor.wild else (),
            as_open3d=not self.to_file,
            nms_3d=False)

        self.masked_image = None

        if self.predictor.can_render:
            rendering, ids = self.predictor.render_meshes(self.meshes)
            mask = ids > 0
            self.masked_image = self.image.copy()
            self.masked_image[mask] = np.clip(
                0.8 * rendering[mask] * 255 + 0.2 * self.masked_image[mask],
                0,
                255
            ).astype(np.uint8)
        return True

    def detectImage(self, image, scene_name=None):
        self.image = image
        self.updateSceneName(scene_name)

        if not self.detect():
            print("[ERROR][ROCADetector::detectImage]")
            print("\t detect failed!")
            return False
        return True

    def detectImageFromPath(self, image_file_path, scene_name=None):
        if not os.path.exists(image_file_path):
            print("[ERROR][ROCADetector::detectImageFromPath]")
            print("\t image file not exist!")
            return False

        image = Image.open(image_file_path)
        return self.detectImage(image, scene_name)

    def getResultDict(self):
        result_dict = {
            "image": self.image,
            "scene_name": self.scene_name,
            "instances": self.instances,
            "cad_ids": self.cad_ids,
            "meshes": self.meshes,
            "masked_image": self.masked_image,
        }
        return result_dict

    def renderResultImage(self):
        if self.predictor.can_render:
            if self.masked_image is None:
                print("[ERROR][ROCADetector::renderResult]")
                print("\t masked_image is None!")
                return False

            img = o3d.geometry.Image(self.masked_image)
            o3d.visualization.draw_geometries([img], height=480, width=640)
        return True

    def renderResult3D(self):
        o3d.visualization.draw_geometries(self.meshes)
        return True

    def renderResultImageWithProcess(self):
        process = Process(target=self.renderResultImage)
        process.start()
        #  process.join()
        #  process.close()
        return True

    def renderResult3DWithProcess(self):
        process = Process(target=self.renderResult3D)
        process.start()
        #  process.join()
        #  process.close()
        return True

    def renderResultWithProcess(self, render_image=True, render_3d=True):
        if render_image:
            self.renderResultImageWithProcess()
        if render_3d:
            self.renderResult3DWithProcess()
        return True

    def saveResult(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        if self.masked_image is not None:
            Image.fromarray(self.masked_image).save(
                output_dir + 'overlay_' + self.scene_name + \
                '_' + str(self.save_idx) + '.jpg')

        out_file = output_dir + 'mesh_' + self.scene_name + \
            '_' + str(self.save_idx) + '.ply'
        export_mesh(stack_meshes(self.meshes), out_file, file_type='ply')

        self.save_idx += 1
        return True

def demo():
    scene_name = "scene0474_02"

    roca_settings = {
        "model_path": "./Models/model_best.pth",
        "data_dir": "./Dataset/Dataset/",
        "config_path": "./Models/config.yaml",
        "wild": False,
        "output_dir": "none",
    }

    roca_detector = ROCADetector()
    roca_detector.updateSceneName(scene_name)
    roca_detector.loadSettings(roca_settings)

    scene_name_dict = {
        '3m': 'scene0474_02',
        'sofa': 'scene0207_00',
        'lab': 'scene0378_02',
        'desk': 'scene0474_02',
    }

    for name in scene_name_dict.keys():
        scene_name = scene_name_dict[name]
        roca_detector.detectImageFromPath('assets/' + name + '.jpg', scene_name)
        result = roca_detector.getResultDict()
        roca_detector.renderResult()
    return True

if __name__ == '__main__':
    demo()

