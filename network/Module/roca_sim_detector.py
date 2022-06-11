#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("./habitat_sim_manage/")

from habitat_sim_manage.Module.sim_manager import SimManager

from Module.roca_detector import ROCADetector

class ROCASimDetector(ROCADetector):
    def __init__(self):
        super(ROCASimDetector, self).__init__()
        self.sim_manager = SimManager()
        return

    def reset(self):
        super(ROCASimDetector, self).reset()
        self.sim_manager.reset()
        return True

    def loadSettings(self, roca_settings, sim_settings):
        super(ROCASimDetector, self).loadSettings(roca_settings)
        self.sim_manager.loadSettings(sim_settings)
        return True

    def detectObservations(self):
        if self.scene_name is None:
            print("[ERROR][ROCASimDetector::detectObservations]")
            print("\t scene_name is None!")
            return False

        observations = self.sim_manager.sim_loader.observations

        if "color_sensor" not in observations.keys():
            print("[ERROR][ROCASimDetector::detectObservations]")
            print("\t color_sensor observation not exist!")
            return False

        rgb_obs = observations["color_sensor"]
        if not self.detectImage(rgb_obs):
            print("[ERROR][ROCASimDetector::detectObservations]")
            print("\t detectImage failed!")
            return False
        return True

def demo():
    scene_name = "scene0474_02"
    glb_file_path = \
        "/home/chli/habitat/scannet/scans/scene0474_02/scene0474_02_vh_clean.glb"
    control_mode = "pose"
    pause_time = 0.001

    roca_settings = {
        "model_path": "../Models/model_best.pth",
        "data_dir": "../Data/Dataset/",
        "config_path": "../Models/config.yaml",
        "wild": False,
        "output_dir": "none",
    }
    sim_settings = {
        "width": 256,
        "height": 256,
        "scene": glb_file_path,
        "default_agent": 0,
        "move_dist": 0.25,
        "rotate_angle": 10.0,
        "sensor_height": 0,
        "color_sensor": True,
        "depth_sensor": True,
        "semantic_sensor": True,
        "seed": 1,
        "enable_physics": False,
    }

    roca_sim_detector = ROCASimDetector()
    roca_sim_detector.updateSceneName(scene_name)
    roca_sim_detector.loadSettings(roca_settings, sim_settings)

    scene_name_dict = {
        '3m': 'scene0474_02',
        'sofa': 'scene0207_00',
        'lab': 'scene0378_02',
        'desk': 'scene0474_02',
    }

    for name in scene_name_dict.keys():
        scene_name = scene_name_dict[name]
        roca_sim_detector.detectImageFromPath('assets/' + name + '.jpg', scene_name)
        roca_sim_detector.renderResult()
    return True

if __name__ == '__main__':
    demo()

