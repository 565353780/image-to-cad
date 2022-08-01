#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
sys.path.append("../habitat_sim_manage")

from getch import getch

from habitat_sim_manage.Data.point import Point
from habitat_sim_manage.Data.rad import Rad
from habitat_sim_manage.Data.pose import Pose
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

    def loadSettings(self, roca_settings, sim_settings=None):
        super(ROCASimDetector, self).loadSettings(roca_settings)
        if sim_settings is not None:
            self.sim_manager.loadSettings(sim_settings)
        return True

    def setControlMode(self, control_mode):
        return self.sim_manager.setControlMode(control_mode)

    def setRenderMode(self, render_mode):
        return self.sim_manager.setRenderMode(render_mode)

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
        rgb_obs = rgb_obs[..., 0:3]
        if not self.detectImage(rgb_obs):
            print("[ERROR][ROCASimDetector::detectObservations]")
            print("\t detectImage failed!")
            return False
        return True

    def startKeyBoardControlRender(self, wait_val):
        #  self.sim_manager.resetAgentPose()
        self.sim_manager.sim_renderer.init()

        while True:
            if not self.sim_manager.sim_renderer.renderFrame(
                    self.sim_manager.sim_loader.observations):
                break
            self.sim_manager.sim_renderer.wait(wait_val)

            agent_state = self.sim_manager.sim_loader.getAgentState()
            print("agent_state: position", agent_state.position,
                  "rotation", agent_state.rotation)

            input_key = getch()
            if input_key == "a":
                self.detectObservations()
                self.renderResultWithProcess()
                continue
            if not self.sim_manager.keyBoardControl(input_key):
                break
        self.sim_manager.sim_renderer.close()
        return True

def demo_roca():
    scene_name = "scene0474_02"

    roca_settings = {
        "model_path": "./Models/model_best.pth",
        "data_dir": "./Dataset/Dataset/",
        "config_path": "./Models/config.yaml",
        "wild": False,
        "output_dir": "none",
    }

    roca_sim_detector = ROCASimDetector()
    roca_sim_detector.updateSceneName(scene_name)
    roca_sim_detector.loadSettings(roca_settings)

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

def demo_roca_sim():
    scene_name = "scene0474_02"
    glb_file_path = \
        "/home/chli/scan2cad/scannet/scans/scene0474_02/scene0474_02_vh_clean.glb"
    control_mode = "pose"
    render_mode = "cv"
    wait_val = 1

    roca_settings = {
        "model_path": "./Models/model_best.pth",
        "data_dir": "./Dataset/Dataset/",
        "config_path": "./Models/config.yaml",
        "wild": False,
        "output_dir": "none",
    }
    sim_settings = {
        "width": 480,
        "height": 360,
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
    roca_sim_detector.loadSettings(roca_settings, sim_settings)
    roca_sim_detector.updateSceneName(scene_name)
    roca_sim_detector.setControlMode(control_mode)
    roca_sim_detector.setRenderMode(render_mode)

    roca_sim_detector.sim_manager.pose_controller.pose = Pose(
        Point(1.7, 1.5, -2.5), Rad(0.2, 0.0))
    roca_sim_detector.sim_manager.sim_loader.setAgentState(
        roca_sim_detector.sim_manager.pose_controller.getAgentState())

    roca_sim_detector.startKeyBoardControlRender(wait_val)
    return True

def demo():
    #  demo_roca()
    demo_roca_sim()
    return True

if __name__ == '__main__':
    demo()

