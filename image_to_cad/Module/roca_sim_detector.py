#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../habitat-sim-manage")
from habitat_sim_manage.Data.point import Point
from habitat_sim_manage.Data.rad import Rad
from habitat_sim_manage.Data.pose import Pose
from habitat_sim_manage.Module.sim_manager import SimManager

from getch import getch

from image_to_cad.Module.roca_detector import ROCADetector


class ROCASimDetector(ROCADetector):

    def __init__(self):
        super(ROCASimDetector, self).__init__()
        self.sim_manager = SimManager()
        return

    def reset(self):
        super(ROCASimDetector, self).reset()
        self.sim_manager.reset()
        return True

    def loadSettings(self, roca_settings, glb_file_path):
        super().loadSettings(roca_settings)
        self.sim_manager.loadSettings(glb_file_path)
        return True

    def setControlMode(self, control_mode):
        return self.sim_manager.setControlMode(control_mode)

    def detectObservations(self):
        assert self.scene_name is not None

        observations = self.sim_manager.sim_loader.observations

        assert "color_sensor" in observations.keys()

        rgb_obs = observations["color_sensor"]
        rgb_obs = rgb_obs[..., 0:3]
        assert self.detectImage(rgb_obs)
        return True

    def startKeyBoardControlRender(self, wait_val):
        #  self.sim_manager.resetAgentPose()
        self.sim_manager.cv_renderer.init()

        while True:
            if not self.sim_manager.cv_renderer.renderFrame(
                    self.sim_manager.sim_loader.observations):
                break
            self.sim_manager.cv_renderer.waitKey(wait_val)

            agent_state = self.sim_manager.sim_loader.getAgentState()
            print("agent_state: position", agent_state.position, "rotation",
                  agent_state.rotation)

            input_key = getch()
            if input_key == "x":
                self.detectObservations()
                self.renderResultWithProcess()
                continue
            if not self.sim_manager.keyBoardControl(input_key):
                break
        self.sim_manager.cv_renderer.close()
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
        roca_sim_detector.detectImageFromPath('assets/' + name + '.jpg',
                                              scene_name)
        roca_sim_detector.renderResult()
    return True


def demo_roca_sim():
    scene_name = "scene0474_02"
    glb_file_path = \
        "/home/chli/chLi/ScanNet/scans/scene0474_02/scene0474_02_vh_clean.glb"
    control_mode = "circle"
    wait_val = 1

    roca_settings = {
        "model_path": "./Models/model_best.pth",
        "data_dir": "./Dataset/Dataset/",
        "config_path": "./Models/config.yaml",
        "wild": False,
        "output_dir": "none",
    }

    roca_sim_detector = ROCASimDetector()
    roca_sim_detector.loadSettings(roca_settings, glb_file_path)
    roca_sim_detector.updateSceneName(scene_name)
    roca_sim_detector.setControlMode(control_mode)

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
