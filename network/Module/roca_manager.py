#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("./habitat_sim_manage/")

from getch import getch

from habitat_sim_manage.Data.point import Point
from habitat_sim_manage.Data.rad import Rad
from habitat_sim_manage.Data.pose import Pose

from Module.roca_sim_detector import ROCASimDetector
from Module.roca_merger import ROCAMerger

import matplotlib.pyplot as plt
import numpy as np

class ROCAManager(object):
    def __init__(self):
        self.roca_sim_detector = ROCASimDetector()
        self.roca_merger = ROCAMerger()
        return

    def reset(self):
        self.roca_sim_detector.reset()
        self.roca_merger.reset()
        return True

    def showPoints(self, point_list):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        for i in range(int(len(point_list) / 2)):
            start = point_list[2 * i]
            end = point_list[2 * i + 1]
            ax.quiver(start[0], start[1], start[2],
                  end[0]-start[0], end[1]-start[1], end[2]-start[2],
                  arrow_length_ratio=0.1)

        plt.show()
        return True

    def loadSettings(self, roca_settings, sim_settings=None):
        return self.roca_sim_detector.loadSettings(roca_settings, sim_settings)

    def updateSceneName(self, scene_name):
        return self.roca_sim_detector.updateSceneName(scene_name)

    def setControlMode(self, control_mode):
        return self.roca_sim_detector.setControlMode(control_mode)

    def setRenderMode(self, render_mode):
        return self.roca_sim_detector.setRenderMode(render_mode)

    def addResult(self):
        result_dict = self.roca_sim_detector.getResultDict()
        self.roca_merger.addResult(
            result_dict, self.roca_sim_detector.sim_manager.pose_controller.pose)
        pose_list = self.roca_merger.getPoseListInWorld()
        self.showPoints(pose_list)
        return True

    def startKeyBoardControlRender(self, wait_val):
        #  self.roca_sim_detector.sim_manager.resetAgentPose()
        self.roca_sim_detector.sim_manager.sim_renderer.init()

        while True:
            if not self.roca_sim_detector.sim_manager.sim_renderer.renderFrame(
                    self.roca_sim_detector.sim_manager.sim_loader.observations):
                break
            self.roca_sim_detector.sim_manager.sim_renderer.wait(wait_val)

            agent_state = self.roca_sim_detector.sim_manager.sim_loader.getAgentState()
            print("agent_state: position", agent_state.position,
                  "rotation", agent_state.rotation)

            input_key = getch()
            if input_key == "a":
                self.roca_sim_detector.detectObservations()
                self.addResult()
                self.roca_sim_detector.renderResultWithProcess()
                continue
            if not self.roca_sim_detector.sim_manager.keyBoardControl(input_key):
                break
        self.roca_sim_detector.sim_manager.sim_renderer.close()
        return True

def demo():
    scene_name = "scene0474_02"
    glb_file_path = \
        "/home/chli/habitat/scannet/scans/scene0474_02/scene0474_02_vh_clean.glb"
    control_mode = "pose"
    render_mode = "cv"
    wait_val = 1

    roca_settings = {
        "model_path": "../Models/model_best.pth",
        "data_dir": "../Data/Dataset/",
        "config_path": "../Models/config.yaml",
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

    roca_manager = ROCAManager()

    roca_manager.loadSettings(roca_settings, sim_settings)
    roca_manager.updateSceneName(scene_name)
    roca_manager.setControlMode(control_mode)
    roca_manager.setRenderMode(render_mode)

    roca_manager.roca_sim_detector.sim_manager.pose_controller.pose = Pose(
        Point(1.7, 1.5, -2.5), Rad(0.2, 0.0))
    roca_manager.roca_sim_detector.sim_manager.sim_loader.setAgentState(
        roca_manager.roca_sim_detector.sim_manager.pose_controller.getAgentState())

    roca_manager.startKeyBoardControlRender(wait_val)
    return True

if __name__ == "__main__":
    demo()
