#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import open3d as o3d

sys.path.append("../habitat-sim-manage/")

from getch import getch

from habitat_sim_manage.Data.point import Point
from habitat_sim_manage.Data.rad import Rad
from habitat_sim_manage.Data.pose import Pose

from image_to_cad.Module.roca_sim_detector import ROCASimDetector
from image_to_cad.Module.roca_merger import ROCAMerger


class ROCAManager(object):

    def __init__(self):
        self.roca_sim_detector = ROCASimDetector()
        self.roca_merger = ROCAMerger()

        self.scene_name = None
        return

    def reset(self):
        self.roca_sim_detector.reset()
        self.roca_merger.reset()
        return True

    def loadSettings(self, roca_settings, glb_file_path):
        return self.roca_sim_detector.loadSettings(roca_settings,
                                                   glb_file_path)

    def updateSceneName(self, scene_name):
        self.scene_name = scene_name
        return self.roca_sim_detector.updateSceneName(scene_name)

    def setControlMode(self, control_mode):
        return self.roca_sim_detector.setControlMode(control_mode)

    def convertResult(self):
        result_dict = self.roca_sim_detector.getResultDict()
        self.roca_merger.addResult(
            result_dict,
            self.roca_sim_detector.sim_manager.pose_controller.pose)
        return True

    def saveRetrievalResult(self):
        save_folder_path = "./output/retrieval/" + self.scene_name + "/"
        os.makedirs(save_folder_path, exist_ok=True)

        mesh_list = self.roca_merger.getInstanceSetListMeanMeshList()

        for i, mesh in enumerate(mesh_list):
            mesh.compute_vertex_normals()

            save_file_path = save_folder_path + str(i) + ".ply"
            o3d.io.write_triangle_mesh(save_file_path, mesh)
        return True

    def startKeyBoardControlRender(self, wait_val):
        #  self.roca_sim_detector.sim_manager.resetAgentPose()
        self.roca_sim_detector.sim_manager.cv_renderer.init()

        while True:
            if not self.roca_sim_detector.sim_manager.cv_renderer.renderFrame(
                    self.roca_sim_detector.sim_manager.sim_loader.observations
            ):
                break
            self.roca_sim_detector.sim_manager.cv_renderer.waitKey(wait_val)

            agent_state = self.roca_sim_detector.sim_manager.sim_loader.getAgentState(
            )
            print("agent_state: position", agent_state.position, "rotation",
                  agent_state.rotation)

            input_key = getch()
            if input_key == "v":
                self.roca_sim_detector.detectObservations()
                self.convertResult()
                #  self.roca_sim_detector.renderResultWithProcess(render_3d=False)
                #  self.roca_merger.renderResultList3DWithProcess()
                self.roca_merger.renderInstanceSetList3DWithProcess()
                self.roca_merger.renderInstanceSetListMean3DWithProcess()
                continue
            if input_key == "x":
                continue
            if not self.roca_sim_detector.sim_manager.keyBoardControl(
                    input_key):
                break
        self.roca_sim_detector.sim_manager.cv_renderer.close()
        return True


def demo():
    scannet_glb_dataset_folder_path = "/home/chli/chLi/ScanNet/glb/"

    valid_scene_name_list = [
        'scene0474_02', 'scene0000_01', 'scene0667_01', 'scene0500_00',
        'scene0247_01', 'scene0644_00', 'scene0231_01', 'scene0653_00',
        'scene0300_00', 'scene0569_00'
    ]

    scene_name = valid_scene_name_list[0]
    glb_file_path = scannet_glb_dataset_folder_path + scene_name + "/" + scene_name + "_vh_clean.glb"
    control_mode = "pose"
    wait_val = 1

    roca_settings = {
        "model_path": "/home/chli/chLi/ROCA/Models/model_best.pth",
        "data_dir": "/home/chli/chLi/ROCA/Data/Dataset/",
        "config_path": "/home/chli/chLi/ROCA/Models/config.yaml",
        "wild": False,
        "output_dir": "none",
    }

    roca_manager = ROCAManager()

    roca_manager.loadSettings(roca_settings, glb_file_path)
    roca_manager.updateSceneName(scene_name)
    roca_manager.setControlMode(control_mode)

    roca_manager.roca_sim_detector.sim_manager.pose_controller.pose = Pose(
        Point(1.7, 1.5, -2.5), Rad(0.2, 0.0))
    roca_manager.roca_sim_detector.sim_manager.sim_loader.setAgentState(
        roca_manager.roca_sim_detector.sim_manager.pose_controller.
        getAgentState())

    roca_manager.startKeyBoardControlRender(wait_val)
    return True
