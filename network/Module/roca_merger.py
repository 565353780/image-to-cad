#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from math import cos, sin, pi

from Data.trans import Trans
from Data.instance import Instance

from Method.nms import getKeepList

class ROCAMerger(object):
    def __init__(self):
        self.instance_list_list = []
        return

    def reset(self):
        self.instance_list_list = []
        return True

    def getInstance(self, result, instance_idx):
        instances = result["instances"]
        cad_ids = result["cad_ids"]
        meshes = result["meshes"]
        if instance_idx >= len(instances):
            print("[ERROR][ROCAMerger::getInstance]")
            print("\t instance_idx out of range!")
            return None

        instance = Instance(
            instances.pred_classes[instance_idx],
            instances.scores[instance_idx],
            Trans(
                instances.pred_translations[instance_idx],
                instances.pred_rotations[instance_idx],
                instances.pred_scales[instance_idx]),
            cad_ids[instance_idx],
            meshes[instance_idx])
        return instance

    def getInstanceList(self, result):
        instances = result["instances"]
        instance_list = [
            self.getInstance(result, i) for i in range(len(instances))
        ]
        return instance_list

    def loadInstanceList(self, result):
        min_dist_3d = 0.4

        instance_list = self.getInstanceList(result)
        keep_list = getKeepList(instance_list, min_dist_3d)
        nms_instance_list = [i for i, k in zip(instance_list, keep_list) if k]
        self.instance_list_list.append(nms_instance_list)
        return True

    def addResult(self, result):
        self.loadInstanceList(result)
        return True

    def getInstanceTransMatrix(self, instance):
        agent_trans_matrix = self.getAgentTransMatrix()
        instance_source_trans_matrix = instance.getTransMatrix()
        instance_trans_matrix = agent_trans_matrix @ instance_source_trans_matrix
        return instance_trans_matrix

    def getPoseListInWorld(self):
        pose_list = []

        if len(self.instance_list_list) == 0:
            return pose_list

        scene_rot = np.array([
            [1, 0, 0, 0],
            [0, cos(pi), -sin(pi), 0],
            [0, sin(pi), cos(pi), 0],
            [0, 0, 0, 1]
        ])
        #  scene_rot = np.linalg.inv(scene_rot)

        init_point_list = [
            [-0.25, -0.25, -0.25, 1.0],
            [-0.25, -0.25, 0.25, 1.0],
            [-0.25, 0.25, -0.25, 1.0],
            [-0.25, 0.25, 0.25, 1.0],
            [0.25, -0.25, -0.25, 1.0],
            [0.25, -0.25, 0.25, 1.0],
            [0.25, 0.25, -0.25, 1.0],
            [0.25, 0.25, 0.25, 1.0]]

        init_line_pair = [
            [0, 1], [2, 3], [4, 5], [6, 7],
            [0, 4], [1, 5], [2, 6], [3, 7],
            [0, 2], [1, 3], [4, 6], [5, 7]]
        
        for instance_list in self.instance_list_list:
            for instance in instance_list:
                trans_matrix = scene_rot @ instance.getTransMatrix()
                for pair in init_line_pair:
                    start = init_point_list[pair[0]]
                    end = init_point_list[pair[1]]
                    pose_list.append(trans_matrix @ start)
                    pose_list.append(trans_matrix @ end)
                mesh_center = instance.mesh.get_center()
                new_point = mesh_center + [1, 0, 0]
                pose_list.append(instance.mesh.get_center())
                pose_list.append(new_point)
        return pose_list

def demo():
    roca_merger = ROCAMerger()
    return True

if __name__ == "__main__":
    demo()

