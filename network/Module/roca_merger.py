#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
from multiprocessing import Process

from Data.result import Result
from Data.instance_set import InstanceSet

from Method.dists import getMatchListFromMatrix

class ROCAMerger(object):
    def __init__(self):
        self.result_list = []

        self.instance_set_list = []
        return

    def reset(self):
        self.result_list = []

        self.instance_set_list = []
        return True

    def addResult(self, result_dict, camera_pose):
        result = Result()
        if not result.loadResultDict(result_dict, camera_pose):
            print("[ERROR][ROCAMerger::addResult]")
            print("\t loadResultDict failed!")
            return False
        self.result_list.append(result)

        if not self.mergeResult(len(self.result_list) - 1):
            print("[ERROR][ROCAMerger::addResult]")
            print("\t mergeResult failed!")
            return False
        return True

    def getResultMeshList(self, result_idx):
        result_mesh_list = []

        if result_idx >= len(self.result_list):
            print("[ERROR][ROCAMerger::getResultMeshList]")
            print("\t result_idx out of range!")
            return result_mesh_list

        result = self.result_list[result_idx]
        for result_keep_instance_idx in result.keep_instance_idx_list:
            instance = result.instance_list[result_keep_instance_idx]
            result_mesh_list.append(instance.world_mesh)
            result_mesh_list.append(instance.getOpen3DBBox([0, 255, 0]))
        if result.camera_instance is not None:
            result_mesh_list.append(result.camera_instance.world_mesh)
        return result_mesh_list

    def getResultListMeshList(self):
        result_list_mesh_list = []
        for i in range(len(self.result_list)):
            result_mesh_list = self.getResultMeshList(i)
            result_list_mesh_list += result_mesh_list
        return result_list_mesh_list

    def mergeResult(self, result_idx):
        if result_idx >= len(self.result_list):
            print("[ERROR][ROCAMerger::mergeResult]")
            print("\t result_idx out of range!")
            return False

        result = self.result_list[result_idx]
        if len(self.instance_set_list) == 0:
            for result_keep_instance_idx in result.keep_instance_idx_list:
                instance = result.instance_list[result_keep_instance_idx]
                instance_set = InstanceSet([instance])
                self.instance_set_list.append(instance_set)
            return True

        instance_dist_matrix = np.zeros([len(self.instance_set_list), result.getKeepInstanceNum()])

        for i in range(len(self.instance_set_list)):
            instance_dist_list = self.instance_set_list[i].getInstanceDistList(result.instance_list,
                                                                               result.keep_list)
            instance_dist_matrix[i] = np.array(instance_dist_list)

        match_list = getMatchListFromMatrix(instance_dist_matrix)

        is_match_list = [False for _ in result.keep_instance_idx_list]
        for match_pair in match_list:
            instance_set_idx = match_pair[0]
            result_keep_instance_idx = match_pair[1]
            is_match_list[result_keep_instance_idx] = True
            result_instance_idx = result.keep_instance_idx_list[result_keep_instance_idx]
            self.instance_set_list[instance_set_idx].instance_list.append(
                result.instance_list[result_instance_idx])
        for is_match, result_keep_instance_idx in zip(is_match_list, result.keep_instance_idx_list):
            if is_match:
                continue
            instance = result.instance_list[result_keep_instance_idx]
            instance_set = InstanceSet([instance])
            self.instance_set_list.append(instance_set)
        return True

    def renderResult3D(self, result_idx):
        if result_idx >= len(self.result_list):
            print("[ERROR][ROCAMerger::renderResult3D]")
            print("\t result_idx out of range!")
            return False

        result_mesh_list = self.getResultMeshList(result_idx)
        o3d.visualization.draw_geometries(result_mesh_list)

    def renderResult3DWithProcess(self, result_idx):
        process = Process(target=self.renderResult3D, args=(result_idx,))
        process.start()
        #  process.join()
        #  process.close()
        return True

    def renderResultList3D(self):
        if len(self.result_list) == 0:
            return True

        result_list_mesh_list = self.getResultListMeshList()
        o3d.visualization.draw_geometries(result_list_mesh_list)
        return True

    def renderResultList3DWithProcess(self):
        process = Process(target=self.renderResultList3D)
        process.start()
        #  process.join()
        #  process.close()
        return True

    def renderInstanceSet3D(self):
        return

def demo():
    roca_merger = ROCAMerger()
    return True

if __name__ == "__main__":
    demo()

