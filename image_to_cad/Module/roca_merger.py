#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
from multiprocessing import Process

from image_to_cad.Data.result import Result
from image_to_cad.Data.instance_set import InstanceSet

from image_to_cad.Method.dists import getMatchListFromMatrix

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

    def getCameraMeshList(self):
        camera_mesh_list = []
        for result in self.result_list:
            if result.camera_instance is None:
                continue
            camera_mesh_list.append(result.camera_instance.world_mesh)
        return camera_mesh_list

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

    def getInstanceSetMeshList(self, instance_set_idx):
        instance_set_mesh_list = []

        if instance_set_idx >= len(self.instance_set_list):
            print("[ERROR][ROCAMerger::getInstanceSetMeshList]")
            print("\t instance_set_idx out of range!")
            return instance_set_mesh_list

        instance_set = self.instance_set_list[instance_set_idx]
        for instance in instance_set.instance_list:
            instance_set_mesh_list.append(instance.world_mesh)
            instance_set_mesh_list.append(instance.getOpen3DBBox([0, 255, 0]))

        instance_set_mesh_list += self.getCameraMeshList()
        return instance_set_mesh_list

    def getInstanceSetListMeshList(self):
        instance_set_list_mesh_list = []
        for i in range(len(self.instance_set_list)):
            instance_set_mesh_list = self.getInstanceSetMeshList(i)
            instance_set_list_mesh_list += instance_set_mesh_list
        return instance_set_list_mesh_list

    def getInstanceSetMeanMeshList(self, instance_set_idx):
        if instance_set_idx >= len(self.instance_set_list):
            print("[ERROR][ROCAMerger::getInstanceSetMeshList]")
            print("\t instance_set_idx out of range!")
            return []

        instance_set = self.instance_set_list[instance_set_idx]
        return [instance_set.getMeanMesh()]

    def getInstanceSetListMeanMeshList(self):
        instance_set_list_mean_mesh_list = []
        for i in range(len(self.instance_set_list)):
            instance_set_mean_mesh_list = self.getInstanceSetMeanMeshList(i)
            instance_set_list_mean_mesh_list += instance_set_mean_mesh_list
        return instance_set_list_mean_mesh_list

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

    def renderInstanceSet3D(self, instance_set_idx):
        if instance_set_idx >= len(self.instance_set_list):
            print("[ERROR][ROCAMerger::renderInstanceSet3D]")
            print("\t instance_set_idx out of range!")
            return False

        instance_set_mesh_list = self.getInstanceSetMeshList(instance_set_idx)
        o3d.visualization.draw_geometries(instance_set_mesh_list)
        return True

    def renderInstanceSet3DWithProcess(self, instance_set_idx):
        process = Process(target=self.renderInstanceSet3D, args=(instance_set_idx,))
        process.start()
        #  process.join()
        #  process.close()
        return True

    def renderInstanceSetList3D(self):
        if len(self.instance_set_list) == 0:
            return True

        instance_set_list_mesh_list = self.getInstanceSetListMeshList()
        o3d.visualization.draw_geometries(instance_set_list_mesh_list)
        return True

    def renderInstanceSetList3DWithProcess(self):
        process = Process(target=self.renderInstanceSetList3D)
        process.start()
        #  process.join()
        #  process.close()
        return True

    def renderInstanceSetMean3D(self, instance_set_idx):
        if instance_set_idx >= len(self.instance_set_list):
            print("[ERROR][ROCAMerger::renderInstanceSetMean3D]")
            print("\t instance_set_idx out of range!")
            return False

        instance_set_mean_mesh_list = self.getInstanceSetMeanMeshList(instance_set_idx)
        o3d.visualization.draw_geometries(instance_set_mean_mesh_list)
        return True

    def renderInstanceSetMean3DWithProcess(self, instance_set_idx):
        process = Process(target=self.renderInstanceSetMean3D, args=(instance_set_idx,))
        process.start()
        #  process.join()
        #  process.close()
        return True

    def renderInstanceSetListMean3D(self):
        if len(self.instance_set_list) == 0:
            return True

        instance_set_list_mean_mesh_list = self.getInstanceSetListMeanMeshList()
        o3d.visualization.draw_geometries(instance_set_list_mean_mesh_list)
        return True

    def renderInstanceSetListMean3DWithProcess(self):
        process = Process(target=self.renderInstanceSetListMean3D)
        process.start()
        #  process.join()
        #  process.close()
        return True

def demo():
    roca_merger = ROCAMerger()
    return True

if __name__ == "__main__":
    demo()

