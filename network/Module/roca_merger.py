#!/usr/bin/env python
# -*- coding: utf-8 -*-

import open3d as o3d
from multiprocessing import Process

from Data.result import Result

class ROCAMerger(object):
    def __init__(self):
        self.result_list = []
        return

    def reset(self):
        self.result_list = []
        return True

    def addResult(self, result_dict, camera_pose):
        result = Result()
        if not result.loadResultDict(result_dict, camera_pose):
            print("[ERROR][ROCAMerger::addResult]")
            print("\t loadResultDict failed!")
            return False
        self.result_list.append(result)
        return True

    def render3D(self):
        if len(self.result_list) == 0:
            return True

        mesh_list = []
        for result in self.result_list:
            for instance, keep in zip(result.instance_list, result.keep_list):
                if not keep:
                    continue
                mesh_list.append(instance.world_mesh)
                mesh_list.append(instance.getXYZBBox([0, 255, 0]))
                mesh_list.append(instance.getOrientedBBox([0, 0, 255]))
            if result.camera_instance is None:
                continue
            mesh_list.append(result.camera_instance.world_mesh)
        o3d.visualization.draw_geometries(mesh_list)
        return True

    def render3DWithProcess(self):
        process = Process(target=self.render3D)
        process.start()
        #  process.join()
        #  process.close()
        return True

def demo():
    roca_merger = ROCAMerger()
    return True

if __name__ == "__main__":
    demo()

