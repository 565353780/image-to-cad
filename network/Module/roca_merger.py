#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

    def getPoseListInWorld(self):
        pose_list = []

        if len(self.result_list) == 0:
            return pose_list

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
        
        for result in self.result_list:
            for instance in result.instance_list:
                trans_matrix = instance.getTransMatrix()
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

