#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Data.trans import Trans
from Data.instance import Instance

from Method.nms import getKeepList

class Result(object):
    def __init__(self, instance_list=[], camera_pose=None):
        self.instance_list = instance_list
        self.camera_pose = camera_pose

        self.keep_list = []
        return

    def getInstance(self, result_dict, instance_idx):
        instances = result_dict["instances"]
        cad_ids = result_dict["cad_ids"]
        meshes = result_dict["meshes"]
        if instance_idx >= len(instances):
            print("[ERROR][Result::getInstance]")
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

    def updateInstanceWorldPose(self, camera_pose):
        for instance in self.instance_list:
            if not instance.updateWorldPose(camera_pose):
                print("[ERROR][Result::updateInstanceWorldPose]")
                print("\t updateWorldPose failed!")
                return False
        return True

    def loadResultDict(self, result_dict, camera_pose, min_dist_3d=0.4):
        self.camera_pose = camera_pose

        instances = result_dict["instances"]
        self.instance_list = [
            self.getInstance(result_dict, i) for i in range(len(instances))]

        if not self.updateInstanceWorldPose(camera_pose):
            print("[ERROR][Result::loadResultDict]")
            print("\t updateInstanceWorldPose failed!")
            return False

        self.keep_list = getKeepList(self.instance_list, min_dist_3d)
        return True

