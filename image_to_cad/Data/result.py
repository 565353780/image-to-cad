#!/usr/bin/env python
# -*- coding: utf-8 -*-

from image_to_cad.Data.trans import Trans
from image_to_cad.Data.instance import Instance

from image_to_cad.Method.nms import getKeepList

class Result(object):
    def __init__(self, instance_list=[], camera_pose=None):
        self.instance_list = instance_list
        self.camera_pose = camera_pose

        self.keep_list = []
        self.keep_instance_idx_list = []
        self.camera_instance = None
        return

    def getInstanceFromDict(self, result_dict, instance_idx):
        instances = result_dict["instances"]
        cad_ids = result_dict["cad_ids"]
        meshes = result_dict["meshes"]
        if instance_idx >= len(instances):
            print("[ERROR][Result::getInstanceFromDict]")
            print("\t instance_idx out of range!")
            return None

        instance = Instance(
            instances.pred_classes[instance_idx],
            instances.scores[instance_idx],
            Trans(
                instances.trans_pred[instance_idx],
                instances.rot_pred[instance_idx],
                instances.scales_pred[instance_idx]),
            cad_ids[instance_idx],
            meshes[instance_idx])
        return instance

    def updateCameraInstance(self, result_dict):
        instances = result_dict["instances"]
        meshes = result_dict["meshes"]
        if len(instances) == len(meshes):
            return True

        self.camera_instance = Instance(mesh=meshes[-1])

        if not self.camera_instance.updateWorldTrans(self.camera_pose):
            print("[ERROR][Result::updateCameraInstance]")
            print("\t updateWorldTrans failed!")
            return False
        return True

    def updateInstanceWorldPose(self):
        for instance in self.instance_list:
            if not instance.updateWorldTrans(self.camera_pose):
                print("[ERROR][Result::updateInstanceWorldPose]")
                print("\t updateWorldTrans failed!")
                return False
        return True

    def loadResultDict(self, result_dict, camera_pose, min_dist_3d=0.4):
        self.camera_pose = camera_pose

        instances = result_dict["instances"]
        self.instance_list = [
            self.getInstanceFromDict(result_dict, i) for i in range(len(instances))]

        if not self.updateCameraInstance(result_dict):
            print("[ERROR][Result::loadResultDict]")
            print("\t updateCameraInstance failed!")
            return False

        if not self.updateInstanceWorldPose():
            print("[ERROR][Result::loadResultDict]")
            print("\t updateInstanceWorldPose failed!")
            return False

        self.keep_list = getKeepList(self.instance_list, min_dist_3d)

        for i in range(len(self.keep_list)):
            if not self.keep_list[i]:
                continue
            self.keep_instance_idx_list.append(i)
        return True

    def getKeepInstanceNum(self):
        keep_true_num = len(self.keep_instance_idx_list)
        return keep_true_num

