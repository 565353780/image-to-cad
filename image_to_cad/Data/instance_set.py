#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d

from image_to_cad.Data.trans import Trans

from image_to_cad.Method.directions import getMatrixFromTrans
from image_to_cad.Method.dists import getInstanceDist

class InstanceSet(object):
    def __init__(self, instance_list=[]):
        self.instance_list = instance_list
        return

    def getInstanceDist(self, new_instance):
        if len(self.instance_list) == 0:
            return -1
        min_instance_dist = float("inf")
        for instance in self.instance_list:
            current_instance_dist = getInstanceDist(new_instance, instance)
            if current_instance_dist >= min_instance_dist:
                continue
            min_instance_dist = current_instance_dist
        return min_instance_dist

    def getInstanceDistList(self, instance_list, keep_list=None):
        if keep_list is None:
            instance_dist_list = [
                self.getInstanceDist(instance) for instance in instance_list]
            return instance_dist_list

        instance_dist_list = [
            self.getInstanceDist(instance) for
            instance, keep in zip(instance_list, keep_list) if keep]
        return instance_dist_list

    def getMeanMesh(self):
        if len(self.instance_list) == 0:
            return None

        if len(self.instance_list) == 1:
            return self.instance_list[0].world_mesh

        mean_mesh = o3d.geometry.TriangleMesh(self.instance_list[0].mesh)
        mean_mesh.transform(self.instance_list[0].getInverseTransMatrix())

        mean_translation = np.array([0.0, 0.0, 0.0])
        mean_rotation = np.array([0.0, 0.0, 0.0, 0.0])
        mean_scale = np.array([0.0, 0.0, 0.0])
        for instance in self.instance_list:
            mean_translation += instance.world_trans.translation
            mean_rotation += instance.world_trans.rotation
            mean_scale += instance.world_trans.scale
        mean_translation /= len(self.instance_list)
        mean_rotation /= len(self.instance_list)
        mean_scale /= len(self.instance_list)

        mean_trans = Trans(mean_translation, mean_rotation, mean_scale)

        mean_trans.outputInfo()

        mean_mesh.transform(getMatrixFromTrans(mean_trans))
        return mean_mesh

