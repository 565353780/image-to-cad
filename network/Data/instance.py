#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d

from Config.matrix import SCENE_ROT, SCENE_ROT_INV

from Data.trans import Trans

from Method.directions import \
    getPoseFromTrans, getPoseMul, getMatrixFromPose

class Instance(object):
    def __init__(self,
                 class_id=-1, score=0, trans=Trans(),
                 cad_id="", mesh=None):
        self.class_id = int(class_id)
        self.score = float(score)
        self.trans = trans
        self.cad_id = cad_id
        self.mesh = mesh

        self.world_pose = None
        self.world_mesh = None
        return

    def updateWorldMesh(self):
        if self.mesh is None:
            return True

        self.world_mesh = o3d.geometry.TriangleMesh(self.mesh)

        trans_matrix = self.trans.getTransMatrix()
        self.world_mesh.transform(np.linalg.inv(trans_matrix))

        trans_matrix = getMatrixFromPose(self.world_pose)
        self.world_mesh.transform(trans_matrix)
        return True

    def updateWorldTrans(self, camera_pose):
        instance_pose = getPoseFromTrans(self.trans)
        self.world_pose = getPoseMul(camera_pose, instance_pose)

        if not self.updateWorldMesh():
            print("[ERROR][Instance::updateWorldTrans]")
            print("\t updateWorldMesh failed!")
            return False
        return True

    def getTransMatrix(self):
        trans_matrix = self.trans.getTransMatrix()
        matrix = SCENE_ROT_INV @ trans_matrix @ SCENE_ROT
        return matrix

    def outputInfo(self, info_level=0):
        line_start = "\t" * info_level

        print(line_start + "[Instance]")
        print(line_start + "\t class_id =", self.class_id)
        print(line_start + "\t score =", self.score)
        print(line_start + "\t cad_id=", self.cad_id)
        self.trans.outputInfo(info_level + 1)
        return True

