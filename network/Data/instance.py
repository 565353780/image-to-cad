#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d

from Config.matrix import INIT_MATRIX, SCENE_ROT

from Data.trans import Trans

from Method.directions import \
    getTransFromMatrix, getMatrixFromTrans, \
    getMatrixFromPose
from Method.bboxes import getOpen3DBBox, \
    getBBoxFromOpen3DBBox, getOpen3DBBoxFromBBox

class Instance(object):
    def __init__(self,
                 class_id=-1, score=0, trans=Trans(),
                 cad_id="", mesh=None):
        self.class_id = int(class_id)
        self.score = float(score)
        self.trans = trans
        self.cad_id = cad_id
        self.mesh = mesh

        self.world_trans = None
        self.world_mesh = None
        self.world_bbox = None
        return

    def updateWorldMesh(self):
        if self.mesh is None:
            return True

        self.world_mesh = o3d.geometry.TriangleMesh(self.mesh)

        inverse_trans_matrix = self.getInverseTransMatrix()
        self.world_mesh.transform(inverse_trans_matrix)

        trans_matrix = getMatrixFromTrans(self.world_trans)
        self.world_mesh.transform(trans_matrix)
        return True

    def updateWorldTrans(self, camera_pose):
        instance_matrix = self.getTransMatrix()

        camera_matrix = getMatrixFromPose(camera_pose)
        trans_matrix = camera_matrix @ instance_matrix
        self.world_trans = getTransFromMatrix(trans_matrix)

        if not self.updateWorldMesh():
            print("[ERROR][Instance::updateWorldTrans]")
            print("\t updateWorldMesh failed!")
            return False

        if not self.updateWorldBBox():
            print("[ERROR][Instance::updateWorldTrans]")
            print("\t updateWorldBBox failed!")
            return False
        return True

    def updateWorldBBox(self):
        xyz_bbox = self.getOpen3DXYZBBox()
        self.world_bbox = getBBoxFromOpen3DBBox(xyz_bbox)
        return True

    def getTransMatrix(self):
        trans_matrix = self.trans.getTransMatrix()
        matrix = SCENE_ROT @ trans_matrix
        return matrix

    def getInverseTransMatrix(self):
        trans_matrix = self.getTransMatrix()
        inverse_trans_matrix = np.linalg.inv(trans_matrix)
        return inverse_trans_matrix

    def getWorldTransMatrix(self):
        if self.world_trans is None:
            print("[WARN][Instance::getWorldTransMatrix]")
            print("\t world_trans is None!")
            return INIT_MATRIX
        world_trans_matrix = getMatrixFromTrans(self.world_trans)
        return world_trans_matrix

    def getInverseWorldTransMatrix(self):
        world_trans_matrix = self.getWorldTransMatrix()
        inverse_world_trans_matrix = np.linalg.inv(world_trans_matrix)
        return inverse_world_trans_matrix

    def getOpen3DTransBBox(self):
        bbox = getOpen3DBBox()
        bbox.transform(self.getWorldTransMatrix())
        return bbox

    def getOpen3DXYZBBox(self, color=[255, 0, 0]):
        xyz_bbox = self.world_mesh.get_axis_aligned_bounding_box()
        xyz_bbox.color = np.array(color, dtype=np.float32) / 255.0
        return xyz_bbox

    def getOpen3DOrientedBBox(self, color=[255, 0, 0]):
        oriented_bbox = self.world_mesh.get_oriented_bounding_box()
        oriented_bbox.color = np.array(color, dtype=np.float32) / 255.0
        return oriented_bbox

    def getOpen3DBBox(self, color=[255, 0, 0]):
        open3d_bbox = getOpen3DBBoxFromBBox(self.world_bbox, color)
        return open3d_bbox

    def outputInfo(self, info_level=0):
        line_start = "\t" * info_level

        print(line_start + "[Instance]")
        print(line_start + "\t class_id =", self.class_id)
        print(line_start + "\t score =", self.score)
        print(line_start + "\t cad_id=", self.cad_id)
        self.trans.outputInfo(info_level + 1)
        return True

