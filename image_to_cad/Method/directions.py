#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")

import numpy as np
from math import sqrt, atan2

from roca.utils.linalg import make_M_from_tqs, decompose_mat4

from habitat_sim.utils.common import \
    quat_from_angle_axis

from habitat_sim_manage.Data.rad import Rad

from image_to_cad.Data.trans import Trans

def getTransFromMatrix(matrix):
    t, q, s = decompose_mat4(matrix)
    trans = Trans(t, q, s)
    return trans

def getMatrixFromTrans(trans):
    t = trans.translation.tolist()
    q = trans.rotation.tolist()
    s = trans.scale.tolist()
    matrix = make_M_from_tqs(t, q, s)
    return matrix

def getRotationFromRad(rad):
    up_rotate_rad, right_rotate_rad, front_rotate_rad = rad.toList()

    up_rotation = quat_from_angle_axis(up_rotate_rad, np.array([0.0, 1.0, 0.0]))
    right_rotation = quat_from_angle_axis(right_rotate_rad, np.array([1.0, 0.0, 0.0]))
    front_rotation = quat_from_angle_axis(front_rotate_rad, np.array([0.0, 0.0, -1.0]))

    rotation = up_rotation * right_rotation * front_rotation
    return rotation

def getRadFromDirection(direction):
    x, y, z = direction.toList()
    xoy_length = sqrt(x*x + y*y)

    up_rotate_rad = atan2(y, x)
    right_rotate_rad = atan2(z, xoy_length)

    rad = Rad(up_rotate_rad, right_rotate_rad)
    return rad

def getMatrixFromPose(pose):
    t = [pose.position.x, pose.position.y, pose.position.z]
    q = getRotationFromRad(pose.rad)
    s = pose.scale
    matrix = make_M_from_tqs(t, q, s)
    return matrix

def getTransFromPose(pose):
    t = [pose.position.x, pose.position.y, pose.position.z]
    q = getRotationFromRad(pose.rad)
    s = pose.scale
    trans = Trans(t, q, s)
    return trans

