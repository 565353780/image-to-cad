#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("./habitat_sim_manage/")

import numpy as np
from math import sqrt, atan2, cos, sin, asin

from habitat_sim.utils.common import \
    quat_from_angle_axis

from roca.utils.linalg import make_M_from_tqs, decompose_mat4

from Data.point import Point
from Data.rad import Rad
from Data.pose import Pose
from Data.trans import Trans

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

def getRadFromRotation(rotation):
    w, x, y, z = rotation

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = atan2(t0, t1)
 
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = asin(t2)
 
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = atan2(t3, t4)

    rad = Rad(yaw_z, -pitch_y, roll_x)
    return rad

def getRotationFromRad(rad):
    up_rotate_rad, right_rotate_rad, front_rotate_rad = rad.toList()

    up_rotation = quat_from_angle_axis(up_rotate_rad, np.array([0.0, 0.0, 1.0]))
    right_rotation = quat_from_angle_axis(right_rotate_rad, np.array([0.0, -1.0, 0.0]))
    front_rotation = quat_from_angle_axis(front_rotate_rad, np.array([1.0, 0.0, 0.0]))

    rotation = up_rotation * right_rotation * front_rotation
    return rotation

def getRadFromDirection(direction):
    x, y, z = direction.toList()
    xoy_length = sqrt(x*x + y*y)

    up_rotate_rad = atan2(y, x)
    right_rotate_rad = atan2(z, xoy_length)

    rad = Rad(up_rotate_rad, right_rotate_rad)
    return rad

def getPoseFromMatrix(matrix):
    t, q, s = decompose_mat4(matrix)
    position = Point(t[0], t[1], t[2])
    rad = getRadFromRotation(q)
    pose = Pose(position, rad)
    pose.scale = s
    return pose

def getMatrixFromPose(pose):
    t = [pose.position.x, pose.position.y, pose.position.z]
    q = getRotationFromRad(pose.rad)
    s = pose.scale
    matrix = make_M_from_tqs(t, q, s)
    return matrix

