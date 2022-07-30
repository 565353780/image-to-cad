#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from math import sqrt
from copy import deepcopy

from habitat_sim_manage.Data.point import Point
from habitat_sim.utils.common import quat_rotate_vector

def getDiffList(point_1, point_2):
    x_diff, y_diff, z_diff = None, None, None
    if isinstance(point_1, Point):
        x_diff = point_1.x - point_2.x
        y_diff = point_1.y - point_2.y
        z_diff = point_1.z - point_2.z
    else:
        x_diff = point_1[0] - point_2[0]
        y_diff = point_1[1] - point_2[1]
        z_diff = point_1[2] - point_2[2]
    return [x_diff, y_diff, z_diff]

def getPointDist2(point_1, point_2):
    x_diff, y_diff, z_diff = getDiffList(point_1, point_2)
    point_dist2 = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff
    return point_dist2

def getPointDist(point_1, point_2):
    point_dist2 = getPointDist2(point_1, point_2)
    point_dist = sqrt(point_dist2)
    return point_dist

def getQuaternionDist(quaternion_1, quaternion_2):
    direction_1 = quat_rotate_vector(quaternion_1, [0, 0, 1])
    direction_2 = quat_rotate_vector(quaternion_2, [0, 0, 1])
    quaternion_dist = getPointDist2(direction_1, direction_2)
    return quaternion_dist

def getTransDist(trans_1, trans_2):
    translation_dist = getPointDist2(trans_1.translation, trans_2.translation)

    quat_1 = trans_1.getQuaternion()
    quat_2 = trans_2.getQuaternion()
    rotation_dist = getQuaternionDist(quat_1, quat_2)

    scale_dist = getPointDist2(trans_1.scale, trans_2.scale)

    trans_dist = translation_dist + rotation_dist + scale_dist
    return trans_dist

def getInstanceDist(instance_1, instance_2):
    trans_dist = getTransDist(instance_1.world_trans, instance_2.world_trans)
    instance_dist = trans_dist
    return instance_dist

def getMatchListFromMatrix(matrix):
    inf = float("inf")

    matrix_copy = deepcopy(matrix)

    match_list = []
    match_value_list = []

    matrix_min_value = np.min(matrix_copy)
    while matrix_min_value != inf:
        if matrix_min_value > 1:
            break

        min_idx = np.where(matrix_copy==matrix_min_value)
        row = min_idx[0][0]
        col = min_idx[1][0]

        if len(match_value_list) > 0:
            match_value_mean = np.mean(match_value_list)

            new_match_value = matrix_copy[row][col]

            if new_match_value > match_value_mean * 10:
                break

        match_list.append([row, col])
        match_value_list.append(matrix_copy[row][col])

        matrix_copy[row, :] = float("inf")
        matrix_copy[:, col] = float("inf")

        matrix_min_value = np.min(matrix_copy)
    return match_list

