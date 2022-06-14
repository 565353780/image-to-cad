#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d

POINTS = np.array([
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
])

LINES = np.array([
    [0, 1], [0, 2],
    [1, 3], [2, 3],
    [4, 5], [4, 6],
    [5, 7], [6, 7],
    [0, 4], [1, 5],
    [2, 6], [3, 7]])

COLORS = np.array([
    [1, 0, 0] for _ in LINES])

def getBBox():
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(POINTS),
        lines=o3d.utility.Vector2iVector(LINES))
    line_set.colors = o3d.utility.Vector3dVector(COLORS)
    return line_set

