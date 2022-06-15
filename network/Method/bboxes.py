#!/usr/bin/env python
# -*- coding: utf-8 -*-

import open3d as o3d

from Config.bbox import POINTS, LINES, COLORS

def getOpen3DBBox():
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(POINTS),
        lines=o3d.utility.Vector2iVector(LINES))
    line_set.colors = o3d.utility.Vector3dVector(COLORS)
    return line_set


def getBBoxDist(bbox_1, bbox_2):
    dist = 0
    return dist

