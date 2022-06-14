#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy

from habitat_sim_manage.Data.point import Point

inf = float("inf")

class BBox(object):
    def __init__(self,
                 min_point=Point(inf, inf, inf),
                 max_point=Point(-inf, -inf, -inf)):
        self.min_point = min_point
        self.max_point = max_point

        self.diff_point = Point(-inf, -inf, -inf)

        self.updateDiffPoint()
        return

    def isValid(self):
        if self.min_point == inf:
            return False
        return True

    def updateDiffPoint(self):
        if not self.isValid():
            self.diff_point = Point(-inf, -inf, -inf)
            return True
        self.diff_point = Point(
            self.max_point.x - self.min_point.x,
            self.max_point.y - self.min_point.y,
            self.max_point.z - self.min_point.z)
        return True

    def addPoint(self, point):
        if not self.isValid():
            self.min_point = deepcopy(point)
            self.max_point = deepcopy(point)
            self.updateDiffPoint()
            return True

        self.min_point.x = min(self.min_point.x, point.x)
        self.min_point.y = min(self.min_point.y, point.y)
        self.min_point.z = min(self.min_point.z, point.z)
        self.max_point.x = max(self.max_point.x, point.x)
        self.max_point.y = max(self.max_point.y, point.y)
        self.max_point.z = max(self.max_point.z, point.z)
        self.updateDiffPoint()
        return True

    def addBBox(self, bbox):
        self.addPoint(bbox.min_point)
        self.addPoint(bbox.max_point)
        self.updateDiffPoint()
        return True

