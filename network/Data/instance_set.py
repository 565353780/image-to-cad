#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Method.dists import getInstanceDist

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

