#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def getKeepList(instance_list, min_dist_3d):
    keeps = [True for _ in instance_list]
    if min_dist_3d <= 0:
        return keeps
    for i in range(len(instance_list)):
        instance = instance_list[i]
        for instance_ in instance_list:
            translation_diff =  np.linalg.norm(
                instance_.trans.translation - instance.trans.translation, ord=2)
            if instance_.class_id == instance.class_id and \
                    instance_.score > instance.score and \
                    translation_diff < min_dist_3d:
                keeps[i] = False
                break
    return keeps

