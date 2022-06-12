#!/usr/bin/env python
# -*- coding: utf-8 -*-

from roca.utils.alignment_errors import translation_diff

def getKeepList(instance_list, min_dist_3d):
    keeps = [True for _ in instance_list]
    if min_dist_3d <= 0:
        return keeps
    for i in range(len(instance_list)):
        instance = instance_list[i]
        for instance_ in instance_list:
            if instance_.class_id == instance.class_id and \
                    instance_.score > instance.score and \
                    translation_diff(
                        instance_.trans.translation,
                        instance.trans.translation) < min_dist_3d:
                keeps[i] = False
                break
    return keeps

