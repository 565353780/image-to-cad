#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("./habitat_sim_manage/")

from Data.trans import Trans
from Data.instance import Instance

class ROCAMerger(object):
    def __init__(self):
        self.instance_list_list = []
        return

    def reset(self):
        self.instance_list_list = []
        return True

    def getInstance(self, instances, instance_idx):
        if instance_idx >= len(instances):
            print("[ERROR][ROCAMerger::getInstance]")
            print("\t instance_idx out of range!")
            return None

        instance = Instance(
            instances.pred_classes[instance_idx],
            instances.scores[instance_idx],
            Trans(
                instances.pred_translations[instance_idx],
                instances.pred_rotations[instance_idx],
                instances.pred_scales[instance_idx]
            )
        )
        return instance

    def getInstanceList(self, instances):
        instance_list = [
            self.getInstance(instances, i) for i in range(len(instances))
        ]
        return instance_list

    def addResult(self, result):
        instances = result["instances"]
        instance_list = self.getInstanceList(instances)
        self.instance_list_list.append(instance_list)
        for instance in instance_list:
            instance.outputInfo()
        return True

    def getPoseListInWorld(self):
        if len(self.instance_list_list) == 0:
            return True
        
        for instance_list in self.instance_list_list:
            for instance in instance_list:
                trans_matrix = instance.getTransMatrix()
                print(trans_matrix)
        return True

def demo():
    roca_merger = ROCAMerger()
    return True

if __name__ == "__main__":
    demo()

