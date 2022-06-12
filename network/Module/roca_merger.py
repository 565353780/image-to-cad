#!/usr/bin/env python
# -*- coding: utf-8 -*-

class ROCAMerger(object):
    def __init__(self):
        self.result_list = []
        return

    def reset(self):
        self.result_list = []
        return True

    def addResult(self, result):
        self.result_list.append(result)
        return True

    def getPositionInWorld(self):
        if len(self.result_list) == 0:
            return True
        for result in self.result_list:
            image = result["image"]
            scene_name = result["scene_name"]
            instances = result["instances"]
            cad_ids = result["cad_ids"]
            meshes = result["meshes"]
            masked_image = result["masked_image"]
        return True

def demo():
    roca_merger = ROCAMerger()
    return True

if __name__ == "__main__":
    demo()

