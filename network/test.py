#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import open3d as o3d
from PIL import Image
from trimesh.exchange.export import export_mesh
from trimesh.util import concatenate as stack_meshes

from roca.engine import Predictor

def demo():
    data_dir = "../Data/Dataset/"
    model_path = "../Models/model_best.pth"
    config_path = "../Models/config.yaml"
    wild = False
    output_dir = 'none'

    predictor = Predictor(data_dir, model_path, config_path, wild=wild)
    to_file = output_dir != 'none'

    scene_name_dict = {
        '3m': 'scene0474_02',
        'sofa': 'scene0207_00',
        'lab': 'scene0378_02',
        'desk': 'scene0474_02',
    }

    scene_name_dict = {
        #  'desk': 'scene0378_02',
        'desk': 'scene0474_02',
    }

    for name in scene_name_dict.keys():
        scene = scene_name_dict[name]

        img = Image.open('assets/' + name + ".jpg")
        img = np.asarray(img)
        instances, cad_ids = predictor(img, scene=scene)
        print("==========")
        print(cad_ids)
        meshes = predictor.output_to_mesh(
            instances, cad_ids,
            excluded_classes={'table'} if wild else (),
            as_open3d=not to_file)

        if to_file:
            os.makedirs(output_dir, exist_ok=True)

        if predictor.can_render:
            rendering, ids = predictor.render_meshes(meshes)
            mask = ids > 0
            overlay = img.copy()
            overlay[mask] = np.clip(
                0.8 * rendering[mask] * 255 + 0.2 * overlay[mask], 0, 255
            ).astype(np.uint8)
            if to_file:
                Image.fromarray(overlay).save(output_dir + 'overlay_' + name + '.jpg')
            else:
                img = o3d.geometry.Image(overlay)
                o3d.visualization.draw_geometries([img], height=480, width=640)

        if to_file:
            out_file = output_dir + 'mesh_' + name + '.ply'
            export_mesh(stack_meshes(meshes), out_file, file_type='ply')
        else:
            o3d.visualization.draw_geometries(meshes)
    return True

if __name__ == '__main__':
    demo()

