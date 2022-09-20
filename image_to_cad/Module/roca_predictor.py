#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import trimesh
import numpy as np

from renderer.scan2cad_rasterizer import Rasterizer

from image_to_cad.Config.roca.constants import CAD_TAXONOMY, COLOR_BY_CLASS
from image_to_cad.Config.roca.roca_config import roca_config

from image_to_cad.Data.roca.cad_manager import CADCatalog
from image_to_cad.Data.roca.datasets import register_scan2cad
from image_to_cad.Data.camera.intrinsics import Intrinsics

from image_to_cad.Model.roca import ROCA

from image_to_cad.Method.nms import getKeepList
from image_to_cad.Method.matrix import make_M_from_tqs

class Predictor(object):
    def __init__(self, data_dir, model_path, config_path, thresh=0.5):
        cfg = roca_config('Scan2CAD', 'Scan2CAD')
        cfg.merge_from_file(config_path)
        cfg.MODEL.INSTANCES_CONFIDENCE_THRESH = thresh

        model = ROCA(cfg)
        model.to(torch.device("cuda"))

        backup = torch.load(model_path)
        model.load_state_dict(backup['model'])

        model.eval()
        model.requires_grad_(False)

        data_name = 'Scan2CADVal'
        register_scan2cad(data_name, {}, '', data_dir, '', '', 'val')

        cad_manager = CADCatalog.get(data_name)
        points, ids = cad_manager.batched_points_and_ids(volumes=True)
        model.set_cad_models(points, ids, cad_manager.scene_alignments)
        model.embed_cads()

        self.model = model
        self.cad_manager = cad_manager

        camera_obj_file_path = "./image_to_cad/Config/roca/assets/camera.obj"
        with open(camera_obj_file_path) as f:
            cam = trimesh.load(f, file_type='obj', force='mesh')
        cam.apply_scale(0.25)
        cam.visual.face_colors = [100, 100, 100, 255]
        self._camera_mesh = cam

        self.scene_rot = np.array([
            [1, 0, 0, 0],
            [0, np.cos(np.pi), -np.sin(np.pi), 0],
            [0, np.sin(np.pi), np.cos(np.pi), 0],
            [0, 0, 0, 1]
        ])

        print('\nDone building predictor\n')
        return

    @torch.no_grad()
    def __call__(self, image_rgb, f=435., scene='scene0474_02'):
        inputs = {
            'scene': scene,
            'image': torch.as_tensor(
                np.ascontiguousarray(image_rgb[:, :, ::-1].transpose(2, 0, 1))),
            'intrinsics': Intrinsics(torch.tensor([
                [f, 0., image_rgb.shape[1] / 2],
                [0., f, image_rgb.shape[0] / 2],
                [0., 0., 1.]])),
            }

        results, _ = self.model([inputs])
        outputs = results[0]
        cad_ids = outputs['cad_ids']
        return outputs['instances'].to('cpu'), cad_ids

    def output_to_mesh(self, instances, cad_ids,
                       min_dist_3d=0.4,
                       excluded_classes=(),
                       nms_3d=True,
                       as_open3d=False):
        meshes = []
        trans_cls_scores = []
        for i in range(len(instances)):
            cad_id = cad_ids[i]
            if cad_id is None:
                continue
            if CAD_TAXONOMY[int(cad_id[0])] in excluded_classes:
                continue

            trans_cls_scores.append((
                instances.trans_pred[i],
                instances.pred_classes[i].item(),
                instances.scores[i].item(),
            ))

            mesh = self.cad_manager.model_by_id(*cad_ids[i], verbose=True)
            mesh = trimesh.Trimesh(
                vertices=mesh.verts_list()[0].numpy(),
                faces=mesh.faces_list()[0].numpy()
            )

            trs = make_M_from_tqs(
                instances.trans_pred[i].tolist(),
                instances.rot_pred[i].tolist(),
                instances.scales_pred[i].tolist()
            )
            mesh.apply_transform(self.scene_rot @ trs)

            color = COLOR_BY_CLASS[int(cad_ids[i][0])]
            if as_open3d:
                mesh = mesh.as_open3d
                mesh.paint_uniform_color(color)
                mesh.compute_vertex_normals()
            else:
                mesh.visual.face_colors = [*(255 * color), 255]
            meshes.append(mesh)

        if nms_3d:
            keeps = getKeepList(trans_cls_scores, min_dist_3d)
            meshes = [m for m, b in zip(meshes, keeps) if b]

        if as_open3d:
            cam = self._camera_mesh
            cam = cam.as_open3d
            cam.compute_vertex_normals()
            meshes.append(cam)
        else:
            meshes.append(self._camera_mesh)
        return meshes

    def render_meshes(self, meshes, f= 435.):
        inv_rot = np.linalg.inv(self.scene_rot)
        raster = Rasterizer(480, 360, f, f, 240., 180, False, True)
        colors = {}
        for i, mesh in enumerate(meshes[:-1], start=1):
            if isinstance(mesh, trimesh.Trimesh):
                mesh = mesh.copy()
                mesh.apply_transform(inv_rot)
                raster.add_model(
                    np.asarray(mesh.faces, dtype=raster.index_dtype),
                    np.asarray(mesh.vertices, dtype=raster.scalar_dtype),
                    i,
                    np.asarray(mesh.face_normals, raster.scalar_dtype)
                )
                colors[i] = np.asarray(mesh.visual.face_colors)[0][:3] / 255
            else:
                mesh = type(mesh)(mesh)
                mesh.compute_triangle_normals()
                mesh.transform(inv_rot)
                raster.add_model(
                    np.asarray(mesh.triangles, dtype=raster.index_dtype),
                    np.asarray(mesh.vertices, dtype=raster.scalar_dtype),
                    i,
                    np.asarray(mesh.triangle_normals, raster.scalar_dtype)
                )
                colors[i] = np.asarray(mesh.vertex_colors[0])[:3]

        raster.rasterize()
        raster.set_colors(colors)
        raster.render_colors(0.2)
        return raster.read_color(), raster.read_idx()

