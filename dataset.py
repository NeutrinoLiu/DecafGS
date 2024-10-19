"""
data set loading and sampling policy

expected file structure:
    - scene_name
        - cam00
            - images
                - 0.png
                - 1.png
                - ...
        - cam01
        - ...
        - cams.json
        - points.bin
    
    number of png images in each cam folder should be the same, starting from 0.

expected cams.json structure:
    {
        "cam00": {
            "intri": [ [f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1] ],
            "c2w_R": [ [r_00, r_01, r_02], [r_10, r_11, r_12], [r_20, r_21, r_22] ],
            "c2w_t": [t_x, t_y, t_z],
            "width": w,
            "height": h
        },
        "cam01": ...
    }
    intri is pixel based, not fov based
    axis direction follows colmap's convention
"""

import os 
from typing import Literal, Tuple
from plyfile import PlyData
import torch
import numpy as np
import json
from PIL import Image
import random

from interface import Camera
from examples.datasets.normalize import (
    similarity_from_cameras,
    transform_cameras,
    transform_points,
    align_principle_axes
)

def dataset_split(all:list , test_every):
    """
    split the dataset into train and test
    """
    train = []
    test = []
    for i in range(len(all)):
        if i % test_every == 0:
            test.append(all[i])
        else:
            train.append(all[i])
    return train, test

def random_init(n, extend):
    """
    randomly sample n points from the unit cubic
    """
    points = torch.rand(n, 3) * 2 - 1
    points *= extend
    colors = torch.rand(n, 3)
    return points, colors

def read_ply(ply_path):
    """
    read point cloud from .ply file
    return two tensors: points and colors
    """
    plydata = PlyData.read(ply_path)
    points = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
    colors = np.stack((np.asarray(plydata.elements[0]["red"]),
                        np.asarray(plydata.elements[0]["green"]),
                        np.asarray(plydata.elements[0]["blue"])), axis=1) / 255.
    return points, colors

class SceneReader:
    def __init__(self, cfg, cache_in_GPU=False, return_dict=False):
        self.cache_device = torch.device("cuda") \
            if torch.cuda.is_available() and cache_in_GPU \
            else torch.device("cpu")
        self.return_dict = return_dict
        self.path = cfg.data_dir

        # 1) load cameras' info
        with open(os.path.join(self.path, "cams.json"), "r") as f:
            all_cams = json.load(f)
        self.cams: list[Camera] = []
        basic_downscale = cfg.resolution
        c2w_batch_np: list[torch.Tensor] = []
        for cname, cpara in all_cams.items():
            intri = torch.tensor(cpara["intri"])
            # intri should scale with resolution
            intri[:2, :] /= basic_downscale
            cam = Camera(
                os.path.join(self.path, cname),
                intri,
                torch.tensor(cpara["c2w_R"]),
                torch.tensor(cpara["c2w_t"]),
                int(cpara["width"])  // basic_downscale,
                int(cpara["height"]) // basic_downscale,
            )
            self.cams.append(cam)
            c2w_batch_np.append(cam.c2w.clone().numpy())
        c2w_batch_np = np.stack(c2w_batch_np, axis=0)

        # 2) camera space normalization
        # codes borrowed from gsplat/examples/datasets/colmap.py
        # hence, this part is numpy based, not torch based
        T1 = similarity_from_cameras(c2w_batch_np)
        c2w_batch_np = transform_cameras(T1, c2w_batch_np)
        if cfg.init_type == "sfm":
            points, points_rgb = read_ply(os.path.join(self.path, "init.ply"))
            points = transform_points(T1, points)
            T2 = align_principle_axes(points)
            c2w_batch_np = transform_cameras(T2, c2w_batch_np)
            points = transform_points(T2, points)
        elif cfg.init_type == "random":
            pass
        else:
            raise ValueError(f"unknown init type {cfg.init_type}")
        # apply normalization
        for cam, new_c2w in zip(self.cams, c2w_batch_np):
            cam.update_c2w(new_c2w)
        # scene scale
        cam_t_batch = c2w_batch_np[:, :3, 3]
        center = np.mean(cam_t_batch, axis=0)
        dist = np.linalg.norm(cam_t_batch - center, axis=1)
        self.scene_scale = np.max(dist)
        if cfg.init_type == "random":
            points, points_rgb = random_init(cfg.init_random_num,
                self.scene_scale * cfg.init_random_extend)
        assert points is not None, "points has not been initialized !"

        # 3) dataset's meta data
        self.init_pts = points
        self.init_pts_rgb = points_rgb
        self.scene_name = os.path.basename(self.path)
        self.cam_num = len(self.cams)
        self.frame_total = cfg.frame_total

        print(f'''
scene <{self.scene_name}> loaded:
cameras: {self.cam_num}
frames: {self.frame_total}
scene scale: {self.scene_scale}
init points: {len(points)}
init type: {cfg.init_type}
''')

        # 4) image cache
        self.cached = {}
        self.max_cache = cfg.max_cached_img

    def _get(self, 
            cam: int, 
            frame: int,
            downscale): # -> Camera, GT image
        """
        storage blind loader, not expected to be used directly
        cam attr and image are loaded separately
        since image is lazy loaded
        """
        cam_obj = self.cams[cam][frame].thumbnail(downscale)
        img = self.cached.get(
            (cam, frame, downscale), None)
        if img is None: # cache miss
            img_path = os.path.join(self.cams[cam].path, "images", f"{frame}.png")
            with Image.open(img_path) as f:
                img = f.resize((cam_obj.width, 
                          cam_obj.height))
                img = np.asarray(img)
                img = torch.tensor(img).permute(2, 0, 1) / 255. # [3, H, W]
            img = img.to(self.cache_device)
            self.cached[(cam, frame, downscale)] = img
        return cam_obj, img
    
    def batch_get(self, triplets: list):
        """
        get a batch of images
        storage awared loader
        image format: [3, H, W]
        """
        assert isinstance(triplets, list), "batch_get expects a list of triplets"
        if len(triplets) > self.max_cache:
            raise ValueError(f"wanted {len(triplets)} images exceeds max {self.max_cache} images in cache")
        ready = set(self.cached.keys())
        want = set(triplets)
        more = want - ready
        uneed = ready - want
        if len(more) + len(ready) > self.max_cache:
            # clear cache
            for tri in uneed:
                del self.cached[tri]
        ret = []
        for tri in want:
            ret.append(self._get(*tri))
        
        # a formatter for gsplat compatibility
        if self.return_dict:
            ret = self.cam_formatter(ret)
        return ret
    
    @staticmethod
    def cam_formatter(get_res):
        """
        format the get_res to a dict
        """
        ret = {
            "camtoworld": torch.stack([cam.c2w for cam, _ in get_res], dim=0),
            "K": torch.stack([cam.intri for cam, _ in get_res], dim=0),
            "image": torch.stack([img.permute(1, 2, 0) for _, img in get_res], dim=0)
        }
        return ret


class CamSampler:
    def __init__(self, 
                 scene: SceneReader,
                 # range
                 cams_idx, 
                 frames_idx,
                 further_downscale=1,
                 # options
                 batch_size = 1, 
                 policy: Literal["random", "sequential"] = "sequential"
                 ):
        self.scene = scene
        self.policy = policy
        self.cams = cams_idx.copy()
        self.frames = frames_idx.copy()
        self.batch_size = batch_size
        self.further_downscale = further_downscale
        # runtime
        self._pool = None
        print(f"data sampler pool size: {len(self.cams) * len(self.frames)}")
    def __len__(self):
        return - (len(self.cams) * len(self.frames) // -self.batch_size)
    def __iter__(self):
        # reset the iterator
        self._pool = sorted([(c, f, self.further_downscale) \
                             for c in self.cams for f in self.frames], key=lambda x: (x[0], x[1]))
        return self
    def __next__(self):
        # return a batch of cams
        if len(self._pool) == 0:
            raise StopIteration
        if self.policy == "random":
            random.shuffle(self._pool)
        batch = self._pool[:self.batch_size]
        self._pool = self._pool[self.batch_size:]
        return self.scene.batch_get(batch)