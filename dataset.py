"""
data set loading and sampling policy

expected file structure:
    - scene_name
        - cam00
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
import cv2
import random
from torch.utils.data import DataLoader, Dataset

from interface import Camera
from helper import timeit
from datasampler import *

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
    def __init__(self, cfg, cache=None):
        if cache == "gpu":
            self.cache_device = torch.device("cuda")
        elif cache == "cpu":
            self.cache_device = torch.device("cpu")
        else:
            print("no cache device set, cache disabled")
            self.cache_device = None
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

        # 4) optional image cache
        self.cached = {}
        self.max_cache = cfg.max_cached_img

    def get_cam(self, triad) -> Camera:
        """
        get the camera object from the triad
        """
        assert len(triad) == 3, "get_cam expects a triad"
        assert triad[0] < self.cam_num, f"cam index {triad[0]} out of range"
        assert triad[1] < self.frame_total, f"frame index {triad[1]} out of range"
        return self.cams[triad[0]][triad[1]].thumbnail(triad[2])

    def get(self, cam, frame, downscale):
        """
        load the image from disk,
        for torch dataloader
        """
        cam_obj = self.get_cam((cam, frame, downscale))
        return self._get(cam_obj)

    def _get(self, cam_obj):
        """
        load the image from disk using OpenCV
        """
        img_path = os.path.join(cam_obj.path, f"{cam_obj.frame}.png")
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"failed to load {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (cam_obj.width, cam_obj.height))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

    # _cached_get and cached_get_batch are self-designed interface to
    # load images from disk and cache them, yet using torch dataloader should 
    # be more convenient
    def _cached_get(self, 
            cam: int, 
            frame: int,
            downscale): # -> Camera, GT image
        """
        storage blind loader, not expected to be used directly
        cam attr and image are loaded separately
        since image is lazy loaded
        """
        assert self.cache_device is not None, "cache_device is not set"
        cam_obj = self.get_cam((cam, frame, downscale))
        img = self.cached.get(
            (cam, frame, downscale), None)
        if img is None: # cache miss
            img = self._get(cam_obj=cam_obj)
            img = img.to(self.cache_device)
            self.cached[(cam, frame, downscale)] = img
        return img
    
    def cached_get_batch(self, triplets: list):
        """
        get a batch of images
        storage awared loader
        image format: [3, H, W]
        for our own loader only
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
            for tri in list(uneed)[:len(more)]:
                del self.cached[tri]
        ret_x = []
        ret_y = []
        for tri in want:
            ret_x.append(tri)
            ret_y.append(self._cached_get(*tri))
        
        # a formatter for gsplat compatibility
        return ret_x, ret_y
    
class BatchLoader:
    def __init__(self, 
                 getter,
                 # range
                 cams_idx, 
                 frames_idx,
                 further_downscale=1,
                 # options
                 batch_size = 1, 
                 policy: Literal["random", "sequential"] = "sequential"
                 ):
        self.getter = getter
        self.policy = policy
        self.cams = cams_idx.copy()
        self.frames = frames_idx.copy()
        self.batch_size = batch_size
        self.further_downscale = further_downscale
        # runtime
        self._pool = None
    def __len__(self):
        return - (len(self.cams) * len(self.frames) // -self.batch_size)
    def __iter__(self):
        # reset the iterator
        self._pool = sorted([(c, f, self.further_downscale) \
                             for c in self.cams for f in self.frames], key=lambda x: (x[0], x[1]))
        if self.policy == "random":
            random.shuffle(self._pool)
        elif self.policy == "sequential":
            pass
        else:
            raise ValueError(f"unknown policy {self.policy}")
        return self
    def __next__(self):
        # return a batch of cams
        if len(self._pool) == 0:
            raise StopIteration
        batch = self._pool[:self.batch_size]
        self._pool = self._pool[self.batch_size:]
        return self.getter(batch)

class DataIter(Dataset):
    def __init__(self,
                 getter,
                 sample_sequence: list,):
        # sample_sequence: [(cam, frame, downscale), ...]
        self.getter = getter
        self.sequence = sample_sequence
    def __getitem__(self, idx):
        cam = torch.tensor(self.sequence[idx], dtype=torch.int32)
        gt = self.getter(*self.sequence[idx])
        return cam, gt
    def __len__(self):
        return len(self.sequence)

class DataManager:
    def __init__(self, 
                 # context
                 scene: SceneReader,
                 # range
                 cams_idx, 
                 frames_idx,
                 further_downscale=1,
                 # options
                 batch_size = 1,
                 policy: Literal["random", 
                                 "sequential",
                                 "max_parallex"] = "sequential",
                 num_workers = 4,
                 use_torch_loader = False,
                 img_proc_fn = None,
                 mini_cache = False,
                 ):
        
        self.scene = scene
        self.policy = policy
        self.num_workers = num_workers
        self.use_torch_loader = use_torch_loader
        self.img_proc_fn = img_proc_fn

        self.cams = cams_idx.copy()
        self.frames = frames_idx.copy()
        self.downscale_base = further_downscale
        self.batch_size_base = batch_size

        self.cached_loader = None
        self.cached_cam_dist = None
        self.cached_img = {} if mini_cache else None
    
    @property
    def cam_dist(self):
        if self.cached_cam_dist is None:
            self.cached_cam_dist = np.zeros((len(self.cams), len(self.cams)))
            for i in range(len(self.cams)):
                for j in range(len(self.cams)):
                    t1 = self.scene.cams[i].c2w_t
                    t2 = self.scene.cams[j].c2w_t
                    self.cached_cam_dist[i, j] = np.linalg.norm(t1 - t2)
        return self.cached_cam_dist
    
    # -------------------------- for our own loader only ------------------------- #
    # our own loader use [cached_get_batch] to load images
    def _get_batch_loader(self):
        if self.cached_loader is None:
            self.cached_loader = BatchLoader(
                self.scene.cached_get_batch,
                self.cams,
                self.frames,
                self.downscale_base,
                self.batch_size_base,
                self.policy
            )
        return self.cached_loader

    # ------------------------- for torch dataloader only ------------------------ #
    # torch dataloader need an iterator to fetch data
    # in this case, simple [get] is used to fetch single image
    def _get_torch_iter(self, policy, info, step, downscale, batch_size, sample_nums):
        frame_range = self.frames
        if isinstance(policy, list):
            p = random.choice(policy)
            return self._get_torch_iter(p, info, step, downscale, batch_size)
        if policy == "random":
            frams = uniform_frame_sampler(frame_range, sample_nums)
            cams = uniform_camera_sampler(self.cams, sample_nums)
            seq = [(c, f, downscale) for c,f in zip(cams, frams)]
        elif policy == "sequential":
            seq = sequential_sampler(
                    self.cams,
                    frame_range,
                    downscale)
        elif policy == "max_parallex":
            frames = uniform_frame_sampler(frame_range, sample_nums)
            cams = max_parallex_sampler(self.cams, sample_nums,
                    self.cam_dist,
                    batch_size)
            seq = [(c, f, downscale) for c,f in zip(cams, frames)]
        elif policy == "screening":
            cams = max_parallex_sampler(self.cams, sample_nums,
                    self.cam_dist,
                    batch_size)
            # copy frames for twice
            frames = np.array(frame_range.copy()).repeat(batch_size)
            seq = [(c, f, downscale) for c,f in zip(cams, frames)]
        else:
            raise ValueError(f"unknown policy {policy}")
        
        if self.cached_img is not None:
            raw_getter = self.cached_get
        else:
            raw_getter = self.scene.get

        if self.img_proc_fn is not None:
            getter = lambda c,f,s: self.img_proc_fn(raw_getter(c, f, s))
        else:
            getter = raw_getter

        return DataIter(getter, seq)
    
    def cached_get(self, cam, frame, downscale):
        """
        load the image from disk,
        for torch dataloader
        """
        if self.cached_img is not None:
            img = self.cached_img.get(
                (cam, frame, downscale), None)
            if img is None:
                img = self.scene.get(cam, frame, downscale)
                self.cached_img[(cam, frame, downscale)] = img
            return img
        else:
            return self.scene.get(cam, frame, downscale)
    
    # ------------------- unified api for training and testing ------------------- #
    def gen_loader(self, info, step):
        # we rebuild the loader every time after exhausted
        # because we want to adjust the sampling order according to (info, step)
        min_iters = max(len(self.frames), 100)
        if self.use_torch_loader:
            if isinstance(self.policy, str):
                sample_nums = min_iters * self.batch_size_base
                data_iter = self._get_torch_iter(
                    self.policy, 
                    info,
                    step,
                    self.downscale_base,
                    self.batch_size_base,
                    sample_nums)
                batch_size = self.batch_size_base
            else:
                data_iter = None
                for s, p in self.policy.items():
                    if step < s:
                        if p.startswith("downsample"):
                            ratio = int(p.split("_")[1])
                            batch_size = self.batch_size_base * ratio
                            downscale = self.downscale_base * ratio
                            sample_nums = min_iters * batch_size
                            data_iter = self._get_torch_iter(
                                ["max_parallex"],
                                info,
                                step,
                                downscale,
                                batch_size,
                                sample_nums)
                        else:
                            sample_nums = min_iters * self.batch_size_base
                            data_iter = self._get_torch_iter(
                                p,
                                info,
                                step,
                                self.downscale_base,
                                self.batch_size_base)
                            batch_size = self.batch_size_base
                        break
                assert data_iter is not None, f"no specified policy for step {step}"

            loader = DataLoader(
                        data_iter,
                        batch_size=batch_size,
                        shuffle=False,                  # follow our own sampling order
                        num_workers=self.num_workers,
                        pin_memory=True,
                    )
        else:
            loader = self._get_batch_loader()
        return loader