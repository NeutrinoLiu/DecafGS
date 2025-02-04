from dataclasses import dataclass
import json
import time

from memory_profiler import profile

import os
import math
from tqdm import tqdm
from functools import reduce
import numpy as np
import hydra
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.utils.tensorboard import SummaryWriter

import random

import nerfview
import viser

from fused_ssim import fused_ssim
from gsplat.rendering import rasterization
from examples.utils import set_random_seed, rgb_to_sh

from strategy import DecafMCMCStrategy
from dataset import SceneReader, DataManager, dataset_split
from interface import Gaussians, Camera

from helper import (LogDirMgr, 
                    save_tensor_images_threaded,
                    save_video,
                    calculate_grads,
                    calculate_blames,
                    calculate_perturb,
                    opacity_analysis,
                    standardize,
                    update_state,
                    gaussian_blur_diff,
                    )
from helper_viewer import ViewerMgr
from helper_routine import RoutineMgrIncremental, RoutineMgrNull, RoutineMgrFence, RoutineMgrFenceSimple

class Runner:
    def __init__(self, cfg, chkpt=None):

        data_cfg = cfg.data
        model_cfg = cfg.model
        train_cfg = cfg.train

        self.cfg = cfg
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        if chkpt is not None:
            print(f"▶ checkpoint saved at step {chkpt['step']}, will resume from step {chkpt['step']+1}")
            self.init_step = chkpt['step'] + 1
        else:
            self.init_step = 0

        # --------------------------------- setup log -------------------------------- #
        self.log = LogDirMgr(self.train_cfg.root)
        with open(self.log.config, 'w') as f:
            f.write(json.dumps({
                "data": OmegaConf.to_container(data_cfg),
                "model": OmegaConf.to_container(model_cfg),
                "train": OmegaConf.to_container(train_cfg)
            }, indent=" "))

        # ------------------------------- data loading ------------------------------- #
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        assert data_cfg.loader in ["torch", "plain"], \
            "only support torch or plain"
        loader_cache = data_cfg.loader == "plain"
        use_torch_loader = data_cfg.loader == "torch"
        self.scene = SceneReader(data_cfg, loader_cache)

        train_cam_idx, test_cam_idx = dataset_split(
            list(range(self.scene.cam_num)),
            train_cfg.test_dataset_split)
        self.min_frame = model_cfg.frame_start
        self.max_frame = min(self.scene.frame_total, model_cfg.frame_end)
        self.all_frames = list(range(self.min_frame, self.max_frame))

        self.train_loader_gen = DataManager(
            self.scene,                         # img reader
            train_cam_idx,                      # train cam only
            self.all_frames,                        # full frame
            batch_size = train_cfg.batch_size,
            policy = "max_parallex",
            num_workers=data_cfg.num_workers,
            use_torch_loader=use_torch_loader,
            )
        self.test_loader_gen = DataManager(
            self.scene,                         # img reader
            test_cam_idx,                       # test cam only
            self.all_frames,                        # full frame
            batch_size = 1,
            policy = "sequential",
            num_workers=data_cfg.num_workers,
            use_torch_loader=use_torch_loader,
            )
        self.train_loader = iter([])
        self.test_loader = iter([])
        
        
        print(f"totally {len(train_cam_idx)}+{len(test_cam_idx)} cams")
        print(f"training frame [{self.min_frame} ~ {self.max_frame})\n")

        # ----------------------------- model & opt init ----------------------------- #

        N = self.scene.init_pts.shape[0]
        frame_length = self.max_frame - self.min_frame
        init_pts_timestamp = ((self.scene.init_pts_frame - self.min_frame) / frame_length) \
            if self.scene.init_pts_frame is not None else (0.5 * np.ones(N))
        # print(f"init_pts_frame: {np.unique(self.scene.init_pts_frame).tolist()}")
        # print(f"min_frame: {self.min_frame}, max_frame: {self.max_frame}")
        # print(f"frame_offset: {np.unique(self.scene.init_pts_frame - self.min_frame).tolist()}")
        # print(f"init_pts_timestamp: {np.unique(init_pts_timestamp).tolist()}")
        
        if train_cfg.routine == "fence" and train_cfg.routine_fence_std_grow:
            # incremental setup for the opacity span std, init span size is 2 (std=1 frame)
            init_pts_timespan = (2 / frame_length) * np.ones(N)
        else:
            init_pts_timespan = (self.scene.init_pts_frame_span / frame_length) \
                if self.scene.init_pts_frame_span is not None else (1.0 * np.ones(N))

        from pipeline import DecafPipeline
        if model_cfg.resfield:
            self.model = DecafPipeline(
                train_cfg=train_cfg,
                model_cfg=model_cfg,
                init_pts=torch.Tensor(self.scene.init_pts).float(),
                init_pts_time=torch.Tensor(init_pts_timestamp).float(),
                init_pts_time_span=torch.Tensor(init_pts_timespan).float(),
                device=self.device,
                T_max=300,
                chkpt=chkpt["model"] if chkpt is not None else None
            )
        else:
            self.model = DecafPipeline(
                train_cfg=train_cfg,
                model_cfg=model_cfg,
                init_pts=torch.Tensor(self.scene.init_pts).float(),
                init_pts_time=torch.Tensor(init_pts_timestamp).float(),
                init_pts_time_span=torch.Tensor(init_pts_timespan).float(),
                device=self.device,
                chkpt=chkpt["model"] if chkpt is not None else None
            )

        # ------------------------- other training schedulers ------------------------ #

        if self.train_cfg.routine == "incremental" or self.train_cfg.routine == "fence" or self.train_cfg.routine == "fence_simple":
            assert self.model_cfg.deform_delta_decoupled, "train.deform_delta_decoupled must be True"
            if self.train_cfg.routine == "incremental":
                self.routine_mgr = RoutineMgrIncremental(
                    first_frame_iters=self.train_cfg.routine_first_frame_iters,
                    stage_1_iters=self.train_cfg.routine_stage_1_iters,
                    stage_2_iters=self.train_cfg.routine_stage_2_iters,
                    runner=self,
                    chkpt=chkpt["routine"] if chkpt is not None else None
                )
            elif self.train_cfg.routine == "fence":
                unique_frames = np.unique(self.scene.init_pts_frame).astype(int).tolist()
                unique_frames = [f for f in unique_frames if f >= self.min_frame and f < self.max_frame]
                self.routine_mgr = RoutineMgrFence(
                    first_frame_iters=self.train_cfg.routine_first_frame_iters,
                    stage_1_iters=self.train_cfg.routine_stage_1_iters,
                    stage_2_iters=self.train_cfg.routine_stage_2_iters,
                    init_frames=unique_frames,
                    runner=self,
                    std_grow=self.train_cfg.routine_fence_std_grow,
                    chkpt=chkpt["routine"] if chkpt is not None else None
                )
            elif self.train_cfg.routine == "fence_simple":
                unique_frames = np.unique(self.scene.init_pts_frame).astype(int).tolist()
                unique_frames = [f for f in unique_frames if f >= self.min_frame and f < self.max_frame]
                self.routine_mgr = RoutineMgrFenceSimple(
                    first_frame_iters=self.train_cfg.routine_first_frame_iters,
                    init_frames=unique_frames,
                    runner=self,
                    chkpt=chkpt["routine"] if chkpt is not None else None
                )
        else:
            self.routine_mgr = RoutineMgrNull()
            
        # ------------------------------- other plugin ------------------------------- #
        self.strategy = DecafMCMCStrategy(
            train_cfg=train_cfg,
            max_cap=model_cfg.anchor_num,
            naive_decay=model_cfg.naive_decay,)
        
        self.state = {}         # running state, not only for strategy
        self.state.update(
            self.strategy.initialize_state())

        # ----------------------------------- eval ----------------------------------- #
        self.eval_funcs = {
            "psnr": PeakSignalNoiseRatio(data_range=1.0).to(self.device),
            "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device),
            "msssim": MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(self.device),
            "lpips": LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True).to(self.device),
        }

        # --------------------------------- visualize -------------------------------- #
        self.viewer_mgr = ViewerMgr(self, self.min_frame, self.max_frame)
        self.writer = SummaryWriter(log_dir=self.log.tb)

    def img_proc_callback(self, img):
        if self.train_cfg.blur_for_motion_learning and self.routine_mgr.means_opt_only():
            # only blur for motion learning
            init_r = self.train_cfg.blur_radius
            final_r = 1
            final_step = self.train_cfg.blur_steps
            cur_step = self.routine_mgr.means_opt_only_lasts()
            if cur_step > final_step:
                return img
            # expotentially decay 
            r = init_r * (final_r / init_r) ** (cur_step / final_step)
            return gaussian_blur_diff(img, r)
        else:
            return img

    def single_render( self,
                pc: Gaussians,
                cam: Camera,
                permute: bool = True,
                perturb_intensity: float = 0.0,
                prior_filter_idx = None,
                **kwargs): # sh_degree is expected to be provided

        # gsplat rasterization requires batched cams
        viewmats_batch = torch.linalg.inv(cam.c2w)[None].to(pc.device)
        Ks_batch = cam.intri[None].to(pc.device)
        width = cam.width
        height = cam.height

        if self.model_cfg.filter_by_ops:
            assert prior_filter_idx is None, "filter_by_ops and prior_filter_mask cannot be both True"
            pc, filter_idx = Gaussians.filter_by_ops(pc, self.train_cfg.reloc_dead_thres)
            # idx[i] = j means, the i-th filtered gaussian is the j-th gaussian in the original pc
        elif prior_filter_idx is not None:
            filter_idx = prior_filter_idx
        else:
            filter_idx = torch.arange(pc.means.shape[0], device=pc.device)

        means = pc.means
        if perturb_intensity > 0:
            noise_per_gs = calculate_perturb(pc, perturb_intensity)
            means += torch.randn_like(pc.means) * noise_per_gs

        img, alpha, info = rasterization(
            # gaussian attrs
            means       =   means,
            quats       =   pc.quats,
            scales      =   pc.scales,
            opacities   =   pc.opacities,
            colors      =   pc.colors,
            # batched cams
            viewmats    =   viewmats_batch,
            Ks          =   Ks_batch,
            width       =   width,
            height      =   height,
            # options
            sh_degree   =   0,
            packed      =   True,                            # to save memory
            absgrad     =   True,                           # return abs grad
            **kwargs)
        
        # output of rasterization is (N, H, W, 3), 
        # we need to permute to (N, 3, H, W) for loss calculation
        if permute:
            img = img.permute(0, 3, 1, 2).clamp(0, 1.)
        
        info["filter_idx"] = filter_idx

        return img[0], alpha[0], info

    def setup_loader(self, frames):
        self.train_loader_gen.frames = frames
        self.test_loader_gen.frames = frames
        self.train_loader = iter([])
        self.test_loader = iter([])

    # @profile(stream=open('memory_profiler.log','w+'))
    def train(self):
        init_step = self.init_step
        max_step = self.train_cfg.max_step
        pbar = tqdm(range(init_step, max_step+1), initial=init_step, total=max_step, desc="training")

        self.train_loader = iter([])
        for step in pbar:
            self.step = step
            tic = self.viewer_mgr.checkin()
            self.routine_mgr.checkin(step)

            # ------------------------- read train data in batch ------------------------- #
            try:
                data = next(self.train_loader)
            except StopIteration:
                self.train_loader = iter(
                    self.train_loader_gen.gen_loader(self.state, step))
                data = next(self.train_loader)
            
            # ------------------------------ forward pass ------------------------------ #
            # for batched, yet cam might lead to different gaussian pc
            # we simple render each image separately
            # no need to combine tensors within the batch, which will lead to costly memory operations (low fps)

            metrics = {
                "l1loss": 0.0,
                "ssimloss": 0.0,
                "l1opacity": 0.0,
                "l1scale": 0.0,
                "volume": 0.0,
                "offset": 0.0
            }
            weights = {
                "l1loss": self.train_cfg.ssim_lambda,
                "ssimloss": 1 - self.train_cfg.ssim_lambda,
                "l1opacity": self.train_cfg.reg_opacity,
                "l1scale": self.train_cfg.reg_scale,
                "volume": self.train_cfg.reg_volume,
                "offset": self.train_cfg.reg_offset
            }

            # running state per batch,
            # self.state is a longer term state over the whole dataset
            key_for_gradient = "means2d"
            batch_state = {
                "ops": [],
                "info": [],
                "img": [],
                "gt": [],
                "frames": [],
            }
            K = self.model.deform.anchor_params['anchor_offsets'].shape[1]
            N = self.model.deform.anchor_params['anchor_offsets'].shape[0]

            # check whether data loader is batched
            cams = [self.scene.get_cam(cam).to(self.device, non_blocking=True) \
                    for cam in data[0]]
            if isinstance(data[1], list):
                gts = [gt.to(self.device, non_blocking=True) for gt in data[1]]
            elif isinstance(data[1], torch.Tensor):
                gts = data[1].to(self.device, non_blocking=True)
            else:
                raise ValueError(f"unexpected data[1] type {type(data[1])}")

            for cam, gt in zip(cams, gts):

                pc, aks = self.model.produce(cam)
                aks_tempo_mask = aks.opacity_tempo_decay > 0.1
                gs_tempo_mask = aks_tempo_mask.unsqueeze(1).expand(-1, K).reshape(-1)
                gs_tempo_mask_idx = torch.nonzero(gs_tempo_mask).flatten()


                xyz_lr = self.model.deform.anchor_lr_sched.get('anchor_offsets', None)
                xyz_lr = xyz_lr.get_last_lr()[0] \
                    if xyz_lr is not None else self.train_cfg.lr_anchor_offsets
                perturb_intensity = self.train_cfg.perturb_intensity * xyz_lr

                if step < self.train_cfg.means_freeze_period:
                    pc = pc.means_freeze_only()
                    perturb_intensity = 0
                elif self.routine_mgr.means_opt_only():
                    pc = pc.means_opt_only()
                
                img, _, info = self.single_render(
                    pc=pc,
                    cam=cam,
                    perturb_intensity=perturb_intensity,
                    prior_filter_idx=gs_tempo_mask_idx if self.model_cfg.filter_by_tempo else None
                ) # img and info are batched actually, but batch size = 1

                img = self.img_proc_callback(img)
                gt = self.img_proc_callback(gt)

                metrics["l1loss"] += F.l1_loss(img[None], gt[None])
                metrics["ssimloss"] += 1 - fused_ssim(img[None], gt[None], padding="valid")
                if self.train_cfg.reg_opacity > 0:
                    metrics["l1opacity"] += torch.abs(pc.opacities).mean()
                if self.train_cfg.reg_scale > 0:
                    metrics["l1scale"] += torch.abs(pc.scales).mean()
                if self.train_cfg.reg_volume > 0:
                    metrics["volume"] += torch.prod(pc.scales, dim=1).mean()
                if self.train_cfg.reg_offset > 0:
                    offsets = aks.childs_xyz - aks.anchor_xyz.unsqueeze(1).expand(-1, K, -1)    # [N, K, 3]
                    gs_offsets = torch.norm(offsets, dim=-1)                                    # [N, K]
                    ak_radius = torch.norm(gs_offsets, dim=-1) / K ** 0.5                       # [N]
                    metrics["offset"] += torch.mean(ak_radius)                                  # [1]

                batch_state["ops"].append(pc.opacities)
                batch_state["frames"].append(cam.frame)
                batch_state["info"].append(info)
                if self.train_cfg.grad2d_alpha > 0:
                    info[key_for_gradient].retain_grad()

                batch_state["img"].append(img.detach())
                batch_state["gt"].append(gt.detach())

            # ------------------------------- loss calculation --------------------------- #

            # losses averaged by batch
            metrics = {k: v / len(data) for k, v in metrics.items()}
            loss =  sum([metrics[k] * weights[k] for k in metrics.keys()])

            # ------------------------------- backward pass ------------------------------ #
            self.strategy.step_pre_backward(
                state=self.state,
                info=info,
            )

            # pc.retain_grad()
            loss.backward()
            if False:
                grads = {
                    "means": pc.means.grad.norm(dim=-1).mean(),
                    "sh0": pc.sh0.grad.norm(dim=-1).mean(),
                    "scales": pc.scales.grad.norm(dim=-1).mean(),
                    "quats": pc.quats.grad.norm(dim=-1).mean(),
                    "opacities": pc.opacities.grad.abs().mean(),
                }
                self.writer.add_scalars("grads/grads", grads, step)


            desc = f"loss={loss.item():.3f}| anchor#={aks.anchor_xyz.shape[0]}"
            pbar.set_description(desc)
            
            # ------------------- use batch state update training state ------------------ #

            (anchor_childs,
             anchor_opacity) = opacity_analysis(batch_state, K,
                                                self.train_cfg.reloc_dead_thres,
                                                per_frame=True)
            (anchor_grad2d, 
             anchor_count) = calculate_grads(batch_state,
                                             N=N, K=K,
                                             device=self.device,
                                             per_frame=True)
            compute_blame = self.train_cfg.blame_alpha > 0
            anchor_blame = calculate_blames(batch_state, 
                                            N=N, K=K, 
                                            device=self.device,
                                            max_gs_per_tile=self.train_cfg.blame_max_gs_per_tile,
                                            per_frame=True) \
                if compute_blame else None


            def ele_wise_max(tensor_list):
                res = tensor_list[0].clone()
                for t in tensor_list[1:]:
                    res = torch.maximum(res, t, out=res)
                return res
            
            def avg_none_zero_only(tensor_list):
                """
                tensorlist: [tensor, tensor, ...], tensor of shape [N,]
                return: tensor of shape [N,], average of non-zero elements
                """
                non_zero_count = reduce(lambda x, y: x + (y > 0).float(), tensor_list, torch.zeros_like(tensor_list[0]))
                sumup = reduce(lambda x, y: x + y, tensor_list, torch.zeros_like(tensor_list[0]))
                return sumup / (non_zero_count + 1e-6)

            if step >= min(self.train_cfg.grow_start_iter, self.train_cfg.reloc_start_iter):
                snapshot = update_state(
                    self.state, 
                    new_history = {
                        "anchor_blame": anchor_blame,
                        "anchor_grad2d": anchor_grad2d,
                        "anchor_count": anchor_count,
                        "anchor_opacity": anchor_opacity,
                        "anchor_childs": anchor_childs,
                    }, 
                    special_aggr = {
                        "anchor_opacity": lambda d: ele_wise_max(list(d.values())),
                        "anchor_blame": lambda d: avg_none_zero_only(list(d.values())),
                    })

                # compensation by frame contribution
                # if self.model_cfg.temporal_opacity:
                #     total_frames = self.max_frame - self.min_frame
                #     active_frames_bin = [1 if (f+self.min_frame) in self.train_loader_gen.frames else 0 for f in range(total_frames)]
                #     active_frames_bin = torch.Tensor(active_frames_bin).to(self.device)
                #     opacity_means = self.model.deform.anchor_params["anchor_opacity_mean"]
                #     opacity_stds = torch.sigmoid(self.model.deform.anchor_params["anchor_opacity_std"])
                #     opacity_span_left = ((opacity_means - opacity_stds).clamp(0, 1) * total_frames).round()
                #     opacity_span_right = ((opacity_means + opacity_stds).clamp(0, 1) * total_frames).round()
                #     contribute_ratio = calculate_ratio(
                #         opacity_span_left, opacity_span_right, active_frames_bin)
                #     assert contribute_ratio.shape[0] == N
                #     self.state["anchor_opacity"] /= contribute_ratio
            else:
                snapshot = update_state({}, {
                    "anchor_blame": anchor_blame,
                    "anchor_grad2d": anchor_grad2d,
                    "anchor_count": anchor_count,
                    "anchor_opacity": anchor_opacity,
                    "anchor_childs": anchor_childs,
                })
            # ------------------------ relocate and densification ------------------------ #

            rescale = standardize
            opacity_thres_fn = lambda x: x["anchor_opacity"] < self.train_cfg.reloc_dead_thres
            opacity_count_thres_fn = lambda x: x["anchor_count"] < 0.05
            blame_thres_fn = lambda x: rescale(x["anchor_blame"]) < 0.1
            mask_lowest_fn = lambda x: torch.zeros(N, dtype=torch.bool).to(self.device).scatter(
                                            0, torch.topk(
                                                rescale(x["anchor_opacity"]), 
                                                N // 20, largest=False).indices, True
                                        )
            grad2d_fn = lambda x: rescale(x["anchor_grad2d"])
            ops_fn = lambda x: rescale(x["anchor_opacity"])
            blame_fn = lambda x: rescale(x["anchor_blame"])

            grad2d_mixing = lambda x: self.train_cfg.grad2d_alpha * grad2d_fn(x) + \
                                (1 - self.train_cfg.grad2d_alpha) * ops_fn(x)
            blame_mixing = lambda x: self.train_cfg.blame_alpha * blame_fn(x) + \
                                (1 - self.train_cfg.blame_alpha) * ops_fn(x)

            self.densify_dead_func = opacity_thres_fn
            if self.train_cfg.blame_alpha > 0 and step > self.train_cfg.blame_start_iter:
                self.densify_prefer_func = blame_mixing
            else:
                self.densify_prefer_func = ops_fn
            
            report = self.strategy.step_post_backward(
                state=self.state,
                aks_params=self.model.deform.anchor_params,
                aks_opts=self.model.deform.anchor_opts,
                step=step,
                dead_func= self.densify_dead_func,
                impact_func= self.densify_prefer_func
            ) if self.state.get("history_length", 0) > 100 else {}
            # ) if not self.routine_mgr.means_opt_only() else {}
            
            self.model.optimize()
            self.model.zero_grad()
            self.model.update_lr(step)

            if self.train_cfg.perturb_anchor_post_densify > 0 and report.get("relocated_idx", None) is not None:
                assert self.model_cfg.naive_decay, "naive_decay is required to perturb anchor after densify"
                perturb_idx = report["relocated_idx"]
                added_idx = report["grew_idx"]
                if added_idx is not None:
                    perturb_idx = torch.cat([perturb_idx, added_idx], dim=0)
                
                with torch.no_grad():
                    anchor_xyz_lr = self.model.deform.anchor_lr_sched.get('anchor_xyz', None)
                    anchor_xyz_lr = anchor_xyz_lr.get_last_lr()[0] \
                        if anchor_xyz_lr is not None else self.train_cfg.lr_anchor_xyz
                    perturb_intensity = self.train_cfg.perturb_anchor_post_densify * anchor_xyz_lr
                    anchors_as_pc = {
                        "means": self.model.deform.anchor_params["anchor_xyz"][perturb_idx],
                        "scales": torch.exp(self.model.deform.anchor_params["anchor_scale_extend"][perturb_idx]),
                        "quats": self.model.deform.anchor_params["anchor_quat"][perturb_idx],
                        "opacities": torch.sigmoid(self.model.deform.anchor_params["anchor_opacity_decay"][perturb_idx]),
                        "sh0": None,
                        "shN": None,
                    }
                    perturb = calculate_perturb(Gaussians(anchors_as_pc), perturb_intensity)
                    self.model.deform.anchor_params["anchor_xyz"][perturb_idx] += \
                        torch.randn_like(self.model.deform.anchor_params["anchor_xyz"][perturb_idx]) * perturb
                    print(f"avg perturb intensity: {perturb.norm(dim=-1).mean()}, std: {perturb.norm(dim=-1).std()}")

            # tensorboard >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
            with torch.no_grad():
                if step % 500 == 0:
                    self.writer.add_histogram("train/tempo_ops_mean", self.model.deform.anchor_params["anchor_opacity_mean"], step)
                    self.writer.add_histogram("train/tempo_ops_std", torch.sigmoid(self.model.deform.anchor_params["anchor_opacity_std"]), step)

                if step % 500 == 0 and self.train_cfg.tb_histogram > 0:
                    self.writer.add_histogram("train/gs_opacities", snapshot["anchor_opacity"], step)
                    self.writer.add_histogram("train/grads2d", snapshot["anchor_grad2d"], step)
                    self.writer.add_histogram("train/childs_offsets", gs_offsets.flatten().clamp(-1,1), step) # last offsets in the batch

                self.writer.add_scalar("loss/loss", loss.item(), step)
                for k, v in metrics.items():
                    if weights[k] > 0:
                        self.writer.add_scalar(f"loss/reg:{k}", v * weights[k], step)

                self.writer.add_scalar("runtime/time", time.time() - tic, step)
                self.writer.add_scalar("runtime/active_gs", torch.sum(snapshot["anchor_childs"]), step)
                self.writer.add_scalar("runtime/average_child", torch.mean(snapshot["anchor_childs"]), step)

                if len(self.train_loader_gen.frames) > 1 and self.model_cfg.frame_embed_dim > 0:
                    frame_embeds = self.model.deform.deform_params["frame_embed"] # [F, N]
                    frame_embeds = torch.stack([frame_embeds[i] for i in range(len(self.train_loader_gen.frames))])
                    frame_embeds_std = torch.std(frame_embeds, dim=0)
                    self.writer.add_scalar("runtime/frame_embeds_std", frame_embeds_std.mean(), step)

                    frame_delta_embeds = self.model.deform.deform_params["frame_delta_embed"] # [F, N]
                    frame_delta_embeds = torch.stack([frame_delta_embeds[i] for i in range(len(self.train_loader_gen.frames))])
                    frame_delta_embeds_std = torch.std(frame_delta_embeds, dim=0)
                    self.writer.add_scalar("runtime/frame_delta_embeds_std", frame_delta_embeds_std.mean(), step)

                anchor_lr = self.model.deform.anchor_lr_sched['anchor_xyz'].get_last_lr()[0]
                self.writer.add_scalar("runtime/anchor_lr", anchor_lr, step)
                offset_lr = self.model.deform.anchor_lr_sched['anchor_offsets'].get_last_lr()[0]
                self.writer.add_scalar("runtime/offset_lr", offset_lr, step)
                if report.get("relocated", None) is not None:
                    self.writer.add_scalar("runtime/relocated", report["relocated"], step)
                opacity_decay = self.model.deform.anchor_params["anchor_opacity_decay"].mean()
                self.writer.add_scalar("runtime/opacity_decay", opacity_decay, step)
            
            # ----------------------------------- eval ----------------------------------- #
            if step + 1 in self.train_cfg.test_steps or \
                (self.train_cfg.test_steps_every > 0 and (step+1) % self.train_cfg.test_steps_every == 0):
                self.eval(step)

            # ---------------------------- viser viewer update --------------------------- #
            num_rays = gt.shape[0] * gt.shape[1] * gt.shape[2]
            self.viewer_mgr.checkout(num_rays, step)

            # -------------------------------- save chkpt -------------------------------- #
            if step + 1 in self.train_cfg.chkpt_steps:
                print(f"save checkpoint of step {step} ...")
                target = self.log.chkpt(step)
                dir = os.path.dirname(target)
                if not os.path.exists(dir):
                    os.makedirs(dir)
                torch.save({
                    "step": step,
                    "model": self.model.dump_chkpt(step),
                    "routine": self.routine_mgr.dump_chkpt(),
                    "config": OmegaConf.to_container(self.cfg)
                }, target)
        
            if step % 1000 == 0:
                with open(self.log.summary, 'w+') as f:
                    f.write(json.dumps(self.model.count_params(), indent=4))
        print("training finished, viewer lingering...")
        time.sleep(10)

    @torch.no_grad()
    def eval(self, step):
        last_step = step == self.train_cfg.max_step - 1

        results = {k:[] for k in self.eval_funcs.keys()}    # frame-wise results
        elapsed = []
        self.test_loader = iter(self.test_loader_gen.gen_loader({}, 0))
        n = len(self.test_loader)
        selected_idx = random.sample(range(n), min(n, 10)) \
            if not last_step else range(n)
        sub_bar = tqdm(self.test_loader, desc="evaluating", leave=False)

        video_buffer = []
        for i, data in enumerate(sub_bar):
            if isinstance(data[0], list):
                assert len(data[0]) == 1, "batch size should be 1"
            elif isinstance(data[0], torch.Tensor):
                assert data[0].shape[0] == 1, "batch size should be 1"
            else:
                raise ValueError(f"unexpected data[0] type {type(data[0])}")

            cam = self.scene.get_cam(data[0][0]).to(
                self.device, non_blocking=True)
            gt = data[1][0].to(
                self.device, non_blocking=True)

            start_time = time.time()
            pc, _ = self.model.produce(cam)
            img, _, _ = self.single_render(
                pc=pc,
                cam=cam,
            )
            elapsed.append(time.time() - start_time)
            img = img
            for k, func in self.eval_funcs.items():
                results[k].append(func(img[None], gt[None]).item())  # metrics func expect batches
            if  (
                self.train_cfg.save_eval_img or \
                last_step or \
                ((step + 1) % self.train_cfg.save_eval_img_every == 0 and self.train_cfg.save_eval_img_every > 0)
                ) and i in selected_idx:
                save_tensor_images_threaded(self.img_proc_callback(img), gt, self.log.render(f"{step}_{i}"))
            if last_step:
                video_buffer.append(self.img_proc_callback(img))
        if last_step:
            save_video(video_buffer, os.path.join(self.log.root, f"{step}.mp4"), fps=30)

        frames = self.test_loader_gen.frames
        results_avg = {k: sum(results[k]) / len(results[k]) for k in results.keys()}
        results_avg['fps'] = len(elapsed) / sum(elapsed)
        psnr_per_frame = {f"psnr_{frames[i]}": results["psnr"][i] for i in range(len(results["psnr"]))}

        # terminal print
        print()
        print(f"step {step}: \n{json.dumps(results_avg, indent=4)}")
        # tb print
        self.writer.add_scalar("eval/psnr", results_avg['psnr'] , step)
        self.writer.add_scalar("eval/msssim", results_avg['msssim'] , step)
        if self.train_cfg.tb_per_frame_psnr:
            self.writer.add_scalars("eval/perframe", psnr_per_frame, step)

        # log print
        results.update(
            {"step": step, 
             "avg": results_avg,
            }
        )
        with open(self.log.stat, 'a') as f:
            f.write(json.dumps(results) + '\n')

@hydra.main(config_path='.', config_name='default', version_base=None)
def main(cfg):
    set_random_seed(cfg.train.random_seed)
    assert True, 'sanity check'
    torch.autograd.set_detect_anomaly(True)

    try:
        if cfg.chkpt is not None:
            chkpt = torch.load(cfg.chkpt, weights_only=True)
        else:
            chkpt = None
    except:
        chkpt = None
    
    # override config than the default
    if chkpt is not None:
        print(f"▶ Loading checkpoint from {cfg.chkpt}, default config overwritten")
        new_cfg = OmegaConf.create(chkpt['config'])
        cfg = OmegaConf.merge(cfg, new_cfg)
    
    # override config than the checkpoint
    override_cfg = "./override.yaml"
    if os.path.exists(override_cfg):
        override = OmegaConf.load(override_cfg)
        print(f"▶ Loading override config from {override_cfg}: {override}")
        cfg = OmegaConf.merge(cfg, override)

    runner = Runner(cfg, chkpt)
    runner.train()


if __name__ == '__main__':
    main()