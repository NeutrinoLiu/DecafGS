from dataclasses import dataclass
import json
import time

from line_profiler import profile

import os
import math
from tqdm import tqdm
import numpy as np
import hydra
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
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
from pipeline import DecafPipeline

from helper import (LogDirMgr, 
                    save_tensor_images_threaded,
                    calculate_grads,
                    calculate_blames,
                    opacity_analysis,
                    normalize,
                    ewma_update,
                    cached_func)
from helper_viewer import ViewerMgr

class Runner:
    def __init__(self, data_cfg, model_cfg, train_cfg):

        self.cfg = train_cfg
        self.model_cfg = model_cfg
        # --------------------------------- setup log -------------------------------- #
        self.log = LogDirMgr(self.cfg.root)
        with open(self.log.config, 'w') as f:
            f.write(json.dumps({
                "data": OmegaConf.to_container(data_cfg),
                "model": OmegaConf.to_container(model_cfg),
                "train": OmegaConf.to_container(train_cfg)
            }, indent=" "))

        # ------------------------------- data loading ------------------------------- #
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        assert data_cfg.cache_device in ["torch", "cpu", "gpu"], \
            "only support cpu/gpu cache or torch dataloader"
        self.scene = SceneReader(data_cfg, data_cfg.cache_device)
        self.use_torch_loader = data_cfg.cache_device == "torch"

        train_cam_idx, test_cam_idx = dataset_split(
            list(range(self.scene.cam_num)),
            train_cfg.test_every)
        min_frame = model_cfg.frame_start
        max_frame = min(self.scene.frame_total, model_cfg.frame_end)
        self.frames = list(range(min_frame, max_frame))

        self.train_loader_gen = DataManager(
            self.scene,                         # img reader
            train_cam_idx,                      # train cam only
            self.frames,                        # full frame
            batch_size = train_cfg.batch_size,
            policy = "max_parallex",
            num_workers=data_cfg.num_workers,
            use_torch_loader=self.use_torch_loader,
            )
        self.test_sampler_gen = DataManager(
            self.scene,                         # img reader
            test_cam_idx,                       # test cam only
            self.frames,                        # full frame
            batch_size = 1,
            policy = "sequential",
            num_workers=data_cfg.num_workers,
            use_torch_loader=self.use_torch_loader
            )
        
        print(f"totally {len(train_cam_idx)}+{len(test_cam_idx)} cams")
        print(f"training frame {min_frame} ~ {max_frame}\n")

        # ----------------------------- model & opt init ----------------------------- #
        self.model = DecafPipeline(
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            init_pts=torch.Tensor(self.scene.init_pts).float(),
            device=self.device
        )

        # ------------------------- other training schedulers ------------------------ #
        self.unlocked_frame = 0
        self.train_frame_embed_only = False
        def freeze_scaffold_and_anchor_deform():
            self.train_frame_embed_only = True
            self.unlocked_frame += 1
            print(f">>>>> unlock training frame {self.unlocked_frame}")
            if self.unlocked_frame > 0 and self.unlocked_frame < len(self.frames):
                with torch.no_grad():
                    self.model.deform.deform_params["frame_embed"][self.unlocked_frame].data += \
                        self.model.deform.deform_params["frame_embed"][self.unlocked_frame - 1].data
            for p in self.model.scaffold.parameters():
                p.requires_grad = False
            for p in self.model.deform.anchor_params.values():
                p.requires_grad = False
            for p in self.model.deform.deform_params["mlp_deform"]:
                p.requires_grad = False
        def unfreeze_scaffold_and_anchor_deform():
            self.train_frame_embed_only = False
            for p in self.model.scaffold.parameters():
                p.requires_grad = True
            for p in self.model.deform.anchor_params.values():
                p.requires_grad = True
            for p in self.model.deform.deform_params["mlp_deform"]:
                p.requires_grad = True
        routine = {}

        iters_frame_only = self.cfg.incremental_routine_frame
        iters_mixing = self.cfg.incremental_routine_mixing
        iters_total = iters_frame_only + self.cfg.incremental_routine_mixing
        for i in range(len(self.frames)):
            f_range = list(range(i + 1))
            routine[i * iters_total] = (f_range, unfreeze_scaffold_and_anchor_deform)
            routine[i * iters_total + iters_mixing] = ([i+1], freeze_scaffold_and_anchor_deform)
        # drop last of the dict
        routine.pop(max(routine.keys()))

        def routine_(step):
            frame_range, fn = routine.get(step, (None, lambda: None))
            if frame_range is not None:
                fn()
                self.train_loader_gen.frames = frame_range
                self.test_sampler_gen.frames = frame_range
        
        self.routine = routine_ if self.cfg.incremental_routine else \
                        lambda x : None

        # ------------------------------- other plugin ------------------------------- #
        self.strategy = DecafMCMCStrategy(
            train_cfg=train_cfg,
            max_cap=model_cfg.anchor_num,
            verbose=False)
        
        self.state = {}         # running state, not only for strategy
        self.state.update(
            self.strategy.initialize_state())

        # ----------------------------------- eval ----------------------------------- #å
        self.eval_funcs = {
            "psnr": PeakSignalNoiseRatio(data_range=1.0).to(self.device),
            "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device),
            "lpips": LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True).to(self.device),
        }

        # --------------------------------- visualize -------------------------------- #
        self.viewer_mgr = ViewerMgr(self, min_frame, max_frame)
        self.writer = SummaryWriter(log_dir=self.log.tb)

    def single_render( self,
                pc: Gaussians,
                cam: Camera,
                permute: bool = True,
                **kwargs): # sh_degree is expected to be provided

        # gsplat rasterization requires batched cams
        viewmats_batch = torch.linalg.inv(cam.c2w)[None].to(pc.device)
        Ks_batch = cam.intri[None].to(pc.device)
        width = cam.width
        height = cam.height

        if self.model_cfg.filter_by_ops:
            pc, filter_idx = Gaussians.filter_by_ops(pc, self.cfg.reloc_dead_thres)
            # idx[i] = j means, the i-th filtered gaussian is the j-th gaussian in the original pc
        else:
            filter_idx = torch.arange(pc.means.shape[0], device=pc.device)

        img, alpha, info = rasterization(
            # gaussian attrs
            means       =   pc.means,
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

    @profile
    def train(self):
        init_step = 0
        max_step = self.cfg.max_step
        pbar = tqdm(range(init_step, max_step))

        train_loader = iter([])
        for step in pbar:
            self.routine(step)

            tic = self.viewer_mgr.checkin()

            # ------------------------- read train data in batch ------------------------- #
            try:
                data = next(train_loader)
            except StopIteration:
                train_loader = iter(
                    self.train_loader_gen.gen_loader(self.state, step))
                data = next(train_loader)
            
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
                "l1loss": self.cfg.ssim_lambda,
                "ssimloss": 1 - self.cfg.ssim_lambda,
                "l1opacity": self.cfg.reg_opacity,
                "l1scale": self.cfg.reg_scale,
                "volume": self.cfg.reg_volume,
                "offset": self.cfg.reg_offset
            }

            # running state per batch,
            # self.state is a longer term state over the whole dataset
            key_for_gradient = "means2d"
            batch_state = {
                "ops": [],
                "info": [],
                "img": [],
                "gt": [],
            }
            K = self.model.deform.anchor_params['anchor_offsets'].shape[1]
            N = self.model.deform.anchor_params['anchor_offsets'].shape[0]


            cams = [self.scene.get_cam(cam).to(self.device, non_blocking=True) \
                    for cam in data[0]]
            gts = data[1].to(self.device, non_blocking=True)

            for cam, gt in zip(cams, gts):

                pc, aks = self.model.produce(cam)
                img, _, info = self.single_render(
                    pc=pc,
                    cam=cam,
                ) # img and info are batched actually, but batch size = 1

                metrics["l1loss"] += F.l1_loss(img[None], gt[None])
                metrics["ssimloss"] += 1 - fused_ssim(img[None], gt[None], padding="valid")
                if self.cfg.reg_opacity > 0:
                    metrics["l1opacity"] += torch.abs(pc.opacities).mean()
                if self.cfg.reg_scale > 0:
                    metrics["l1scale"] += torch.abs(pc.scales).mean()
                if self.cfg.reg_volume > 0:
                    metrics["volume"] += torch.prod(pc.scales, dim=1).mean()
                if self.cfg.reg_offset > 0:
                    offsets = aks.childs_xyz - aks.anchor_xyz.unsqueeze(1).expand(-1, K, -1)    # [N, K, 3]
                    gs_offsets = torch.norm(offsets, dim=-1)                                    # [N, K]
                    ak_radius = torch.norm(gs_offsets, dim=-1) / K ** 0.5                       # [N]
                    metrics["offset"] += torch.mean(ak_radius)                                  # [1]

                batch_state["ops"].append(pc.opacities)
                batch_state["info"].append(info)
                if self.cfg.grad2d_for_impact > 0:
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

            loss.backward()
            desc = f"loss={loss.item():.3f}| anchor#={aks.anchor_xyz.shape[0]}"
            pbar.set_description(desc)
            
            # ------------------- use batch state update training state ------------------ #

            (anchor_avg_childs,
             anchor_opacity) = opacity_analysis(batch_state, K, self.cfg.reloc_dead_thres)
            (anchor_grad2d, 
             anchor_count) = calculate_grads(batch_state, N=N, K=K, device=self.device)
            anchor_blame = calculate_blames(batch_state, N=N, K=K, device=self.device)

            ewma_update(self.state, {
                "anchor_opacity": anchor_opacity,
                "anchor_childs": anchor_avg_childs,
                "anchor_grad2d": anchor_grad2d,
                "anchor_count": anchor_count,
                "anchor_blame": anchor_blame
            }, self.cfg.state_ewma_alpha)

            # compute_blame = (step < self.strategy.refine_stop_iter          # densification in this step
            #                     and step >= self.strategy.refine_start_iter
            #                     and step % self.strategy.refine_every == 0) or self.cfg.always_compute_blame

            # self.state["anchor_blame"] = normalize(calculate_blames(batch_state, N=N, K=K, device=self.device)) \
            #     if compute_blame else None

            # ------------------------ relocate and densification ------------------------ #
            opacity_thres_fn = lambda x: x["anchor_opacity"] < self.cfg.reloc_dead_thres
            blame_thres_fn = lambda x: normalize(x["anchor_blame"]) < 0.1
            mask_lowest_fn = lambda x: torch.zeros(N, dtype=torch.bool).to(self.device).scatter(
                                            0, 
                                            torch.topk(
                                                normalize(x["anchor_opacity"]) + x["anchor_blame"], 
                                                N // 20, largest=False).indices,
                                            True
                                        )
            grad_ops_mixing_fn = lambda x: self.cfg.grad2d_for_impact * normalize(x["anchor_grad2d"]) + \
                                (1 - self.cfg.grad2d_for_impact) * normalize(x["anchor_opacity"])
            ops_fn = lambda x: normalize(x["anchor_opacity"])
            blame_fn = lambda x: normalize(x["anchor_blame"])

            # ---------------------------- different policies ---------------------------- #
            self.densify_dead_func = lambda x: torch.logical_and(opacity_thres_fn(x), blame_thres_fn(x))
            if step < 1000:
                self.densify_prefer_func = ops_fn
            else:
                self.densify_prefer_func = blame_fn
            
            xyz_lr = self.model.deform.anchor_lr_sched.get('anchor_xyz', None)
            xyz_lr = xyz_lr.get_last_lr()[0] \
                if xyz_lr is not None else self.cfg.lr_anchor_xyz
            report = self.strategy.step_post_backward(
                state=self.state,
                aks_params=self.model.deform.anchor_params,
                aks_opts=self.model.deform.anchor_opts,
                step=step,
                anchor_xyz_lr=xyz_lr,
                dead_func= cached_func(self.densify_dead_func, self.state),
                impact_func= cached_func(self.densify_prefer_func, self.state),
            ) if not self.train_frame_embed_only else {}
            
            self.model.optimize()
            self.model.zero_grad()
            self.model.update_lr(step)

            # tensorboard >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
            with torch.no_grad():

                if step % 400 == 0 and self.cfg.tb_histogram > 0:
                    self.writer.add_histogram("train/gs_opacities", anchor_opacity, step)
                    self.writer.add_histogram("train/grads2d", anchor_grad2d, step)
                    self.writer.add_histogram("train/childs_offsets", gs_offsets.flatten().clamp(-1,1), step) # last offsets in the batch

                self.writer.add_scalar("loss/loss", loss.item(), step)
                for k, v in metrics.items():
                    if weights[k] > 0:
                        self.writer.add_scalar(f"loss/reg:{k}", v * weights[k], step)

                self.writer.add_scalar("runtime/time", time.time() - tic, step)
                self.writer.add_scalar("runtime/active_gs", torch.sum(anchor_avg_childs), step)
                self.writer.add_scalar("runtime/average_child", torch.mean(anchor_avg_childs), step)
                frame_embeds = self.model.deform.deform_params["frame_embed"] # [F, N]
                frame_embeds = torch.stack([frame_embeds[i] for i in range(len(self.train_loader_gen.frames))])
                frame_embeds_std = torch.std(frame_embeds, dim=1)
                self.writer.add_scalar("runtime/frame_embeds_std", frame_embeds_std.mean(), step)
                anchor_lr = self.model.deform.anchor_lr_sched['anchor_xyz'].get_last_lr()[0]
                self.writer.add_scalar("runtime/anchor_lr", anchor_lr, step)
                offset_lr = self.model.deform.anchor_lr_sched['anchor_offsets'].get_last_lr()[0]
                self.writer.add_scalar("runtime/offset_lr", offset_lr, step)
                if report.get("relocated", None) is not None:
                    self.writer.add_scalar("runtime/relocated", report["relocated"], step)
                opacity_decay = self.model.deform.anchor_params["anchor_opacity_decay"].mean()
                self.writer.add_scalar("runtime/opacity_decay", opacity_decay, step)
            
            # ----------------------------------- eval ----------------------------------- #
            if step in [i - 1 for i in self.cfg.test_steps] or \
                (self.cfg.test_steps_every > 0 and (step+1) % self.cfg.test_steps_every == 0):
                self.eval(step)

            # ---------------------------- viser viewer update --------------------------- #
            num_rays = gt.shape[0] * gt.shape[1] * gt.shape[2]
            self.viewer_mgr.checkout(num_rays, step)
        
        with open(self.log.summary, 'a') as f:
            f.write(json.dumps(self.model.count_params(), indent=4))
        print("training finished, viewer lingering...")
        time.sleep(10)

    @torch.no_grad()
    def eval(self, step):
        results = {k:[] for k in self.eval_funcs.keys()}    # frame-wise results
        elapsed = []
        test_loader = iter(self.test_sampler_gen.gen_loader({}, 0))
        n = len(test_loader)
        selected_idx = random.sample(range(n), min(n, 10))
        sub_bar = tqdm(test_loader, desc="evaluating", leave=False)

        for i, data in enumerate(sub_bar):
            assert data[0].shape[0] == 1, "batch size should be 1"

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
                self.cfg.save_eval_img or \
                step == self.cfg.max_step - 1 or \
                ((step + 1) % self.cfg.save_eval_img_every == 0 and self.cfg.save_eval_img_every > 0)
                ) and i in selected_idx:
                save_tensor_images_threaded(img, gt, self.log.render(f"{step}_{i}"))

        frames = self.test_sampler_gen.frames
        results_avg = {k: sum(results[k]) / len(results[k]) for k in results.keys()}
        results_avg['fps'] = len(elapsed) / sum(elapsed)
        psnr_per_frame = {f"psnr_{frames[i]}": results["psnr"][i] for i in range(len(results["psnr"]))}

        # terminal print
        print(f"\ntraining frames: {self.train_loader_gen.frames[0]}-{self.train_loader_gen.frames[-1]}")
        print(f"test frames: {self.test_sampler_gen.frames[0]}-{self.test_sampler_gen.frames[-1]}")
        print(f"step {step}: \n{json.dumps(results_avg, indent=4)}")
        # tb print
        if not self.train_frame_embed_only:
            self.writer.add_scalar("eval/psnr", results_avg['psnr'] , step)
            self.writer.add_scalar("eval/ssim", results_avg['ssim'] , step)
        self.writer.add_scalars("eval/perframe", psnr_per_frame, step)

        # log print
        results.update(
            {"step": step, "fps": results_avg['fps']}
        )
        with open(self.log.stat, 'a') as f:
            f.write(json.dumps(results) + '\n')

@hydra.main(config_path='.', config_name='default', version_base=None)
def main(cfg):
    set_random_seed(cfg.train.random_seed)
    assert True, 'sanity check'
    torch.autograd.set_detect_anomaly(True)
    runner = Runner(cfg.data, cfg.model, cfg.train)
    runner.train()


if __name__ == '__main__':
    main()