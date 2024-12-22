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
                    threaded,
                    calculate_grads,
                    calculate_blames,
                    opacity_analysis,
                    normalize,
                    standardize,
                    ewma_update,
                    update_state,
                    gaussian_blur,
                    )
from helper_viewer import ViewerMgr
from helper_routine import RoutineMgr

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
        
        assert data_cfg.loader in ["torch", "plain"], \
            "only support torch or plain"
        loader_cache = data_cfg.loader == "plain"
        use_torch_loader = data_cfg.loader == "torch"
        self.scene = SceneReader(data_cfg, loader_cache)

        train_cam_idx, test_cam_idx = dataset_split(
            list(range(self.scene.cam_num)),
            train_cfg.test_every)
        min_frame = model_cfg.frame_start
        max_frame = min(self.scene.frame_total, model_cfg.frame_end)
        self.all_frames = list(range(min_frame, max_frame))

        self.train_loader_gen = DataManager(
            self.scene,                         # img reader
            train_cam_idx,                      # train cam only
            self.all_frames,                        # full frame
            batch_size = train_cfg.batch_size,
            policy = "max_parallex",
            num_workers=data_cfg.num_workers,
            use_torch_loader=use_torch_loader,
            img_proc_fn=self.img_proc_callback,
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
        
        print(f"totally {len(train_cam_idx)}+{len(test_cam_idx)} cams")
        print(f"training frame {min_frame} ~ {max_frame}\n")

        # ----------------------------- model & opt init ----------------------------- #
        from pipeline import DecafPipeline
        if model_cfg.resfield:
            self.model = DecafPipeline(
                train_cfg=train_cfg,
                model_cfg=model_cfg,
                init_pts=torch.Tensor(self.scene.init_pts).float(),
                device=self.device,
                T_max=300
            )
        else:
            self.model = DecafPipeline(
                train_cfg=train_cfg,
                model_cfg=model_cfg,
                init_pts=torch.Tensor(self.scene.init_pts).float(),
                device=self.device,
            )

        # ------------------------- other training schedulers ------------------------ #

        self.routine_mgr = RoutineMgr(self.model, self.all_frames)
        routine_schedule = None
        if self.cfg.routine == "perframe":
            routine_schedule = self.routine_mgr.frame_by_frame_routine(
                init_phase      =   self.cfg.perframe_routine_init,
                freeze_phase    =   self.cfg.perframe_routine_freeze,
                mixing_phase    =   self.cfg.perframe_routine_mixing
            )
        elif self.cfg.routine == "fence":
            routine_schedule = self.routine_mgr.fence_by_fence_routine(
                fence_interval = self.cfg.fenced_routine_interval,
                iters_shift = self.cfg.fenced_routine_init,
                iters_per_fence = self.cfg.fenced_routine_iters
            )
        def routine_(step):
            frame_range, fn = routine_schedule.get(step, (None, lambda: None))
            if frame_range is not None:
                fn()
                self.train_loader_gen.frames = frame_range
                self.test_loader_gen.frames = frame_range
        
        self.routine = routine_ if routine_schedule is not None else lambda x: None

        # ------------------------------- other plugin ------------------------------- #
        self.strategy = DecafMCMCStrategy(
            train_cfg=train_cfg,
            max_cap=model_cfg.anchor_num)
        
        self.state = {}         # running state, not only for strategy
        self.state.update(
            self.strategy.initialize_state())

        # ----------------------------------- eval ----------------------------------- #å
        self.eval_funcs = {
            "psnr": PeakSignalNoiseRatio(data_range=1.0).to(self.device),
            "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device),
            "msssim": MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(self.device),
            "lpips": LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True).to(self.device),
        }

        # --------------------------------- visualize -------------------------------- #
        self.viewer_mgr = ViewerMgr(self, min_frame, max_frame)
        self.writer = SummaryWriter(log_dir=self.log.tb)

    def img_proc_callback(self, img):
        if self.cfg.gradual_opt:
            init_r = 5
            final_r = 1
            final_step = self.cfg.gradual_opt_steps
            if self.step > final_step:
                return img
            # expotentially decay 
            r = init_r * (final_r / init_r) ** (self.step / final_step)
            return gaussian_blur(img, r)
        else:
            return img

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
            self.step = step
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
                "frames": [],
            }
            K = self.model.deform.anchor_params['anchor_offsets'].shape[1]
            N = self.model.deform.anchor_params['anchor_offsets'].shape[0]

            cams = [self.scene.get_cam(cam).to(self.device, non_blocking=True) \
                    for cam in data[0]]
            if isinstance(data[1], list):
                gts = [gt.to(self.device, non_blocking=True) for gt in data[1]]
            elif isinstance(data[1], torch.Tensor):
                gts = data[1].to(self.device, non_blocking=True)
            else:
                raise ValueError(f"unexpected data[1] type {type(data[1])}")

            for cam, gt in zip(cams, gts):

                # random save once gt
                # if random.random() < 0.1:
                #     save_tensor_images_threaded(gt, None, f"./blured_{step}.png")

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
                batch_state["frames"].append(cam.frame)
                batch_state["info"].append(info)
                if self.cfg.grad2d_alpha > 0:
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
                                                self.cfg.reloc_dead_thres,
                                                per_frame=True)
            (anchor_grad2d, 
             anchor_count) = calculate_grads(batch_state,
                                             N=N, K=K,
                                             device=self.device,
                                             per_frame=True)
            compute_blame = self.cfg.blame_alpha > 0
            anchor_blame = calculate_blames(batch_state, 
                                            N=N, K=K, 
                                            device=self.device,
                                            max_gs_per_tile=self.cfg.blame_max_gs_per_tile,
                                            per_frame=True) \
                if compute_blame else None

            snapshot = update_state(self.state, {
                "anchor_blame": anchor_blame,
                "anchor_grad2d": anchor_grad2d,
                "anchor_count": anchor_count,
                "anchor_opacity": anchor_opacity,
                "anchor_childs": anchor_childs,
            })

            # ------------------------ relocate and densification ------------------------ #

            rescale = standardize
            opacity_thres_fn = lambda x: x["anchor_opacity"] < self.cfg.reloc_dead_thres
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

            grad2d_mixing = lambda x: self.cfg.grad2d_alpha * grad2d_fn(x) + \
                                (1 - self.cfg.grad2d_alpha) * ops_fn(x)
            blame_mixing = lambda x: self.cfg.blame_alpha * blame_fn(x) + \
                                (1 - self.cfg.blame_alpha) * ops_fn(x)

            self.densify_dead_func = opacity_thres_fn
            if self.cfg.blame_alpha > 0 and step > self.cfg.blame_start_iter:
                self.densify_prefer_func = blame_mixing
            else:
                self.densify_prefer_func = ops_fn
            
            xyz_lr = self.model.deform.anchor_lr_sched.get('anchor_xyz', None)
            xyz_lr = xyz_lr.get_last_lr()[0] \
                if xyz_lr is not None else self.cfg.lr_anchor_xyz
            report = self.strategy.step_post_backward(
                state=self.state,
                aks_params=self.model.deform.anchor_params,
                aks_opts=self.model.deform.anchor_opts,
                step=step,
                anchor_xyz_lr=xyz_lr,
                dead_func= self.densify_dead_func,
                impact_func= self.densify_prefer_func,
            ) if not self.routine_mgr.freezed else {}
            
            self.model.optimize()
            self.model.zero_grad()
            self.model.update_lr(step)

            # tensorboard >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
            with torch.no_grad():

                if step % 400 == 0 and self.cfg.tb_histogram > 0:
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

                if len(self.train_loader_gen.frames) > 1:
                    frame_embeds = self.model.deform.deform_params["frame_embed"] # [F, N]
                    frame_embeds = torch.stack([frame_embeds[i] for i in range(len(self.train_loader_gen.frames))])
                    frame_embeds_std = torch.std(frame_embeds, dim=0)
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
        test_loader = iter(self.test_loader_gen.gen_loader({}, 0))
        n = len(test_loader)
        selected_idx = random.sample(range(n), min(n, 10))
        sub_bar = tqdm(test_loader, desc="evaluating", leave=False)

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
                self.cfg.save_eval_img or \
                step == self.cfg.max_step - 1 or \
                ((step + 1) % self.cfg.save_eval_img_every == 0 and self.cfg.save_eval_img_every > 0)
                ) and i in selected_idx:
                save_tensor_images_threaded(self.img_proc_callback(img), gt, self.log.render(f"{step}_{i}"))

        frames = self.test_loader_gen.frames
        results_avg = {k: sum(results[k]) / len(results[k]) for k in results.keys()}
        results_avg['fps'] = len(elapsed) / sum(elapsed)
        psnr_per_frame = {f"psnr_{frames[i]}": results["psnr"][i] for i in range(len(results["psnr"]))}

        # terminal print
        # print(f"\ntraining frames: {self.train_loader_gen.frames}")
        # print(f"test frames: {self.test_loader_gen.frames}")
        print(f"step {step}: \n{json.dumps(results_avg, indent=4)}")
        # tb print
        if not self.routine_mgr.freezed:
            self.writer.add_scalar("eval/psnr", results_avg['psnr'] , step)
            self.writer.add_scalar("eval/msssim", results_avg['msssim'] , step)
        if self.cfg.tb_per_frame_psnr:
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
    runner = Runner(cfg.data, cfg.model, cfg.train)
    runner.train()


if __name__ == '__main__':
    main()