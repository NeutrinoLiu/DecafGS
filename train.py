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

from helper import LogDirMgr, save_tensor_images_threaded, calculate_grads, opacity_analysis, normalize, ewma_update



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

        # ------------------------- other training schedulers ------------------------ #
        policy_complex = {
            # 1000 : "downsample_2",
            train_cfg.max_step : "max_parallex"
        }
        sustained_release = None
        if train_cfg.frame_sustained_release:
            sustained_release = [1000] 
            for i in range(1, max_frame - min_frame):
                interval = 5 * int(i ** 0.5)
                sustained_release.append(sustained_release[-1] + interval)
        # ------------------------------------- - ------------------------------------ #

        self.train_loader_gen = DataManager(
            self.scene,                         # img reader
            train_cam_idx,                      # train cam only
            list(range(min_frame, max_frame)),  # full frame
            batch_size = train_cfg.batch_size,
            policy = policy_complex,
            num_workers=data_cfg.num_workers,
            use_torch_loader=self.use_torch_loader,
            sustained_release=sustained_release
            )
        self.test_sampler_gen = DataManager(
            self.scene,                         # img reader
            test_cam_idx,                       # test cam only
            list(range(min_frame, max_frame)),  # full frame
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

        # ------------------------------- online viewer ------------------------------ #
        self.server = viser.ViserServer(port=8080, verbose=False)
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self.viewer_callback,
            mode="training",
            min_frame=min_frame,
            max_frame=max_frame
        )

        # ----------------------------- setup tensorboard ---------------------------- #
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

        train_loader = iter(
            self.train_loader_gen.gen_loader({}, init_step))

        for step in pbar:
            
            while self.viewer.state.status == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()
            tic = time.time()

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
            ewma_update(self.state, {
                "anchor_opacity": anchor_opacity,
                "anchor_childs": anchor_avg_childs,
                "anchor_grad2d": anchor_grad2d,
                "anchor_count": anchor_count,
            }, self.cfg.state_ewma_alpha)

            # ------------------------ relocate and densification ------------------------ #
            cur_anchor_xyz_lr = self.model.deform.anchor_lr_sched['anchor_xyz'].get_last_lr()[0]
            dead_fn = lambda x: x["anchor_opacity"] < self.cfg.reloc_dead_thres
            impact_fn = lambda x: self.cfg.grad2d_for_impact * normalize(x["anchor_grad2d"]) + \
                                (1 - self.cfg.grad2d_for_impact) * normalize(x["anchor_opacity"])
            report = self.strategy.step_post_backward(
                state=self.state,
                aks_params=self.model.deform.anchor_params,
                aks_opts=self.model.deform.anchor_opts,
                step=step,
                anchor_xyz_lr=cur_anchor_xyz_lr,
                dead_func=dead_fn,
                impact_func=impact_fn
            )
            
            self.model.optimize()
            self.model.zero_grad()
            self.model.update_lr(step)

            # tensorboard >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
            with torch.no_grad():
                frame_embeds = self.model.deform.deform_params["frame_embed"] # [F, N]
                frame_embeds_std = torch.std(frame_embeds, dim=1)
                self.writer.add_scalar("train/frame_embeds_std", frame_embeds_std.mean(), step)
                if step % 400 == 0 and self.cfg.tb_histogram > 0:
                    self.writer.add_histogram("train/gs_opacities", anchor_opacity, step)
                    self.writer.add_histogram("train/grads2d", anchor_grad2d, step)
                    self.writer.add_histogram("train/childs_offsets", gs_offsets.flatten().clamp(-1,1), step) # last offsets in the batch
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", metrics["l1loss"].item(), step)
                self.writer.add_scalar("train/ssimloss", metrics["ssimloss"].item(), step)
                self.writer.add_scalar("train/time", time.time() - tic, step)
                self.writer.add_scalar("train/active_gs", torch.sum(anchor_avg_childs), step)
                self.writer.add_scalar("train/average_child", torch.mean(anchor_avg_childs), step)
                if report.get("relocated", None) is not None:
                    self.writer.add_scalar("train/relocated", report["relocated"], step)
            
            # ----------------------------------- eval ----------------------------------- #
            if step in [i - 1 for i in self.cfg.test_steps]:
                self.eval(step)

            # ---------------------------- viser viewer update --------------------------- #
            self.viewer.lock.release()
            num_train_rays_per_step = (
                gt.shape[0] * gt.shape[1] * gt.shape[2]
            )
            num_train_steps_per_sec = 1.0 / (time.time() - tic)
            num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
            # static update
            self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
            self.viewer.update(step, num_train_rays_per_step)
        
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
            if (self.cfg.save_eval_img or step == self.cfg.max_step - 1) and i in selected_idx:
                save_tensor_images_threaded(img, gt, self.log.render(f"{step}_{i}"))

        results_avg = {k: sum(results[k]) / len(results[k]) for k in results.keys()}
        results_avg['fps'] = len(elapsed) / sum(elapsed)

        # terminal print
        print(f"step {step}: \n{json.dumps(results_avg, indent=4)}")
        # tb print
        self.writer.add_scalar("eval/psnr", results_avg['psnr'] , step)
        self.writer.add_scalar("eval/ssim", results_avg['ssim'] , step)

        # log print
        results.update(
            {"step": step, "fps": results_avg['fps']}
        )
        with open(self.log.stat, 'a') as f:
            f.write(json.dumps(results) + '\n')

    @torch.no_grad()
    def viewer_callback(
        self, 
        camera_state: nerfview.CameraState,
        img_wh: "tuple[int, int]",
        frame=0,
        mode="RGB",
        init_scale=0.02
    ):
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float()
        K = torch.from_numpy(K).float()
        c2w_R = c2w[:3, :3]
        c2w_t = c2w[:3, 3]

        cam = Camera(
            path=None,
            intri=K,
            c2w_R=c2w_R,
            c2w_t=c2w_t,
            width=W,
            height=H,
            frame=frame
        )
        # --------------------------- pure image rendering --------------------------- #
        if mode == "RGB":
            pc, _ = self.model.produce(cam)
        # ------------------- anchor location and opacity rendering ------------------ #
        else:
            gs, aks = self.model.produce(cam)
            num_childs = aks.childs_xyz.shape[1]
            if mode == "ops":
                vis_ops = gs.opacities.reshape(-1, num_childs).mean(dim=1)
            elif mode == "avg-ops":
                vis_ops = self.state["anchor_opacity"]
            elif mode == "avg-grad":
                vis_ops = self.state["anchor_grad2d"]
            elif mode == "avg-impact":
                impact_fn = lambda x: self.cfg.grad2d_for_impact * normalize(x["anchor_grad2d"]) + \
                                    (1 - self.cfg.grad2d_for_impact) * normalize(x["anchor_opacity"])
                vis_ops = impact_fn(self.state)
            if vis_ops is None:
                return self.buffered_viewer_img.cpu().numpy()
            vis_ops = normalize(vis_ops)
            quats = torch.zeros(aks.anchor_xyz.shape[0], 4, device=self.device)
            quats[:, 0] = 1.0
            scales = torch.ones(aks.anchor_xyz.shape[0], 3, device=self.device) * init_scale
            colors = torch.ones(aks.anchor_xyz.shape[0], 3, device=self.device) 

            pc_dict = {
                "means" : aks.anchor_xyz,
                "quats" : quats,
                "scales" : scales,
                "opacities" : vis_ops,
                "sh0": rgb_to_sh(colors).unsqueeze(1),
                "shN": torch.zeros(aks.anchor_xyz.shape[0], 0, 3, device=self.device)
            }
            pc = Gaussians(pc_dict)


        img, _, _ = self.single_render(
            pc=pc,
            cam=cam,
            permute=False,      # no need to permute, viewer need (H, W, 3)
            )

        self.buffered_viewer_img = img
        return img.cpu().numpy()


@hydra.main(config_path='.', config_name='default', version_base=None)
def main(cfg):
    set_random_seed(cfg.train.random_seed)
    assert True, 'sanity check'
    torch.autograd.set_detect_anomaly(True)
    runner = Runner(cfg.data, cfg.model, cfg.train)
    runner.train()


if __name__ == '__main__':
    main()