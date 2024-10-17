"""
train a single frame, make a reference from gsplat/simple_train.py
"""

import os
import time
import math
import json
from typing import Literal
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import tqdm
import nerfview
import viser

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy 
from examples.datasets.colmap import Dataset, Parser
from examples.utils import knn, rgb_to_sh, set_random_seed

from dataset import SceneReader, CamSampler, dataset_split

from fused_ssim import fused_ssim

@dataclass
class Config:
    # training
    batch_size: int = 4
    max_step: int = 13_000
    eval_steps = range(0, max_step, 1000)
    test_every: int = 25

    # rendering
    sh_degree_interval: int = 1000
    sh_degree: int = 3
    near_plane: float = 0.01
    far_plane: float = 1e10

    # data
    data_dir: str = "scenes/coffee"
    data_factor: float = 2
    resolution = 2
    init_type = "sfm"
    init_random_num = 6000
    init_random_extend = 3.0
    frame_total = 1
    max_cached_img = 100
    normalize_world_space: bool = True

    # optim
    lr = {
        'xyz' : 1.6e-4,
        'scale' : 5e-3,
        'rotate' : 1e-3,
        'opacity' : 5e-2,
        'sh0': 2.5e-3,
        'shN': 2.5e-3 / 20,
    }
    ssim_lambda: float = 0.2
    opacity_reg: float = 0.01
    scale_reg: float = 0.01

def init_gs_and_opt(
        scene_parser: SceneReader,
        scene_scale: float = 1.0,
        device: str = "cuda",
        batch_size: int = 1, # BS will impact learning rate of optimizor
        # init method
        init_type: Literal["random", "sfm"] = "sfm",
        init_rand_pts: int = 100_000,
        init_rand_extent: float = 3.0,
        # init attr
        init_scale: float = 1.0,
        init_opacity: float = 0.1,
        sh_degree: int = 3,
    ):

    if init_type == "random":
        xyz = (torch.rand(init_rand_pts, 3) * 2 - 1) * \
            init_rand_extent * scene_scale
        rgb = torch.rand(init_rand_pts, 3)
    elif init_type == "sfm":
        xyz = torch.from_numpy(scene_parser.init_pts).float()
        rgb = torch.from_numpy(scene_parser.init_pts_rgb).float()

    N = xyz.shape[0]
    dist2_avg = (knn(xyz, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scale = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    rotate = torch.rand(N, 4)
    opacity = torch.logit(torch.full((N,), init_opacity)) # reverse sigmoid
    sh0 = rgb_to_sh(rgb).unsqueeze(1) # [N, 1, 3]
    shN = torch.zeros(N, (sh_degree + 1) ** 2 - 1, 3)      # [N, TOTAL-1, 3]

    params = [
        ("means", torch.nn.Parameter(xyz), cfg.lr['xyz'] * scene_scale),
        ("scales", torch.nn.Parameter(scale), cfg.lr['scale']),
        ("quats", torch.nn.Parameter(rotate), cfg.lr['rotate']),
        ("opacities", torch.nn.Parameter(opacity), cfg.lr['opacity']),
        ("sh0", torch.nn.Parameter(sh0), cfg.lr['sh0']),
        ("shN", torch.nn.Parameter(shN), cfg.lr['shN']),
    ]
    gs = torch.nn.ParameterDict({k: v for k, v, _ in params}).to(device)
    opt_cali = batch_size # calibrate learning rate by batch size
    opt = {
        attr_name: torch.optim.Adam(
            [{
                'name'  : attr_name,
                'params': gs[attr_name],
                'lr'    : attr_lr * math.sqrt(opt_cali),
            }],
            eps     = 1e-15 / math.sqrt(opt_cali),
            betas   = (1 - opt_cali * (1 - 0.9), 1 - opt_cali * (1 - 0.999))
            )
        for attr_name, _, attr_lr in params # independent optimizer for each attr
    }
    return gs, opt


class Runner:
    def __init__(self, cfg):

        set_random_seed(42)
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # -------------------------------- stats dirs -------------------------------- #
        os.makedirs('output', exist_ok=True)
        self.output_dir = f"output/{time.strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)

        # -------------------------------- dataloader -------------------------------- #
        
        self.scene = SceneReader(cfg, True, True)

        train_cam_idx, test_cam_idx = dataset_split(
            list(range(self.scene.cam_num)),
            cfg.test_every)
        self.trainset = CamSampler(
            self.scene,
            train_cam_idx,
            [0],
            batch_size = cfg.batch_size,
            policy = "random",
            )
        self.valset = CamSampler(
            self.scene,
            test_cam_idx,
            [0]
            )

        # ----------------------- create GS attrs and optimizer ---------------------- #
        self.scene_scale = self.scene.scene_scale * 1.1
        self.gs, self.opt = init_gs_and_opt(
            scene_parser=self.scene,
            scene_scale=self.scene_scale,       # box of canonical space
            device=self.device,
            batch_size=cfg.batch_size,
            init_type=cfg.init_type,
            sh_degree=cfg.sh_degree,
        )
        # strategy for densification
        self.strategy = MCMCStrategy(verbose=True, cap_max=200_000)
        self.strategy_state = self.strategy.initialize_state()

        # ------------------------------- eval metrics ------------------------------- #
        # TODO
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", 
            normalize=True).to(self.device)

        # ---------------------------------- viewer ---------------------------------- #
        self.server = viser.ViserServer(port=8090, verbose=False)
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self.viewer_callback,
            mode="training",
        )


    def render(self,
                c2w_batch: torch.Tensor,
                K_batch: torch.Tensor,
                width: int,
                height: int,
                **kwargs):
        
        means = self.gs['means']       # [N, 3]
        quats = self.gs['quats'] # [N, 4]
        scales = torch.exp(self.gs['scales'])   # [N, 3]
        opacities = torch.sigmoid(self.gs['opacities']) # [N,]
        sh0 = self.gs['sh0']       # [N, 1, 3]
        shN = self.gs['shN']       # [N, TOTAL-1, 3]
        color = torch.cat([sh0, shN], dim=1) # [N, TOTAL, 3]
        w2c_batch = torch.linalg.inv(c2w_batch)

        # TODO, are they activated or not?
        img, alpha, info = rasterization(
            # attrs
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=color,
            # cams
            viewmats=w2c_batch,
            Ks=K_batch,
            width=width,
            height=height,
            # options
            packed=True,    # to save memory
            absgrad=True,   # return abs grad
            rasterize_mode="classic", # no anti-aliasing
            **kwargs)
        
        return img, alpha, info

    def train(self):
        init_step = 0
        max_step = self.cfg.max_step
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.opt['means'], gamma=0.01 ** (1.0 / max_step))
        ]
        # ----------------------------- train dataloader ----------------------------- #
        train_loader = iter(self.trainset)

        # ---------------------------- training profiling ---------------------------- #
        pbar = tqdm.tqdm(range(init_step, max_step))

        for step in pbar:
            while self.viewer.state.status == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()
            
            tic = time.time()
            try:
                data = next(train_loader)
            except StopIteration:
                train_loader = iter(self.trainset)
                data = next(train_loader)
            
            # ------------------------- read train data in batch ------------------------- #
            c2w_batch = data['camtoworld'].to(self.device)
            K_batch = data['K'].to(self.device)
            gt_batch = data['image'].to(self.device)
            height, width = gt_batch.shape[1:3]
            num_train_rays_per_step = (
                gt_batch.shape[0] * height * width
            )
            active_sh_degrees = min(
                step // self.cfg.sh_degree_interval,
                self.cfg.sh_degree
            )

            # ------------------------------ forward pass ------------------------------ #
            img_batch, alpha_batch, info_batch = self.render(
                c2w_batch=c2w_batch,
                K_batch=K_batch,
                width=width,
                height=height,
                # other options
                sh_degree=active_sh_degrees
            )

            self.strategy.step_pre_backward(
                params=self.gs,
                optimizers=self.opt,
                state=self.strategy_state,
                step=step,
                info=info_batch
            )
            
            # --------------------------------- loss calc -------------------------------- #
            l1loss = F.l1_loss(img_batch, gt_batch)
            ssimloss = 1 - fused_ssim(
                img_batch.permute(0, 3, 1, 2),
                gt_batch.permute(0, 3, 1, 2),
                padding="valid")
            loss = self.cfg.ssim_lambda * ssimloss + (1 - self.cfg.ssim_lambda) * l1loss

            # regs
            if self.cfg.opacity_reg > 0:
                loss += self.cfg.opacity_reg * \
                    torch.abs(torch.sigmoid(self.gs["opacities"])).mean()
            if self.cfg.scale_reg > 0:
                loss += self.cfg.scale_reg * \
                    torch.abs(torch.exp(self.gs["scales"])).mean()

            # ------------------------------- backward pass ------------------------------ #
            loss.backward()
            desc = f"loss={loss.item():.3f}| " f"sh degree={active_sh_degrees}| "
            pbar.set_description(desc)

            self.strategy.step_post_backward(
                params=self.gs,
                optimizers=self.opt,
                state=self.strategy_state,
                step=step,
                info=info_batch,
                lr=schedulers[0].get_last_lr()[0] # decaying xyz lr 
            )
            
            for optimizer in self.opt.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # ----------------------------------- eval ----------------------------------- #
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)

            # ------------------------------- update viewer ------------------------------ #
            self.viewer.lock.release()
            num_train_steps_per_sec = 1.0 / (time.time() - tic)
            num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
            # static update
            self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
            self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step, label="val"):
        print(f"Eval at step {step}")
        val_loader = iter(self.valset)
        metrics = {
            "psnr": [],
            "ssim": [],
            "lpips": []
        }
        timer = 0

        for i, data in enumerate(val_loader):
            c2w = data['camtoworld'].to(self.device)
            K = data['K'].to(self.device)
            gt = data['image'].to(self.device)
            height, width = gt.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            img, _, _ = self.render(
                c2w_batch=c2w,
                K_batch=K,
                width=width,
                height=height,
                sh_degree=self.cfg.sh_degree,
            )
            img = img.clamp(0., 1.)
            torch.cuda.synchronize()
            timer += time.time() - tic

            img_permuted = img.permute(0, 3, 1, 2)
            gt_permuted = gt.permute(0, 3, 1, 2)

            metrics["psnr"].append(self.psnr(img_permuted, gt_permuted))
            metrics["ssim"].append(self.ssim(img_permuted, gt_permuted))
            metrics["lpips"].append(self.lpips(img_permuted, gt_permuted))

        timer /= len(val_loader)

        stats = {
            k: torch.stack(v).mean().item()
            for k, v in metrics.items()
        }
        stats.update({
            "time": timer,
            "num of gs": len(self.gs['means'])
        })
        print(f"""
            PSNR: {stats["psnr"]:.3f}
            SSIM: {stats["ssim"]:.3f}
            LPIPS: {stats["lpips"]:.3f}
            Time: {stats["time"]:.3f}
            Num of GS: {stats["num of gs"]}
        """)
        with open(f"{self.output_dir}/{label}_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
    

    @torch.no_grad()
    def viewer_callback(
        self, camera_state: nerfview.CameraState, img_wh: "tuple[int, int]"
    ):
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        img, _, _ = self.render(
            c2w_batch=c2w[None],
            K_batch=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,   # active all SH degrees
            radius_clip=3.0,                # boost rendering speed
        )  # [1, H, W, 3]
        return img[0].cpu().numpy()


if __name__ == '__main__':
    cfg = Config()
    runner = Runner(cfg)
    runner.train()