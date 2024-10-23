from dataclasses import dataclass
import json
import time
import os

from tqdm import tqdm
import hydra
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.utils.tensorboard import SummaryWriter


import nerfview
import viser

from fused_ssim import fused_ssim
from gsplat.rendering import rasterization
from examples.utils import set_random_seed

from strategy import DecafMCMCStrategy
from dataset import SceneReader, CamSampler, dataset_split
from interface import Gaussians, Camera
from pipeline import DecafPipeline

from helper import LogDirMgr, save_tensor_images, mem_profile_start, mem_profile_end

class Runner:

    def __init__(self, data_cfg, model_cfg, train_cfg):

        self.cfg = train_cfg
        # --------------------------------- setup log -------------------------------- #
        self.log = LogDirMgr(self.cfg.log_dir)
        with open(self.log.config, 'w') as f:
            f.write(json.dumps({
                "data": OmegaConf.to_container(data_cfg),
                "model": OmegaConf.to_container(model_cfg),
                "train": OmegaConf.to_container(train_cfg)
            }, indent=" "))

        # ------------------------------- data loading ------------------------------- #
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scene = SceneReader(data_cfg, True)

        train_cam_idx, test_cam_idx = dataset_split(
            list(range(self.scene.cam_num)),
            train_cfg.test_every)
        min_frame = model_cfg.frame_start
        max_frame = min(self.scene.frame_total, model_cfg.frame_end)
        self.train_sampler = CamSampler(
            self.scene,
            train_cam_idx,                      # train cam only
            list(range(min_frame, max_frame)),  # full frame
            batch_size = train_cfg.batch_size,
            policy = "random")
        self.test_sampler = CamSampler(
            self.scene,
            test_cam_idx,                       # test cam only
            list(range(min_frame, max_frame))   # full frame
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
        self.state = {}
        self.state.update(self.strategy.initialize_state())

        # ----------------------------------- eval ----------------------------------- #å
        self.eval_funcs = {
            "psnr": PeakSignalNoiseRatio(data_range=1.0).to(self.device),
            "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device),
            "lpips": LearnedPerceptualImagePatchSimilarity(
                net_type="alex",
                normalize=True).to(self.device),
        }

        # ------------------------------- online viewer ------------------------------ #
        self.server = viser.ViserServer(port=8080, verbose=False)
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self.viewer_callback,
            mode="training",
        )

        # ----------------------------- setup tensorboard ---------------------------- #
        self.writer = SummaryWriter(log_dir=self.log.tb)

    @staticmethod
    def render( pc: Gaussians,
                cam: Camera,
                permute: bool = True,
                **kwargs): # sh_degree is expected to be provided

        viewmats = torch.linalg.inv(cam.c2w)[None].to(pc.device)
        Ks = cam.intri[None].to(pc.device)
        width = cam.width
        height = cam.height

        img, alpha, info = rasterization(
            # gaussian attrs
            means       =   pc.means,
            quats       =   pc.quats,
            scales      =   pc.scales,
            opacities   =   pc.opacities,
            colors      =   pc.colors,
            # batched cams
            viewmats    =   viewmats,
            Ks          =   Ks,
            width       =   width,
            height      =   height,
            # options
            packed      =   True,                            # to save memory
            absgrad     =   True,                           # return abs grad
            **kwargs)
        
        # output of rasterization is (N, H, W, 3), 
        # we need to permute to (N, 3, H, W) for loss calculation
        if permute:
            img = img.permute(0, 3, 1, 2).clamp(0, 1.)

        return img, alpha, info


    def train(self):
        init_step = 0
        max_step = self.cfg.max_step
        pbar = tqdm(range(init_step, max_step))

        train_loader = iter(self.train_sampler)
        for step in pbar:
            
            while self.viewer.state.status == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()
            tic = time.time()

            # ------------------------- read train data in batch ------------------------- #
            try:
                data = next(train_loader)
            except StopIteration:
                # self.train_sampler.learn(self.state) # TODO adaptive data loader
                train_loader = iter(self.train_sampler)
                data = next(train_loader)
            
            # ------------------------------ forward pass ------------------------------ #
            assert len(data) == 1, "batch size should be 1"
            cam, gt = data[0]
            pc = self.model.produce(cam)

            img, _, info = self.render(
                pc=pc,
                cam=cam,
                sh_degree=0
            ) # img and info are batched actually, but batch size = 1

            # losses
            gt = gt[None].to(self.device)
            l1loss = F.l1_loss(img, gt)
            ssimloss = 1 - fused_ssim(
                img,
                gt,
                padding="valid")
            loss = self.cfg.ssim_lambda * ssimloss + (1 - self.cfg.ssim_lambda) * l1loss
        
            # regs
            if self.cfg.reg_opacity > 0:
                loss += self.cfg.reg_opacity * \
                    torch.abs(pc.opacities).mean()
            if self.cfg.reg_scale > 0:
                loss += self.cfg.reg_scale * \
                    torch.abs(pc.scales).mean()

            # ------------------------------- backward pass ------------------------------ #
            # update the states
            self.strategy.step_pre_backward()

            loss.backward()
            desc = f"loss={loss.item():.3f}| " f"sh degree={0}| "
            pbar.set_description(desc)
            
            # relocate and densification
            cur_anchor_xyz_lr = self.model.deform.lr_sched['anchor_xyz'].get_last_lr()[0]
            self.strategy.step_post_backward(
                gs=pc,
                aks_params=self.model.deform.raw_params,
                aks_opts=self.model.deform.opts,
                state=self.state,
                step=step,
                anchor_xyz_lr=cur_anchor_xyz_lr
            )
            
            self.model.optimize()
            self.model.zero_grad()
            self.model.update_lr(step)

            # ----------------------------------- eval ----------------------------------- #
            self.writer.add_scalar("train/loss", loss.item(), step)
            self.writer.add_scalar("train/l1loss", l1loss.item(), step)
            self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
            # print distributions
            if step % 400 == 0:
                self.writer.add_histogram("train/gs_opacities", pc.opacities, step)
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
        
        print("training finished, viewer lingering...")
        time.sleep(5)

    @torch.no_grad()
    def eval(self, step):
        test_loader = iter(self.test_sampler)
        results = {k:0.0 for k in self.eval_funcs.keys()}
        for i, data in enumerate(test_loader):
            assert len(data) == 1, "batch size should be 1 for test"
            cam, gt = data[0]
            pc = self.model.produce(cam)
            img, _, _ = self.render(
                pc=pc,
                cam=cam,
                sh_degree=0
            )
            gt = gt[None].to(self.device)
            for k, func in self.eval_funcs.items():
                results[k] += func(img, gt).item()
            save_tensor_images(img[0], gt[0], self.log.render(f"{step}_{i}"))

        for k in results.keys():
            results[k] /= len(self.test_sampler)
        
        # terminal print
        print(f"step {step}: \n{json.dumps(results, indent=4)}")
        # tb print
        self.writer.add_scalar("eval/psnr", results['psnr'] , step)
        self.writer.add_scalar("eval/ssim", results['ssim'] , step)
        # log print
        results['step'] = step
        with open(self.log.stat, 'a') as f:
            f.write(json.dumps(results) + '\n')

    @torch.no_grad()
    def viewer_callback(
        self, camera_state: nerfview.CameraState, img_wh: "tuple[int, int]"
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
            frame=0
        )

        pc = self.model.produce(cam)
        img, _, _ = self.render(
            pc=pc,
            cam=cam,
            permute=False,
            sh_degree=0,
            radius_clip=3.0,                # boost rendering speed
            )

        return img[0].cpu().numpy()


@hydra.main(config_path='.', config_name='config', version_base=None)
def main(cfg):
    set_random_seed(cfg.train.random_seed)
    assert True, 'sanity check'
    torch.autograd.set_detect_anomaly(True)
    runner = Runner(cfg.data, cfg.model, cfg.train)
    runner.train()


if __name__ == '__main__':
    main()