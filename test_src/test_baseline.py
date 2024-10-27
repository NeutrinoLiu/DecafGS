from dataclasses import dataclass
import math
import os
import json

from tqdm import tqdm
import hydra
import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


from fused_ssim import fused_ssim
from gsplat.strategy import MCMCStrategy
from gsplat.rendering import rasterization
from examples.utils import knn, rgb_to_sh, set_random_seed

from dataset import SceneReader, CamSampler, dataset_split
from interface import Gaussians, Camera
from pipeline import DummyPipeline

def init_model_and_opt(
        scene: SceneReader,
        train_cfg: dict,
        model_cfg: dict,
        device: str = "cuda",
    ):

    # ------------------------------ gaussian attrs ------------------------------ #
    xyz = torch.from_numpy(scene.init_pts).float()
    rgb = torch.from_numpy(scene.init_pts_rgb).float()

    N = xyz.shape[0]
    dist2_avg = (knn(xyz, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scale = torch.log(dist_avg * train_cfg.init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    rotate = torch.rand(N, 4)
    opacity = torch.logit(torch.full((N,), train_cfg.init_opacity)) # reverse sigmoid
    sh0 = rgb_to_sh(rgb).unsqueeze(1) # [N, 1, 3]
    shN = torch.zeros(N, (train_cfg.sh_degree + 1) ** 2 - 1, 3)      # [N, TOTAL-1, 3]

    params = [
        ("means", torch.nn.Parameter(xyz), train_cfg.lr_mean * scene.scene_scale),
        ("scales", torch.nn.Parameter(scale), train_cfg.lr_scale),
        ("quats", torch.nn.Parameter(rotate), train_cfg.lr_quat),
        ("opacities", torch.nn.Parameter(opacity), train_cfg.lr_opacity),
        ("sh0", torch.nn.Parameter(sh0), train_cfg.lr_sh0),
        ("shN", torch.nn.Parameter(shN), train_cfg.lr_shN),
    ]
    
    # ----------------- optimizer binding and learning rate setup ---------------- #
    gs_attr_dict = torch.nn.ParameterDict({k: v for k, v, _ in params}).to(device)
    opt_cali = train_cfg.batch_size # calibrate learning rate by batch size
    opts = {
        attr_name: torch.optim.Adam(
            [{
                'name'  : attr_name,
                'params': gs_attr_dict[attr_name],
                'lr'    : attr_lr * math.sqrt(opt_cali),
            }],
            eps     = 1e-15 / math.sqrt(opt_cali),
            betas   = (1 - opt_cali * (1 - 0.9), 1 - opt_cali * (1 - 0.999))
            )
        for attr_name, _, attr_lr in params # independent optimizer for each attr
    }
    gs = Gaussians(gs_attr_dict)
    model = DummyPipeline(gs, model_cfg)
    return model, opts

class Runner:
    def __init__(self, data_cfg, model_cfg, train_cfg):

        self.cfg = train_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scene = SceneReader(data_cfg, True)

        # ------------------------------- data loading ------------------------------- #å
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
        self.model, self.opts = init_model_and_opt(
            self.scene,
            train_cfg,
            model_cfg,
            device=self.device
            )
        
        # ------------------------------- other plugin ------------------------------- #
        self.strategy = MCMCStrategy(cap_max=200_000)
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

        # TODO online viewer
    
    @staticmethod
    def render( pc: Gaussians,
                cam: Camera,
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
        img = img.permute(0, 3, 1, 2).clamp(0, 1.)

        return img, alpha, info

    def train(self):

        init_step = 0
        max_step = self.cfg.max_step
        pbar = tqdm(range(init_step, max_step))

        # lr decaying
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.opts['means'], gamma=0.01 ** (1.0 / max_step))
        ]

        train_loader = iter(self.train_sampler)
        for step in pbar:
            active_sh_degrees = min(
                step // self.cfg.sh_degree_interval,
                self.cfg.sh_degree
            )
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
            pc: Gaussians = self.model.produce(cam)
            img, _, info = self.render(
                pc=pc,
                cam=cam,
                sh_degree=active_sh_degrees
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
            self.strategy.step_pre_backward(
                params=pc._params,
                optimizers=self.opts,
                state=self.state,
                step=step,
                info=info
            )

            loss.backward()
            desc = f"loss={loss.item():.3f}| " f"sh degree={active_sh_degrees}| "
            pbar.set_description(desc)
            
            # relocate and densification
            self.strategy.step_post_backward(
                params=pc._params,
                optimizers=self.opts,
                state=self.state,
                step=step,
                info=info,
                lr=schedulers[0].get_last_lr()[0]
            )
            
            for optimizer in self.opts.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # ----------------------------------- eval ----------------------------------- #
            if step in [i - 1 for i in self.cfg.test_steps]:
                self.eval(step)

            # TODO save model

    @torch.no_grad()
    def eval(self, step):
        test_loader = iter(self.test_sampler)
        results = {k:0.0 for k in self.eval_funcs.keys()}
        for data in test_loader:
            assert len(data) == 1, "batch size should be 1 for test"
            cam, gt = data[0]
            pc: Gaussians = self.model.produce(cam)
            img, _, _ = self.render(
                pc=pc,
                cam=cam,
                sh_degree=self.cfg.sh_degree
            )
            gt = gt[None].to(self.device)
            for k, func in self.eval_funcs.items():
                results[k] += func(img, gt).item()
        for k in results.keys():
            results[k] /= len(self.test_sampler)
        print(f"step {step}: \n{json.dumps(results, indent=4)}")



@hydra.main(config_path='.', config_name='config', version_base=None)
def main(cfg):
    assert True, 'sanity check'
    runner = Runner(cfg.data, cfg.model, cfg.train)
    runner.train()


if __name__ == '__main__':
    main()