from dataclasses import dataclass
from tqdm import tqdm
import hydra
import math

import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


from fused_ssim import fused_ssim
from gsplat.strategy import MCMCStrategy
from gsplat.rendering import rasterization
from dataset import SceneReader, CamSampler, dataset_split
from interface import Gaussian, Camera
from examples.utils import knn, rgb_to_sh, set_random_seed


def init_model_and_opt(
        scene: SceneReader,
        train_cfg: dict,
        model_cfg: dict,
        device: str = "cuda",
    ):
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
    
    gs = torch.nn.ParameterDict({k: v for k, v, _ in params}).to(device)
    opt_cali = train_cfg.batch_size # calibrate learning rate by batch size
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
    gs_obj = Gaussian(
        means=gs['means'],
        scales=gs['scales'],
        quats=gs['quats'],
        opacities=gs['opacities'],
        sh0=gs['sh0'],
        shN=gs['shN']
    )
    return gs_obj, opt

class Runner:
    def __init__(self, data_cfg, model_cfg, train_cfg):
        # data
        self.cfg = train_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scene = SceneReader(data_cfg, True)

        train_cam_idx, test_cam_idx = dataset_split(
            list(range(self.scene.cam_num)),
            train_cfg.test_every)
        min_frame = model_cfg.frame_start
        max_frame = min(self.scene.frame_num, model_cfg.frame_end)
        self.train_sampler = CamSampler(
            self.scene,
            train_cam_idx,
            list(range(min_frame, max_frame)),
            train_cfg.batch_size,
            "random")
        self.test_sampler = CamSampler(
            self.scene,
            test_cam_idx,
            list(range(min_frame, max_frame))
            )
        
        print(f"totally {len(train_cam_idx)}+{len(test_cam_idx)} cams")
        print(f"training frame {min_frame} ~ {max_frame}")

        # model
        self.model, self.opt = init_model_and_opt(
            self.scene,
            train_cfg,
            model_cfg,
            device=self.device
            )
        
        # plugin
        self.strategy = MCMCStrategy(verbose=True, cap_max=200_000)
        self.state = {}
        self.state.update(self.strategy.initialize_state())

        # eval
        self.eval_funcs = {
            "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device),
            "psnr": PeakSignalNoiseRatio(data_range=1.0).to(self.device),
            "lpips": LearnedPerceptualImagePatchSimilarity(
                net_type="alex",
                normalize=True).to(self.device),
        }

        # TODO online viewer
    
    @staticmethod
    def render( pc: Gaussian,
                cam: Camera,
                **kwargs):
        
        means = pc.means                            # [N, 3]
        quats = pc.quats                            # [N, 4]
        scales = torch.exp(pc.scales)               # [N, 3]
        opacities = torch.sigmoid(pc.opacities)     # [N,]
        sh0 = pc.sh0                                # [N, 1, 3]
        shN = pc.shN                                # [N, TOTAL-1, 3]
        colors = torch.cat([sh0, shN], dim=1)       # [N, TOTAL, 3]

        viewmats = torch.linalg.inv(cam.c2w)[None].to(means.device)
        Ks = cam.intri[None].to(means.device)
        width = cam.width
        height = cam.height

        img, alpha, info = rasterization(
            # gaussian attrs
            means       =   means,
            quats       =   quats,
            scales      =   scales,
            opacities   =   opacities,
            colors      =   colors,
            # batched cams
            viewmats    =   viewmats,
            Ks          =   Ks,
            width       =   width,
            height      =   height,
            # options
            packed      =   True,                            # to save memory
            absgrad     =   True,                           # return abs grad
            **kwargs)
        
        return img, alpha, info

    def train(self):
        init_step = 0
        max_step = self.cfg.max_step
        pbar = tqdm(range(init_step, max_step))

        # lr decaying
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.opt['means'], gamma=0.01 ** (1.0 / max_step))
        ]

        # data loading
        train_iter = iter(self.train_sampler)

        for step in pbar:
            active_sh_degrees = min(
                step // self.cfg.sh_degree_interval,
                self.cfg.sh_degree
            )
            # ------------------------- read train data in batch ------------------------- #
            try:
                data = next(train_iter)
            except StopIteration:
                self.train_sampler.learn(self.state) # adaptive data loader
                train_iter = iter(self.train_sampler)
                data = next(train_iter)
            
            # ------------------------------ forward pass ------------------------------ #
            assert len(data) == 1, "batch size should be 1"
            cam, gt = data[0]
            pc: Gaussian = self.model.deform(cam)
            img, _, info = self.render(
                pc=pc,
                cam=cam,
                sh_degree=active_sh_degrees
            ) # img and info are batched actually, but batch size = 1

            # losses
            gt = gt[None].to(self.device)
            l1loss = F.l1_loss(img, gt)
            ssimloss = 1 - fused_ssim(
                img.permute(0, 3, 1, 2),
                gt.permute(0, 3, 1, 2),
                padding="valid")
            loss = self.cfg.ssim_lambda * ssimloss + (1 - self.cfg.ssim_lambda) * l1loss
        
            # regs
            if self.cfg.opacity_reg > 0:
                loss += self.cfg.opacity_reg * \
                    torch.abs(torch.sigmoid(pc.opacities)).mean()
            if self.cfg.scale_reg > 0:
                loss += self.cfg.scale_reg * \
                    torch.abs(torch.exp(pc.scales)).mean()

            # ------------------------------- backward pass ------------------------------ #
            # update the states
            self.strategy.step_pre_backward(self.state)  

            loss.backward()
            desc = f"loss={loss.item():.3f}| " f"sh degree={active_sh_degrees}| "
            pbar.set_description(desc)
            
            # relocate and densification
            self.strategy.step_post_backward(self.state)  
            
            for optimizer in self.opt.values():
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
        test_iter = iter(self.test_sampler)
        results = {k:0.0 for k in self.eval_funcs.keys()}
        for data in test_iter:
            assert len(data) == 1, "batch size should be 1"
            cam, gt = data[0]
            pc: Gaussian = self.model.deform(cam)
            img, _, info = self.render(
                pc=pc,
                cam=cam,
                sh_degree=self.cfg.sh_degree
            )
            gt = gt[None].to(self.device)
            for k, func in self.eval_funcs.items():
                results[k] += func(img, gt).item()
        for k in results.keys():
            results[k] /= len(self.test_sampler)
        print(f"step {step}: {results}")



@hydra.main(config_path='.', config_name='config')
def main(cfg):
    assert True, 'sanity check'
    runner = Runner(cfg.data, cfg.model, cfg.train)
    runner.train()


if __name__ == '__main__':
    main()