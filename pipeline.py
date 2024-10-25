"""
main decaf (deformable scaffold) nn module
"""

from typing import Tuple
import torch
from torch import nn
import math

from examples.utils import rgb_to_sh
from helper import get_adam_and_lr_sched
from interface import Camera, Gaussians, Anchors


def random_init(N, dim, scale=1.):
    return (torch.randn(N, dim) * 2. - 1.) * scale

class DummyPipeline(nn.Module):
    """
    pass through module,
    test only
    """
    def __init__(self, content, cfg):
        self.content = content
        self.cfg = cfg
    def produce(self, cam: Camera) -> Gaussians:
        return self.content

class ExpLayer(nn.Module):
    """
    a nn module wapper for torch.exp
    """
    def __init__(self):
        super(ExpLayer, self).__init__()
    def forward(self, x):
        return torch.exp(x)

class IngoreCam(nn.Module):
    """
    ignore the last three dim of batched input features
    """
    def __init__(self):
        super(IngoreCam, self).__init__()
    def forward(self, x):
        return x[:, :-3]

class Deformable(nn.Module):
    """
    A temporal deformation module
    yet currently it simply takes anchors' embeddings into account
    deform nn module
        :input [frame]
        :output [aks]
        :params frame_embed, anchor_embed, anchor_attrs, deform_mlp
    """
    def __init__(self, 
                 train_cfg,
                 model_cfg,
                 init_pts,
                 device):
        super(Deformable, self).__init__()
        self.frame_base = model_cfg.frame_start
        self.frame_length = model_cfg.frame_end - model_cfg.frame_start
        
        # ---------------------------------- params ---------------------------------- #
        # embeddings
        para_frame_embed = torch.nn.Parameter(
            torch.zeros(self.frame_length,
                        model_cfg.frame_dim).to(device))
        # anchor & childs xyz
        assert init_pts.shape[-1] == 3, "init_pts should be [N, 3]"
        anchor_xyz = init_pts
        N = anchor_xyz.shape[0]
        anchor_xyz += random_init(N, 3, 0.001) # add noise
        print(f"init with {N} anchors\n")
        para_anchor_xyz = torch.nn.Parameter(
            anchor_xyz.to(device))
        para_anchor_offsets = torch.nn.Parameter(
            torch.zeros(N, model_cfg.anchor_child_num, 3, dtype=torch.float32, device=device))
        # anchor attributes
        para_anchor_offset_extend = torch.nn.Parameter(
            torch.zeros(N, 3, dtype=torch.float32, device=device))
        para_anchor_scale_extend = torch.nn.Parameter(
            torch.zeros(N, 3, dtype=torch.float32, device=device))
        para_anchor_opacity_decay = torch.nn.Parameter(
            torch.ones(N, dtype=torch.float32, device=device))
        para_anchor_embed = torch.nn.Parameter(
            torch.zeros(N, model_cfg.anchor_feature_dim, dtype=torch.float32, device=device))
        
        # ----------------------------------- MLPs ----------------------------------- #
        # TODO temporal defomable model
        
        # --------------------------------- optimizor -------------------------------- #
        opt_cali = train_cfg.batch_size
        to_be_optimized = [
            ("anchor_embed", para_anchor_embed, train_cfg.lr_anchor_embed),
            ("anchor_xyz", para_anchor_xyz, train_cfg.lr_anchor_xyz),
            ("anchor_offsets", para_anchor_offsets, train_cfg.lr_anchor_offsets),
            ("anchor_offset_extend", para_anchor_offset_extend, train_cfg.lr_anchor_offset_extend),
            ("anchor_scale_extend", para_anchor_scale_extend, train_cfg.lr_anchor_scale_extend),
            ("anchor_opacity_decay", para_anchor_opacity_decay, train_cfg.lr_anchor_opacity_decay)
        ]
        self.raw_params = {k: v for k, v, _ in to_be_optimized}
        self.opts, self.lr_sched = get_adam_and_lr_sched(to_be_optimized, opt_cali, train_cfg.max_step)

    def forward(self, frame) -> Anchors:
        """
        forward pass
        """
        frame = frame - self.frame_base
        assert frame < self.frame_length, "frame out of range"
        aks_dict = {
            "feature": self.raw_params["anchor_embed"],
            "xyz": self.raw_params["anchor_xyz"],
            "offsets": self.raw_params["anchor_offsets"],
            "offset_extend": self.raw_params["anchor_offset_extend"],
            "scale_extend": self.raw_params["anchor_scale_extend"],
            "opacity_decay": self.raw_params["anchor_opacity_decay"]
        }
        return Anchors(aks_dict)

def MLP_builder(in_dim, hidden_dim, out_dim, out_act, view_dependent=True):
    """
    build a 2 layer mlp module, refer to scaffold-gs
    """
    return nn.Sequential(
        nn.Identity() if view_dependent else IngoreCam(),
        nn.Linear(in_dim if view_dependent else in_dim - 3,
                  hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
        out_act
    )

def decay(precursor, exp_decay):
    """
    decay opacities
    to facilitize opacity split during densification
    check: 
    https://arxiv.org/abs/2404.06109
    https://arxiv.org/abs/2404.09591
    """
    eps = torch.finfo(precursor.dtype).eps
    eps = torch.tensor(eps, device=precursor.device)
    return 1 - torch.exp(
        - precursor / torch.max(exp_decay, eps))

class Scaffold(nn.Module):
    """
    scaffold nn module
        :input [aks] and [cam]
        :output [gs]
        :params mlp of scales, quats, opacities, colors
    """
    def __init__(self, 
                 train_cfg, 
                 model_cfg,
                 device):
        super(Scaffold, self).__init__()

        # ----------------------------------- mlps ----------------------------------- #
        in_dim = model_cfg.anchor_feature_dim + 3
        hidden_dim = model_cfg.hidden_dim
        k = model_cfg.anchor_child_num
        self.mlp = nn.ModuleDict({
            "scales": MLP_builder(in_dim, hidden_dim, 3 * k, nn.Sigmoid(), model_cfg.view_dependent).to(device), # scale of offset
            "quats": MLP_builder(in_dim, hidden_dim, 4 * k, nn.Identity(), model_cfg.view_dependent).to(device),
            "opacities": MLP_builder(in_dim, hidden_dim, 1 * k, ExpLayer(), model_cfg.view_dependent).to(device),
            "colors": MLP_builder(in_dim, hidden_dim, 3 * k, nn.Sigmoid(), True).to(device)
        })

        # --------------------------------- optimizer -------------------------------- #
        opt_cali = train_cfg.batch_size
        to_be_optimized = [
            (f"mlp_{k}", v.parameters(), train_cfg[f"lr_mlp_{k}"])
            for k, v in self.mlp.items()
        ]
        self.opts, self.lr_sched = get_adam_and_lr_sched(to_be_optimized, opt_cali, train_cfg.max_step)

    def forward(self, aks: Anchors, cam: Camera) -> Gaussians:
        """
        forward pass
        """
        # anchor
        means = aks.childs_xyz                                  # [N, K, 3]
        N = means.shape[0]
        K = means.shape[1]
        anchor_xyz = aks.anchor_xyz                             # [N, 3]
        scales_extend = aks.scale_extend                        # [N, 3]
        opacity_decay = aks.opacity_decay                       # [N,]
        opacity_decay = opacity_decay.unsqueeze(1).expand(-1, K).flatten()

        feature = aks.feature                                   # [N, D]
        ob_view = cam.c2w_t.float().to(aks.device) - anchor_xyz         # [N, 3]
        ob_view = ob_view / torch.norm(ob_view, dim=-1, keepdim=True)
        fea_ob = torch.cat([feature, ob_view], dim=-1)          # [N, D + 3]

        # attrs
        means = means.reshape(-1, 3)                            # [N * K, 3]
        scales = self.mlp["scales"](fea_ob).reshape(N, -1, 3)
        scales = scales * scales_extend.unsqueeze(1).expand(-1, K, -1)    # [N, K, 3]
        scales = scales.reshape(-1, 3)                          # [N * K, 3]
        quats = self.mlp["quats"](fea_ob).reshape(-1, 4)        # [N * K, 4]
        opacities = self.mlp["opacities"](fea_ob).reshape(-1)   # [N * K]
        opacities = decay(opacities, opacity_decay)
        colors = self.mlp["colors"](fea_ob).reshape(-1, 3)      # [N * K, 3]

        sh_degree = 0
        # since gs_dict is simply an interface
        # dont have to wrap it with torch.nn.ParameterDict
        gs_dict = {
            "means": means,
            "scales": scales,
            "quats": quats,
            "opacities": opacities,
            "sh0": rgb_to_sh(colors).unsqueeze(1),
            "shN": torch.zeros(means.shape[0], 
                               (sh_degree + 1) ** 2 - 1,
                               3,
                               device=means.device)
        }
        return Gaussians(gs_dict)

class DecafPipeline(nn.Module):
    """
    a wrapper of system pipeline
    parameterless nn module
    """
    def __init__(self, train_cfg, model_cfg, init_pts, device):
        super(DecafPipeline, self).__init__()
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.deform = Deformable(train_cfg,
                                 model_cfg,
                                 init_pts,
                                 device)
        self.scaffold = Scaffold(train_cfg,
                                 model_cfg,
                                 device)
        self.produce = self.forward
    
    def forward(self, cam: Camera) -> Tuple[Gaussians, Anchors]:
        frame = cam.frame
        aks: Anchors = self.deform(frame)
        gs: Gaussians = self.scaffold(aks, cam)
        return gs, aks
    
    def optimize(self):
        for opt in self.scaffold.opts.values():
            opt.step()
        for opt in self.deform.opts.values():
            opt.step()

    def zero_grad(self):
        # zero grad
        # dont zero grad before all opt has been stepped
        for opt in self.scaffold.opts.values():
            opt.zero_grad()
        for opt in self.deform.opts.values():
            opt.zero_grad()
    
    def update_lr(self, step):
        for lr_sched in self.scaffold.lr_sched.values():
            lr_sched.step()
        for lr_sched in self.deform.lr_sched.values():
            lr_sched.step()
