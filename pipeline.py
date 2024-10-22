"""
main decaf (deformable scaffold) nn module
"""
from pytorch_memlab import (LineProfiler, clear_global_line_profiler, profile,
                            profile_every, set_target_gpu)

from interface import Camera, Gaussians, Anchors
import torch
from torch import nn
import math

from examples.utils import rgb_to_sh
from helper import get_adam_and_lr_sched

def random_sample(points, K):
    M = points.shape[0]
    indices = torch.randint(0, M, (K,))
    return points[indices]

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
        self._frame_embed = torch.nn.Parameter(
            torch.zeros(self.frame_length,
                        model_cfg.frame_dim).to(device))
        # anchor & childs xyz
        assert init_pts.shape[-1] == 3, "init_pts should be [N, 3]"
        anchor_xyz = random_sample(init_pts, model_cfg.anchor_num)
        N = anchor_xyz.shape[0]
        anchor_xyz += random_init(N, 3, 0.001) # add noise
        print(f"sampling {N} anchors from {init_pts.shape[0]} init points\n")
        self._anchor_xyz = torch.nn.Parameter(
            anchor_xyz.to(device))
        self._anchor_offsets = torch.nn.Parameter(
            torch.zeros(N, model_cfg.anchor_child_num, 3, dtype=torch.float32, device=device))
        # anchor attributes
        self._anchor_offset_extend = torch.nn.Parameter(
            torch.zeros(N, 3, dtype=torch.float32, device=device))
        self._anchor_scale_extend = torch.nn.Parameter(
            torch.zeros(N, 3, dtype=torch.float32, device=device))
        self._anchor_opacity_decay = torch.nn.Parameter(
            torch.ones(N, dtype=torch.float32, device=device))
        self._anchor_embed = torch.nn.Parameter(
            torch.zeros(N, model_cfg.anchor_feature_dim, dtype=torch.float32, device=device))
        
        # ----------------------------------- MLPs ----------------------------------- #
        # TODO temporal defomable model
        
        # --------------------------------- optimizor -------------------------------- #
        opt_cali = train_cfg.batch_size
        to_be_optimized = [
            ("frame_embed", self._frame_embed, train_cfg.lr_frame_embed),
            ("anchor_embed", self._anchor_embed, train_cfg.lr_anchor_embed),
            ("anchor_xyz", self._anchor_xyz, train_cfg.lr_anchor_xyz),
            ("anchor_offsets", self._anchor_offsets, train_cfg.lr_anchor_offsets),
            ("anchor_offset_extend", self._anchor_offset_extend, train_cfg.lr_anchor_offset_extend),
            ("anchor_scale_extend", self._anchor_scale_extend, train_cfg.lr_anchor_scale_extend),
            ("anchor_opacity_decay", self._anchor_opacity_decay, train_cfg.lr_anchor_opacity_decay)
        ]
        self.opts, self.lr_sched = get_adam_and_lr_sched(to_be_optimized, opt_cali)

    def forward(self, frame) -> Anchors:
        """
        forward pass
        """
        frame = frame - self.frame_base
        assert frame < self.frame_length, "frame out of range"
        aks_dict = {
            "feature": self._anchor_embed,
            "xyz": self._anchor_xyz,
            "offsets": self._anchor_offsets,
            "offset_extend": self._anchor_offset_extend,
            "scale_extend": self._anchor_scale_extend,
            "opacity_decay": self._anchor_opacity_decay
        }
        return Anchors(aks_dict)

def MLP_builder(in_dim, hidden_dim, out_dim, out_act):
    """
    build a 2 layer mlp module, refer to scaffold-gs
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
        out_act
    )

def decay(opacities, decay):
    """
    decay opacities
    to facilitize opacity split during densification
    check: 
    https://arxiv.org/abs/2404.06109
    https://arxiv.org/abs/2404.09591
    """
    # TODO implement decay after we implement relocation
    return opacities

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
            "scales": MLP_builder(in_dim, hidden_dim, 3 * k, nn.Identity()).to(device), # scale of offset
            "quats": MLP_builder(in_dim, hidden_dim, 4 * k, nn.Identity()).to(device),
            "opacities": MLP_builder(in_dim, hidden_dim, 1 * k, nn.Identity()).to(device),
            "colors": MLP_builder(in_dim, hidden_dim, 3 * k, nn.Sigmoid()).to(device)
        })

        # --------------------------------- optimizer -------------------------------- #
        opt_cali = train_cfg.batch_size
        to_be_optimized = [
            (f"mlp_{k}", v.parameters(), train_cfg[f"lr_mlp_{k}"])
            for k, v in self.mlp.items()
        ]
        self.opts, self.lr_sched = get_adam_and_lr_sched(to_be_optimized, opt_cali)

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

        feature = aks.feature                                   # [N, D]
        ob_view = cam.c2w_t.float().to(aks.device) - anchor_xyz         # [N, 3]
        ob_view = ob_view / torch.norm(ob_view, dim=-1, keepdim=True)
        fea_ob = torch.cat([feature, ob_view], dim=-1)          # [N, D + 3]

        # attrs
        means = means.reshape(-1, 3)                            # [N * K, 3]
        scales = self.mlp["scales"](fea_ob).reshape(N, -1, 3)
        scales *= scales_extend.unsqueeze(1).repeat(1, K, 1)    # [N, K, 3]
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
        self.deform = Deformable(train_cfg,
                                 model_cfg,
                                 init_pts,
                                 device)
        self.scaffold = Scaffold(train_cfg,
                                 model_cfg,
                                 device)
        self.produce = self.forward
    
    def forward(self, cam: Camera) -> Gaussians:
        frame = cam.frame
        aks = self.deform(frame)
        gs = self.scaffold(aks, cam)
        return gs
    
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
