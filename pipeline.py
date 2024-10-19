"""
main decaf (deformable scaffold) nn module
"""
from interface import Camera, Gaussians, Anchors
import torch
from torch import nn
import math
from examples.utils import rgb_to_sh

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
    deform nn module
        :input [frame]
        :output [aks]
        :params frame_embed, anchor_embed, anchor_attrs, deform_mlp
    """
    def __init__(self, cfg, frame_start, frame_end, device):
        super(Deformable, self).__init__()
        self.frame_base = frame_start
        self.frame_length = frame_end - frame_start
        
        # ---------------------------------- params ---------------------------------- #
        self._frame_embed = torch.nn.Parameter(
            torch.randn(self.frame_length,
                        cfg.frame_dim)).to(device)
        self._anchor_embed = torch.nn.Parameter(
            torch.randn(cfg.anchor_num,
                        cfg.anchor_feature_dim)).to(device)
        self._anchor_xyz = torch.nn.Parameter(
            torch.randn(cfg.anchor_num, 3)).to(device)
        self._anchor_offsets = torch.nn.Parameter(
            torch.randn(cfg.anchor_num,
                        cfg.anchor_child_num,
                        3)).to(device)
        self._anchor_offset_extend = torch.nn.Parameter(
            torch.randn(cfg.anchor_num, 3)).to(device)
        self._anchor_scale_extend = torch.nn.Parameter(
            torch.randn(cfg.anchor_num, 3)).to(device)
        self._anchor_opcity_decay = torch.nn.Parameter(
            torch.randn(cfg.anchor_num)).to(device)
        
        # ----------------------------------- MLPs ----------------------------------- #
        # TODO temporal defomable model
        
        # --------------------------------- optimizor -------------------------------- #
        opt_cali = cfg.batch_size
        to_be_optimized = [
            ("frame_embed", self._frame_embed, cfg.lr_frame_embed),
            ("anchor_embed", self._anchor_embed, cfg.lr_anchor_embed),
            ("anchor_xyz", self._anchor_xyz, cfg.lr_anchor_xyz),
            ("anchor_offsets", self._anchor_offsets, cfg.lr_anchor_offsets),
            ("anchor_offset_extend", self._anchor_offset_extend, cfg.lr_anchor_offset_extend),
            ("anchor_scale_extend", self._anchor_scale_extend, cfg.lr_anchor_scale_extend),
            ("anchor_opcity_decay", self._anchor_opcity_decay, cfg.lr_anchor_opcity_decay)
        ]
        self.opts = {
            attr_name: torch.optim.Adam(
                [{
                    'name': attr_name,
                    'params': attr,
                    'lr': attr_lr * math.sqrt(opt_cali)
                }],
                eps=1e-15 / math.sqrt(opt_cali),
                betas=(1 - opt_cali * (1 - 0.9), 1 - opt_cali * (1 - 0.999))
            )
            for attr_name, attr, attr_lr in to_be_optimized
        }

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
            "opacity_decay": self._anchor_opcity_decay
        }
        return Anchors(aks_dict)
    
    def optimize(self):
        """
        optimize all parameters
        """
        for opt in self.opts.values():
            opt.step()
            opt.zero_grad()

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
    return None

class Scaffold(nn.Module):
    """
    scaffold nn module
        :input [aks] and [cam]
        :output [gs]
        :params mlp of scales, quats, opacities, colors
    """
    def __init__(self, cfg, device):
        super(Scaffold, self).__init__()

        # ----------------------------------- mlps ----------------------------------- #
        in_dim = cfg.anchor_feature_dim + 3
        hidden_dim = cfg.hidden_dim
        k = cfg.anchor_child_num
        self.mlp = nn.ModuleDict({
            "scales": MLP_builder(in_dim, hidden_dim, 3 * k, nn.Sigmoid()).to(device), # scale of offset
            "quats": MLP_builder(in_dim, hidden_dim, 4 * k, nn.Identity()).to(device),
            "opacities": MLP_builder(in_dim, hidden_dim, 1 * k, nn.Sigmoid()).to(device),
            "colors": MLP_builder(in_dim, hidden_dim, 3 * k, nn.Sigmoid()).to(device)
        })

        # --------------------------------- optimizer -------------------------------- #
        opt_cali = cfg.batch_size
        to_be_optimized = [
            (f"mlp_{k}", v, cfg[f"lr_mlp_{k}"])
            for k, v in self.mlp.items()
        ]
        self.opts = {
            attr_name: torch.optim.Adam(
                [{
                    'name': attr_name,
                    'params': attr,
                    'lr': attr_lr * math.sqrt(opt_cali)
                }],
                eps=1e-15 / math.sqrt(opt_cali),
                betas=(1 - opt_cali * (1 - 0.9), 1 - opt_cali * (1 - 0.999))
            )
            for attr_name, attr, attr_lr in to_be_optimized
        }

    def forward(self, aks: Anchors, cam: Camera) -> Gaussians:
        """
        forward pass
        """
        # anchor
        means = aks.childs_xyz
        anchor_xyz = aks.anchor_xyz
        scales_extend = aks.scale_extend
        opacity_decay = aks.opacity_decay

        feature = aks.feature
        ob_view = cam.c2w_t.to(aks.device) - anchor_xyz
        ob_view = ob_view / torch.norm(ob_view, dim=-1, keepdim=True)
        fea_ob = torch.cat([feature, ob_view], dim=-1)

        print(f"fea_ob: {fea_ob.shape}")

        # mlp
        scales = self.mlp["scales"](fea_ob)
        scales = scales * scales_extend
        quats = self.mlp["quats"](fea_ob)
        opacities = self.mlp["opacities"](fea_ob)
        opacities = decay(opacities, opacity_decay)
        colors = self.mlp["colors"](fea_ob)

        sh_degree = 3
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
    
    def optimize(self):
        """
        optimize all parameters
        """
        for opt in self.opts.values():
            opt.step()
            opt.zero_grad()

class DecafPipeline(nn.Module):
    """
    a wrapper of system pipeline
    """
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.deform = Deformable(cfg, cfg.frame_start, cfg.frame_end, device)
        self.scaffold = Scaffold(cfg, device)
        # TODO add optimizer
    
    def produce(self, cam: Camera) -> Gaussians:
        frame = cam.frame
        aks = self.deform(frame)
        gs = self.scaffold(aks, cam)
        return gs
    
    def optimize(self):
        # TODO: does the order of optimization matter?
        self.deform.optimize()
        self.scaffold.optimize()
