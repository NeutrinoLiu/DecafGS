"""
main decaf (deformable scaffold) nn module
"""
from interface import Camera, Gaussians, Anchors
import torch
from torch import nn
import math
from examples.utils import rgb_to_sh

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
                        model_cfg.frame_dim)).to(device)
        # anchor & childs xyz
        assert init_pts.shape[-1] == 3, "init_pts should be [N, 3]"
        anchor_xyz = random_sample(init_pts, model_cfg.anchor_num)
        N = anchor_xyz.shape[0]
        anchor_xyz += random_init(N, 3, 0.001) # add noise
        print(f"sampling {N} anchors from {init_pts.shape[0]} init points")
        self._anchor_xyz = torch.nn.Parameter(
            anchor_xyz).to(device)
        self._anchor_offsets = torch.nn.Parameter(
            torch.zeros(N, model_cfg.anchor_child_gs, 3)).to(device)
        # anchor attributes
        self._anchor_offset_extend = torch.nn.Parameter(
            torch.ones(N, 3)).to(device)
        self._anchor_scale_extend = torch.nn.Parameter(
            torch.ones(N, 3)).to(device)
        self._anchor_opcity_decay = torch.nn.Parameter(
            torch.ones(N)).to(device)
        self._anchor_embed = torch.nn.Parameter(
            torch.zeros(N, model_cfg.anchor_feature_dim)).to(device)
        
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
            ("anchor_opcity_decay", self._anchor_opcity_decay, train_cfg.lr_anchor_opcity_decay)
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
    return nn.Sigmoid()(opacities)

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
            "scales": MLP_builder(in_dim, hidden_dim, 3 * k, nn.Sigmoid()).to(device), # scale of offset
            "quats": MLP_builder(in_dim, hidden_dim, 4 * k, nn.Identity()).to(device),
            "opacities": MLP_builder(in_dim, hidden_dim, 1 * k, nn.Sigmoid()).to(device),
            "colors": MLP_builder(in_dim, hidden_dim, 3 * k, nn.Sigmoid()).to(device)
        })

        # --------------------------------- optimizer -------------------------------- #
        opt_cali = train_cfg.batch_size
        to_be_optimized = [
            (f"mlp_{k}", v, train_cfg[f"lr_mlp_{k}"])
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

class DecafPipeline(nn.Module):
    """
    a wrapper of system pipeline
    parameterless nn module
    """
    def __init__(self, train_cfg, model_cfg, init_pts, device):
        self.deform = Deformable(train_cfg,
                                 model_cfg,
                                 init_pts,
                                 device)
        self.scaffold = Scaffold(train_cfg,
                                 model_cfg,
                                 device)
        # renaming forward
        self.produce = self.forward
    
    def forward(self, cam: Camera) -> Gaussians:
        frame = cam.frame
        aks = self.deform(frame)
        gs = self.scaffold(aks, cam)
        return gs
    
    def optimize(self):
        # TODO: does the order of optimization matter?
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
