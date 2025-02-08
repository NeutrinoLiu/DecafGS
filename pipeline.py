"""
main decaf (deformable scaffold) nn module
"""

from typing import Tuple
import torch
from torch import nn
import math

from examples.utils import rgb_to_sh
from helper import get_adam_and_lr_sched, count_opt_params, calc_tempo_decay, inverse_sigmoid, mul_quat, decay
from interface import Camera, Gaussians, Anchors
from helper_layers import MLP_builder, TempoMixture

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

def set_requires_grad(paralist, target):
    if isinstance(paralist, nn.Parameter):
        paralist.requires_grad = target
    else:
        # as a list or iterable
        for p in paralist:
            p.requires_grad = target

class Deformable(nn.Module):
    """
    A temporal deformation module
    yet currently it simply takes anchors' embeddings into account
    deform nn module
        :input [frame]
        :output [aks]
        :params frame_embed, anchor_embed, anchor_attrs, deform_mlp
    """
    def unfreeze(self):
        """
        unfreeze all params
        """
        for p in self.parameters():
            p.requires_grad = True
    def freeze_only(self, params: list[str]):
        """
        freeze params
        """
        # first unfreeze all
        self.unfreeze()
        for p in params:
            if p in self.anchor_params:
                set_requires_grad(self.anchor_params[p], False)
            elif p in self.deform_params:
                set_requires_grad(self.deform_params[p], False)
            # for mlp_deform sub modules specificlly
            elif p == "mlp_deform_feature":
                set_requires_grad(self.deform_mlp.embed_feature_mlp.parameters(), False)
            elif p == "mlp_deform_delta":
                set_requires_grad(self.deform_mlp.embed_delta_mlp.parameters(), False)
                set_requires_grad(self.deform_mlp.delta_mlps.parameters(), False)
            else: 
                assert False, f"param {p} not found in Deformable Module, no freezing"

    def __init__(self, 
                 train_cfg,
                 model_cfg,
                 init_pts,
                 init_pts_time,
                 init_pts_time_span,
                 device,
                 T_max=None):
        super(Deformable, self).__init__()
        self.cfg = model_cfg
        self.resfield = True if T_max is not None else False
        self.delta_decoupled = True if model_cfg.deform_delta_decoupled > 0 else False
        self.frame_length = model_cfg.frame_end - model_cfg.frame_start
        
        # ------------------------------- anchor params ------------------------------ #
        assert init_pts.shape[-1] == 3, "init_pts should be [N, 3]"
        assert init_pts_time.shape[0] == init_pts.shape[0], "init_pts_frame should have the same length as init_pts"
        assert init_pts_time_span.shape[0] == init_pts_time.shape[0], "init_pts_frame_span should have the same length as init_pts_frame"
        anchor_xyz = init_pts
        N = anchor_xyz.shape[0]
        anchor_xyz += random_init(N, 3, 0.001) # add noise
        print(f"init with {N} anchors\n")
        para_anchor_xyz = nn.Parameter(
            anchor_xyz.to(device))
        para_anchor_offsets = nn.Parameter(
            torch.zeros(N, model_cfg.anchor_child_num, 3, dtype=torch.float32, device=device))
        para_anchor_offset_extend = nn.Parameter(
            torch.zeros(N, 3, dtype=torch.float32, device=device))
        para_anchor_scale_extend = nn.Parameter(
            torch.zeros(N, 3, dtype=torch.float32, device=device))
        para_anchor_anchor_quat = nn.Parameter(
            torch.zeros(N, 4, dtype=torch.float32, device=device))
        para_anchor_anchor_quat.data[:, 0] = 1.0
        ops_decay_init = 4 if model_cfg.naive_decay else 0
        para_anchor_opacity_decay = nn.Parameter(
            torch.ones(N, dtype=torch.float32, device=device) * ops_decay_init)
        para_anchor_embed = nn.Parameter(
            torch.zeros(N, model_cfg.anchor_embed_dim, dtype=torch.float32, device=device))
        anchor_params = [
            ("anchor_embed", para_anchor_embed, train_cfg.lr_anchor_embed),
            ("anchor_xyz", para_anchor_xyz, train_cfg.lr_anchor_xyz),
            ("anchor_offsets", para_anchor_offsets, train_cfg.lr_anchor_offsets),
            ("anchor_offset_extend", para_anchor_offset_extend, train_cfg.lr_anchor_offset_extend),
            ("anchor_scale_extend", para_anchor_scale_extend, train_cfg.lr_anchor_scale_extend),
            ("anchor_quat", para_anchor_anchor_quat, train_cfg.lr_anchor_quat),
            ("anchor_opacity_decay", para_anchor_opacity_decay, train_cfg.lr_anchor_opacity_decay)
        ]
        if model_cfg.temporal_opacity:
            para_anchor_opacity_mean = nn.Parameter(
                init_pts_time.to(device)
                )
            para_anchor_opacity_std = nn.Parameter(
                inverse_sigmoid(init_pts_time_span.to(device) / 2)
                ) # half span is the std
            para_anchor_opacity_steepness = nn.Parameter(
                torch.zeros(N, dtype=torch.float32, device=device))
            anchor_params.extend([
                ("anchor_opacity_mean", para_anchor_opacity_mean, train_cfg.lr_anchor_opacity_mean),
                ("anchor_opacity_std", para_anchor_opacity_std, train_cfg.lr_anchor_opacity_std),
                ("anchor_opacity_steepness", para_anchor_opacity_steepness, train_cfg.lr_anchor_opacity_steepness)])

        # -------------------------- and anchor delta embed -------------------------- #
        if self.delta_decoupled:
            para_anchor_delta_embed = nn.Parameter(
                torch.zeros(N, model_cfg.anchor_delta_embed_dim, dtype=torch.float32, device=device))
            anchor_params.append(
                ("anchor_delta_embed", para_anchor_delta_embed, train_cfg.lr_anchor_embed))

        if model_cfg.anchor_per_frame_dxyz:
            para_anchor_frame_dxyz = nn.Parameter(
                torch.zeros(N, self.frame_length, 3, dtype=torch.float32, device=device))
            anchor_params.append(
                ("anchor_frame_dxyz", para_anchor_frame_dxyz, train_cfg.lr_anchor_frame_dxyz))
        
        self.anchor_opts, self.anchor_lr_sched = get_adam_and_lr_sched(
            anchor_params,
            train_cfg.batch_size, 
            train_cfg.max_step)

        self.anchor_params = {k: v for k, v, _ in anchor_params}


        # ------------------------------- deform params ------------------------------ #
        K = model_cfg.anchor_child_num
        para_frame_embed = nn.ParameterList([
            nn.Parameter(torch.zeros(model_cfg.frame_embed_dim, dtype=torch.float32, device=device))
            for _ in range(self.frame_length)
        ]) 
        para_frame_delta_embed = nn.ParameterList([
            nn.Parameter(torch.zeros(model_cfg.frame_delta_embed_dim, dtype=torch.float32, device=device))
            for _ in range(self.frame_length)
        ]) if self.delta_decoupled else None

        if model_cfg.deform_depth > 0:
            delta_dims = [3, 3 * K, 3, 3, 4]
                # d_xyz, d_offsets, d_offset_extend, d_scale_extend, d_quat
            delta_embed_dim = model_cfg.anchor_delta_embed_dim + model_cfg.frame_delta_embed_dim \
                if self.delta_decoupled else 0
            
            self.deform_mlp = TempoMixture(
                in_dim      =   model_cfg.anchor_embed_dim + model_cfg.frame_embed_dim,
                hidden_dim  =   model_cfg.deform_hidden_dim, 
                feature_dim =   model_cfg.anchor_feature_dim, 
                further_dims=   delta_dims, 
                skip        =   model_cfg.deform_skip,
                depth       =   model_cfg.deform_depth,
                delta_embed_dim   = delta_embed_dim,
                deform_mixing_pypass = model_cfg.deform_mixing_pypass,
                T_max       =   T_max
            ).to(device)
            deform_params = [
                ("frame_embed", list(para_frame_embed.parameters()), train_cfg.lr_frame_embed),  # list the para to keep a reference
                ("mlp_deform", list(self.deform_mlp.parameters()), train_cfg.lr_mlp_deform)
            ]
            if self.delta_decoupled:
                deform_params.append(
                    ("frame_delta_embed", list(para_frame_delta_embed.parameters()), train_cfg.lr_frame_embed)
                )

            self.deform_opts, self.deform_lr_sched = get_adam_and_lr_sched(
                deform_params,
                train_cfg.batch_size,
                train_cfg.max_step)

            self.deform_params = {k: v for k, v, _ in deform_params if k!="mlp_deform"} # no need to explicit stores deform mlp

        else:
            self.deform_mlp = None
            self.deform_opts = {}
            self.deform_lr_sched = {}

    def copy_frame_embed(self, src_frame, dst_frame, offseted=True):
        """
        copy frame embed from src to dst
        """
        with torch.no_grad():
            if not offseted:
                src_frame -= self.cfg.frame_start
                dst_frame -= self.cfg.frame_start
            self.deform_params["frame_embed"][dst_frame].data\
                .copy_(self.deform_params["frame_embed"][src_frame].data)
            if self.delta_decoupled:
                self.deform_params["frame_delta_embed"][dst_frame].data\
                    .copy_(self.deform_params["frame_delta_embed"][src_frame].data)
    
    def increase_all_std(self, value):
        """
        grow all std
        """
        with torch.no_grad():
            cur_std = torch.sigmoid(self.anchor_params["anchor_opacity_std"])
            new_std = cur_std + value
            self.anchor_params["anchor_opacity_std"].data = inverse_sigmoid(new_std)

    def get_frame_embed(self, frame):
        """
        get frame embed, return a tensor of shape [1, D]
        """
        if frame < 0 or frame >= self.frame_length:
            return None
        
        embeds = self.deform_params["frame_embed"]
        if self.cfg.embed_accumulative:
            raise NotImplementedError("accumulative frame embed not implemented")
        elif self.cfg.frame_embed_entangle_range > 0:
            r = self.cfg.frame_embed_entangle_range
            weighted = embeds[frame]
            weights = 1.
            for i in range(-r, r + 1):
                idx = frame + i
                if idx == frame or idx < 0 or idx >= self.frame_length:
                    continue
                w = 2 ** (-abs(i))
                weighted = weighted + embeds[idx] * w
                weights += w
            weighted = weighted / weights
            return weighted.unsqueeze(0)
        else:
            return embeds[frame].unsqueeze(0)

    def forward(self, frame) -> Anchors:
        """
        forward pass
        """
        assert frame < self.frame_length, "frame out of range"

        # ------------------------------ deform by frame ----------------------------- #
        frame_embed = self.get_frame_embed(frame)

        N = self.anchor_params["anchor_xyz"].shape[0]

        embeds = torch.cat([self.anchor_params["anchor_embed"],
                            frame_embed.expand(N, -1)],
                            dim=-1)
        
        delta_embed = torch.cat([self.anchor_params["anchor_delta_embed"],
                                    self.deform_params["frame_delta_embed"][frame].expand(N, -1)],
                                    dim=-1) \
            if self.delta_decoupled else None
        
        t = frame if self.resfield else None

        (features, 
         d_xyz, 
         d_offsets, 
         d_offset_extend, 
         d_scale_extend,
         d_quat) = self.deform_mlp(embeds, delta_embed, t)
        
        if self.cfg.anchor_per_frame_dxyz:
            d_xyz = self.anchor_params["anchor_frame_dxyz"][:, frame] # [N, 3]
        
        # print(f"frame {frame} embed: {frame_embed}")
        # print(f"shape of embeds: {embeds.shape}")
        # print(f"first anchor embed: {self.anchor_params['anchor_embed'][0]}")
        # print(f"first mlp layer: {self.deform_mlp.layers[0][0].weight}")
        # print(f"first mlp layer shpae: {self.deform_mlp.layers[0][0].weight.shape}")

        # ----------------------------- anchor attributes ---------------------------- #
        K = self.anchor_params["anchor_offsets"].shape[1]
        xyz = self.anchor_params["anchor_xyz"] + \
            (d_xyz if self.cfg.deform_anchor_xyz else 0)
        offsets = self.anchor_params["anchor_offsets"] + \
            (d_offsets.reshape(-1, K, 3) if self.cfg.deform_child_offsets else 0)
        offset_extend = self.anchor_params["anchor_offset_extend"] + \
            (d_offset_extend if self.cfg.deform_child_offsets else 0)
        scale_extend = self.anchor_params["anchor_scale_extend"] + \
            (d_scale_extend if self.cfg.deform_child_scales else 0)
        anchor_quat = self.anchor_params["anchor_quat"] + \
            (d_quat if self.cfg.deform_child_quats else 0)
        if self.cfg.deform_child_quats:
            anchor_quat = d_quat
        else:
            anchor_quat = torch.zeros(N, 4, dtype=torch.float32, device=xyz.device)
            anchor_quat[:, 0] = 1.0
        if self.cfg.temporal_opacity:
            tt  = frame / self.frame_length
            opacity_tempo_decay = calc_tempo_decay(
                tt, 
                self.anchor_params["anchor_opacity_mean"],
                self.anchor_params["anchor_opacity_std"],
                self.anchor_params["anchor_opacity_steepness"],
                wider=1.2
            )
        else:
            opacity_tempo_decay = torch.ones(N, dtype=torch.float32, device=xyz.device)

        aks_dict = {
            "feature"       : torch.cat([features, self.anchor_params["anchor_embed"]], dim=-1),
            "xyz"           : xyz,
            "offsets"       : offsets,
            "offset_extend" : offset_extend, 
            "scale_extend"  : scale_extend,
            "anchor_quat"   : anchor_quat,
            "opacity_decay" : self.anchor_params["anchor_opacity_decay"],
            "opacity_tempo_decay": opacity_tempo_decay,
        }
        return Anchors(aks_dict)

class Scaffold(nn.Module):
    """
    scaffold nn module
        :input [aks] and [cam]
        :output [gs]
        :params mlp of scales, quats, opacities, colors
    """
    def unfreeze(self):
        """
        unfreeze all params
        """
        for para in self.parameters():
            para.requires_grad = True
    def freeze_only(self, params: list[str]):
        """
        freeze params
        """
        # first unfreeze all
        self.unfreeze()
        for p in params:
            mlp_name = p.replace("mlp_", "")
            if mlp_name in self.mlp:
                set_requires_grad(self.mlp[mlp_name].parameters(), False)
            else:
                assert False, f"param {p} not found in Scaffold Module, no freezing"

    def __init__(self, 
                 train_cfg, 
                 model_cfg,
                 device,
                 T_max=None):
        super(Scaffold, self).__init__()

        self.cfg = model_cfg
        self.resfield= True if T_max is not None else False
        # ----------------------------------- mlps ----------------------------------- #
        in_dim = (model_cfg.anchor_feature_dim if model_cfg.deform_depth > 0 else 0 ) \
                + (3 if model_cfg.spawn_xyz_bypass else 0) \
                + model_cfg.anchor_embed_dim + 3 # feature, embed, ob_view, xyz
        
        hidden_dim = model_cfg.spawn_hidden_dim
        k = model_cfg.anchor_child_num
        self.mlp = nn.ModuleDict({
            "scales":    MLP_builder(in_dim, hidden_dim, 3 * k, nn.Sigmoid(),
                                     model_cfg.view_dependent,
                                     T_max,
                                     model_cfg.spawn_mlp_deeper).to(device), # scale of offset
            "quats":     MLP_builder(in_dim, hidden_dim, 4 * k, nn.Identity(),
                                     model_cfg.view_dependent,
                                     T_max,
                                     model_cfg.spawn_mlp_deeper).to(device),
            "opacities": MLP_builder(in_dim, hidden_dim, 1 * k, nn.Sigmoid(),
                                     model_cfg.view_dependent,
                                     T_max,
                                     model_cfg.spawn_mlp_deeper).to(device),
            "colors":    MLP_builder(in_dim, hidden_dim, 3 * k, nn.Sigmoid(),
                                     True,
                                     T_max,
                                     model_cfg.spawn_mlp_deeper).to(device)     # color must be view dependent
        })

        # --------------------------------- optimizer -------------------------------- #
        opt_cali = train_cfg.batch_size
        to_be_optimized = [
            (f"mlp_{k}", v.parameters(), train_cfg[f"lr_mlp_{k}"])
            for k, v in self.mlp.items()
        ]
        self.opts, self.lr_sched = get_adam_and_lr_sched(to_be_optimized, opt_cali, train_cfg.max_step)

    def forward(self, aks: Anchors, cam: Camera, t=None) -> Gaussians:
        """
        forward pass
        """
        assert self.resfield == (t is not None), "t should be provided if resfield is True"

        # --------------------------- read value of anchor --------------------------- #
        means = aks.childs_xyz                                  # [N, K, 3]
        N = means.shape[0]
        K = means.shape[1]
        anchor_xyz = aks.anchor_xyz                             # [N, 3]
        scales_extend = aks.scale_extend                        # [N, 3]
        anchor_quat = aks.anchor_quat                           # [N, 4]
        child_quat_multiplier = anchor_quat.unsqueeze(1).expand(-1, K, -1).reshape(-1, 4)   # [N * K, 4]
        opacity_decay = aks.opacity_decay.unsqueeze(1).expand(-1, K).flatten()
        opacity_tempo_decay = aks.opacity_tempo_decay.unsqueeze(1).expand(-1, K).flatten()

        # --------------------------- neural gaussian spawn -------------------------- #
        feature = aks.feature                                   # [N, D]
        ob_view = cam.c2w_t.float().to(aks.device) - anchor_xyz         # [N, 3]
        ob_view = ob_view / torch.norm(ob_view, dim=-1, keepdim=True)
        if self.cfg.spawn_xyz_bypass:
            fea_ob = torch.cat([feature, anchor_xyz, ob_view], dim=-1)
        else:
            fea_ob = torch.cat([feature, ob_view], dim=-1)          # [N, D + 3]

        # attrs
        means = means.reshape(-1, 3)                            # [N * K, 3]

        if self.resfield:
            mlp_scales = self.mlp["scales"](fea_ob, t).reshape(N, -1, 3)
            mlp_quats = self.mlp["quats"](fea_ob, t).reshape(-1, 4)        # [N * K, 4]
            mlp_opacities = self.mlp["opacities"](fea_ob, t).reshape(-1)   # [N * K]
            mlp_colors = self.mlp["colors"](fea_ob, t).reshape(-1, 3)      # [N * K, 3]
        else:
            mlp_scales = self.mlp["scales"](fea_ob).reshape(N, -1, 3)
            mlp_quats = self.mlp["quats"](fea_ob).reshape(-1, 4)        # [N * K, 4]
            mlp_opacities = self.mlp["opacities"](fea_ob).reshape(-1)   # [N * K]
            mlp_colors = self.mlp["colors"](fea_ob).reshape(-1, 3)      # [N * K, 3]

        if aks.feature_freezed:
            # when anchor feature is freezed, we also freeze the attributs mlp
            mlp_scales = mlp_scales.detach()
            mlp_quats = mlp_quats.detach()
            mlp_opacities = mlp_opacities.detach()
            mlp_colors = mlp_colors.detach()

        # post process
        scales = mlp_scales * scales_extend.unsqueeze(1).expand(-1, K, -1)    # [N, K, 3]
        scales = scales.reshape(-1, 3)                          # [N * K, 3]
        quats = mul_quat(mlp_quats, child_quat_multiplier)
        opacities = decay(mlp_opacities,  opacity_decay, naive=self.cfg.naive_decay) * opacity_tempo_decay

        # since gs_dict is simply an interface
        # dont have to wrap it with torch.nn.ParameterDict
        gs_dict = {
            "means": means,
            "scales": scales,
            "quats": quats,
            "opacities": opacities,
            "sh0": rgb_to_sh(mlp_colors).unsqueeze(1),
            "shN": torch.zeros(means.shape[0], 
                               0, 3,
                               device=means.device)
        }
        return Gaussians(gs_dict)

class DecafPipeline(nn.Module):
    """
    a wrapper of system pipeline
    parameterless nn module
    """
    def unfreeze(self):
        """
        unfreeze all params
        """
        self.deform.unfreeze()
        self.scaffold.unfreeze()
    def freeze_only(self, deformable_paras: list[str], scaffold_paras: list[str]):
        """
        freeze params
        """
        self.deform.freeze_only(deformable_paras)
        self.scaffold.freeze_only(scaffold_paras)
    def __init__(self, train_cfg, model_cfg,
                 init_pts,
                 init_pts_time,
                 init_pts_time_span,
                 device,
                 T_max=None, chkpt=None):
        super(DecafPipeline, self).__init__()
        self.cfg = model_cfg
        self.train_cfg = train_cfg
        self.deform = Deformable(train_cfg,
                                 model_cfg,
                                 init_pts,
                                 init_pts_time,
                                 init_pts_time_span,
                                 device,
                                 T_max)
        self.scaffold = Scaffold(train_cfg,
                                 model_cfg,
                                 device, 
                                 T_max)
        self.produce = self.forward
        self.resfield = True if T_max is not None else False
        if chkpt is not None:
            self.parse_chkpt(chkpt)
    
    def gen_aks(self, cam: Camera):
        # frame index will be remapped to [0, T_max]
        frame_idx = cam.frame - self.cfg.frame_start
        aks: Anchors = self.deform(frame_idx)
        return aks
    
    def gen_gs(self, aks: Anchors, cam: Camera):
        # frame index will be remapped to [0, T_max]
        frame_idx = cam.frame - self.cfg.frame_start
        gs: Gaussians = self.scaffold(aks, cam, frame_idx) \
            if self.resfield else self.scaffold
        return gs

    def forward(self, cam: Camera) -> Tuple[Gaussians, Anchors]:
        aks = self.gen_aks(cam)
        gs = self.gen_gs(aks, cam)
        return gs, aks
    
    def optimize(self):
        for opt in self.scaffold.opts.values():
            opt.step()
        for opt in self.deform.anchor_opts.values():
            opt.step()
        for opt in self.deform.deform_opts.values():
            opt.step()

    def zero_grad(self):
        # zero grad
        # dont zero grad before all opt has been stepped
        for opt in self.scaffold.opts.values():
            opt.zero_grad()
        for opt in self.deform.anchor_opts.values():
            opt.zero_grad()
        for opt in self.deform.deform_opts.values():
            opt.zero_grad()
    
    def update_lr(self, step):
        for lr_sched in self.scaffold.lr_sched.values():
            lr_sched.step()
        for lr_sched in self.deform.anchor_lr_sched.values():
            lr_sched.step()
        for lr_sched in self.deform.deform_lr_sched.values():
            lr_sched.step()
    
    def count_params(self):
        """
        count params
        """
        ret = {}
        for k, v in self.deform.anchor_opts.items():
            ret[k] = count_opt_params(v)
        for k, v in self.deform.deform_opts.items():
            ret[k] = count_opt_params(v)
        for k, v in self.scaffold.opts.items():
            ret[k] = count_opt_params(v)
        total = 0
        for k, v in ret.items():
            total += v
        ret["total"] = total
        ret["total_mem"] = total * 4 / 1024 / 1024
        return ret

    def dump_chkpt(self, step, save_opt=True, save_lr_sched=True):
        """
        save checkpoint
        {
            "step": step,
            "deform": deform model state_dict,
            "scaffold": scaffold model state_dict,
            "scaffold_opt": scaffold optimizer state_dict,
            "deform_anchor_opt": deform anchor optimizer state_dict,
            "deform_deform_opt": deform mlp optimizer state_dict,
        }
        """
        chkpt = {
            "step": step,
            "deform": self.deform.state_dict(),
            "deform_anchor_params": self.deform.anchor_params,
            "deform_deform_params": self.deform.deform_params,
            "scaffold": self.scaffold.state_dict()
        }
        if save_opt:
            chkpt["scaffold_opt"] = {}
            for k, v in self.scaffold.opts.items():
                chkpt["scaffold_opt"][k] = v.state_dict()
            chkpt["deform_anchor_opt"] = {}
            for k, v in self.deform.anchor_opts.items():
                chkpt["deform_anchor_opt"][k] = v.state_dict()
            chkpt["deform_deform_opt"] = {}
            for k, v in self.deform.deform_opts.items():
                chkpt["deform_deform_opt"][k] = v.state_dict()
        if save_lr_sched:
            chkpt["scaffold_lr_sched"] = {}
            for k, v in self.scaffold.lr_sched.items():
                chkpt["scaffold_lr_sched"][k] = v.state_dict()
            chkpt["deform_anchor_lr_sched"] = {}
            for k, v in self.deform.anchor_lr_sched.items():
                chkpt["deform_anchor_lr_sched"][k] = v.state_dict()
            chkpt["deform_deform_lr_sched"] = {}
            for k, v in self.deform.deform_lr_sched.items():
                chkpt["deform_deform_lr_sched"][k] = v.state_dict()
        return chkpt

    def parse_chkpt(self, chkpt, load_opt=True, load_lr_sched=True):
        """
        load checkpoint
        """
        self.deform.load_state_dict(chkpt["deform"])
        self.scaffold.load_state_dict(chkpt["scaffold"])

        chkpt_anchor_params = chkpt["deform_anchor_params"]
        chkpt_defrom_params = chkpt["deform_deform_params"]
        for k, v in self.deform.anchor_params.items():
            v.data = chkpt_anchor_params[k].data
        for k, model_para_list in self.deform.deform_params.items():
            chkpt_para_list = chkpt_defrom_params[k]
            for p, c in zip(model_para_list, chkpt_para_list):
                p.data = c.data

        if load_opt:
            for k, v in self.scaffold.opts.items():
                v.load_state_dict(chkpt["scaffold_opt"][k])
            for k, v in self.deform.anchor_opts.items():
                v.load_state_dict(chkpt["deform_anchor_opt"][k])
            for k, v in self.deform.deform_opts.items():
                v.load_state_dict(chkpt["deform_deform_opt"][k])
            
        if load_lr_sched:
            for k, v in self.scaffold.lr_sched.items():
                v.load_state_dict(chkpt["scaffold_lr_sched"][k])
            for k, v in self.deform.anchor_lr_sched.items():
                v.load_state_dict(chkpt["deform_anchor_lr_sched"][k])
            for k, v in self.deform.deform_lr_sched.items():
                v.load_state_dict(chkpt["deform_deform_lr_sched"][k])
            
        return chkpt["step"]