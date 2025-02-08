"""
interface definition
for better code readability
"""

import torch
from dataclasses import dataclass
from typing import NamedTuple   # immutable

@dataclass
class Camera:
    """
    Camera parameters
    """
    path: str
    intri: torch.Tensor
    c2w_R: torch.Tensor
    c2w_t: torch.Tensor
    width: int
    height: int
    frame: int = None

    def to(self, device, **kwargs):
        self.intri = self.intri.to(device, **kwargs)
        self.c2w_R = self.c2w_R.to(device, **kwargs)
        self.c2w_t = self.c2w_t.to(device, **kwargs)
        return self

    def __getitem__(self, frame:int):
        return Camera(
            self.path,
            self.intri,
            self.c2w_R,
            self.c2w_t,
            self.width,
            self.height,
            frame
        )
    def thumbnail(self, scale):
        if scale == 1:
            return self
        new_intri = self.intri.clone()
        new_intri[:2, :] /= scale
        return Camera(
            self.path,
            new_intri,
            self.c2w_R,
            self.c2w_t,
            self.width // scale,
            self.height // scale,
            self.frame
        )
    def update_c2w(self, new_c2w):
        new_c2w = torch.tensor(new_c2w)
        self.c2w_R = new_c2w[:3, :3]
        self.c2w_t = new_c2w[:3, 3]
    @property
    def c2w(self) -> torch.Tensor:
        Rt = torch.eye(4)
        Rt[:3, :3] = self.c2w_R
        Rt[:3, 3] = self.c2w_t
        return Rt

class Anchors:
    """
    Higher layer wrapper of anchor dict
    """
    required = ["feature", "xyz", "offsets", "offset_extend", 
                "scale_extend", "anchor_quat",
                "opacity_decay", "opacity_tempo_decay"]
    def __init__(self, params):
        assert all([k in params for k in self.required]), f"missing key in Anchors: {self.required}"
        self._params = params
    @property
    def N(self):
        return self.offsets.shape[0]
    @property
    def K(self):
        return self.offsets.shape[1]
    @property
    def feature(self):
        return self._params["feature"]
    @property
    def device(self):
        return self._params["xyz"].device
    @property
    def anchor_xyz(self):
        # [N, 3]
        return self._params["xyz"]
    @property
    def offsets(self):
        # [N, K, 3]
        return self._params["offsets"]
    @property
    def offset_extend(self):
        # [N, 3]
        return torch.exp(self._params["offset_extend"])
    @property
    def childs_xyz(self):
        # [N, K, 3]
        K = self.offsets.shape[1]
        return self.anchor_xyz.unsqueeze(1).repeat(1, K, 1) + \
              self.offsets * self.offset_extend.unsqueeze(1).repeat(1, K, 1)
    @property
    def scale_extend(self):
        # [N, 3]
        return torch.exp(self._params["scale_extend"])
    @property
    def anchor_quat(self):
        # [N, 4]
        return torch.nn.functional.normalize(self._params["anchor_quat"], p=2, dim=-1)
    @property
    def opacity_decay(self):
        # [N,]
        return self._params["opacity_decay"]
    @property
    def opacity_tempo_decay(self):
        # [N,]
        return self._params["opacity_tempo_decay"]

    @property
    def feature_freezed(self):
        return self._params.get("feature_freezed", False)

    def feat_opt_only(self):
        return Anchors({
            "feature": self.feature,
            "xyz": self.anchor_xyz.detach(),
            "offsets": self.offsets.detach(),
            "offset_extend": self.offset_extend.detach(),
            "scale_extend": self.scale_extend.detach(),
            "anchor_quat": self.anchor_quat.detach(),
            "opacity_decay": self.opacity_decay.detach(),
            "opacity_tempo_decay": self.opacity_tempo_decay.detach()
        })

    def feat_freeze_only(self):
        return Anchors({
            "feature": self.feature.detach(),
            "feature_freezed": True,
            "xyz": self.anchor_xyz,
            "offsets": self.offsets,
            "offset_extend": self.offset_extend,
            "scale_extend": self.scale_extend,
            "anchor_quat": self.anchor_quat,
            "opacity_decay": self.opacity_decay,
            "opacity_tempo_decay": self.opacity_tempo_decay
        })


class Gaussians:
    """
    higher layer warpper of Guassian para dict
    """
    required = ["means", "scales", "quats", "opacities", "sh0", "shN"]
    def __init__(self, params):
        assert all([k in params for k in self.required]), f"missing key in Gaussians: {self.required}"
        self._params = params
    @property
    def device(self):
        return self._params["means"].device
    @property
    def means(self):
        # [N, 3]
        return self._params["means"]
    @property
    def quats(self):
        # [N, 4]
        return self._params["quats"]
    @property
    def scales(self):
        # [N, 3]
        return self._params["scales"]
    @property
    def opacities(self):
        # [N,]
        return self._params["opacities"]
    @property
    def sh0(self):
        # [N, 1, 3]
        return self._params["sh0"]
    @property
    def shN(self):
        # [N, TOTAL-1, 3]
        return self._params["shN"]
    @property
    def colors(self):
        # [N, TOTAL, 3]
        sh0 = self.sh0
        shN = self.shN
        return torch.cat([sh0, shN], dim=1)
    @classmethod
    def filter_by_ops(cls, gs, thres=0.):
        """
        filter gaussians by opacities
        """
        mask = gs.opacities > thres
        idx = torch.nonzero(mask).squeeze()
        return Gaussians({
            "means": gs.means[mask],
            "scales": gs.scales[mask],
            "quats": gs.quats[mask],
            "opacities": gs.opacities[mask],
            "sh0": gs.sh0[mask],
            "shN": gs.shN[mask]
        }), idx
    
    def retain_grad(self):
        for k, v in self._params.items():
            if v.requires_grad:
                v.retain_grad()

    def means_opt_only(self):
        return Gaussians({
            "means": self.means,
            "scales": self.scales.detach(),
            "quats": self.quats.detach(),
            "opacities": self.opacities.detach(),
            "sh0": self.sh0.detach(),
            "shN": self.shN.detach()
        })
    
    def means_freeze_only(self):
        return Gaussians({
            "means": self.means.detach(),
            "scales": self.scales,
            "quats": self.quats,
            "opacities": self.opacities,
            "sh0": self.sh0,
            "shN": self.shN
        })