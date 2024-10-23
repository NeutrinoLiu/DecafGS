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

class SpacetimeEmbeds:
    def __init__(self, cfg):
        pass

class Anchors:
    """
    Higher layer wrapper of anchor dict
    """
    required = ["feature", "xyz", "offsets", "offset_extend", "scale_extend", "opacity_decay"]
    def __init__(self, anchor):
        assert all([k in anchor for k in self.required]), f"missing key in Anchors: {self.required}"
        self._anchor = anchor
    @property
    def N(self):
        return self.offsets.shape[0]
    @property
    def K(self):
        return self.offsets.shape[1]
    @property
    def feature(self):
        return self._anchor["feature"]
    @property
    def device(self):
        return self._anchor["xyz"].device
    @property
    def anchor_xyz(self):
        # [N, 3]
        return self._anchor["xyz"]
    @property
    def offsets(self):
        # [N, K, 3]
        return self._anchor["offsets"]
    @property
    def offset_extend(self):
        # [N, 3]
        return torch.exp(self._anchor["offset_extend"])
    @property
    def childs_xyz(self):
        # [N, K, 3]
        K = self.offsets.shape[1]
        return self.anchor_xyz.unsqueeze(1).repeat(1, K, 1) + \
              self.offsets * self.offset_extend.unsqueeze(1).repeat(1, K, 1)
    @property
    def scale_extend(self):
        # [N, 3]
        return torch.exp(self._anchor["scale_extend"])
    @property
    def opacity_decay(self):
        # [N,]
        return torch.exp(self._anchor["opacity_decay"])


class Gaussians:
    """
    higher layer warpper of Guassian para dict
    """
    required = ["means", "scales", "quats", "opacities", "sh0", "shN"]
    def __init__(self, gs):
        assert all([k in gs for k in self.required]), f"missing key in Gaussians: {self.required}"
        self._gs = gs
    @property
    def device(self):
        return self._gs["means"].device
    @property
    def means(self):
        # [N, 3]
        return self._gs["means"]
    @property
    def quats(self):
        # [N, 4]
        return self._gs["quats"]
    @property
    def scales(self):
        # [N, 3]
        return self._gs["scales"]
    @property
    def opacities(self):
        # [N,]
        return self._gs["opacities"]
    @property
    def sh0(self):
        # [N, 1, 3]
        return self._gs["sh0"]
    @property
    def shN(self):
        # [N, TOTAL-1, 3]
        return self._gs["shN"]
    @property
    def colors(self):
        # [N, TOTAL, 3]
        sh0 = self.sh0
        shN = self.shN
        return torch.cat([sh0, shN], dim=1)