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

class Gaussian:
    """
    higher layer warpper of Guassian para dict
    """
    def __init__(self, gs:dict):
        self.gs = gs
    @property
    def device(self):
        return self.gs["means"].device
    @property
    def means(self):
        # [N, 3]
        return self.gs["means"]
    @property
    def quats(self):
        # [N, 4]
        return self.gs["quats"]
    @property
    def scales(self):
        # [N, 3]
        return torch.exp(self.gs["scales"])
    @property
    def opacities(self):
        # [N,]
        return torch.sigmoid(self.gs["opacities"])
    @property
    def sh0(self):
        # [N, 1, 3]
        return self.gs["sh0"]
    @property
    def shN(self):
        # [N, TOTAL-1, 3]
        return self.gs["shN"]
    @property
    def colors(self):
        # [N, TOTAL, 3]
        sh0 = self.sh0
        shN = self.shN
        return torch.cat([sh0, shN], dim=1)