"""
interface definition
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

class Gaussian(NamedTuple):
    """
    Gaussian parameters
    """
    means: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    opacities: torch.Tensor
    sh0: torch.Tensor
    shN: torch.Tensor