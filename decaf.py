"""
main decaf (deformable scaffold) nn module
"""
from interface import Camera, Gaussian

class DeformableScaffold:
    def __init__(self, cfg):
        self.cfg = cfg
    def deform(self, cam: Camera) -> Gaussian:
        """
        deform the scaffold to the given camera
        """
        raise NotImplementedError