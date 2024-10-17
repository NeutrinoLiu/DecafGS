"""
main decaf (deformable scaffold) nn module
"""
from interface import Camera, Gaussian

class DeformableScaffold:
    def __init__(self, content, cfg):
        self.content = content
        self.cfg = cfg
    def deform(self, cam: Camera) -> Gaussian:
        """
        deform the scaffold to the given camera
        """
        return self.content