import numpy as np

class Renormalizer:
    """Corrects raw calculated colors to maximize dynamic range."""
    def __init__(self):
        self._scale_xyz = 1
    
    def find_scaling_factor(self, edge_xyz: np.ndarray):
        """Finds approriate scaling factor for extreme colors based on an array of shape (edgecolors, 3). Scaling method is global channel saturation to keep colors in [0..1]³."""
        max_component = np.max(edge_xyz)
        self._scale_xyz = 1.0 / max_component

    def normalize(self, xyz):
        return xyz * self._scale_xyz
