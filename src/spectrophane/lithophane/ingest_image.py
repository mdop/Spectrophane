from PIL import Image, ImageFile
import numpy as np

from spectrophane.inverse.inverter import Inverter
from spectrophane.core.dataclasses import StackCandidates, VoxelGeometry

def format_image(image:ImageFile, resolution: tuple[int,int]) -> np.ndarray:
    pass

def image_to_stackmap(image: np.ndarray, inverter: Inverter, convert_xyz: bool = True) -> tuple[StackCandidates, np.ndarray, np.ndarray]:
    pass

def stackmap_to_voxelmap(stacks: StackCandidates, indexes: np.ndarray) -> VoxelGeometry:
    pass
