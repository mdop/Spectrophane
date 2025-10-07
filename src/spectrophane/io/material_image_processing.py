import rawpy
import numpy as np
from typing import Tuple

def raw_to_linear_rgb(path: str) -> np.ndarray:
    """Takes path to raw image file and converts content to linear RGB image in numpy float32 array in [0,1]"""
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=False,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1,1)
        ).astype(np.float32) / 65535.0
    return rgb

def decode_srgb_img(image: np.ndarray) -> np.ndarray:
    """Takes srgb image in [0,1] and decodes to linear RGB image according to Rec. 709. Returns image in numpy float32 array in [0,1]"""
    #TODO: Add customizable gamma
    return np.clip(np.where(image < 0.081, image/4.5, np.pow((image+0.099)/1.099,1/0.45)), 0,1)
    
def roi_median(image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Takes image numpy array and roi tuple (x0, y0, x1, y1)"""
    x0, y0, x1, y1 = roi
    patch = image[y0:y1, x0:x1, :]
    # robust mean (ignore saturated)
    valid = (patch < 1).all(axis=2)
    #TODO: Add warning for large fraction of saturated values
    return np.median(patch[valid], axis=0)

def image_to_linrgb(path: str) -> np.ndarray:
    """Takes image path and transforms it into linear rgb"""
    
    pass

def aggregate_color_region_from_lin_image():
    pass