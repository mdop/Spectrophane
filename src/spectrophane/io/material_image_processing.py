import rawpy
import numpy as np
from PIL import Image
from pathlib import Path
import json
from typing import Sequence, Callable
from dataclasses import dataclass

from spectrophane.io.data_io import get_resource_path
from spectrophane.core.color_transformations import linrgb_to_xyz
from spectrophane.io.stack_io import stack_json_to_array, StackData

@dataclass
class TrainingRefImageData():
    transmission_stacks: StackData
    transmission_xyz: np.ndarray

def raw_to_linear_rgb(filename: str) -> np.ndarray:
    """Takes filename of the raw image file and converts content to linear RGB image in numpy float32 array in [0,1]"""
    total_path = get_resource_path("material_data/images/" + filename)
    with rawpy.imread(total_path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=False,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1,1)
        ).astype(np.float32) / 65535.0
    return rgb

def decode_rgb_img(image: np.ndarray) -> np.ndarray:
    """Takes srgb image in [0,1] and decodes to linear RGB image according to Rec. 709. Returns image in numpy float32 array in [0,1]"""
    #TODO: Add customizable gamma
    return np.clip(np.where(image < 0.081, image/4.5, np.pow((image+0.099)/1.099,1/0.45)), 0,1)

def rgb_image_to_linrgb(filename: str) -> np.ndarray:
    """Takes image path and transforms it into linear rgb. Assumes an 8-bit image."""
    total_path = get_resource_path("material_data/images/" + filename)
    image = np.array(Image.open(total_path), dtype=np.float32) / 255.0
    lin_image = decode_rgb_img(image)
    return lin_image

def import_image(filename: str) -> np.ndarray:
    """Takes the filename and imports the raw image file (jnp, raw) or rgb image. Returns numpy array with rgb intensities in [0,1]"""
    path = Path(filename)
    match path.suffix.lstrip("."):
        case "jnp" | "raw" | "nef" | "cr2" | "arw" | "raf":
            return raw_to_linear_rgb(filename)
        case _:
            return rgb_image_to_linrgb(filename)

def calibrate_spatial_brightness(whiteimage: np.ndarray, whitefield: Sequence[int]) -> Callable:
    """Takes an image whith the measurement area being homogeneously white and finds a brightness imhomogeneity correction function"""
    #TODO: Implement
    pass
    
def roi_filter(image: np.ndarray, roi: Sequence[int]) -> np.ndarray:
    """returns valid pixel values inside the specified roi. Clips rois outside of image size."""
    x0 = roi[0]
    y0 = roi[1]
    x1 = roi[2]+1
    y1 = roi[3]+1
    patch = image[y0:y1, x0:x1, :]
    valid = (patch < 1).all(axis=2)
    return patch[valid]

def aggregate_rois(image: np.ndarray, rois: Sequence[Sequence[int]], mode: str = "median") -> np.ndarray:
    """Aggregate pixels across all rectangular ROIs together. Available aggregation modes: median, average."""
    all_pixels = np.concatenate([roi_filter(image, roi) for roi in rois], axis=0)
    if mode == "median":
        return np.median(all_pixels, axis=0)
    elif mode == "average":
        return np.mean(all_pixels, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {mode}")

def aggregate_image_rois(linrgb_image: np.ndarray, white_rois: Sequence[Sequence[int]], black_rois: Sequence[Sequence[int]], color_rois: Sequence[Sequence[int]], 
                         aggregation: str = "median"):
    """Takes linear rgb image and evaluates aggregation of regions of interest. Will return the aggregated rgb color for white and black reference regions and color regions."""
    white_ref = aggregate_rois(linrgb_image, white_rois, aggregation)
    black_ref = aggregate_rois(linrgb_image, black_rois, aggregation)
    color_refs = np.array([aggregate_rois(linrgb_image, [color_roi], aggregation) for color_roi in color_rois])
    return white_ref, black_ref, color_refs

def process_image_to_xyz(linrgb_image: np.ndarray, white_rois: Sequence[Sequence[int]], black_rois: Sequence[Sequence[int]], color_rois: Sequence[Sequence[int]], 
                         aggregation: str = "median", rgb_xyz_matrix: str|np.ndarray = "sRGB") -> np.ndarray:
    """Takes linear rgb image and rois of white and black patches and color patches. Normalizes aggreagted colors and transforms them into an XYZ-like color space. 
    If no tranformation matrix is supplied the standard D65 transformation matrix is used. Colors are then normalized to (1,1,1) for the white reference."""
    white_rgb, black_rgb, color_rgb = aggregate_image_rois(linrgb_image, white_rois, black_rois, color_rois, aggregation)
    #reduce blacklevel and do white correction per channel
    color_rgb_corr = (color_rgb - black_rgb)/(white_rgb-black_rgb)
    white_corr = (white_rgb-black_rgb)
    white_xyz = linrgb_to_xyz(white_corr, rgb_xyz_matrix)
    color_xyz = linrgb_to_xyz(color_rgb_corr, rgb_xyz_matrix)
    xyz_correction = np.array([1,1,1]) / white_xyz
    color_xyz_corr = color_xyz / xyz_correction
    return color_xyz_corr

def parse_image_data(input_data):
    """Takes json material characterization file data and returns stack data arrays and a corresponding color array"""
    stack_dictlist = []
    xyz_colors = []
    for image_data in input_data["images"]["measurement_images"]["transmission"]:
        white_rois = image_data["white_refs"]
        black_rois = image_data["black_refs"]
        color_rois = [area["roi"]               for area in image_data["measurement_areas"]]
        image_stack_dictlist = [area["stack"]   for area in image_data["measurement_areas"]]
        image_array = import_image(image_data["filename"])
        xyz_imagecolors = process_image_to_xyz(image_array, white_rois, black_rois, color_rois)
        xyz_colors.extend(xyz_imagecolors)
        stack_dictlist.extend(image_stack_dictlist)
    xyz_array = np.array(xyz_colors)
    stack_data = stack_json_to_array(input_data["materials"], stack_dictlist)
    return TrainingRefImageData(stack_data, xyz_array)