from PIL import Image, ImageFile
import numpy as np

from spectrophane.inverse.inverter import Inverter
from spectrophane.core.dataclasses import StackCandidates, StackData, VoxelGeometry
from spectrophane.color.conversions import linrgb_to_xyz, decode_rgb

def format_image(image: ImageFile, resolution: tuple[int,int]) -> np.ndarray:
    """Preprocess image for inverter. Returns the image as a numpy array"""
    image = image.resize(resolution)
    #TODO: Implement color curve/brightness/contrast changes to optimize for color gamut
    return np.array(image)

def image_to_stackmap(image: np.ndarray, inverter: Inverter, convert_xyz: bool = True) -> tuple[StackCandidates, np.ndarray, np.ndarray]:
    """Transforms pre-processed image into a stack representation. Returns StackCandidates of unique colors and an index map for the image and calculated scores of the image."""
    #get unique colors, map unique color index to image spot
    h, w, c = image.shape
    assert c==3
    pixels = image.reshape(-1, c)
    unique_colors, inverse_indices = np.unique(pixels, axis=0, return_inverse=True)
    image_unique_color_indexes = inverse_indices.reshape(h, w)
    #optional xyz conversion
    if convert_xyz:
        unique_colors = linrgb_to_xyz(decode_rgb(unique_colors))
    #invert
    space = "xyz" if convert_xyz else "rgb"
    unique_stacks, _, unique_scores = inverter.invert_color(unique_colors, max_suggested_stacks=1, color_space=space) #TODO: Add support for multiple stack suggestions and pool with score penalty
    
    calc_score = unique_scores[image_unique_color_indexes]
    return unique_stacks, image_unique_color_indexes, calc_score

def _pixel_voxel_stack_height_matching(cumulative_voxel_heights: np.ndarray, cumulative_stack_heights: np.ndarray, stack_material_nums: np.ndarray) -> np.ndarray:
    """Transforms compressed stack height maps to a voxel material index format. 
    Assumes increasing height values and that stack layer boundaries match up with voxel boundaries, otherwise undefined behavior."""
    pixel_materials = np.full(len(cumulative_voxel_heights), -1)

    z_running_bottom_index = 0
    last_cumulative_stack_thickness = 0
    for stack_index in range(len(cumulative_stack_heights)):
        cumulative_voxels = np.sum(cumulative_voxel_heights <= cumulative_stack_heights[stack_index]+1e-5)
        if cumulative_stack_heights[stack_index] >= last_cumulative_stack_thickness + 1e-5:
            pixel_materials[z_running_bottom_index:cumulative_voxels] = stack_material_nums[stack_index]

        z_running_bottom_index = cumulative_voxels
        last_cumulative_stack_thickness = cumulative_stack_heights[stack_index]
    return pixel_materials


def stackmap_to_voxelmap(layer_thicknesses: np.ndarray, voxel_size_xy: tuple[float, float] | float, stacks: StackData, image_stack_indexes: np.ndarray, material_names: list[str]) -> VoxelGeometry:
    cumulative_thicknesses = np.cumsum(layer_thicknesses)
    cumulative_stack_heights = np.cumsum(stacks.thicknesses, axis=1)

    stack_pixel_map = np.full((cumulative_stack_heights.shape[0], len(cumulative_thicknesses)), -1)
    for stack_index in range(cumulative_stack_heights.shape[0]):
        stack_pixel_map[stack_index] = _pixel_voxel_stack_height_matching(cumulative_thicknesses, cumulative_stack_heights[stack_index], stacks.material_nums[stack_index])

    if isinstance(voxel_size_xy, float):
        voxel_size_xy = (voxel_size_xy, voxel_size_xy)
    
    material_map = np.full((image_stack_indexes.shape[0], image_stack_indexes.shape[1], len(layer_thicknesses)), -1)
    #loop over x, y, z
    for x in range(material_map.shape[0]):
        for y in range(material_map.shape[1]):
            stack_index = image_stack_indexes[x,y]
            material_map[x,y] = stack_pixel_map[stack_index]
    
    assert not np.any(material_map == -1), "Mapping color to material stacks failed. Unassigned voxel position detected!"

    return VoxelGeometry(materials=material_map, layer_thickness=layer_thicknesses, voxel_size_xy=voxel_size_xy, material_names=material_names)
