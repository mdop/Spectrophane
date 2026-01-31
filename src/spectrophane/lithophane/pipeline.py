import numpy as np
from PIL import ImageFile, Image
import os

from spectrophane.core.dataclasses import VoxelGeometry
from spectrophane.lithophane.ingest_image import format_image, image_to_stackmap, stackmap_to_voxelmap
from spectrophane.lithophane.solid_generation import SolidBuilder, PerVoxelBoxBuilder
from spectrophane.lithophane.export import SolidBackend, STLTessellationBackend
from spectrophane.inverse.inverter import Inverter, LUTInverter

def export_geometry(geometry: VoxelGeometry, builder: SolidBuilder, export_backend: SolidBackend) -> list[str]:
    """Orchestrator for voxel map -> 3D model file"""
    for material_id in np.unique(geometry.materials):
        export_backend.begin(material_id)
        for solid in builder.solids_for_material(geometry, material_id):
            export_backend.add(solid)

    return export_backend.end()

def generate_lithophane_from_image(image: ImageFile, resolution: tuple[int, int], 
                                    inverter: Inverter, 
                                    layer_thicknesses: np.ndarray, voxel_size_xy: tuple[float, float], 
                                    material_names: list[str],
                                    builder: SolidBuilder,
                                    export_backend: SolidBackend,
                                    ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Complete lithophane generation pipeline from image to model file. Returns a list of strings with paths to the files, the expected image and a scoremap.
    Resolution should be (width, height), but return arrays are (height, width)!
    Output path should contain the output format. If multiple files are produced the material name is inserted with _ before the extension."""
    #process image data to voxel map
    image_array = format_image(image, resolution)
    convert_xyz = inverter.preferred_color_space == "xyz"
    unique_stacks, image_unique_color_indexes, calc_score_arr = image_to_stackmap(image_array, inverter, convert_xyz)
    voxelmap = stackmap_to_voxelmap(layer_thicknesses, voxel_size_xy, unique_stacks, image_unique_color_indexes, material_names)

    #turn voxel map to lithophane models
    output_paths = export_geometry(voxelmap, builder, export_backend)

    #output image array preparation
    expected_rgb_arr = unique_stacks.rgb[image_unique_color_indexes].astype(np.uint8)
    expected_rgb_img = Image.fromarray(expected_rgb_arr, mode="RGB")
    calc_score_arr = (calc_score_arr * 255).astype(np.uint8)
    calc_score_img = Image.fromarray(calc_score_arr, mode="L")
    return output_paths, expected_rgb_img, calc_score_img