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

def transform_image(image: ImageFile, resolution: tuple[int, int], 
                    inverter: Inverter, 
                    layer_thicknesses: np.ndarray, voxel_size_xy: tuple[float, float], 
                    material_names: list[str],
                    output_path: str, model_config: dict = {},
                    voxel_pool_algorithm: str = "single",
                    ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Complete lithophane generation pipeline from image to model file. Returns a list of strings with paths to the files, the expected image and a scoremap.
    Resolution should be (width, height), but return arrays are (height, width)!
    Output path should contain the output format. If multiple files are produced the material name is inserted with _ before the extension."""
    #process image data to voxel map
    image_array = format_image(image, resolution)
    if isinstance(inverter, LUTInverter):
        convert_xyz = False
    else:
        convert_xyz = True
    unique_stacks, image_unique_color_indexes, calc_score_arr = image_to_stackmap(image_array, inverter, convert_xyz)
    voxelmap = stackmap_to_voxelmap(layer_thicknesses, voxel_size_xy, unique_stacks, image_unique_color_indexes, material_names)

    #generate 3D model and save
    if voxel_pool_algorithm == "single":
        builder = PerVoxelBoxBuilder()
    else:
        raise NotImplementedError(f"Unknown voxel pooling algorithm {voxel_pool_algorithm}.")
    
    filename, file_extension = os.path.splitext(output_path)
    if file_extension == ".stl":
        binary = model_config.get("binary", True)
        export_backend = STLTessellationBackend(filename, material_names, binary)
    else:
        raise NotImplementedError(f"Unknown model format {file_extension}.")
    
    output_paths = export_geometry(voxelmap, builder, export_backend)

    #output image array preparation
    expected_rgb_arr = unique_stacks.rgb[image_unique_color_indexes].astype(np.uint8)
    expected_rgb_img = Image.fromarray(expected_rgb_arr, mode="RGB")
    calc_score_arr = (calc_score_arr * 255).astype(np.uint8)
    calc_score_img = Image.fromarray(calc_score_arr, mode="L")
    return output_paths, expected_rgb_img, calc_score_img