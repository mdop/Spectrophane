from pathlib import PosixPath, Path
from PIL import ImageFile
import json
import numpy as np

from spectrophane.core.dataclasses import MaterialParams, LightSources, Observers, EvaluatorSpec, InverterSpec, StackTopologyRules, LithophaneConfig, StackCandidates
from spectrophane.color.spectral_helper import parse_light_sources, parse_observers
from spectrophane.inverse.inverter import Inverter
from spectrophane.inverse.stack_generation import StackGenerator
from spectrophane.io.resources import get_resource_path
from spectrophane.lithophane.pipeline import generate_lithophane_from_image
from spectrophane.training.material_parameter import deserialize_parameter
from spectrophane.pipeline.lithophane_factories import generate_evaluator, generate_inverter, generate_lithophane_solid_builder, generate_lithophane_export_backend

def file_to_parameter(path: str | PosixPath, local_path: bool = True, material_filter: list[str] | None = None) -> tuple[list[dict], MaterialParams, dict]:
    """Returns a material data list as saved in the parameter file and de-serialized material parameter. Materials may be filtered by material_filter, unknown entries will be silently ignored."""
    if local_path:
        total_path = get_resource_path(path)
    else:
        total_path = Path(path)
    
    with open(total_path, "r") as file:
        data = json.load(file)
    
    parameter, metadata = deserialize_parameter(data)

    #filter
    available_names = [m["name"] for m in data["materials"]]
    if material_filter is None:
        indexes = np.array(range(len(available_names)))
    else:
        indexes = np.array([available_names.index(name) for name in material_filter if name in available_names])
    filter_parameter = parameter.take(indexes)
    filter_material_data = [data["materials"][i] for i in indexes]
    return filter_material_data, filter_parameter, metadata

def file_to_spectral_helper(path: str | PosixPath | None = None, local_path: bool = True) -> tuple[LightSources, Observers]:
    """Get Spectral data for light sources and observers from a configuration file (or CIE data). If CIE data only are used the path may be passed as None"""
    if path is None:
        data = {}
    else:
        if local_path:
            total_path = get_resource_path(path)
        else:
            total_path = Path(path)
        with open(total_path, "r") as file:
            data = json.load(file)
    
    light_sources = parse_light_sources(data)
    observers = parse_observers(data)
    return light_sources, observers

def parameter_to_inverter(material_parameter: MaterialParams, 
                          illuminators: LightSources, 
                          observers: Observers,
                          stack_generator: StackGenerator,
                          evaluator_config: EvaluatorSpec,
                          inverter_config: InverterSpec,
                          normalization_stacks: StackCandidates = None) -> Inverter:
    if not evaluator_config.normalize:
        edge_stacks = None #if config says no ignore argument
    elif normalization_stacks is None:
        edge_stacks = stack_generator.generate("single material")
    else:
        edge_stacks = normalization_stacks
    evaluator = generate_evaluator(material_parameter=material_parameter,
                                   illuminators=illuminators,
                                   observers=observers,
                                   config=evaluator_config,
                                   edge_stacks=edge_stacks)
    inverter = generate_inverter(stack_generator=stack_generator,
                                 evaluator=evaluator,
                                 config=inverter_config)
    return inverter

def image_to_lithophane(image: ImageFile,
                        output_base_path: str | PosixPath,
                        material_names: list[str],
                        inverter: Inverter,
                        stack_rules: StackTopologyRules,
                        config: LithophaneConfig) -> tuple[list[str], np.ndarray, np.ndarray]:
    builder = generate_lithophane_solid_builder(config)
    backend = generate_lithophane_export_backend(output_base_path, config)
    output_paths, calc_image, score_img = generate_lithophane_from_image(image=image, 
                                                                        resolution=config.resolution,
                                                                        inverter=inverter,
                                                                        layer_thicknesses=stack_rules.layer_thicknesses,
                                                                        voxel_size_xy=config.pixel_xy_dimension,
                                                                        material_names=material_names,
                                                                        builder=builder,
                                                                        export_backend=backend)
    return output_paths, calc_image, score_img