import numpy as np
from typing import Sequence
from pathlib import PosixPath

from spectrophane.core.dataclasses import MaterialParams, EvaluatorSpec, LightSources, Observers, TopologyBlock, StackTopologyRules, WavelengthAxis, SpectrumBlock, InverterSpec, LithophaneConfig, StackCandidates
from spectrophane.core.jax_utils import register_with_jax
from spectrophane.evaluation.evaluator import Evaluator
from spectrophane.evaluation.cache import ForwardCache
from spectrophane.inverse.stack_generation import StackGenerator
from spectrophane.inverse.inverter import Inverter, LUTInverter
from spectrophane.lithophane.solid_generation import SolidBuilder, PerVoxelBoxBuilder
from spectrophane.lithophane.export import SolidBackend, STLTessellationBackend

def generate_homogeneous_topology_block(layer_thickness: int, layer_count: int, material_count: int, material_layer_count_limits: np.ndarray | None = None):
    if material_layer_count_limits is None:
        material_layer_count_limits = np.array([layer_count]*material_count)
    if len(material_layer_count_limits) != material_count:
        raise ValueError(f"Inconsistent input data to create a topology block: material count is {material_count}, but layer limits has length {len(material_layer_count_limits)}!")
    
    return TopologyBlock(max_layers_per_material=material_layer_count_limits, thicknesses=np.array([layer_thickness] * layer_count))

def generate_stack_rules_homogeneous_blocks(blocks: Sequence[TopologyBlock], ordered: bool = False) -> StackTopologyRules:
    return StackTopologyRules(blocks = blocks, ordered=ordered)

def generate_stack_rules_single_homogeneous_block(layer_thickness: int, layer_count: int, material_count: int, material_layer_count_limits: np.ndarray | None = None, ordered: bool = False) -> StackTopologyRules:
    block = generate_homogeneous_topology_block(layer_thickness=layer_thickness, layer_count=layer_count, material_count=material_count, material_layer_count_limits=material_layer_count_limits)
    return generate_stack_rules_homogeneous_blocks(blocks = [block], ordered=ordered)

def generate_evaluator(material_parameter: MaterialParams, illuminators: LightSources, observers: Observers, config: EvaluatorSpec, edge_stacks: StackCandidates | None = None) -> Evaluator:
    register_with_jax() #in case renormalizer is called
    #interpolate/expand backing, illuminator and observer to parameter wavelength axis
    param_axis = WavelengthAxis(start=material_parameter.wl_start, step=material_parameter.wl_step, length=material_parameter.absorption_coeff.shape[1])
    if isinstance(config.background, SpectrumBlock):
        target_background = config.background.resample(param_axis)
    else:
        target_background = SpectrumBlock(param_axis.start, param_axis.step, np.array([[config.background]*param_axis.length]))
    target_illuminator = illuminators.take_names(config.illuminator).spectra[0].resample(param_axis)
    target_observer = observers.take_names(config.observer).spectra[0].resample(param_axis)

    cache = ForwardCache(config.cache_backend)

    evaluator = Evaluator(theory=config.theory, 
                          view_geometry=config.view_geometry, 
                          cache=cache,
                          material_parameters=material_parameter,
                          illuminator=target_illuminator.values[0],
                          observer=target_observer.values[0],
                          step_wavelength=param_axis.step,
                          backing=target_background.values,
                          calc_backend=config.calc_backend,
                          edge_stacks=None)
    #edge stacks if renormalized
    if not edge_stacks is None:
        evaluator.set_renormalizer(edge_stacks)
    
    return evaluator

def generate_inverter(stack_generator: StackGenerator, evaluator: Evaluator, config: InverterSpec) -> Inverter:
    if config.algorithm == "lut":
        inverter = LUTInverter(lut_compression_factor=config.lut_compression_factor, stack_generator=stack_generator, evaluator=evaluator)
    else:
        raise ValueError(f"Unknown inverter algorithm {config.algorithm}")
    return inverter

def generate_lithophane_solid_builder(config: LithophaneConfig) -> SolidBuilder:
    if config.builder_algorithm == "voxel":
        builder = PerVoxelBoxBuilder()
    else:
        raise ValueError(f"Unknown lithophane solid builder algorithm {config.builder_algorithm}")
    return builder

def generate_lithophane_export_backend(base_path: str | PosixPath, config: LithophaneConfig) -> SolidBackend:
    if config.export_backend_format == "stl":
        bin = config.export_stl_type == "binary"
        backend = STLTessellationBackend(base_path=str(base_path),
                                         material_names=config.material_names,
                                         binary=bin)
    else:
        raise ValueError(f"Unknown lithophane export algorithm {config.export_backend_format}")
    
    return backend