import pytest
from spectrophane.pipeline.lithophane_factories import (
    generate_homogeneous_topology_block,
    generate_stack_rules_homogeneous_blocks,
    generate_stack_rules_single_homogeneous_block,
    generate_evaluator,
    generate_inverter,
    generate_lithophane_solid_builder,
    generate_lithophane_export_backend
)
from spectrophane.core.dataclasses import (
    MaterialParams, EvaluatorSpec, SpectrumBlock, LightSources, Observers, TopologyBlock, StackTopologyRules, WavelengthAxis, SpectrumBlock, InverterSpec, LithophaneConfig, StackCandidates
)
from spectrophane.inverse.stack_generation import StackGenerator
from spectrophane.evaluation.evaluator import Evaluator
from spectrophane.inverse.inverter import LUTInverter
from spectrophane.lithophane.solid_generation import SolidBuilder, PerVoxelBoxBuilder
from spectrophane.lithophane.export import SolidBackend, STLTessellationBackend
import numpy as np
from pathlib import PosixPath


def test_generate_homogeneous_topology_block():
    block = generate_homogeneous_topology_block(layer_thickness=10, layer_count=5, material_count=2)
    assert isinstance(block, TopologyBlock)
    assert block.max_layers_per_material.shape == (2,)
    assert block.thicknesses.shape == (5,)

def test_generate_stack_rules_homogeneous_blocks():
    blocks = [TopologyBlock(max_layers_per_material=np.array([3, 2]), thicknesses=np.array([10, 20, 30, 40, 50]))]
    rules = generate_stack_rules_homogeneous_blocks(blocks=blocks, ordered=False)
    assert isinstance(rules, StackTopologyRules)
    assert rules.blocks == blocks
    assert rules.ordered is False

def test_generate_stack_rules_single_homogeneous_block():
    rules = generate_stack_rules_single_homogeneous_block(layer_thickness=10, layer_count=5, material_count=2, ordered=False)
    assert isinstance(rules, StackTopologyRules)
    assert len(rules.blocks) == 1
    assert rules.ordered is False


@pytest.fixture
def material_params():
    return MaterialParams(wl_start=400, wl_step=5, absorption_coeff=np.random.rand(2, 100), scattering_coeff=np.random.rand(2, 100))

@pytest.fixture
def illuminators():
    return LightSources(["D65", "b"], [SpectrumBlock(350, 10,np.random.rand(1,100)), SpectrumBlock(350, 10,np.random.rand(1,100))])

@pytest.fixture
def observers():
    return Observers(["CIE1931", "bb"], [SpectrumBlock(350, 10,np.random.rand(1,3,100))]*2)

@pytest.fixture
def evaluator(material_params, illuminators, observers):
    config = EvaluatorSpec()
    evaluator = generate_evaluator(material_params, illuminators, observers, config)
    return evaluator

def test_generate_evaluator(evaluator):
    assert isinstance(evaluator, Evaluator)

@pytest.fixture
def stack_generator():
    rules = generate_stack_rules_single_homogeneous_block(layer_thickness=0.1, layer_count=5, material_count=2, ordered=False)
    return StackGenerator(rules)

@pytest.fixture()
def inverter(evaluator, stack_generator):
    config = InverterSpec(algorithm="lut", lut_compression_factor=8)
    inverter = generate_inverter(stack_generator=stack_generator, evaluator=evaluator, config=config)
    return inverter

def test_generate_inverter(inverter):
    assert isinstance(inverter, LUTInverter)

@pytest.fixture
def lithophane_config():
    return LithophaneConfig(["mat_a", "mat_b"], builder_algorithm="voxel", export_backend_format="stl")

def test_generate_lithophane_solid_builder(lithophane_config):
    builder = generate_lithophane_solid_builder(lithophane_config)
    assert isinstance(builder, PerVoxelBoxBuilder)

def test_generate_lithophane_export_backend(lithophane_config):
    base_path = PosixPath("/some/path")
    backend = generate_lithophane_export_backend(base_path, lithophane_config)
    assert isinstance(backend, STLTessellationBackend)