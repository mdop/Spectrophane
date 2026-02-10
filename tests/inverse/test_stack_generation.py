import numpy as np
import pytest
from dataclasses import dataclass

from spectrophane.inverse.stack_generation import StackGenerator, StackTopologyRules, TopologyBlock

@pytest.fixture
def two_layer_block():
    block = TopologyBlock(max_layers_per_material=np.array([2,2,0,0]), thicknesses=np.array([0.1,0.1]))
    return block #3 combinations

@pytest.fixture
def three_layer_restricted_block():
    block = TopologyBlock(max_layers_per_material=np.array([3,1,2,0]), thicknesses=np.array([0.1,0.1,0.1]))
    return block #6 combinations

@pytest.fixture
def four_layer_block():
    block = TopologyBlock(max_layers_per_material=np.array([4,4,0,4]), thicknesses=np.array([0.2,0.2,0.2,0.2]))
    return block #comb(materials+layers-1, materials-1)=15 combinations

@pytest.fixture
def stack_rules(two_layer_block, three_layer_restricted_block, four_layer_block):
    return StackTopologyRules(material_indexes=np.array([0,1,2,3]), blocks=[two_layer_block, three_layer_restricted_block, four_layer_block], ordered=False) #270 total combinations

@pytest.fixture
def stack_generator(stack_rules):
    return StackGenerator(stack_rules)

@pytest.mark.parametrize("index, expected_thickness, expected_shape", [
    (0, 0.1, (3, 2)),
    (1, 0.1, (6, 3)),
    (2, 0.2, (15, 3)),
])
def test_single_block_unordered(stack_generator: StackGenerator, index, expected_thickness, expected_shape):
    combinations, thickness = stack_generator._complete_unordered_block(index)
    unique_combinations = np.unique(combinations, axis=0)

    assert thickness == expected_thickness
    assert combinations.shape == expected_shape
    assert combinations.shape == unique_combinations.shape  # no duplicates

def test_single_block_multithickness_unordered(stack_generator: StackGenerator):
    stack_generator._rules.blocks[0].thicknesses[0] = 0.1
    stack_generator._rules.blocks[0].thicknesses[1] = 1.0
    with pytest.raises(AssertionError):
        stack_generator._complete_unordered_block(0)

def test_complete_stack_generation_unordered(stack_generator: StackGenerator):
    material_nums, thicknesses = stack_generator._complete_unordered_stackset()
    #material_nums contain duplicates, thicknesses should not because they encode layer count
    unique_combinations = np.unique(thicknesses, axis=0)

    assert material_nums.shape == thicknesses.shape
    assert unique_combinations.shape == thicknesses.shape
    assert material_nums.shape == (270, 8)
    assert np.allclose(np.sum(thicknesses, axis=1), np.sum(thicknesses, axis=1)[0])

def test_generate_complete_mode(stack_generator: StackGenerator):
    candidates = stack_generator.generate(mode="complete")
    
    assert candidates.material_nums.shape == candidates.thicknesses.shape
    assert candidates.material_nums.shape == (270, 8)
    assert np.all(candidates.material_nums >= 0)
    assert np.all(candidates.material_nums <= 3)


def test_single_color_stacks(stack_generator: StackGenerator):
    material_nums, thicknesses = stack_generator._single_material_unordered_edge_stacks()

    assert material_nums.shape == thicknesses.shape
    assert material_nums.shape == (4,9)
    assert np.all(material_nums[:,0] == np.array([0,1,2,3]))
    assert np.all(material_nums[:,0] == np.moveaxis(material_nums, 0, 1))
    assert np.all(thicknesses[0] == np.array([0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.2]))
    assert np.all(thicknesses == thicknesses[0])