import pytest
import numpy as np
from spectrophane.inverse.stack_generation import StackGenerator, StackTopologyRules, TopologyBlock

def layer_2_block():
    block = TopologyBlock(
        allowed_materials=np.array([1, 2]),
        max_layers_per_allowed_material=np.array([2, 2]),
        thicknesses=np.array([1.0, 1.0]),
    )
    return block #3 combinations

def layer_3_block_restricted():
    block = TopologyBlock(
        allowed_materials=np.array([0, 3, 2]),
        max_layers_per_allowed_material=np.array([3, 1, 2]),
        thicknesses=np.array([2.0, 2.0, 2.0]),
    )
    return block #6 combinations

def layer_4_block():
    block = TopologyBlock(
        allowed_materials=np.array([0, 1, 2]),
        max_layers_per_allowed_material=np.array([4, 4, 4]),
        thicknesses=np.array([1.0, 1.0, 1.0, 1.0]),
    )
    return block #stars and bars, comb(material_num+layer_num-1, material_num-1)= 15 combinations


@pytest.fixture
def singlelayer_rule():
    rules = StackTopologyRules(
        material_indexes=np.array([0, 1, 2, 3]),
        blocks=[layer_2_block()],
        ordered=False
    )
    return rules

@pytest.fixture
def multilayer_rule():
    rules = StackTopologyRules(
        material_indexes=np.array([0, 1, 2, 3]),
        blocks=[layer_2_block(), layer_3_block_restricted(), layer_4_block()],
        ordered=False
    )
    return rules

def test_stack_generator_complete_unordered_mode(singlelayer_rule: StackTopologyRules):
    generator = StackGenerator(singlelayer_rule)
    result = generator.generate("complete")

    assert result.material_nums.shape[1] == 2  # 2 layers in the block
    assert result.thicknesses.shape[1] == 2
    assert len(result.material_nums) > 0
    assert np.all(result.score == 0)

def test_complete_unordered_block(multilayer_rule: StackTopologyRules):
    generator = StackGenerator(multilayer_rule)
    counts2, thickness2 = generator._complete_unordered_block(0)
    counts3, thickness3 = generator._complete_unordered_block(1)
    counts4, thickness4 = generator._complete_unordered_block(2)

    assert thickness2 == 1.0
    assert thickness3 == 2.0
    assert thickness4 == 1.0
    assert counts2.shape == (3, 2)
    assert counts3.shape == (6, 3)
    assert counts4.shape == (15, 3)
    assert np.all(counts2 >= 0)
    assert np.all(counts3 >= 0)
    assert np.all(counts4 >= 0)
    assert np.all(np.sum(counts2, axis=1) == 2)
    assert np.all(np.sum(counts3, axis=1) == 3)
    assert np.all(np.sum(counts4, axis=1) == 4)

def test_complete_unordered_block_invalid_thickness():
    # Setup test block with invalid thickness
    block = TopologyBlock(
        allowed_materials=np.array([0, 1]),
        max_layers_per_allowed_material=np.array([2, 2]),
        thicknesses=np.array([1.0, 2.0]),
    )
    rules = StackTopologyRules(
        material_indexes=np.array([0, 1]),
        blocks=[block],
        ordered=False
    )
    generator = StackGenerator(rules)

    # Test that this raises an error
    with pytest.raises(AssertionError):
        generator._complete_unordered_block(0)

def test_complete_unordered_stackset(singlelayer_rule: StackTopologyRules, multilayer_rule: StackTopologyRules):
    generator_single = StackGenerator(singlelayer_rule)
    generator_multi = StackGenerator(multilayer_rule)

    material_nums_single, thicknesses_single = generator_single._complete_unordered_stackset()
    material_nums_multi, thicknesses_multi = generator_multi._complete_unordered_stackset()

    # Check output
    assert material_nums_single.shape == thicknesses_single.shape
    assert material_nums_single.shape == (3,2)
    assert material_nums_multi.shape == thicknesses_multi.shape
    assert material_nums_multi.shape == (270,8) #cartesian product 15*6*3
