import pytest
from spectrophane.training.ingest_stacks import stack_json_to_array, StackData
import numpy as np

def test_stack_json_to_array():
    material_list = [
        {"id": "Material1"},
        {"id": "Material2"}
    ]
    
    stack_data = [
        [{"id":"Material1", "d": 0.5}, {"id":"Material2", "d":1.0}],
        [{"id":"Material1", "d":0.3}]
    ]

    expected_output = StackData(
        np.array([[0, 1], [0, 0]]),
        np.array([[0.5, 1.0], [0.3, 0]]),
        np.array([2, 1])
    )

    output = stack_json_to_array(material_list, stack_data)
    
    assert np.array_equal(output.material_nums, expected_output.material_nums)
    assert np.array_equal(output.thicknesses, expected_output.thicknesses)
    assert np.array_equal(output.stack_counts, expected_output.stack_counts)


def test_stack_json_to_array_empty_stack():
    material_list = [
        {"id": 1, "name": "Material1"}
    ]
    
    stack_data = [
        []
    ]

    expected_output = StackData(
        np.array([[]], dtype=np.uint16),
        np.array([[]], dtype = np.uint8),
        np.array([0], dtype=np.uint16)
    )

    output = stack_json_to_array(material_list, stack_data)
    
    assert np.array_equal(output.material_nums, expected_output.material_nums)
    assert np.array_equal(output.thicknesses, expected_output.thicknesses)
    assert np.array_equal(output.stack_counts, expected_output.stack_counts)


def test_stack_json_to_array_single_material():
    material_list = [
        {"id": "Material1"}
    ]
    
    stack_data = [
        [{"id":"Material1", "d": 0.5}]
    ]

    expected_output = StackData(
        np.array([[0]]),
        np.array([[0.5]]),
        np.array([1])
    )

    output = stack_json_to_array(material_list, stack_data)
    
    assert np.array_equal(output.material_nums, expected_output.material_nums)
    assert np.array_equal(output.thicknesses, expected_output.thicknesses)
    assert np.array_equal(output.stack_counts, expected_output.stack_counts)