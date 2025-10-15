import unittest
from spectrophane.io.stack_io import stack_json_to_array, StackData
import numpy as np

class TestStackIO(unittest.TestCase):

    def test_stack_json_to_array(self):
        material_list = [
            {"id": "Material1"},
            {"id": "Material2"}
        ]
        
        stack_data = [
            [("Material1", 0.5), ("Material2", 1.0)],
            [("Material1", 0.3)]
        ]

        expected_output = StackData(
            material_list,
            np.array([[0, 1], [0, 0]]),
            np.array([[0.5, 1.0], [0.3, 0]]),
            np.array([2, 1])
        )

        output = stack_json_to_array(material_list, stack_data)
        
        self.assertEqual(output.material_list, expected_output.material_list)
        self.assertTrue(np.array_equal(output.material_nums, expected_output.material_nums))
        self.assertTrue(np.array_equal(output.thicknesses, expected_output.thicknesses))
        self.assertTrue(np.array_equal(output.stack_count, expected_output.stack_count))

    def test_stack_json_to_array_empty_stack(self):
        material_list = [
            {"id": 1, "name": "Material1"}
        ]
        
        stack_data = [
            []
        ]

        expected_output = StackData(
            material_list,
            np.array([[]], dtype=np.uint16),
            np.array([[]], dtype = np.uint8),
            np.array([0], dtype=np.uint16)
        )

        output = stack_json_to_array(material_list, stack_data)
        
        self.assertEqual(output.material_list, expected_output.material_list)
        self.assertTrue(np.array_equal(output.material_nums, expected_output.material_nums))
        self.assertTrue(np.array_equal(output.thicknesses, expected_output.thicknesses))
        self.assertTrue(np.array_equal(output.stack_count, expected_output.stack_count))

    def test_stack_json_to_array_single_material(self):
        material_list = [
            {"id": "Material1"}
        ]
        
        stack_data = [
            [("Material1", 0.5)]
        ]

        expected_output = StackData(
            material_list,
            np.array([[0]]),
            np.array([[0.5]]),
            np.array([1])
        )

        output = stack_json_to_array(material_list, stack_data)
        
        self.assertEqual(output.material_list, expected_output.material_list)
        self.assertTrue(np.array_equal(output.material_nums, expected_output.material_nums))
        self.assertTrue(np.array_equal(output.thicknesses, expected_output.thicknesses))
        self.assertTrue(np.array_equal(output.stack_count, expected_output.stack_count))

if __name__ == '__main__':
    unittest.main()