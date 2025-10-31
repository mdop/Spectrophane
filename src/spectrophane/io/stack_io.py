from typing import Sequence, Dict
import numpy as np
from dataclasses import dataclass

@dataclass
class StackData():
    """Dataclass that contains all relevant data for material stack representation"""
    material_list: Sequence[Dict]
    material_nums: np.ndarray
    thicknesses: np.ndarray
    stack_counts: np.ndarray

def stack_json_to_array(material_list: Sequence[Dict], stack_data: Sequence[Dict]) -> StackData:
    """Takes a material list and a list of material names + thicknesses and morphs them into a padded numpy material and thickness, and a stacklength array."""
    max_layers = max(len(stack) for stack in stack_data)
    n_stacks = len(stack_data)

    
    material_nums = np.zeros((n_stacks, max_layers), dtype=np.uint16)
    thicknesses = np.zeros((n_stacks, max_layers))
    lengths = np.zeros(n_stacks, dtype=np.uint8)

    material_id_list = [material["id"] for material in material_list]
    for i, stack in enumerate(stack_data):
        for j, layer_data in enumerate(stack):
            material_nums[i, j] = material_id_list.index(layer_data["id"])
            thicknesses[i, j] = layer_data["d"]
        lengths[i] = len(stack)

    stack_data = StackData(material_list, material_nums, thicknesses, lengths)
    return stack_data