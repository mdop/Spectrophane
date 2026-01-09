import numpy as np
from math import comb
from dataclasses import dataclass
from itertools import product
from functools import reduce
from operator import mul

from spectrophane.core.dataclasses import StackCandidates, TopologyBlock, StackTopologyRules

class StackGenerator: 
    """Generates stack combinations based on a general configuration in the constructor. Individual functions output stack combinations based on the function rules."""
    def __init__(self, rules: StackTopologyRules):
        self._rules = rules

    def generate(self, mode: str, **kwargs):
        """Generates Stack combinations based on initialized rules and a function specified mode. Available modes:
        - "complete": all possible combinations generated
        """
        if mode == "complete" and not self._rules.ordered:
            material_nums, thicknesses = self._complete_unordered_stackset()
        else:
            raise ValueError(f"Error generating stacks: Unknown mode {mode}")
        
        score = np.zeros(len(material_nums), dtype=np.float32)
        ref_id = np.zeros(len(material_nums), dtype=np.int32)
        return StackCandidates(material_nums=material_nums, thicknesses=thicknesses, request_id=ref_id, score=score)

    def _complete_unordered_stackset(self):
        """Assembles unordered stack data from all blocks to a complete array for stack data."""
        block_data = []
        total_layers = 0

        # enumerate each block independently
        for block in range(len(self._rules.blocks)):
            counts, thickness = self._complete_unordered_block(block)
            local = self._rules.blocks[block]

            Lb = len(local.allowed_materials)
            total_layers += Lb

            block_data.append({
                "block_comb": counts,                     # (C_b, M_b)
                "allowed": local.allowed_materials,   # (M_b,)
                "thickness": thickness,
                "layers": Lb
            })

        total_combination_count = reduce(mul, [entry["block_comb"].shape[0] for entry in block_data], 1)
        material_nums = np.zeros((total_combination_count, total_layers))
        thicknesses = np.zeros((total_combination_count, total_layers))

        for i, selection in enumerate(product(*(range(len(bd["block_comb"])) for bd in block_data))):
            offset = 0
            for block, combination_index in enumerate(selection):
                bd = block_data[block]
                material_nums[i, offset:offset+bd["layers"]] = bd["allowed"]
                thicknesses  [i, offset:offset+bd["layers"]] = bd["block_comb"][combination_index] * bd["thickness"]
                offset += bd["layers"]
        return material_nums, thicknesses


    def _complete_unordered_block(self, block: int):
        """Creates all allowed combinations of layer counts for the allowed materials of the specified stack block."""
        local = self._rules.blocks[block]

        allowed = local.allowed_materials
        max_layers = local.max_layers_per_allowed_material
        thicknesses = local.thicknesses

        thickness = thicknesses[0]
        assert np.all(thicknesses == thickness), "For unordered blocks all thicknesses in a block must be equal."

        # total layers in this block
        L = len(thicknesses)
        M = len(allowed)

        results = []
        current = np.zeros(M, dtype=int)

        # precompute suffix sums for pruning
        max_suffix = np.zeros(M + 1, dtype=int)
        for i in range(M - 1, -1, -1):
            max_suffix[i] = max_suffix[i + 1] + max_layers[i]

        def dfs(i, remaining):
            # prune if impossible to fill remaining layers
            if remaining < 0 or remaining > max_suffix[i]:
                return

            if i == M:
                if remaining == 0:
                    results.append(current.copy())
                return

            max_assign = min(max_layers[i], remaining)
            for n in range(max_assign + 1):
                current[i] = n
                dfs(i + 1, remaining - n)
            current[i] = 0  # cleanup

        dfs(0, L)

        return np.asarray(results, dtype=int), thickness
