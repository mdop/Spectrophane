from abc import ABC, abstractmethod
import numpy as np
from math import ceil

from spectrophane.core.dataclasses import StackCandidates
from spectrophane.color.conversions import decode_rgb, linrgb_to_xyz
from spectrophane.evaluation.evaluator import Evaluator
from spectrophane.inverse.stack_generation import StackGenerator

class Inverter(ABC):

    @abstractmethod
    def invert_rgb(self, rgb: np.ndarray) -> StackCandidates:
        ...


class LUTInverter(Inverter):
    """Inverter that initially runs stack combinations to create a lookup table for RGB values"""

    def __init__(self, lut_compression_factor: int, stack_generator: StackGenerator, evaluator: Evaluator):
        self._stack_generator = stack_generator
        self._compression = lut_compression_factor
        self._eval = evaluator

        self._lut = None
        self._stacks = None
        self._steps = ceil(256.0 / self._compression)

        self._generate_lut()

    def _generate_xyz_space(self) -> np.ndarray:
        """Generates xyz values for the compressed rgb space voxel center points"""
        rgb_space = np.indices((self._steps, self._steps, self._steps), dtype=np.float32)
        rgb_space = rgb_space.transpose(1, 2, 3, 0)

        # voxel center, normalized to [0, 1]
        rgb_space = np.clip((rgb_space + 0.5) * self._compression / 255.0, 0.0, 1.0)

        xyz_list = linrgb_to_xyz(decode_rgb(rgb_space.reshape(-1, 3)))
        return xyz_list.reshape(self._steps, self._steps, self._steps, 3)  # shape: (steps, steps, steps, 3)

    def _generate_lut(self):
        """Generates lookup table for color requests"""
        self._stacks = self._stack_generator.generate("complete")
        stack_xyz = self._eval.evaluate(stacks=self._stacks)
        xyz_space = self._generate_xyz_space()

        # Flatten voxel grid for vectorized distance computation
        xyz_flat = xyz_space.reshape(-1, 3)  # (n_voxels, 3)

        # Compute squared L2 distances: (n_voxels, n_stacks)
        diff = xyz_flat[:, None, :] - stack_xyz[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)

        # Best stack per voxel
        best_stack_idx = np.argmin(dist2, axis=1)

        # Reshape back to LUT grid
        self._lut = best_stack_idx.reshape(self._steps, self._steps, self._steps)

    def invert_rgb(self, rgb: np.ndarray) -> StackCandidates:
        """
        Invert RGB using LUT.
        """
        # Map to compressed LUT index
        idx = np.floor(rgb / self._compression).astype(int)
        idx = np.clip(idx, 0, self._steps - 1)

        stack_idx = self._lut[idx[0], idx[1], idx[2]]
        return self._stacks.take(stack_idx)
