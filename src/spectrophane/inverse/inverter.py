from abc import ABC, abstractmethod
import numpy as np
from math import ceil

from spectrophane.core.dataclasses import StackCandidates
from spectrophane.color.conversions import decode_rgb, encode_rgb, linrgb_to_xyz, xyz_to_linrgb, xyz_to_lab
from spectrophane.evaluation.evaluator import Evaluator
from spectrophane.inverse.stack_generation import StackGenerator

class Inverter(ABC):
    preferred_color_space = "xyz"

    @abstractmethod
    def invert_color(self, colors: np.ndarray, max_suggested_stacks: int, color_space: str | None) -> tuple[StackCandidates, np.ndarray, np.ndarray]:
        """Returns StackCandidates, request index array, and score array for a requested color batch."""
        ...


class LUTInverter(Inverter):
    """Inverter that initially runs stack combinations to create a lookup table for RGB values"""
    preferred_color_space = "rgb"

    def __init__(self, lut_compression_factor: int, stack_generator: StackGenerator, evaluator: Evaluator):
        self._stack_generator = stack_generator
        self._compression = lut_compression_factor
        self._eval = evaluator

        self._lut = None
        self._lut_score = None
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
        stack_xyz = self._eval.evaluate(stacks=self._stacks).astype(np.float32)
        white_point = self._eval.get_whitepoint()
        stack_lab = xyz_to_lab(stack_xyz, white_point)
        self._stacks.rgb = np.rint(encode_rgb(xyz_to_linrgb(stack_xyz))*255)
        xyz_space = self._generate_xyz_space()
        lab_space = xyz_to_lab(xyz_space, white_point)

        #to find distances use ∣∣a−b∣∣^2=∣∣a∣∣^2+∣∣b∣∣^2−2a⋅b
        lab_flat = lab_space.reshape(-1, 3)  # (n_voxels, 3)
        a2 = np.sum(lab_flat**2, axis=1, keepdims=True)      # (n_voxels, 1)
        b2 = np.sum(stack_lab**2, axis=1, keepdims=True).T   # (1, n_stacks)

        # Compute pairwise squared distances
        dist2 = a2 + b2 - 2 * lab_flat @ stack_lab.T         # (n_voxels, n_stacks)
        # Flatten voxel grid , calculate L2 distance

        # Best stack per voxel
        best_stack_idx = np.argmin(dist2, axis=1) #(n_voxels,)
        #best_stack_score = 1 - dist2[:,best_stack_idx] / np.sqrt(3) # normalize to 0..1 with 1 being best fit #doing this somehow crashes due to RAM usage
        scores = np.zeros_like(best_stack_idx, dtype=np.float16)
        for i in range(len(scores)):
            scores[i] = dist2[i, best_stack_idx[i]]/np.sqrt(3)
        scores = scores/np.max(scores)

        # Reshape back to LUT grid
        self._lut       = best_stack_idx.reshape(self._steps, self._steps, self._steps)
        self._lut_score = scores.reshape(self._steps, self._steps, self._steps)

    def invert_color(self, colors: np.ndarray, max_suggested_stacks: int = 1, color_space: str | None = None) -> tuple[StackCandidates, np.ndarray, np.ndarray]:
        """
        Invert RGB using LUT. Returns StackCandidates, request index array, and score array. Can only return 1 stack per color.
        """
        # Map to compressed LUT index
        if color_space is None:
            color_space = self.preferred_color_space
        
        if color_space == "rgb":
            encoded_rgb = colors
        elif color_space == "xyz":
            encoded_rgb = encode_rgb(xyz_to_linrgb(colors))
        lut_index = np.floor(encoded_rgb / self._compression).astype(int)
        lut_index = np.clip(lut_index, 0, self._steps - 1)

        stack_idx = self._lut[lut_index[:,0], lut_index[:,1], lut_index[:,2]]
        return self._stacks.take(stack_idx), np.indices([len(stack_idx)]).ravel(), self._lut_score[lut_index[:,0], lut_index[:,1], lut_index[:,2]]
