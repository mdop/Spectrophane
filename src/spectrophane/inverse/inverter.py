from abc import ABC, abstractmethod
import numpy as np
from math import ceil

from spectrophane.core.dataclasses import StackCandidates
from spectrophane.color.conversions import decode_rgb, encode_rgb, linrgb_to_xyz, xyz_to_linrgb, color_distance
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

    def __init__(self, lut_compression_factor: int, stack_generator: StackGenerator, evaluator: Evaluator, chunk_size = 16384):
        self._stack_generator = stack_generator
        self._compression = lut_compression_factor
        self._eval = evaluator

        self._lut = None
        self._lut_score = None
        self._stacks = None
        self._steps = ceil(256.0 / self._compression)
        self._chunk_size = chunk_size

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
        #stack_lab = xyz_to_lab(stack_xyz, white_point)
        self._stacks.rgb = np.rint(encode_rgb(xyz_to_linrgb(stack_xyz))*255)
        xyz_space = self._generate_xyz_space().astype(np.float32)
        xyz_space_flat = xyz_space.reshape(-1, 3)
        #lab_space = xyz_to_lab(xyz_space, white_point)

        best_stack_idx = np.zeros(len(xyz_space_flat), dtype=np.int32)
        scores = np.zeros(len(xyz_space_flat), dtype=np.float32)
        #chunk calculation to balance RAM usage (may be a many GB) and calculation time
        for start in range(0, len(xyz_space_flat), self._chunk_size):
            end = min(start+self._chunk_size, len(xyz_space_flat))
            dist = color_distance(xyz_space_flat[start:end], stack_xyz, white=white_point) #shape (chunk_size,stack_size)
            idx = np.argmin(dist, axis=1)
            best_stack_idx[start:end] = idx
            scores[start:end] = dist[np.arange(len(idx)), idx]

        try:
            scores = scores/np.max(scores)
        except:
            pass    #in case of divide by 0

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
