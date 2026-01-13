from typing import Tuple, Sequence, Dict, Optional
import numpy as np
from numbers import Number
from dataclasses import dataclass

@dataclass
class LightSources:
    """Dataclass representing light sources with their names and spectra.
    Defaults to NumPy arrays. If used with jax, register as a pytree and convert arrays at the boundary."""
    names: Tuple[str]
    spectra: np.ndarray

@dataclass
class Observers:
    """Dataclass representing observers with their names and spectra.
    Defaults to NumPy arrays. If used with jax, register as a pytree and convert arrays at the boundary."""
    names: Tuple[str]
    spectra: np.ndarray

@dataclass
class StackData:
    """
    Pure backend-neutral dataclass describing a material stack.
    Defaults to NumPy arrays. If used with jax register as a pytree and convert arrays at the boundary.
    """
    material_nums: np.ndarray
    thicknesses: np.ndarray
    def take(self, indices: np.ndarray) -> "StackData":
        return StackData(
            material_nums=self.material_nums[indices],
            thicknesses=self.thicknesses[indices],
        )

@dataclass
class StackCandidates(StackData):
    """
    Dataclass describing StackData with associated color. Score is in range 0..1
    """
    rgb: np.ndarray
    def take(self, indices: np.ndarray) -> "StackData":
        return StackCandidates(
            material_nums=self.material_nums[indices],
            thicknesses=self.thicknesses[indices],
            rgb=self.rgb[indices],
        )

@dataclass
class TrainingRefSpectraData:
    """Dataclass containing reference spectra and corresponding stack data for training.
    Defaults to NumPy arrays. If used with jax, register as a pytree and convert arrays at the boundary."""
    transmission_stacks: StackData
    transmission_spectra: np.ndarray
    reflection_stacks: StackData
    reflection_spectra: np.ndarray
    reflection_background: np.ndarray
    min_wavelength: Number
    step_wavelength: Number
    fallback_spectrumlength: Number

@dataclass
class TrainingRefImageData:
    """Dataclass containing reference image data and corresponding stack information for training.
    Defaults to NumPy arrays. If used with jax, register as a pytree and convert arrays at the boundary."""
    transmission_stacks: StackData
    transmission_xyz: np.ndarray
    transmission_light_source_indexes: np.ndarray


@dataclass
class MaterialParams:
    """
    Backend-neutral material parameter container.
    Defaults to NumPy arrays. If used with jax register as a pytree and convert arrays at the boundary.
    """
    absorption_coeff: Optional[np.ndarray] = None
    scattering_coeff: Optional[np.ndarray] = None
    model_type: Optional[str] = None  # "kubelka_munk", "saunderson", "monte_carlo"


@dataclass
class TopologyBlock:
    allowed_materials: np.ndarray #order may be used in unordered stack construction
    max_layers_per_allowed_material: np.ndarray
    thicknesses: np.ndarray

@dataclass
class StackTopologyRules:
    material_indexes: np.ndarray
    blocks: list[TopologyBlock]
    ordered: bool #Describes if order of layers matters. Decides max_layers for StackData shapes