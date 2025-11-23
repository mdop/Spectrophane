from typing import Tuple, Sequence, Dict
import numpy as np
from numbers import Number
from dataclasses import dataclass

@dataclass
class LightSources:
    names: Tuple[str]
    spectra: np.ndarray

@dataclass
class Observers:
    names: Tuple[str]
    spectra: np.ndarray

@dataclass
class StackData():
    """Dataclass that contains all relevant data for material stack representation"""
    material_list: Sequence[Dict]
    material_nums: np.ndarray
    thicknesses: np.ndarray
    stack_counts: np.ndarray

@dataclass
class TrainingRefSpectraData:
    transmission_stacks: StackData
    transmission_spectra: np.ndarray
    reflection_stacks: StackData
    reflection_spectra: np.ndarray
    reflection_background: np.ndarray
    min_wavelength: Number
    step_wavelength: Number
    fallback_spectrumlength: Number

@dataclass
class TrainingRefImageData():
    transmission_stacks: StackData
    transmission_xyz: np.ndarray
    transmission_light_source_indexes: np.ndarray