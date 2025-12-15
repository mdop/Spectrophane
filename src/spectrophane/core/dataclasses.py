from typing import Tuple, Sequence, Dict, Optional
import numpy as np
from numbers import Number
from functools import partial
import jax
import jax.numpy as jnp
from dataclasses import dataclass

@dataclass
class LightSources:
    names: Tuple[str]
    spectra: np.ndarray

@dataclass
class Observers:
    names: Tuple[str]
    spectra: np.ndarray

@partial(jax.tree_util.register_dataclass)
@dataclass
class StackData():
    """Dataclass that contains all relevant data for material stack representation. Only valid in a context of a known material order"""
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


@partial(jax.tree_util.register_dataclass,
         data_fields=["absorption_coeff", "scattering_coeff"],
         meta_fields=["model_type"])
@dataclass
class MaterialParams:
    # Fundamental physical descriptors
    absorption_coeff: Optional[jnp.ndarray] = None  # [μ_a(λ)] or K(λ)
    scattering_coeff: Optional[jnp.ndarray] = None  # [μ_s(λ)] or S(λ)
    #refractive_index: Optional[jnp.ndarray] = None  # n(λ)
    #extinction_coeff: Optional[jnp.ndarray] = None  # k(λ)
    #anisotropy: Optional[float] = None              # g in Henyey–Greenstein phase fn
    
    ## Theory-specific corrections
    #saunderson_A: Optional[float] = None
    #saunderson_B: Optional[float] = None
    
    # Meta info
    model_type: Optional[str] = None  # "kubelka_munk", "saunderson", "monte_carlo"


@dataclass(frozen=True)
class SpectralBlock:
    wavelengths: np.ndarray
    values: np.ndarray
    material_id: str
    material_name: str
    plotcolor: str
    parameter: str  # e.g. "absorption", "scattering"