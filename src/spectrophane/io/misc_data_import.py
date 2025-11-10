import numpy as np
from numbers import Number
from typing import Sequence, Tuple
from dataclasses import dataclass

from spectrophane.io.material_spectrum_processing import reshape_spectrum

@dataclass
class LightSources:
    names: Sequence[str]
    spectra: np.ndarray

def parse_light_sources(config_data: dict, min_wavelength: Number, step_wavelength: Number, spectrum_length: int) -> Tuple[Sequence, np.ndarray]:
    """Returns a tuple of light source names and a numpy array containing harmonized spectra as specified in input data"""
    if "light_source" not in config_data:
        return LightSources((), np.array([]))
    output_arr = np.zeros((len(config_data["light_source"]), spectrum_length))
    source_names = (source["id"] for source in config_data["light_source"])
    for i, source in enumerate(config_data["light_source"]):
        output_arr[i, :] = reshape_spectrum(source["min_wavelength"], source["step_wavelength"], source["value"], min_wavelength, step_wavelength, spectrum_length)
    return LightSources(source_names, output_arr)

@dataclass
class Observers:
    names: Sequence[str]
    spectra: np.ndarray

def parse_observers(config_data: dict, min_wavelength: Number, step_wavelength: Number, spectrum_length: int) -> Tuple[Sequence, np.ndarray]:
    """Returns a tuple of observer names and a numpy array containing harmonized spectra as specified in input data"""
    if "observer" not in config_data:
        return Observers((), np.array([]))
    output_arr = np.zeros((len(config_data["observer"]), 3, spectrum_length))
    source_names = (source["id"] for source in config_data["observer"])
    for i, observer in enumerate(config_data["observer"]):
        output_arr[i, :] = reshape_spectrum(observer["min_wavelength"], observer["step_wavelength"], observer["value"], min_wavelength, step_wavelength, spectrum_length)
    return Observers(source_names, output_arr)
