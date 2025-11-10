import numpy as np
from numbers import Number
from typing import Sequence, Tuple
from dataclasses import dataclass

from spectrophane.io.material_spectrum_processing import reshape_spectrum

@dataclass
class LightSources:
    names: Tuple[str]
    spectra: np.ndarray

def parse_light_sources(config_data: dict, min_wavelength: Number, step_wavelength: Number, spectrum_length: int) -> LightSources:
    """Returns a tuple of light source names and a numpy array containing harmonized spectra as specified in input data"""
    if "light_sources" not in config_data:
        return LightSources((), np.array([]))
    output_arr = np.zeros((len(config_data["light_sources"]), spectrum_length))
    source_names = tuple(source["id"] for source in config_data["light_sources"])
    for i, source in enumerate(config_data["light_sources"]):
        output_arr[i, :] = reshape_spectrum(source["wl_start"], source["wl_step"], source["value"], min_wavelength, step_wavelength, spectrum_length)
    return LightSources(source_names, output_arr)

@dataclass
class Observers:
    names: Tuple[str]
    spectra: np.ndarray

def parse_observers(config_data: dict, min_wavelength: Number, step_wavelength: Number, spectrum_length: int) -> Observers:
    """Returns a tuple of observer names and a numpy array containing harmonized spectra as specified in input data"""
    if "observer" not in config_data:
        return Observers((), np.array([]))
    output_arr = np.zeros((len(config_data["observer"]), 3, spectrum_length))
    source_names = tuple(source["id"] for source in config_data["observer"])
    for i, observer in enumerate(config_data["observer"]):
        output_arr[i, 0, :] = reshape_spectrum(observer["wl_start"], observer["wl_step"], observer["value"][0], min_wavelength, step_wavelength, spectrum_length)
        output_arr[i, 1, :] = reshape_spectrum(observer["wl_start"], observer["wl_step"], observer["value"][1], min_wavelength, step_wavelength, spectrum_length)
        output_arr[i, 2, :] = reshape_spectrum(observer["wl_start"], observer["wl_step"], observer["value"][2], min_wavelength, step_wavelength, spectrum_length)
    return Observers(source_names, output_arr)
