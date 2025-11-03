import json
from typing import Dict, Tuple, Sequence
from numbers import Number
import numpy as np
from dataclasses import dataclass

from spectrophane.io.data_io import get_resource_path
from spectrophane.io.stack_io import stack_json_to_array, StackData

@dataclass
class TrainingRefSpectraData:
    transmission_stacks: StackData
    transmission_spectra: np.ndarray
    reflection_stacks: StackData
    reflection_spectra: np.ndarray
    min_wavelength: Number
    step_wavelength: Number
    fallback_spectrumlength: Number


def get_common_wavelength_space(min_wavelengths: Sequence[Number], step_wavelengths: Sequence[Number], spectrum_lengths: Sequence[int]) -> Tuple[Number, Number, int] | Tuple[None, None, None]:
    """Takes a set of spectra and finds a suitable common wavelength space for analysis expressed as (min_wavelength, step_wavelength, values)"""
    if len(min_wavelengths) != len(step_wavelengths) or len(min_wavelengths) != len(spectrum_lengths):
        raise IndexError(f"get_common_wavelength_space input values do not have the same length. min_wavelength len {len(min_wavelengths)}, step_wavelength len {len(step_wavelengths)}, spectrum_lengths len {len(spectrum_lengths)}")
    end_wavelengths = [0] * len(min_wavelengths)
    for i in range(len(min_wavelengths)):
        end_wavelengths[i] = min_wavelengths[i] + (spectrum_lengths[i] - 1) * step_wavelengths[i]
    #clip wavelength range and choose smallest step size
    com_min = max(min_wavelengths)
    com_step = min(step_wavelengths)
    com_end = min(end_wavelengths)
    com_length = int((com_end-com_min)/com_step) + 1
    return com_min, com_step, com_length

def reshape_spectrum(old_min_wavelength: Number, old_step_wavelength: Number, old_values: Sequence[Number], new_min_wavelength: Number, new_step_wavelength: Number, new_length: int) -> np.ndarray:
    """Takes a raw spectrum as presented in the input data file and interpolates it to fit a different wavelength space. Uses linear interpolation"""
    old_end_wavelength = old_min_wavelength+old_step_wavelength*(len(old_values)-1)
    wavelengths_old = np.linspace(old_min_wavelength, old_end_wavelength, num=len(old_values))
    new_end_wavelength = new_min_wavelength+new_step_wavelength*(new_length-1)
    wavelengths_new = np.linspace(new_min_wavelength, new_end_wavelength, num=new_length)
    return np.interp(wavelengths_new, wavelengths_old, old_values)

def process_spectrum_list(spectrum_data_list: Sequence[Dict], materials: Sequence[Dict], min_wavelength: Number, step_wavelength: Number, spectrum_length: int) -> Tuple[StackData, np.ndarray] | Tuple[None, None]:
    """Takes data from a list of spectrum data from the source file and returns output grade stack and spectrum data. If given an empty spectrum list will return Null, Null"""
    if len(spectrum_data_list) == 0:
        return None, None
    output_spectra = np.zeros((len(spectrum_data_list), spectrum_length),dtype=np.float64)
    stack_data_list = [entry["stack"] for entry in spectrum_data_list]
    stack_output = stack_json_to_array(materials, stack_data_list)
    for i, spectrum_dict in enumerate(spectrum_data_list):
        spectrum_output = reshape_spectrum(spectrum_dict["wl_start"], spectrum_dict["wl_step"], spectrum_dict["value"], min_wavelength, step_wavelength, spectrum_length)
        output_spectra[i, :] = spectrum_output
    return stack_output, output_spectra

def prepare_spectrum_data(input_data) -> TrainingRefSpectraData | None:
    """Takes the raw input data file content and parses data, transforms spectra to common denominator, and returns a harmonized spectral dataset and associated stacks. If no data are found returns None"""
    spectra_dict = input_data.get("spectra", {})
    materials = input_data.get("materials", {})

    #find common wavelength space
    min_wavelengths = []
    step_wavelengths = []
    spectrum_lengths = []
    for spectrum_type in spectra_dict.values():
        for spectrum_entry in spectrum_type:
            min_wavelengths.append(spectrum_entry["wl_start"])
            step_wavelengths.append(spectrum_entry["wl_step"])
            spectrum_lengths.append(len(spectrum_entry["value"]))
    if len(min_wavelengths) == 0:
        return None
    com_min_wavelength, com_step_wavelength, com_spectrum_length = get_common_wavelength_space(min_wavelengths, step_wavelengths, spectrum_lengths)

    #transform and compile spectrum data
    transmission_stacks, transmission_spectra = process_spectrum_list(spectra_dict.get("transmission", {}), materials, com_min_wavelength, com_step_wavelength, com_spectrum_length)
    reflection_stacks, reflection_spectra = process_spectrum_list(spectra_dict.get("reflection", {}), materials, com_min_wavelength, com_step_wavelength, com_spectrum_length)
    return TrainingRefSpectraData(transmission_stacks, transmission_spectra, reflection_stacks, reflection_spectra, com_min_wavelength, com_step_wavelength, com_spectrum_length)