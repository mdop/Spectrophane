from typing import Dict, Tuple, Sequence
from numbers import Number
import numpy as np

from spectrophane.core.dataclasses import TrainingRefSpectraData, WavelengthAxis, SpectrumBlock
from spectrophane.training.ingest_stacks import stack_json_to_array, StackData


def process_spectrum_list(spectrum_data_list: Sequence[Dict], materials: Sequence[Dict]) -> Tuple[StackData, list[SpectrumBlock], list[SpectrumBlock]]:
    """Takes data from a list of spectrum data from the source file and returns output grade stack and spectrum data. If given an empty spectrum list will return Null, Null"""
    output_spectra = []
    background_spectra = []
    if len(spectrum_data_list) == 0:
        return StackData(np.array([]), np.array([])), output_spectra, background_spectra
    stack_data_list = [entry["stack"] for entry in spectrum_data_list]
    stack_output = stack_json_to_array(materials, stack_data_list)
    for spectrum_dict in spectrum_data_list:
        output_spectra.append(SpectrumBlock(start=spectrum_dict["wl_start"], step=spectrum_dict["wl_step"], values=np.array([spectrum_dict["value"]], dtype=np.float64)))
        #reflectance of the background. "b" for ideally black, "w" for ideally white, or a spectrum if known. Defaults to black.
        background_spectra.append(SpectrumBlock(start=spectrum_dict["wl_start"],
                                                step=spectrum_dict["wl_step"],
                                                values=np.zeros_like([spectrum_dict["value"]], dtype=np.float64)))
        if "background" in spectrum_dict:
            if spectrum_dict["background"] == "w":
                background_spectra[-1].values=np.ones_like([spectrum_dict["value"]], dtype=np.float64)
            elif isinstance(spectrum_dict["background"], dict):
                background_spectra[-1].values=np.array([spectrum_dict["background"]["value"]], dtype=np.float64)

    return stack_output, output_spectra, background_spectra

def prepare_spectrum_data(input_data) -> TrainingRefSpectraData:
    """Takes the raw input data file content and parses data, transforms spectra to common denominator, and returns a harmonized spectral dataset and associated stacks. If no data are found returns None"""
    spectra_dict = input_data.get("spectra", {})
    materials = input_data.get("materials", {})

    #transform and compile spectrum data
    transmission_stacks, transmission_spectra, _ = process_spectrum_list(spectra_dict.get("transmission", {}), materials)
    reflection_stacks, reflection_spectra, reflection_backgrounds = process_spectrum_list(spectra_dict.get("reflection", {}), materials)
    wavelength_axes = [spectrum.axis for spectrum in transmission_spectra] + [spectrum.axis for spectrum in reflection_spectra] + [spectrum.axis for spectrum in reflection_backgrounds]
    common_axis = WavelengthAxis.common(wavelength_axes)

    transmission_spectra = SpectrumBlock.merge_resample_spectra(transmission_spectra, common_axis)
    reflection_spectra = SpectrumBlock.merge_resample_spectra(reflection_spectra, common_axis)
    reflection_backgrounds = SpectrumBlock.merge_resample_spectra(reflection_backgrounds, common_axis)
    return TrainingRefSpectraData(transmission_stacks, transmission_spectra.values, 
                                  reflection_stacks, reflection_spectra.values, reflection_backgrounds.values,
                                  common_axis.start, common_axis.step)