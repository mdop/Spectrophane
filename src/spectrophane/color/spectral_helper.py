import numpy as np
from numbers import Number
from typing import Tuple

from spectrophane.core.dataclasses import LightSources, Observers
from spectrophane.io.resources import get_resource_path, get_json_resource
from spectrophane.training.ingest_spectra import reshape_spectrum


def _import_CIE_light_sources(min_wavelength: Number, step_wavelength: Number, spectrum_length: int) -> Tuple[str]:
    """Imports CIE light sources for use in light source parsing."""
    CIE_metadata = get_json_resource("CIE/data.json")
    ids = [""] * len(CIE_metadata["light_sources"])
    intensities = np.zeros((len(CIE_metadata["light_sources"]), spectrum_length))
    for i, (id, filename) in enumerate(CIE_metadata["light_sources"].items()):
        ids[i] = id
        #parse and reshape intensities
        resource_path = get_resource_path("CIE/" + filename)
        if get_resource_path("CIE/" + filename) is None:
            raise FileNotFoundError(f"Could not find CIE/{filename}. It seems the installation script did not run properly. Need external data from CIE.")
        data = np.loadtxt(resource_path, delimiter=",", dtype=np.float32)
        raw_min_wavelength = data[0,0]
        raw_step_wavelength = data[1,0] - data[0,0]
        intensities[i] = reshape_spectrum(raw_min_wavelength, raw_step_wavelength, data[:,1], min_wavelength, step_wavelength, spectrum_length)
    return tuple(ids), intensities

def _import_CIE_observers(min_wavelength: Number, step_wavelength: Number, spectrum_length: int) -> Tuple[str]:
    """Imports CIE observer for use in light source parsing."""
    CIE_metadata = get_json_resource("CIE/data.json")
    ids = [""] * len(CIE_metadata["observers"])
    intensities = np.zeros((len(CIE_metadata["observers"]), 3, spectrum_length))
    for i, (id, filename) in enumerate(CIE_metadata["observers"].items()):
        ids[i] = id
        #parse and reshape intensities
        resource_path = get_resource_path("CIE/" + filename)
        if get_resource_path("CIE/" + filename) is None:
            raise FileNotFoundError(f"Could not find CIE/{filename}. It seems the installation script did not run properly. Need external data from CIE.")
        data = np.loadtxt(resource_path, delimiter=",", dtype=np.float32)
        raw_min_wavelength = data[0,0]
        raw_step_wavelength = data[1,0] - data[0,0]
        intensities[i,0] = reshape_spectrum(raw_min_wavelength, raw_step_wavelength, data[:,1], min_wavelength, step_wavelength, spectrum_length)
        intensities[i,1] = reshape_spectrum(raw_min_wavelength, raw_step_wavelength, data[:,2], min_wavelength, step_wavelength, spectrum_length)
        intensities[i,2] = reshape_spectrum(raw_min_wavelength, raw_step_wavelength, data[:,3], min_wavelength, step_wavelength, spectrum_length)
    return tuple(ids), intensities

def parse_light_sources(config_data: dict, min_wavelength: Number, step_wavelength: Number, spectrum_length: int) -> LightSources:
    """Returns a tuple of light source names and a numpy array containing harmonized spectra as specified in input data"""
    default_ids, default_intensities = _import_CIE_light_sources(min_wavelength, step_wavelength, spectrum_length)
    if "light_sources" not in config_data:
        return LightSources(default_ids, default_intensities)
    output_arr = np.zeros((len(config_data["light_sources"]), spectrum_length))
    source_names = tuple(source["id"] for source in config_data["light_sources"])
    for i, source in enumerate(config_data["light_sources"]):
        output_arr[i, :] = reshape_spectrum(source["wl_start"], source["wl_step"], source["value"], min_wavelength, step_wavelength, spectrum_length)
    return LightSources(default_ids + source_names, np.vstack([default_intensities, output_arr]))

def parse_observers(config_data: dict, min_wavelength: Number, step_wavelength: Number, spectrum_length: int) -> Observers:
    """Returns a tuple of observer names and a numpy array containing harmonized spectra as specified in input data"""
    cie_observer_list, cie_observer_arr = _import_CIE_observers(min_wavelength,step_wavelength, spectrum_length)
    if "observer" not in config_data:
        return Observers(cie_observer_list, cie_observer_arr)
    observer_arr = np.zeros((len(config_data["observer"]), 3, spectrum_length))
    observer_names = tuple(source["id"] for source in config_data["observer"])
    for i, observer in enumerate(config_data["observer"]):
        observer_arr[i, 0, :] = reshape_spectrum(observer["wl_start"], observer["wl_step"], observer["value"][0], min_wavelength, step_wavelength, spectrum_length)
        observer_arr[i, 1, :] = reshape_spectrum(observer["wl_start"], observer["wl_step"], observer["value"][1], min_wavelength, step_wavelength, spectrum_length)
        observer_arr[i, 2, :] = reshape_spectrum(observer["wl_start"], observer["wl_step"], observer["value"][2], min_wavelength, step_wavelength, spectrum_length)
    total_observer_names = cie_observer_list + observer_names
    total_observer_arr = np.vstack([cie_observer_arr, observer_arr])
    return Observers(total_observer_names, total_observer_arr)
