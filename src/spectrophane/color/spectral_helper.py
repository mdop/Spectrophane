import numpy as np
from numbers import Number

from spectrophane.core.dataclasses import LightSources, Observers, WavelengthAxis, SpectrumBlock
from spectrophane.io.resources import get_resource_path, get_json_resource


def _import_CIE_light_sources() -> tuple[tuple[str], tuple[SpectrumBlock]]:
    """Imports CIE light sources for use in light source parsing."""
    CIE_metadata = get_json_resource("CIE/data.json")
    ids = [""] * len(CIE_metadata["light_sources"])
    spectra = []
    for i, (id, filename) in enumerate(CIE_metadata["light_sources"].items()):
        ids[i] = id
        #parse and reshape intensities
        resource_path = get_resource_path("CIE/" + filename)
        if get_resource_path("CIE/" + filename) is None:
            raise FileNotFoundError(f"Could not find CIE/{filename}. It seems the installation script did not run properly. Need external data from CIE.")
        data = np.loadtxt(resource_path, delimiter=",", dtype=np.float32)
        raw_min_wavelength = data[0,0]
        raw_step_wavelength = data[1,0] - data[0,0]
        intensity_values = data[:,1].reshape(1,data.shape[0])
        spectra.append(SpectrumBlock(start=raw_min_wavelength, step=raw_step_wavelength, values=intensity_values))
    return tuple(ids), tuple(spectra)

def _import_CIE_observers() -> tuple[tuple[str], tuple[SpectrumBlock]]:
    """Imports CIE observer for use in light source parsing."""
    CIE_metadata = get_json_resource("CIE/data.json")
    ids = [""] * len(CIE_metadata["observers"])
    observer = []
    for i, (id, filename) in enumerate(CIE_metadata["observers"].items()):
        ids[i] = id
        #parse and reshape intensities
        resource_path = get_resource_path("CIE/" + filename)
        if get_resource_path("CIE/" + filename) is None:
            raise FileNotFoundError(f"Could not find CIE/{filename}. It seems the installation script did not run properly. Need external data from CIE.")
        data = np.loadtxt(resource_path, delimiter=",", dtype=np.float32)
        raw_min_wavelength = data[0,0]
        raw_step_wavelength = data[1,0] - data[0,0]
        sensitivity = np.zeros((1,3,data.shape[0]))
        sensitivity[0,0] = data[:,1]
        sensitivity[0,1] = data[:,2]
        sensitivity[0,2] = data[:,3]
        observer.append(SpectrumBlock(start=raw_min_wavelength, step=raw_step_wavelength, values=sensitivity))
    return tuple(ids), observer

def parse_light_sources(config_data: dict, wavelength_axis: WavelengthAxis) -> LightSources:
    """Returns a tuple of light source names and a numpy array containing harmonized spectra as specified in input data"""
    ids, intensities = _import_CIE_light_sources()
    if "light_sources" in config_data:
        for source in config_data["light_sources"]:
            ids += (source["id"],)
            vals = np.array(source["value"])
            vals = vals.reshape(1, vals.shape[0])
            intensities += (SpectrumBlock(start=source["wl_start"], step=source["wl_step"], values=vals),)
    return LightSources(ids, SpectrumBlock.merge_resample_spectra(intensities, axis=wavelength_axis))


def parse_observers(config_data: dict, wavelength_axis: WavelengthAxis) -> Observers:
    """Returns a tuple of observer names and a numpy array containing harmonized spectra as specified in input data"""
    ids, observer_data = _import_CIE_observers()
    if "observer" in config_data:
        for observer in config_data["observer"]:
            sensitivity = np.array(observer["value"])
            sensitivity = np.reshape(sensitivity, (1, 3, sensitivity.shape[1]))

            ids += (observer["id"],)
            observer_data += (SpectrumBlock(start=observer["wl_start"], step=observer["wl_step"], values=sensitivity),)
    
    return Observers(ids, SpectrumBlock.merge_resample_spectra(observer_data, wavelength_axis))