import pytest
import numpy as np
import json
import copy
from unittest.mock import ANY
from spectrophane.io.material_spectrum_processing import (
        get_common_wavelength_space,
        reshape_spectrum,
        process_spectrum_list,
        prepare_spectrum_data,
        TrainingRefSpectraData
)
from spectrophane.io.stack_io import StackData

@pytest.fixture
def mock_empty_source_file():
    data = {
        "materials": [
            {
            "id": "bl_jwhite",
            },
            {
            "id": "bl_yellow",
            }
        ],

        "spectra": { }
    }
    return data

@pytest.fixture
def mock_transmission_source_file(mock_empty_source_file):
    data = copy.deepcopy(mock_empty_source_file)
    data["spectra"] = {
        "transmission": [
            {
                "wl_start": 380,
                "wl_step": 1,
                "stack": [{"id": "bl_jwhite", "d":0.20}],
                "value": [1] * 200
            },
            {
                "wl_start": 400,
                "wl_step": 2,
                "stack": [{"id": "bl_jwhite", "d":0.20}, {"id": "bl_yellow", "d":0.10}, {"id": "bl_jwhite", "d":0.30}, {"id": "bl_yellow", "d":0.40}],
                "value": [1] * 110
            },
            {
                "wl_start": 380,
                "wl_step": 2,
                "stack": [{"id": "bl_jwhite", "d":0.20}],
                "value": [1] * 100
            },
            {
                "wl_start": 410,
                "wl_step": 2,
                "stack": [{"id": "bl_jwhite", "d":0.20}],
                "value": [1] * 100
            },
            {
                "wl_start": 410,
                "wl_step": 2,
                "stack": [{"id": "bl_jwhite", "d":0.20}],
                "value": [1] * 90
            }
        ]
    }
    return data

@pytest.fixture
def mock_reflection_source_file(mock_empty_source_file):
    data = copy.deepcopy(mock_empty_source_file)
    data["spectra"] = {
        "reflection": [
            {
                "wl_start": 380,
                "wl_step": 1,
                "stack": [{"id": "bl_jwhite", "d":0.20}],
                "value": [1] * 200
            },
            {
                "wl_start": 400,
                "wl_step": 2,
                "stack": [{"id": "bl_jwhite", "d":0.20}],
                "value": [1] * 110
            }
        ]
    }
    return data

@pytest.fixture
def mock_combined_source_file(mock_transmission_source_file, mock_reflection_source_file):
    data = copy.deepcopy(mock_transmission_source_file)
    data["spectra"]["reflection"] = mock_reflection_source_file["spectra"]["reflection"]
    return data


@pytest.mark.parametrize(("min", "step", "lengths", "ref_min", "ref_step", "ref_length"), [
                            ((0,0,0,0),(1,1,1,1),(10,20,30,40),0,1,10),
                            ((0,3,2,1),(1,1,1,1),(10,10,10,10),3,1,7),
                            ((0,0,0,0),(4,2,3,1),(10,10,10,10),0,1,10),
                            ((100,90,110,105),(1,2,3,1),(30,20,40,50),110,1,19),
                            ((100,90,110,105),(5,2,10,5),(11,100,10,50),110,2,21)
                        ])
def test_get_common_wavelength_space(min, step, lengths, ref_min, ref_step, ref_length):
    res_min, res_step, res_length = get_common_wavelength_space(min, step, lengths)
    assert res_min == ref_min
    assert res_step == ref_step
    assert res_length == ref_length

@pytest.mark.parametrize(("old_min_wavelength, old_step_wavelength, old_values, new_min_wavelength, new_step_wavelength, new_length, new_values"), [
                            (0,1,[1,2,3,4,5,6,7,8,9,10], 4,2,3,[5,7,9]),
                            (0,2,[1,2,3,4,5,6,7,8,9,10], 2,2,4,[2,3,4,5])
                        ])
def test_reshape_spectrum(old_min_wavelength, old_step_wavelength, old_values, new_min_wavelength, new_step_wavelength, new_length, new_values):
    result = reshape_spectrum(old_min_wavelength, old_step_wavelength, old_values, new_min_wavelength, new_step_wavelength, new_length)
    new_values_arr = np.array(new_values)
    assert (result == new_values_arr).all()

def test_process_spectrum_list(mock_empty_source_file, mock_transmission_source_file):
    empty_result = process_spectrum_list(mock_empty_source_file["spectra"].get("transmission", {},), mock_empty_source_file.get("materials", {}), None, None, None)
    assert empty_result == (None, None)

    transmission_result = process_spectrum_list(mock_transmission_source_file["spectra"].get("transmission", {},), mock_transmission_source_file.get("materials", {}), 410, 1, 80)
    assert isinstance(transmission_result[0], StackData)
    assert transmission_result[1].shape == (5,80)

def test_prepare_spectrum_data_empty(mocker, mock_empty_source_file):
    empty_result = prepare_spectrum_data(mock_empty_source_file)
    assert empty_result is None

def test_prepare_spectrum_data_transmission(mocker, mock_transmission_source_file):
    transmission_result = prepare_spectrum_data(mock_transmission_source_file)
    assert isinstance(transmission_result, TrainingRefSpectraData)
    assert transmission_result.reflection_spectra is None
    assert transmission_result.reflection_stacks is None
    assert isinstance(transmission_result.transmission_stacks, StackData)
    assert isinstance(transmission_result.transmission_spectra, np.ndarray)

def test_prepare_spectrum_data_reflection(mocker, mock_reflection_source_file):
    reflection_result = prepare_spectrum_data(mock_reflection_source_file)
    assert isinstance(reflection_result, TrainingRefSpectraData)
    assert reflection_result.transmission_spectra is None
    assert reflection_result.transmission_stacks is None
    assert isinstance(reflection_result.reflection_stacks, StackData)
    assert isinstance(reflection_result.reflection_spectra, np.ndarray)

def test_prepare_spectrum_data_combined(mocker, mock_combined_source_file):
    combined_result = prepare_spectrum_data(mock_combined_source_file)
    assert isinstance(combined_result, TrainingRefSpectraData)
    assert isinstance(combined_result.transmission_stacks, StackData)
    assert isinstance(combined_result.transmission_spectra, np.ndarray)
    assert isinstance(combined_result.reflection_stacks, StackData)
    assert isinstance(combined_result.reflection_spectra, np.ndarray)
    assert combined_result.transmission_spectra.shape == (5,ANY)
    assert combined_result.reflection_spectra.shape == (2,ANY)