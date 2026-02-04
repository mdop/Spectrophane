import pytest
import numpy as np
import json
import copy
from unittest.mock import ANY
from spectrophane.core.dataclasses import WavelengthAxis
from spectrophane.training.ingest_spectra import (
        process_spectrum_list,
        prepare_spectrum_data,
        TrainingRefSpectraData
)
from spectrophane.training.ingest_stacks import StackData

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
                "value": [1] * 110,
                "background": "w"
            },
            {
                "wl_start": 400,
                "wl_step": 2,
                "stack": [{"id": "bl_jwhite", "d":0.20}],
                "value": [1] * 110,
                "background": "b"
            },
            {
                "wl_start": 400,
                "wl_step": 2,
                "stack": [{"id": "bl_jwhite", "d":0.20}],
                "value": [1] * 110,
                "background": {
                    "wl_start": 400,
                    "wl_step": 2,
                    "value": [0.5] * 110
                }
            }
        ]
    }
    return data

@pytest.fixture
def mock_combined_source_file(mock_transmission_source_file, mock_reflection_source_file):
    data = copy.deepcopy(mock_transmission_source_file)
    data["spectra"]["reflection"] = mock_reflection_source_file["spectra"]["reflection"]
    return data

def test_process_spectrum_list_empty(mock_empty_source_file):
    empty_stacks, empty_spectra, empty_background = process_spectrum_list(mock_empty_source_file["spectra"].get("transmission", []), mock_empty_source_file.get("materials", []))
    assert isinstance(empty_stacks, StackData)
    assert len(empty_spectra) == 0
    assert len(empty_background) == 0

def test_process_spectrum_list_empty(mock_transmission_source_file):
    transmission_stacks, transmission_spectra, _ = process_spectrum_list(mock_transmission_source_file["spectra"].get("transmission", {},), mock_transmission_source_file.get("materials", {}))
    assert isinstance(transmission_stacks, StackData)
    assert len(transmission_spectra) == 5

def test_process_spectrum_list_background(mock_reflection_source_file):
    reflection_stacks, reflection_spectra, reflection_backgrounds = process_spectrum_list(mock_reflection_source_file["spectra"].get("reflection", []), mock_reflection_source_file.get("materials", []))
    assert len(reflection_backgrounds) == len(reflection_spectra)
    assert np.all(reflection_backgrounds[0].values == 0)
    assert np.all(reflection_backgrounds[1].values == 1)
    assert np.all(reflection_backgrounds[2].values == 0)
    assert np.all(reflection_backgrounds[3].values == 0.5)

def test_prepare_spectrum_data_empty(mock_empty_source_file):
    empty_result = prepare_spectrum_data(mock_empty_source_file)
    assert empty_result.min_wavelength > 0
    assert empty_result.transmission_spectra.size == 0
    assert empty_result.reflection_spectra.size == 0

def test_prepare_spectrum_data_transmission(mocker, mock_transmission_source_file):
    transmission_result = prepare_spectrum_data(mock_transmission_source_file)
    assert isinstance(transmission_result, TrainingRefSpectraData)
    assert transmission_result.reflection_spectra.size == 0
    assert isinstance(transmission_result.transmission_stacks, StackData)
    assert isinstance(transmission_result.transmission_spectra, np.ndarray)

def test_prepare_spectrum_data_reflection(mocker, mock_reflection_source_file):
    reflection_result = prepare_spectrum_data(mock_reflection_source_file)
    assert isinstance(reflection_result, TrainingRefSpectraData)
    assert reflection_result.transmission_spectra.size == 0
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
    assert combined_result.reflection_spectra.shape == (4,ANY)