import numpy as np
from unittest.mock import MagicMock

from spectrophane.color.spectral_helper import parse_light_sources, parse_observers
from spectrophane.core.dataclasses import LightSources, Observers, WavelengthAxis, SpectrumBlock


import pytest
import numpy as np
from spectrophane.color.spectral_helper import _import_CIE_light_sources, _import_CIE_observers

@pytest.fixture
def mock_get_json_resource(mocker):
    mock_data = {
        "light_sources": {
            "source1": "file1.csv",
            "source2": "file2.csv"
        },
        "observers": {
            "source3": "file3.csv",
            "source4": "file4.csv"
        }
    }
    return mocker.patch("spectrophane.color.spectral_helper.get_json_resource", return_value=mock_data)

@pytest.fixture
def mock_get_path(mocker):
    return mocker.patch("spectrophane.color.spectral_helper.get_resource_path", return_value = True)

@pytest.fixture
def mock_csv_loadtext_light_source(mocker):
    data = np.array([[300, 1.1], [305, 1.2], [310, 1.3], [315, 1.4]], dtype=np.float32)
    mock = mocker.MagicMock()
    mock.return_value = data
    return mocker.patch("numpy.loadtxt", mock)
    
def test_import_CIE_light_csv_import(mock_get_json_resource, mock_get_path, mock_csv_loadtext_light_source):
    ids, intensities = _import_CIE_light_sources()

    assert ids == ("source1", "source2")
    assert len(ids) == len(intensities)
    assert len(ids) == 2
    expected_intensities = np.array([[1.1, 1.2, 1.3, 1.4]])
    assert np.allclose(intensities[0].values, expected_intensities, 0.01)
    assert np.allclose(intensities[1].values, expected_intensities, 0.01)

def test_import_CIE_light_sources_empty_light_sources(mock_get_json_resource, mock_csv_loadtext_light_source):
    mock_get_json_resource.return_value = {"light_sources": {}}

    ids, intensities = _import_CIE_light_sources()

    assert len(ids) == 0
    assert len(intensities) == len(ids)


@pytest.fixture
def mock_csv_loadtext_observers(mocker):
    data = np.array([[300, 1.1, 2.1, 3.1], [305, 1.2, 2.2, 3.2], [310, 1.3, 2.3, 3.3], [315, 1.4, 2.4, 3.4]], dtype=np.float32)
    mock = mocker.MagicMock()
    mock.return_value = data
    return mocker.patch("numpy.loadtxt", mock)
    
def test_import_CIE_observers(mock_get_json_resource, mock_get_path, mock_csv_loadtext_observers):
    ids, intensities = _import_CIE_observers()

    assert ids == ("source3", "source4")
    expected_intensities = np.array([ [[1.1, 1.2, 1.3, 1.4], [2.1, 2.2, 2.3, 2.4], [3.1, 3.2, 3.3, 3.4]] ])
    assert np.allclose(intensities[0].values, expected_intensities, 0.01)

def test_import_CIE_observers_empty_light_sources(mock_get_json_resource, mock_csv_loadtext_light_source):
    mock_get_json_resource.return_value = {"observers": {}}

    ids, intensities = _import_CIE_observers()

    assert len(ids) == 0
    assert len(intensities) == 0


@pytest.fixture
def mock_CIE_light_sources(mocker):
    block = SpectrumBlock(start=350,step=2,values=np.array([[1.0]*100]))
    output = (("D1", "D2"), (block, block))
    return mocker.patch("spectrophane.color.spectral_helper._import_CIE_light_sources", return_value = output)

def test_parse_light_sources(mock_CIE_light_sources):
    data = {
        "light_sources": [
            {
                "id": "D0",
                "wl_start": 300,
                "wl_step": 1,
                "value": [0.5]*300,
            },
            {
                "id": "D1",
                "wl_start": 350,
                "wl_step": 2,
                "value": [1]*50,
            }
        ]
    }
    result = parse_light_sources(data)
    assert isinstance(result, LightSources)
    assert len(result.names) == 4
    assert isinstance(result.names, tuple)
    assert isinstance(result.spectra[2], SpectrumBlock)
    assert result.spectra[0].values.shape == (1,100)
    assert np.all(result.spectra[2].values[0,:] == 0.5)
    assert np.all(result.spectra[3].values[0,:] == 1)

@pytest.fixture
def mock_CIE_observer(mocker):
    block = SpectrumBlock(start=350,step=2,values=np.array([[[1.0]*100]*3]))
    output = (("C1", "C2"), (block,block))
    return mocker.patch("spectrophane.color.spectral_helper._import_CIE_observers", return_value = output)

def test_parse_observer(mock_CIE_observer):
    data = {
        "observer": [
            {
            "id": "CIE1",
            "wl_start": 360,
            "wl_step": 2,
            "value": [[0.5]*100,[0.5]*100,[0.5]*100]
            },
            {
            "id": "CIE2",
            "wl_start": 380,
            "wl_step": 1,
            "value": [[1]*100,[1]*100,[1]*100]
            }
        ],
    }
    result = parse_observers(data)
    assert isinstance(result, Observers)
    assert len(result.names) == 4
    assert isinstance(result.names, tuple)
    assert isinstance(result.spectra[2], SpectrumBlock)
    assert result.spectra[2].values.shape == (1,3,100)
    assert np.all(result.spectra[2].values[0,:] == 0.5)
    assert np.all(result.spectra[3].values[0,:] == 1)