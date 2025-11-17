import numpy as np
from unittest.mock import MagicMock

from spectrophane.io.misc_data_import import parse_light_sources, parse_observers, LightSources, Observers


import pytest
import numpy as np
from spectrophane.io.misc_data_import import _import_CIE_light_sources, _import_CIE_observers

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
    return mocker.patch("spectrophane.io.misc_data_import.get_json_resource", return_value=mock_data)

@pytest.fixture
def mock_get_resource_path(mocker):
    return mocker.patch(
        "spectrophane.io.data_io.get_resource_path",
        side_effect=lambda path: f"mocked/{path}"  # optional: preserve input
    )

@pytest.fixture
def mock_csv_loadtext_light_source(mocker):
    data = np.array([[300, 1.1], [305, 1.2], [310, 1.3], [315, 1.4]], dtype=np.float32)
    mock = mocker.MagicMock()
    mock.return_value = data
    return mocker.patch("numpy.loadtxt", mock)
    

@pytest.fixture
def mock_reshape_spectrum_light_source(mocker):
    return mocker.patch(
        "spectrophane.io.misc_data_import.reshape_spectrum",
        return_value=np.array([1.0, 2.0, 3.0])
    )

def test_import_CIE_light_sources(mock_get_json_resource, mock_csv_loadtext_light_source, mock_reshape_spectrum_light_source):
    min_wavelength = 380.0
    step_wavelength = 5.0
    spectrum_length = 3

    ids, intensities = _import_CIE_light_sources(min_wavelength, step_wavelength, spectrum_length)

    assert ids == ("source1", "source2")
    expected_intensities = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0]
    ])
    assert np.allclose(intensities, expected_intensities, 0.01)

def test_import_CIE_light_csv_import(mock_get_json_resource, mock_csv_loadtext_light_source):
    min_wavelength = 300
    step_wavelength = 5.0
    spectrum_length = 4

    _, intensities = _import_CIE_light_sources(min_wavelength, step_wavelength, spectrum_length)
    expected_intensities = np.array([
        [1.1, 1.2, 1.3, 1.4],
        [1.1, 1.2, 1.3, 1.4]
    ])
    assert np.allclose(intensities, expected_intensities, 0.01)

def test_import_CIE_light_sources_empty_light_sources(mock_get_json_resource, mock_csv_loadtext_light_source, mock_reshape_spectrum_light_source):
    mock_get_json_resource.return_value = {"light_sources": {}}
    min_wavelength = 380.0
    step_wavelength = 5.0
    spectrum_length = 3

    ids, intensities = _import_CIE_light_sources(min_wavelength, step_wavelength, spectrum_length)

    assert ids == ()
    assert intensities.shape == (0, 3)


@pytest.fixture
def mock_csv_loadtext_observers(mocker):
    data = np.array([[300, 1.1, 2.1, 3.1], [305, 1.2, 2.2, 3.2], [310, 1.3, 2.3, 3.3], [315, 1.4, 2.4, 3.4]], dtype=np.float32)
    mock = mocker.MagicMock()
    mock.return_value = data
    return mocker.patch("numpy.loadtxt", mock)
    

@pytest.fixture
def mock_reshape_spectrum_observers(mocker):
    return mocker.patch(
        "spectrophane.io.misc_data_import.reshape_spectrum",
        return_value=np.array([1.0, 2.0, 3.0])
    )

def test_import_CIE_observers(mock_get_json_resource, mock_csv_loadtext_observers, mock_reshape_spectrum_observers):
    min_wavelength = 380.0
    step_wavelength = 5.0
    spectrum_length = 3

    ids, intensities = _import_CIE_observers(min_wavelength, step_wavelength, spectrum_length)

    assert ids == ("source3", "source4")
    expected_intensities = np.array([
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    ])
    assert np.allclose(intensities, expected_intensities, 0.01)

def test_import_CIE_observers_csv_import(mock_get_json_resource, mock_csv_loadtext_observers):
    min_wavelength = 300
    step_wavelength = 5.0
    spectrum_length = 4

    _, intensities = _import_CIE_observers(min_wavelength, step_wavelength, spectrum_length)
    expected_intensities = np.array([
        [[1.1, 1.2, 1.3, 1.4],[2.1, 2.2, 2.3, 2.4],[3.1, 3.2, 3.3, 3.4]],
        [[1.1, 1.2, 1.3, 1.4],[2.1, 2.2, 2.3, 2.4],[3.1, 3.2, 3.3, 3.4]]
    ])
    assert np.allclose(intensities, expected_intensities, 0.01)

def test_import_CIE_observers_empty_light_sources(mock_get_json_resource, mock_csv_loadtext_light_source, mock_reshape_spectrum_observers):
    mock_get_json_resource.return_value = {"observers": {}}
    min_wavelength = 380.0
    step_wavelength = 5.0
    spectrum_length = 3

    ids, intensities = _import_CIE_observers(min_wavelength, step_wavelength, spectrum_length)

    assert ids == ()
    assert intensities.shape == (0, 3, 3)


@pytest.fixture
def mock_CIE_light_sources(mocker):
    output = (("D1", "D2"), np.array([[1.0]*100]*2))
    return mocker.patch("spectrophane.io.misc_data_import._import_CIE_light_sources", return_value = output)

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
    result = parse_light_sources(data, 400, 1, 100)
    assert isinstance(result, LightSources)
    assert len(result.names) == 4
    assert isinstance(result.names, tuple)
    assert isinstance(result.spectra, np.ndarray)
    assert result.spectra.shape == (4,100)
    assert np.all(result.spectra[2,:] == 0.5)
    assert np.all(result.spectra[3,:] == 1)

@pytest.fixture
def mock_CIE_observer(mocker):
    output = (("C1", "C2"), np.array([[[1.0]*100]*3]*2))
    return mocker.patch("spectrophane.io.misc_data_import._import_CIE_observers", return_value = output)

def test_parse_observer():
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
    result = parse_observers(data, 400, 1, 50)
    assert isinstance(result, Observers)
    assert len(result.names) == 4
    assert isinstance(result.names, tuple)
    assert isinstance(result.spectra, np.ndarray)
    assert result.spectra.shape == (4,3,50)
    assert np.all(result.spectra[2,:] == 0.5)
    assert np.all(result.spectra[3,:] == 1)