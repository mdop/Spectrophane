import numpy as np

from spectrophane.io.misc_data_import import parse_light_sources, parse_observers, LightSources, Observers

def test_parse_light_sources():
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
    assert len(result.names) == 2
    assert isinstance(result.names, tuple)
    assert isinstance(result.spectra, np.ndarray)
    assert result.spectra.shape == (2,100)
    assert np.all(result.spectra[0,:] == 0.5)
    assert np.all(result.spectra[1,:] == 1)

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
    assert len(result.names) == 2
    assert isinstance(result.names, tuple)
    assert isinstance(result.spectra, np.ndarray)
    assert result.spectra.shape == (2,3,50)
    assert np.all(result.spectra[0,:] == 0.5)
    assert np.all(result.spectra[1,:] == 1)