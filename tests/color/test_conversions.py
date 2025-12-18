import numpy as np
import jax.numpy as jnp
import jax

from spectrophane.color.conversions import linrgb_to_xyz, spectrum_to_xyz
from spectrophane.color.spectral_helper import _import_CIE_light_sources, _import_CIE_observers

def test_linrgb_to_xyz_single():
    rgb = np.random.rand(3)
    result = linrgb_to_xyz(rgb)
    assert result.shape == (3,)
    assert np.all(result <= 1)
    assert np.all(result >= 0)

def test_linrgb_to_xyz_multi():
    rgb = np.random.rand(100,3)
    result = linrgb_to_xyz(rgb)
    assert result.shape == (100,3)
    assert np.all(result <= 1)
    assert np.all(result >= 0)

    matrix = np.random.rand(3,3)
    result_custom_matrix = linrgb_to_xyz(rgb, matrix)
    assert result_custom_matrix.shape == (100,3)

def test_spectrum_to_xyz_np():
    light = np.random.rand(100)
    observer = np.random.rand(3, 100)
    material = np.random.rand(100)
    result = spectrum_to_xyz(material, light, observer, 1)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)

def test_spectrum_to_xyz_jnp():
    light = np.random.rand(100)
    light = jnp.asarray(light)
    observer = np.random.rand(3, 100)
    observer = jnp.asarray(observer)
    material = np.random.rand(100)
    material = jnp.asarray(material)
    result = spectrum_to_xyz(material, light, observer, 1)
    assert isinstance(result, jnp.ndarray)
    assert result.shape == (3,)

def test_spectrum_to_xyz_validate_calculation():
    """THIS TEST RELIES ON EXTERNAL DATA! Verifies calculation of xyz values by calculating xyz value of D65 illuminant with CIE1931 observer in wavelength range 360nm to 830nm."""
    light_list, light_arr = _import_CIE_light_sources(360, 1, 471)
    obs_list, CIE_obs = _import_CIE_observers(360, 1, 471)
    D65_index = light_list.index("D65")
    CIE1931_index = obs_list.index("CIE1931")
    D65 = light_arr[D65_index]
    CIE1931 = CIE_obs[CIE1931_index]
    material = np.ones_like(D65)
    result = spectrum_to_xyz(material, D65, CIE1931, 1)
    #according to wikipedia XYZ coordinates of D65 with 2° observer is (95.047, 100, 108.883)
    target = np.array([95.047, 100, 108.883])/100
    assert np.allclose(result, target, 0.01)