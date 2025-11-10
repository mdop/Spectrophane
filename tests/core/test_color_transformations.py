import numpy as np
import jax.numpy as jnp
import jax

from spectrophane.core.color_transformations import linrgb_to_xyz, spectrum_to_xyz

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