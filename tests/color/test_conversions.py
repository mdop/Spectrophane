import numpy as np
import jax.numpy as jnp
import pytest

from spectrophane.color.conversions import linrgb_to_xyz, xyz_to_linrgb, spectrum_to_xyz, decode_rgb, encode_rgb, xyz_to_lab
from spectrophane.color.spectral_helper import _import_CIE_light_sources, _import_CIE_observers
from spectrophane.core.dataclasses import SpectrumBlock, WavelengthAxis

@pytest.mark.parametrize("func", [
    (linrgb_to_xyz),
    (xyz_to_linrgb),
])
def test_linrgb_to_xyz_single(func):
    rgb = np.random.rand(3)
    result = func(rgb)
    assert result.shape == (3,)
    assert np.all(result <= 1)
    assert np.all(result >= 0)

@pytest.mark.parametrize("func", [
    (linrgb_to_xyz),
    (xyz_to_linrgb),
])
def test_linrgb_to_xyz_multi(func):
    rgb = np.random.rand(100,3)
    result = func(rgb)
    assert result.shape == (100,3)
    assert np.all(result <= 1)
    assert np.all(result >= 0)

    matrix = np.random.rand(3,3)
    result_custom_matrix = linrgb_to_xyz(rgb, matrix)
    assert result_custom_matrix.shape == (100,3)

def test_linrgb_xyz_roundtrip():
    rgb = np.random.rand(100,3)
    roundtrip = xyz_to_linrgb(linrgb_to_xyz(rgb, clip=False), clip=False)
    assert np.allclose(roundtrip, rgb, rtol=0.01)

@pytest.fixture
def mock_rgb_singlearr():
    return np.astype(np.random.rand(3)*255, np.uint8)

@pytest.fixture
def mock_rgb_2D():
    return np.random.rand(100, 100, 3)

@pytest.mark.parametrize("func", [
    (decode_rgb),
    (encode_rgb),
])
def test_deencode_rgb_single(mock_rgb_singlearr, func):
    result_single = func(mock_rgb_singlearr)
    assert result_single.shape == (3,)

def test_deencode_rgb_multi(mock_rgb_2D):
    low = np.clip(mock_rgb_2D / 4.5, 0, 1)
    high = np.clip(np.power((mock_rgb_2D + 0.099) / 1.099, 1/0.45), 0, 1)
    expected = np.where(mock_rgb_2D < 0.081, low, high)
    result = decode_rgb(mock_rgb_2D)
    assert result.shape == (100,100,3)
    assert np.allclose(result, expected)

    low2 = np.clip(mock_rgb_2D * 4.5, 0, 1)
    high2 = np.clip(1.099*np.pow(mock_rgb_2D, 0.45)-0.099, 0, 1)
    expected2 = np.where(mock_rgb_2D < 0.018, low2, high2)
    result2 = encode_rgb(mock_rgb_2D)
    assert result.shape == (100,100,3)
    assert np.allclose(result2, expected2)

    roundtrip = encode_rgb(decode_rgb(mock_rgb_2D))
    assert np.allclose(roundtrip, mock_rgb_2D, rtol=0.004)

def test_xyz_lab_white_point():
    xyz = np.array([95.047, 100.0, 108.883])
    white = np.array([95.047, 100.0, 108.883])

    lab = xyz_to_lab(xyz, white)

    assert np.allclose(lab, [100.0, 0.0, 0.0], atol=1e-4)

def test_xyz_lab_known_value_1():
    #color coordinates calculates by Lindbloom with D65 illuminant
    xyz = np.array([0.4124, 0.2126, 0.0193])
    white = np.array([0.95047, 1.000, 1.08883])

    lab = xyz_to_lab(xyz, white)

    expected = np.array([53.2329, 80.1093, 67.2201])
    assert np.allclose(lab, expected, atol=1e-3)

def test_xyz_lab_neutral_gray():
    white = np.array([0.95047, 1.000, 1.08883])
    grey = np.array([0.200, 0.200, 0.200]) * white / 100

    lab = xyz_to_lab(grey, white)

    assert abs(lab[1]) < 1e-4
    assert abs(lab[2]) < 1e-4

def test_xyz_lab_batch_input():
    xyz = np.array([
        [0.95047, 1.000, 1.08883],
        [0.4124, 0.2126, 0.0193]
    ])
    white = np.array([0.95047, 1.000, 1.08883])

    lab = xyz_to_lab(xyz, white)

    assert lab.shape == (2, 3)

def test_xyz_to_lab_arbitrary_dims():
    xyz = np.random.rand(2, 3, 4, 3)
    white = np.array([0.95047, 1.0, 1.08883])

    lab = xyz_to_lab(xyz, white)

    assert lab.shape == xyz.shape

def test_xyz_lab_small_values_branch():
    xyz = np.array([0.001, 0.001, 0.001])
    white = np.array([0.95047, 1.000, 1.08883])

    lab = xyz_to_lab(xyz, white)

    # Should not produce NaNs or inf
    assert np.all(np.isfinite(lab))

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
    light_list, light_blocks = _import_CIE_light_sources()
    harmonized_light_blocks = SpectrumBlock.merge_resample_spectra(light_blocks, WavelengthAxis(360, 1, 471))
    obs_list, CIE_obs_blocks = _import_CIE_observers()
    harmonized_obs_blocks = SpectrumBlock.merge_resample_spectra(CIE_obs_blocks, WavelengthAxis(360, 1, 471))
    D65_index = light_list.index("D65")
    CIE1931_index = obs_list.index("CIE1931")
    D65 = harmonized_light_blocks.values[D65_index]
    CIE1931 = harmonized_obs_blocks.values[CIE1931_index]
    material = np.ones_like(D65)
    result = spectrum_to_xyz(material, D65, CIE1931, 1)
    #according to wikipedia XYZ coordinates of D65 with 2° observer is (95.047, 100, 108.883)
    target = np.array([95.047, 100, 108.883])/100
    assert np.allclose(result, target, 0.01)