import pytest
import numpy as np

from spectrophane.core.dataclasses import WavelengthAxis, SpectrumBlock

def test_axis_properties():
    start= 0.5
    step = 3
    length = 8
    result = WavelengthAxis(start=start, step=step, length=length)
    
    expected_end = 21.5
    expected_wavelengths = np.array([0.5,3.5,6.5,9.5,12.5,15.5,18.5,21.5])
    assert result.end == expected_end
    assert np.allclose(result.wavelengths, expected_wavelengths)

@pytest.mark.parametrize(("wavelength_axes", "ref_axis"), [
    ([WavelengthAxis(  0, 1, 10), WavelengthAxis( 0, 1,  20), WavelengthAxis(  0,  1, 30), WavelengthAxis(  0, 1, 40)], WavelengthAxis(  0, 1, 10)),
    ([WavelengthAxis(  0, 1, 10), WavelengthAxis( 3, 1,  10), WavelengthAxis(  0,  1, 10), WavelengthAxis(  0, 1, 10)], WavelengthAxis(  3, 1,  7)),
    ([WavelengthAxis(  0, 4, 10), WavelengthAxis( 0, 2,  10), WavelengthAxis(  0,  3, 10), WavelengthAxis(  0, 1, 10)], WavelengthAxis(  0, 1, 10)),
    ([WavelengthAxis(100, 1, 30), WavelengthAxis(90, 2,  20), WavelengthAxis(110,  3, 40), WavelengthAxis(105, 1, 50)], WavelengthAxis(110, 1, 19)),
    ([WavelengthAxis(100, 5, 11), WavelengthAxis(90, 2, 100), WavelengthAxis(110, 10, 10), WavelengthAxis(105, 5, 50)], WavelengthAxis(110, 2, 21))
])
def test_axis_get_common_wavelength_axis(wavelength_axes, ref_axis):
    result_axis = WavelengthAxis.common(wavelength_axes)
    assert result_axis.start == ref_axis.start
    assert result_axis.step == ref_axis.step
    assert result_axis.length == ref_axis.length

@pytest.mark.parametrize("original_values, expected_values",
                         [(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), np.array([[1.5, 2.0, 2.5, 3.0],[4.5, 5.0, 5.5, 6.0]])),
                          (np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]]]),
                           np.array([[[1.5,2,2.5,3],[4.5,5,5.5,6]],[[7.5,8,8.5,9],[10.5,11,11.5,12]],[[13.5,14,14.5,15],[16.5,17,17.5,18]]]))])
def test_spectrum_block_resample_2D(original_values, expected_values):
    original_block = SpectrumBlock(start=0.0, step=1.0, values=original_values)

    target_axis = WavelengthAxis(start=0.5, step=0.5, length=4)
    resampled_block = original_block.resample(target_axis)

    assert resampled_block.axis.start == target_axis.start
    assert resampled_block.axis.step == target_axis.step
    assert resampled_block.axis.length == target_axis.length

    np.testing.assert_allclose(resampled_block.values, expected_values)

def test_spectrum_block_merge_resample_spectra():
    block1_values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    block1 = SpectrumBlock(start=0.0, step=1.0, values=block1_values)

    block2_values = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [10.0, 11.0, 12.0]])
    block2 = SpectrumBlock(start=0.0, step=1.0, values=block2_values)

    target_axis = WavelengthAxis(start=0.5, step=0.5, length=4)
    merged_block = SpectrumBlock.merge_resample_spectra([block1, block2], target_axis)

    assert merged_block.axis.start == target_axis.start
    assert merged_block.axis.step == target_axis.step
    assert merged_block.axis.length == target_axis.length
    expected_values = np.array([
        [1.5, 2.0, 2.5, 3.0],
        [4.5, 5.0, 5.5, 6.0],
        [7.5, 8.0, 8.5, 9.0],
        [10.5, 11.0, 11.5, 12.0],
        [10.5, 11.0, 11.5, 12.0]
    ])
    np.testing.assert_allclose(merged_block.values, expected_values)

def test_spectrum_block_merge_resample_common_axis():
    block1_values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    block1 = SpectrumBlock(start=0.0, step=1.0, values=block1_values)

    block2_values = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    block2 = SpectrumBlock(start=0.0, step=1.0, values=block2_values)

    merged_block = SpectrumBlock.merge_resample_spectra([block1, block2])

    assert merged_block.axis.start == 0.0
    assert merged_block.axis.step == 1.0
    assert merged_block.axis.length == 3
    expected_values = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ])
    np.testing.assert_allclose(merged_block.values, expected_values)

def test_spectrum_block_merge_resample_multidim():
    block1_values = np.random.rand(4,3,100)
    block1 = SpectrumBlock(start=0.0, step=1.0, values=block1_values)

    block2_values = np.random.rand(1,3,100)
    block2 = SpectrumBlock(start=0.0, step=1.0, values=block2_values)

    merged_block = SpectrumBlock.merge_resample_spectra([block1, block2])

    assert merged_block.values.shape == (5,3,100)

def test_spectrum_block_merge_resample_empty_spectra():
    block_completely_empty = SpectrumBlock.merge_resample_spectra(spectra=[], axis=None)
    axis = WavelengthAxis(start=0, step=1, length=100)
    block_only_axis = SpectrumBlock.merge_resample_spectra(spectra=[], axis=axis)
    default_axis = WavelengthAxis.empty()

    assert block_completely_empty.start == default_axis.start
    assert block_completely_empty.start == default_axis.start
    assert block_completely_empty.values.shape == (0, default_axis.length)
    assert block_only_axis.start == axis.start
    assert block_only_axis.start == axis.start
    assert block_only_axis.values.shape == (0, axis.length)