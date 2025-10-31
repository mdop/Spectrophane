import pytest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
import numpy as np
from PIL import Image
import json
from spectrophane.io.material_image_processing import (
    raw_to_linear_rgb,
    decode_rgb_img,
    rgb_image_to_linrgb,
    import_image,
    calibrate_spatial_brightness,
    roi_filter,
    aggregate_rois,
    aggregate_image_rois,
    process_image_to_xyz,
    parse_material_characterization_data
)
from spectrophane.io.stack_io import StackData

# Fixtures for common test cases
@pytest.fixture
def mock_raw_image():
    return np.random.rand(100, 100, 3).astype(np.float32)

@pytest.fixture
def mock_rgb_image():
    return np.astype(np.random.rand(100, 100, 3)*255, np.uint8)

@pytest.fixture
def mock_roi():
    return (10, 10, 20, 20)

@pytest.fixture
def mock_color_image():
    return np.random.rand(120, 100, 3)

@pytest.fixture
def mock_linrgb_image():
    return np.random.rand(120, 100, 3)

@pytest.fixture
def mock_linrgb_processing_image_rois():
    """mock image for processing with white and black roi. Returns a Tuple (image, black_roi, white_roi, color_roi1, color_roi2)"""
    img = np.random.rand(120, 100, 3)
    img[0:10, 0:20] = 0.001
    img[10:20,10:20] = 0.999
    return (img, (0,0,9,9), (0,9,9,19), (10,10,19,19), (20,20,100,90), (20,0,39,19))

# Test raw_to_linear_rgb
def test_raw_to_linear_rgb(mocker, mock_raw_image):
    mock_raw = mocker.MagicMock()
    mock_raw.__enter__.return_value = mock_raw
    mock_raw.__exit__.return_value = None
    mock_raw.postprocess.return_value = (mock_raw_image * 65535).astype(np.uint16)

    mocker.patch('rawpy.imread', return_value=mock_raw)
    mocker.patch('spectrophane.io.material_image_processing.get_resource_path', return_value='/fake/path/test.raw')
    result = raw_to_linear_rgb('test.raw')

    mock_raw.postprocess.assert_called_once_with(
        use_camera_wb=False,
        no_auto_bright=True,
        output_bps=ANY,
        gamma=(1, 1)
    )
    assert np.allclose(result, mock_raw_image, atol=1e-4)

# Test decode_rgb_img
def test_decode_rgb_img(mock_rgb_image):
    low = np.clip(mock_rgb_image[:5, :5, 0] / 4.5, 0, 1)
    high = np.clip(np.power((mock_rgb_image[:5, :5, 0] + 0.099) / 1.099, 1/0.45), 0, 1)
    expected = np.where(mock_rgb_image[:5, :5, 0] < 0.081, low, high)
    result = decode_rgb_img(mock_rgb_image)
    assert np.array_equal(result[:5, :5, 0], expected)

# Test image_to_linrgb
def test_rgb_image_to_linrgb(mocker, mock_rgb_image):
    mock_get_resource_path = mocker.patch('spectrophane.io.material_image_processing.get_resource_path', return_value='/fake/path/test.raw')
    mock_image_open = mocker.patch('PIL.Image.open', return_value=mock_rgb_image)
    mock_decode_rgb_img = mocker.patch('spectrophane.io.material_image_processing.decode_rgb_img', return_value=mock_rgb_image)
    
    result = rgb_image_to_linrgb("rest.raw")
    
    mock_get_resource_path.assert_called_once_with("material_data/images/rest.raw")
    mock_image_open.assert_called_once()
    mock_decode_rgb_img.assert_called_once()
    encoded_array = mock_decode_rgb_img.call_args[0][0]
    assert np.all((encoded_array >= 0) & (encoded_array <= 1))
    assert isinstance(result, np.ndarray)

# Test import_image
@pytest.mark.parametrize("extension,type", [("jnp", "raw"), ("jpg", "rgb"), ("raw", "raw"), ("png", "rgb"), ("jpeg", "rgb")])
def test_import_image(mocker, mock_raw_image, extension, type):
    mock_raw_import = mocker.patch('spectrophane.io.material_image_processing.raw_to_linear_rgb', return_value=mock_raw_image)
    mock_rgb_import = mocker.patch('spectrophane.io.material_image_processing.rgb_image_to_linrgb', return_value=mock_raw_image)

    result = result = import_image("test." + extension)

    if extension == "raw":
        mock_raw_import.assert_called_once()
        mock_rgb_import.assert_not_called()
    elif extension == "rgb":
        mock_raw_import.assert_not_called()
        mock_rgb_import.assert_called_once()
    assert np.array_equal(mock_raw_image, result)

# Test roi_aggregate
def test_roi_filter(mock_linrgb_image, mock_roi):
    result = roi_filter(mock_linrgb_image, mock_roi)
    assert isinstance(result, np.ndarray)
    assert result.shape == (11*11, 3)

# Test aggregate_rois
def test_aggregate_rois(mock_linrgb_image, mock_roi):
    rois = (mock_roi, (0, 0, 10, 10))
    result = aggregate_rois(mock_linrgb_image, rois)
    assert result.shape == (3,)

# Test aggregate_image_rois
def test_aggregate_image_rois(mock_linrgb_image, mock_roi):
    white_rois = [mock_roi]
    black_rois = [mock_roi]
    color_rois = [mock_roi]
    result = aggregate_image_rois(mock_linrgb_image, white_rois, black_rois, color_rois)
    assert isinstance(result, tuple)
    assert len(result) == 3

# Test process_image_to_xyz
def test_process_image_to_xyz(mock_linrgb_processing_image_rois):
    img = mock_linrgb_processing_image_rois[0]
    black_rois = (mock_linrgb_processing_image_rois[1],mock_linrgb_processing_image_rois[2])
    white_rois = (mock_linrgb_processing_image_rois[3],)
    color_rois = (mock_linrgb_processing_image_rois[4], mock_linrgb_processing_image_rois[5])
    result = process_image_to_xyz(img, white_rois, black_rois, color_rois)
    assert result.shape == (2,3)

# Test parse_material_characterization_data
def test_parse_material_characterization_data(mocker, mock_linrgb_processing_image_rois):
    mock_data = {
        "materials": [
            {
                "id": "mat1",
            },
            {
                "id": "mat2",
            }
        ],
        "images": {
            "measurement_images": [
                {
                    "filename": "test.jpg",
                    "white_refs": [mock_linrgb_processing_image_rois[3]],
                    "black_refs": [mock_linrgb_processing_image_rois[1],mock_linrgb_processing_image_rois[2]],
                    "measurement_areas": [
                        {
                            "stack": [{"id": "mat1", "d":0.03}],
                            "roi": mock_linrgb_processing_image_rois[4]
                        },
                        {
                            "stack": [{"id": "mat2", "d":0.03}],
                            "roi": mock_linrgb_processing_image_rois[5]
                        },
                        {
                            "stack": [{"id": "mat1", "d":0.03}, {"id": "mat2", "d":0.03}],
                            "roi": mock_linrgb_processing_image_rois[4]
                        },
                        {
                            "stack": [{"id": "mat2", "d":0.03}],
                            "roi": mock_linrgb_processing_image_rois[5]
                        }
                    ]
                }
            ]
        }
    }
    mock_file = mocker.mock_open(read_data=json.dumps(mock_data))
    mocker.patch("spectrophane.io.material_image_processing.import_image", return_value=mock_linrgb_processing_image_rois[0])
    mocker.patch('builtins.open', mock_file)
    transform_mock = mocker.patch('spectrophane.io.material_image_processing.import_image', return_value=mock_linrgb_processing_image_rois[0])
    
    result_stack_data, result_xyz_array = parse_material_characterization_data()
    print(result_stack_data)
    print(result_xyz_array)
    assert np.all(np.isfinite(result_xyz_array))
    assert np.all(np.isfinite(result_stack_data.material_nums))
    assert np.all(np.isfinite(result_stack_data.thicknesses))
    assert np.all(np.isfinite(result_stack_data.stack_counts))
    assert result_xyz_array.shape == (4,3)
    assert isinstance(result_stack_data, StackData)
    assert len(result_stack_data.material_list) == 2
    assert result_stack_data.material_nums.shape == (4,2)
    assert result_stack_data.thicknesses.shape == (4,2)
    assert result_stack_data.stack_counts.shape == (4,)
