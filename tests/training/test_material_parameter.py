import jax.numpy as jnp
import numpy as np
from pytest import CaptureFixture, MonkeyPatch
import pytest
import os
import json
from pathlib import Path
from dataclasses import dataclass, fields

from spectrophane.training.material_parameter import SpectrumPlotLineData, print_color_comparison, color_str, save_parameter, extract_spectral_plot_series, plot_parameter, load_parameter
from spectrophane.core.dataclasses import MaterialParams


@dataclass
class MockMaterialParam:
    param1: int
    param2: float
    param3: np.ndarray
    param4: np.ndarray
    absorption_coeff: jnp.ndarray
    wl_start: float
    wl_step: float

@pytest.fixture
def mock_material_param():
    return MockMaterialParam(
        param1=10,
        param2=3.14,
        param3=np.array([[1.0,1.1,1.3,1.5,1.7,1.9],[10.01,10.1,10.3,10.5,10.7,10.9]]),
        param4=np.random.rand(2,6),
        absorption_coeff=jnp.array([[1,2,3,4,5,6]]*2),
        wl_start=400,
        wl_step=10
    )

@pytest.fixture
def mock_material_metadata():
    return [
                {"id": "1", "name": "Material A", "plotcolor": "FF0000"},
                {"id": "2", "name": "Material B", "plotcolor": "00FF00"},
            ]

def test_terminal_width_wrapping(monkeypatch: MonkeyPatch, capfd: CaptureFixture[str]):
    # Mock terminal width to 20 characters
    monkeypatch.setattr('shutil.get_terminal_size', lambda *args, **kwargs: (12, 20))

    # Create example arrays of calculated and actual colors
    calculated_colors = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    actual_colors =     jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    # Call the function
    print_color_comparison(calculated_colors, actual_colors)

    # Capture output
    captured = capfd.readouterr()
    output = captured.out

    print(output)
    assert len(output.strip().split('\n')) == 4

def test_no_wrapping_needed(monkeypatch: MonkeyPatch, capfd: CaptureFixture[str]):
    # Mock terminal width to a large value (no wrapping needed)
    monkeypatch.setattr('shutil.get_terminal_size', lambda: (1000, 100))

    calculated_colors = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    actual_colors = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    # Call the function
    print_color_comparison(calculated_colors, actual_colors)

    # Capture output
    captured = capfd.readouterr()
    output = captured.out

    # Check that exactly two lines were printed
    assert len(output.strip().split('\n')) == 2


def test_extract_spectral_blocks(mock_material_param, mock_material_metadata):
    blocks = extract_spectral_plot_series(mock_material_param, mock_material_metadata)
    wavelengths = [400,410,420,430,440,450]

    assert len(blocks) == len(mock_material_metadata) * 3
    assert blocks[0].parameter == "param3"
    assert blocks[0].material_id == "1"
    assert blocks[0].material_name == "Material A"
    assert blocks[0].plotcolor == "FF0000"
    assert np.array_equal(blocks[0].values, mock_material_param.param3[0])
    assert np.array_equal(blocks[0].wavelengths, wavelengths)

    assert blocks[1].parameter == "param4"
    assert blocks[1].material_id == "1"
    assert blocks[1].material_name == "Material A"
    assert blocks[1].plotcolor == "FF0000"
    assert np.array_equal(blocks[1].values, mock_material_param.param4[0])
    assert np.array_equal(blocks[1].wavelengths, wavelengths)

    assert blocks[-1].material_name == "Material B"
    assert np.array_equal(blocks[-1].values, mock_material_param.absorption_coeff[-1])

def test_plot_parameter():
    # Mock SpectralBlock instances
    block1 = SpectrumPlotLineData(
        wavelengths=np.array([400, 500, 600]),
        values=np.array([1, 2, 3]),
        material_id="1",
        material_name="Material A",
        plotcolor="FF0000",
        parameter="field1",
    )
    block2 = SpectrumPlotLineData(
        wavelengths=np.array([400, 500, 600]),
        values=np.array([4, 5, 6]),
        material_id="2",
        material_name="Material B",
        plotcolor="00FF00",
        parameter="field2",
    )
    blocks = [block1, block2]

    fig = plot_parameter(blocks)

    assert fig is not None
    assert fig.layout.title.text is None  # No title by default
    assert fig.layout.xaxis.title.text == "Wavelength (nm)"
    assert fig.layout.yaxis.title.text == "Value"
    assert fig.layout.legend.title.text == "Material"
    assert len(fig.data) == 2
    assert fig.data[0].x is not None
    assert fig.data[0].y is not None
    assert fig.data[0].name == "Material A"
    assert fig.data[1].name == "Material B"


def test_save_parameter(tmpdir, mocker, mock_material_param, mock_material_metadata):
    filename = os.path.join(tmpdir, "test_params.json")
    no_overwrite = True
    mocker.patch("spectrophane.io.resources.get_user_resource_path", return_value=Path(filename))

    save_parameter(filename, mock_material_metadata, mock_material_param, no_overwrite)

    with open(filename, "r") as f:
        result = json.load(f)

    assert result["materials"] == mock_material_metadata
    assert result["parameter"]["param1"] == mock_material_param.param1
    assert result["parameter"]["param2"] == mock_material_param.param2
    assert result["parameter"]["param3"] == mock_material_param.param3.tolist()
    assert result["parameter"]["absorption_coeff"] == mock_material_param.absorption_coeff.tolist()

def test_save_parameter_overwrite(tmpdir, mocker, mock_material_param, mock_material_metadata):
    filename = os.path.join(tmpdir, "test_params.json")
    no_overwrite = False
    mocker.patch("spectrophane.io.resources.get_user_resource_path", return_value=Path(filename))

    save_parameter(filename, mock_material_metadata, mock_material_param, no_overwrite)
    mock_material_param.param1 = 20
    save_parameter(filename, mock_material_metadata, mock_material_param, no_overwrite)

    with open(filename, "r") as f:
        result = json.load(f)

    assert result["parameter"]["param1"] == 20

    no_overwrite = True
    with pytest.raises(FileExistsError):
        save_parameter(filename, mock_material_metadata, mock_material_param, no_overwrite)

def test_load_parameter(mocker, mock_material_metadata):
    file_dict = {}
    file_dict["materials"] = mock_material_metadata
    file_dict["parameter"] = {  
                                "absorption_coeff": [[1, 2, 3, 4, 5, 6]] * 2,
                                "wl_start": 400,
                                "wl_step": 10,
                             }
    mocker.patch("spectrophane.training.material_parameter.get_json_resource", return_value=file_dict)
    
    result = load_parameter("mock.json")
    print("TYPE:", type(result.absorption_coeff))
    assert isinstance(result.absorption_coeff, np.ndarray)
    assert np.allclose(result.absorption_coeff, np.array([[1, 2, 3, 4, 5, 6]] * 2))
    assert result.scattering_coeff is None

def test_parameter_save_loading(mocker, mock_material_metadata, tmp_path):
    parameter = MaterialParams( wl_start=400,
                                wl_step=10,
                                absorption_coeff=np.array([[1,2,3,4,5,6]]*2))
    mock_path = tmp_path / "test.json"
    mocker.patch("spectrophane.io.resources.get_user_resource_path", return_value=mock_path)

    save_parameter("test", mock_material_metadata, parameter)
    result = load_parameter("mocked_test") # should complain if patching failed

    for field in fields(result):
        if isinstance(getattr(parameter, field.name), np.ndarray):
            assert np.allclose(getattr(parameter, field.name), getattr(result, field.name))
        else:
            assert getattr(parameter, field.name) == getattr(result, field.name)
