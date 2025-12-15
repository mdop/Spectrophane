import jax.numpy as jnp
import numpy as np
from pytest import CaptureFixture, MonkeyPatch
import pytest
import os
import json
from pathlib import Path
from dataclasses import dataclass
from spectrophane.io.material_parameter import print_color_comparison, color_str, save_parameter, extract_spectral_blocks, plot_parameter
from spectrophane.core.dataclasses import SpectralBlock



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


def test_extract_spectral_blocks():
    # Mock parameters and metadata
    class MockParam:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    params = [MockParam(field1=[1, 2, 3], field2=None), MockParam(field3=[4, 5, 6])]
    metadata = [
        {"id": "1", "name": "Material A", "plotcolor": "FF0000"},
        {"id": "2", "name": "Material B", "plotcolor": "00FF00"},
    ]
    wavelengths = np.array([400, 500, 600])

    blocks = extract_spectral_blocks(params, metadata, wavelengths)

    assert len(blocks) == 2
    assert blocks[0].parameter == "field1"
    assert blocks[0].material_id == "1"
    assert blocks[0].material_name == "Material A"
    assert blocks[0].plotcolor == "FF0000"
    assert np.array_equal(blocks[0].values, np.array([1, 2, 3]))
    assert np.array_equal(blocks[0].wavelengths, wavelengths)

    assert blocks[1].parameter == "field3"
    assert blocks[1].material_id == "2"
    assert blocks[1].material_name == "Material B"
    assert blocks[1].plotcolor == "00FF00"
    assert np.array_equal(blocks[1].values, np.array([4, 5, 6]))
    assert np.array_equal(blocks[1].wavelengths, wavelengths)

def test_plot_parameter():
    # Mock SpectralBlock instances
    block1 = SpectralBlock(
        wavelengths=np.array([400, 500, 600]),
        values=np.array([1, 2, 3]),
        material_id="1",
        material_name="Material A",
        plotcolor="FF0000",
        parameter="field1",
    )
    block2 = SpectralBlock(
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


@dataclass
class MaterialParams:
    param1: int
    param2: float
    param3: jnp.ndarray

def test_save_parameter(tmpdir, mocker):
    # Setup
    material_data = ["material1", "material2"]
    parameter = MaterialParams(param1=10, param2=3.14, param3=jnp.array([1.0, 2.0, 3.0]))
    filename = os.path.join(tmpdir, "test_params.json")
    no_overwrite = True
    mocker.patch("spectrophane.io.data_io.get_user_resource_path", return_value=Path(filename))

    # Act
    save_parameter(filename, material_data, parameter, no_overwrite)

    # Assert
    with open(filename, "r") as f:
        result = json.load(f)

    assert result["materials"] == material_data
    assert result["parameter"]["param1"] == 10
    assert result["parameter"]["param2"] == 3.14
    assert result["parameter"]["param3"] == [1.0, 2.0, 3.0]

def test_save_parameter_overwrite(tmpdir, mocker):
    # Setup
    material_data = ["material1", "material2"]
    parameter = MaterialParams(param1=10, param2=3.14, param3=jnp.array([1.0, 2.0, 3.0]))
    filename = os.path.join(tmpdir, "test_params.json")
    no_overwrite = False
    mocker.patch("spectrophane.io.data_io.get_user_resource_path", return_value=Path(filename))

    save_parameter(filename, material_data, parameter, no_overwrite)
    parameter.param1 = 20
    save_parameter(filename, material_data, parameter, no_overwrite)

    # Assert
    with open(filename, "r") as f:
        result = json.load(f)

    assert result["parameter"]["param1"] == 20

    no_overwrite = True
    with pytest.raises(FileExistsError):
        save_parameter(filename, material_data, parameter, no_overwrite)