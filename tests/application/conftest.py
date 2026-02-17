import pytest
from pathlib import Path
from typer.testing import CliRunner
from PIL import Image
import numpy as np


@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def dummy_image(tmp_path: Path):
    img_path = tmp_path / "img.png"
    img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    img.save(img_path)
    return img_path

@pytest.fixture
def dummy_parameter_file(tmp_path: Path):
    path = tmp_path / "params.json"
    path.write_text("{}")
    return path


@pytest.fixture
def dummy_spectral_file(tmp_path: Path):
    path = tmp_path / "spectral.json"
    path.write_text("{}")
    return path



@pytest.fixture
def mock_parameter(monkeypatch):
    def _apply(material_names=("A", "B")):
        def mock_file_to_parameter(path, local_path=False, material_filter=None):
            return [{"name": n} for n in material_names], object()

        monkeypatch.setattr(
            "spectrophane.application.cli.lithophane.file_to_parameter",
            mock_file_to_parameter,
        )

    return _apply


@pytest.fixture
def mock_spectral(monkeypatch):
    def _apply(illuminators=("D65",), observers=("CIE1931",)):
        def mock_file_to_spectral_helper(path, local_path=False):
            class LS:
                names = list(illuminators)

            class OBS:
                names = list(observers)

            return LS(), OBS()

        monkeypatch.setattr(
            "spectrophane.application.cli.lithophane.file_to_spectral_helper",
            mock_file_to_spectral_helper,
        )

    return _apply


@pytest.fixture
def mock_inverter(monkeypatch):
    monkeypatch.setattr(
        "spectrophane.application.cli.lithophane.parameter_to_inverter",
        lambda **kwargs: "INVERTER",
    )


@pytest.fixture
def mock_image_generation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "spectrophane.application.cli.lithophane.image_to_lithophane",
        lambda **kwargs: ([str(tmp_path / "out.stl")], None, None),
    )