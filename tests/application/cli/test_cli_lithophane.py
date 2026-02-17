import pytest
from spectrophane.application.cli.main import app


# ------------------------------------------------------------------
# CLI argument builder
# ------------------------------------------------------------------

def build_cli_args(
    dummy_parameter_file,
    dummy_spectral_file,
    dummy_image,
    tmp_path,
    **overrides,
):
    args = [
        "--parameter-file", str(dummy_parameter_file),
        "--spectral-config", str(dummy_spectral_file),
        "--image-path", str(dummy_image),
        "--output-base", str(tmp_path / "out"),
        "--layer-count", "3",
        "--layer-thickness", "1",
        "--resolution", "10", "10",
        "--pixel-size", "0.1", "0.1",
    ]

    for key, value in overrides.items():
        cli_flag = f"--{key.replace('_', '-')}"

        if isinstance(value, bool):
            if value:
                args.append(cli_flag)
        elif isinstance(value, (list, tuple)):
            args.extend([cli_flag, *map(str, value)])
        else:
            args.extend([cli_flag, str(value)])

    return args


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_lithophane_success(
    runner,
    mock_parameter,
    mock_spectral,
    mock_inverter,
    mock_image_generation,
    dummy_image,
    dummy_parameter_file,
    dummy_spectral_file,
    tmp_path,
):
    mock_parameter()
    mock_spectral()

    result = runner.invoke(
        app,
        build_cli_args(
            dummy_parameter_file,
            dummy_spectral_file,
            dummy_image,
            tmp_path,
        ),
    )

    assert result.exit_code == 0
    assert "Lithophane generation complete" in result.stdout


def test_invalid_material_name(
    runner,
    mock_parameter,
    dummy_image,
    dummy_parameter_file,
    dummy_spectral_file,
    tmp_path,
):
    mock_parameter(material_names=("A", "B"))

    result = runner.invoke(
        app,
        build_cli_args(
            dummy_parameter_file,
            dummy_spectral_file,
            dummy_image,
            tmp_path,
            material_names="INVALID",
        ),
    )

    assert result.exit_code > 0
    assert "Invalid material names" in result.stdout


def test_invalid_illuminator(
    runner,
    mock_parameter,
    mock_spectral,
    dummy_image,
    dummy_parameter_file,
    dummy_spectral_file,
    tmp_path,
):
    mock_parameter(material_names=("A",))
    mock_spectral(illuminators=("D50",))

    result = runner.invoke(
        app,
        build_cli_args(
            dummy_parameter_file,
            dummy_spectral_file,
            dummy_image,
            tmp_path,
            layer_count=2,
            illuminator="D65",
        ),
    )

    assert result.exit_code > 0
    assert "Illuminator" in result.stdout
