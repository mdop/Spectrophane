from spectrophane.application.cli.main import app


def test_training_success(
    runner,
    monkeypatch,
    tmp_path,
):
    dummy_calib = tmp_path / "calib.json"
    dummy_calib.write_text("{}")

    def mock_pipeline(**kwargs):
        return {"terminal_color_str": "OK"}

    monkeypatch.setattr(
        "spectrophane.application.cli.training.parameter_training_pipeline",
        mock_pipeline,
    )

    result = runner.invoke(
        app,
        [
            "training",
            "--calibration-file", str(dummy_calib),
            "--training-steps", "10",
            "--lr", "0.01",
        ],
    )

    assert result.exit_code == 0
    assert "Training completed successfully" in result.stdout
    assert "OK" in result.stdout
