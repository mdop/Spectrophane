from typer.testing import CliRunner
from pathlib import Path

from spectrophane.application.cli.main import app

runner = CliRunner()
calibration_file = "training_data/default.json"
output_path = Path.home() / "Spectrophane" / "material_parameter" / "default.json"

result = runner.invoke(
    app,
    [
        "training",
        "--calibration-file", calibration_file,
        "--output-path", output_path,
        "--training-steps", 2000,
        "--lr", 0.1,
    ],
)
print(result.exit_code)
print(result.stdout)
print(result.stderr)