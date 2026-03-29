from typer.testing import CliRunner
from pathlib import Path

from spectrophane.application.cli.main import app

runner = CliRunner()
image_path = Path(__file__).parent / "test_image.jpg"
output_base_path = Path.home() / "Spectrophane" / "test_output" / "test"
output_base_path.parent.mkdir(parents=True, exist_ok=True)

result = runner.invoke(
    app,
    [
        "lithophane",
        "--image-path", str(image_path),
        "--output-base", str(output_base_path),
        "--layer-count", 10,
        "--layer-thickness", 0.03,
        "--top-thickness", 0.03,
        "--resolution", 200,300,
        "--pixel-size", 0.4,0.4,
    ],
)
print(result.exit_code)
print(result.stdout)
print(result.stderr)