"""
Make dev_tools.sanitize_images importable in tests without packaging it.

This loads dev_tools/sanitize_images.py by filename and places the loaded
module into sys.modules under the name 'dev_tools.sanitize_images', so
tests can use:

    from dev_tools.sanitize_images import clean_exif, is_supported_image

This approach does NOT require dev_tools/__init__.py and does not modify your package.
"""

import importlib.util
import sys
from pathlib import Path
import pytest
from PIL import Image
import subprocess

# Resolve repository root: tests/ is inside repo; parent of tests/ is repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Path to the sanitize_images.py file in your repo (adjust if different)
MODULE_PATH = REPO_ROOT / "dev_tools" / "sanitize_images.py"

if not MODULE_PATH.exists():
    # Fail early with a clear error (pytest will show this)
    raise FileNotFoundError(
        f"dev_tools.sanitize_images not found at expected path: {MODULE_PATH}"
    )

MODULE_NAME = "dev_tools.sanitize_images"

# Only load and inject if not already present (prevents double-loading)
if MODULE_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(MODULE_NAME, str(MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    # Put it in sys.modules *before* executing to allow intra-module imports that refer to itself
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)



@pytest.fixture
def tmp_image(tmp_path):
    """
    Create a small JPEG image that exiftool can modify.
    Returns the path.
    """
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (32, 32), color="red")
    img.save(img_path, "JPEG")
    return img_path


@pytest.fixture
def ensure_exiftool():
    """Skip tests that require exiftool if it's not installed."""
    if subprocess.call(["which", "exiftool"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        pytest.skip("exiftool not installed")