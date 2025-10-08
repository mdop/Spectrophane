from importlib import resources
from pathlib import Path

#default data
#TODO: customize user data dir
USER_DATA_DIR = Path.home / ".Spectrophane"
PACKAGE_RESOURCES_ROOT = "spectrophane.resources"

#resources.files("spectrophane.resources.material_data").joinpath("default.json").read_text()

