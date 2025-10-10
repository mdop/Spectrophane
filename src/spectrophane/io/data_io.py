from importlib import resources
from pathlib import Path
import pathlib

#default data
#TODO: customize user data dir
USER_DATA_DIR = Path.home / ".Spectrophane"
PACKAGE_RESOURCES_ROOT = "spectrophane.resources"

#resources.files("spectrophane.resources.material_data").joinpath("default.json").read_text()

def get_package_resource_path(project_resource_path: str) -> pathlib.Path | None:
    """Determines Path object of a project resource from the root resources directory. E.g. "material_data/images/001.jnp". Returns None if resource does not exist """
    resource_path_obj = str(Path(project_resource_path).parent)
    resource_filename = Path(project_resource_path).name
    resource_anchor = resource_path_obj.replace("/", ".").replace("\\", ".")
    resource = resources.files(PACKAGE_RESOURCES_ROOT + "." + resource_anchor).joinpath(resource_filename)
    if resource.exists():
        return resource
    else:
        return None

def get_user_resource_path(project_resource_path: str) -> pathlib.Path | None:
    """Determines Path object of a user supplied resource. Returns if resource does not exist."""
    path = USER_DATA_DIR / project_resource_path
    if path.exists():
        return path
    else:
        return None

def get_resource_path(project_resource_path: str) -> pathlib.Path | None:
    """Determines Path object for a requested resource path. User supplied data are prioritized over package provided resources. Returns None if no matching resource in either user data or package data is found."""
    path = get_user_resource_path(project_resource_path)
    if path is None:
        path = get_package_resource_path(project_resource_path)
    return path

