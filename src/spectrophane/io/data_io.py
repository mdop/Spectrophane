from importlib import resources
from pathlib import Path
#default data
#TODO: customize user data dir
USER_DATA_DIR = Path.home() / ".Spectrophane"
PACKAGE_RESOURCES_ROOT = "spectrophane.resources"

USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_package_resource_path(project_resource_path: str) -> Path | None:
    """Return a concrete filesystem path to a package resource, or None if it doesn't exist."""
    project_path = Path(project_resource_path)
    resource_anchor = project_path.parent.as_posix().replace("/", ".")
    resource_filename = project_path.name
    resource_root = f"{PACKAGE_RESOURCES_ROOT}.{resource_anchor}" if resource_anchor else PACKAGE_RESOURCES_ROOT
    resource = resources.files(resource_root).joinpath(resource_filename)
    if not resource.exists():
        return None
    with resources.as_file(resource) as real_path:
        return real_path

def get_user_resource_path(project_resource_path: str) -> Path | None:
    """Return a path to a user-supplied resource, or None if it doesn't exist."""
    path = USER_DATA_DIR / project_resource_path
    return path if path.exists() else None

def get_resource_path(project_resource_path: str) -> Path | None:
    """
    Return path for a requested resource.
    User-supplied data take priority over package resources.
    Returns None if neither exists.
    """
    user_path = get_user_resource_path(project_resource_path)
    if user_path:
        return user_path
    return get_package_resource_path(project_resource_path)


