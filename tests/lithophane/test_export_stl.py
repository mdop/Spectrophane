import os
import struct
import numpy as np
import pytest

from spectrophane.lithophane.export import STLTessellationBackend, TriangleTessellatingBackend
from spectrophane.core.dataclasses import AffineTransform, Box

@pytest.fixture
def transform():
    return AffineTransform(1,1, 0.001)

@pytest.fixture
def material_data():
    return [{"name": "mat"}]

def test_triangle_mesh_box_count(material_data):
    box = Box(0, 1, 0, 1, 0, 1)
    triangle_mesh = TriangleTessellatingBackend._mesh_box(box)

    assert triangle_mesh.vertices.shape == (8, 3)
    assert triangle_mesh.triangles.shape == (12, 3)

def test_stl_box_tessellation_normals_are_unit_length(tmp_path, material_data):
    base_path = tmp_path / "model"
    backend = STLTessellationBackend(base_path=str(base_path), material_data=material_data, binary=False)
    backend.begin(0, AffineTransform(1,1,0.001))

    box = Box(0, 2, 0, 2, 0, 2)
    triangles = backend._mesh_box(box)
    stl_triangle_array = backend.add_mesh(triangles)

    normals = stl_triangle_array[:, 9:12]
    lengths = np.linalg.norm(normals, axis=1)

    assert np.allclose(lengths, 1.0)

def test_ascii_stl_output(tmp_path, transform, material_data):
    base_path = tmp_path / "model"

    backend = STLTessellationBackend(base_path=str(base_path), material_data=material_data, binary=False)

    backend.begin(0, transform)
    backend.add(Box(0, 1, 0, 1, 0, 1))
    backend.add(Box(1, 3, 2, 3, 1, 3))
    paths = backend.end()

    assert len(paths) == 1
    stl_path = paths[0]
    assert os.path.exists(stl_path)

    text = open(stl_path).read()

    assert text.startswith("solid material_mat")
    assert text.strip().endswith("endsolid material_mat")

    # One facet per triangle
    assert text.count("facet normal") == 24
    assert text.count("vertex") == 72


@pytest.mark.parametrize("filename",  ["model", "model.stl"])
def test_binary_stl_output(tmp_path, filename, transform, material_data):
    base_path = tmp_path / filename

    backend = STLTessellationBackend(base_path=str(base_path), material_data=material_data, binary=True)

    backend.begin(0, transform)
    backend.add(Box(0, 2, 0, 3, 1, 5))
    paths = backend.end()

    assert len(paths) == 1
    stl_path = paths[0]
    assert os.path.exists(stl_path)

    with open(stl_path, "rb") as f:
        data = f.read()

    # Binary STL layout
    header = data[:80]
    tri_count = struct.unpack("<I", data[80:84])[0]

    assert "mat".encode("ascii") in header
    assert tri_count == 12

    # Each triangle is exactly 50 bytes
    expected_size = 80 + 4 + 12 * 50
    assert len(data) == expected_size
