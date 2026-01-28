import os
import struct
import numpy as np
import pytest

from spectrophane.lithophane.export import STLTessellationBackend, Box

def test_box_tessellation_triangle_count():
    backend = STLTessellationBackend(base_path="dummy", material_names=["mat"], binary=False)

    box = Box(0, 1, 0, 1, 0, 1)
    triangles = backend._tessellate_box(box)

    # STL box = 6 faces * 2 triangles
    assert triangles.shape == (12, 12)

def test_box_tessellation_normals_are_unit_length():
    backend = STLTessellationBackend(base_path="dummy", material_names=["mat"], binary=False)

    box = Box(0, 2, 0, 2, 0, 2)
    triangles = backend._tessellate_box(box)

    normals = triangles[:, 9:12]
    lengths = np.linalg.norm(normals, axis=1)

    assert np.allclose(lengths, 1.0)

def test_ascii_stl_output(tmp_path):
    base_path = tmp_path / "model"

    backend = STLTessellationBackend(base_path=str(base_path), material_names=["white"], binary=False)

    backend.begin(0)
    backend.add(Box(0, 1, 0, 1, 0, 1))
    backend.add(Box(1, 3, 2, 3, 1, 3))
    paths = backend.end()

    assert len(paths) == 1
    stl_path = paths[0]
    assert os.path.exists(stl_path)

    text = open(stl_path).read()

    assert text.startswith("solid material_white")
    assert text.strip().endswith("endsolid material_white")

    # One facet per triangle
    assert text.count("facet normal") == 24
    assert text.count("vertex") == 72


def test_binary_stl_output(tmp_path):
    base_path = tmp_path / "model"

    backend = STLTessellationBackend(base_path=str(base_path), material_names=["white"], binary=True)

    backend.begin(0)
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

    assert "white".encode("ascii") in header
    assert tri_count == 12

    # Each triangle is exactly 50 bytes
    expected_size = 80 + 4 + 12 * 50
    assert len(data) == expected_size
