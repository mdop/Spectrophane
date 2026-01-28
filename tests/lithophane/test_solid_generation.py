import pytest
import numpy as np

from spectrophane.core.dataclasses import VoxelGeometry, SolidPrimitive, Box
from spectrophane.lithophane.solid_generation import PerVoxelBoxBuilder

@pytest.fixture
def simple_geometry():
    return VoxelGeometry(
                materials=np.array([[[0, 1], [3, 2]], [[4, 2], [2, 5]]]),
                layer_thickness=np.array([0.1, 0.2]),
                voxel_size_xy=(1.0, 1.0),
                material_names=["A", "B", "C", "D", "E", "F"]
            )

def test_per_voxel_solids_returns_boxes(simple_geometry):
    builder = PerVoxelBoxBuilder()
    
    for i in range(len(simple_geometry.material_names)):
        solids = list(builder.solids_for_material(simple_geometry, i))
        assert len(solids) == np.sum(simple_geometry.materials == i)
        for solid in solids:
            assert isinstance(solid, Box)

def test_per_voxel_solids_skips_non_matches(simple_geometry):
    builder = PerVoxelBoxBuilder()

    solids = list(builder.solids_for_material(simple_geometry, 10))

    assert len(solids) == 0

@pytest.mark.parametrize("material_id, expected_x0, expected_x1, expected_y0, expected_y1, expected_z0, expected_z1", [
    (0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.1),
    (1, 0.0, 1.0, 0.0, 1.0, 0.1, 0.3),
    (3, 0.0, 1.0, 1.0, 2.0, 0.0, 0.1),
    (4, 1.0, 2.0, 0.0, 1.0, 0.0, 0.1),
    (5, 1.0, 2.0, 1.0, 2.0, 0.1, 0.3),
])
def test_per_voxel_solids_box_geometry_check(
    simple_geometry, material_id, expected_x0, expected_x1, expected_y0, expected_y1, expected_z0, expected_z1
):
    builder = PerVoxelBoxBuilder()

    solid = list(builder.solids_for_material(simple_geometry, material_id))[0]
    assert isinstance(solid, Box)
    assert solid.x0 == pytest.approx(expected_x0)
    assert solid.x1 == pytest.approx(expected_x1)
    assert solid.y0 == pytest.approx(expected_y0)
    assert solid.y1 == pytest.approx(expected_y1)
    assert solid.z0 == pytest.approx(expected_z0)
    assert solid.z1 == pytest.approx(expected_z1)