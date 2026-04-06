import pytest
import numpy as np

from spectrophane.core.dataclasses import VoxelGeometry, SolidPrimitive, Box
from spectrophane.lithophane.solid_generation import PerVoxelBoxBuilder, GreedyMeshingBoxBuilder

def voxel_count_from_boxes(boxes, geometry):
    """helper to compare total box coverage"""
    pixel_x, pixel_y = geometry.voxel_size_xy
    cumulative_z = np.concatenate([[0.0], np.cumsum(geometry.layer_thickness)])

    count = 0
    for b in boxes:
        dx = int(round((b.x1 - b.x0) / pixel_x))
        dy = int(round((b.y1 - b.y0) / pixel_y))

        z0 = np.searchsorted(cumulative_z, b.z0, side="left")
        z1 = np.searchsorted(cumulative_z, b.z1, side="left")
        dz = z1 - z0

        count += dx * dy * dz
    return count

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

def test_greedy_meshing_preserves_voxel_count(simple_geometry):
    greedy = GreedyMeshingBoxBuilder()
    naive = PerVoxelBoxBuilder()

    for material_id in range(len(simple_geometry.material_names)):
        greedy_boxes = list(greedy.solids_for_material(simple_geometry, material_id))
        naive_boxes = list(naive.solids_for_material(simple_geometry, material_id))

        greedy_count = voxel_count_from_boxes(greedy_boxes, simple_geometry)
        naive_count = len(naive_boxes)

        assert greedy_count == naive_count

def test_greedy_meshing_reduces__box_count(simple_geometry):
    greedy = GreedyMeshingBoxBuilder()
    naive = PerVoxelBoxBuilder()

    for material_id in range(len(simple_geometry.material_names)):
        greedy_boxes = list(greedy.solids_for_material(simple_geometry, material_id))
        naive_boxes = list(naive.solids_for_material(simple_geometry, material_id))

        assert len(greedy_boxes) <= len(naive_boxes)

def test_greedy_meshing_single_block():
    geometry = VoxelGeometry(
        materials=np.ones((2, 2, 2), dtype=int),
        layer_thickness=np.array([1.0, 1.0]),
        voxel_size_xy=(1.0, 1.0),
        material_names=["A", "B"]
    )

    builder = GreedyMeshingBoxBuilder()
    boxes = list(builder.solids_for_material(geometry, 1))

    assert len(boxes) == 1

    b = boxes[0]
    assert b.x0 == 0.0 and b.x1 == 2.0
    assert b.y0 == 0.0 and b.y1 == 2.0
    assert b.z0 == 0.0 and b.z1 == 2.0

def test_greedy_meshing_merges_rectangle():
    geometry = VoxelGeometry(
        materials=np.ones((2, 2, 1), dtype=int),
        layer_thickness=np.array([1.0]),
        voxel_size_xy=(1.0, 1.0),
        material_names=["A", "B"]
    )

    builder = GreedyMeshingBoxBuilder()
    boxes = list(builder.solids_for_material(geometry, 1))

    assert len(boxes) == 1

    b = boxes[0]
    assert b.x1 - b.x0 == 2.0
    assert b.y1 - b.y0 == 2.0