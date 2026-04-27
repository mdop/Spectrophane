import pytest
import numpy as np

from spectrophane.core.dataclasses import VoxelGeometry, SolidPrimitive, Box
from spectrophane.lithophane.solid_generation import PerVoxelBoxBuilder, GreedyMeshingBoxBuilder, PrismBuilder

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



#-------------------------------------------------POLYGONS-------------------------------------------

#helper
def normalize_loop(loop):
    """Rotate loop so smallest point is first (for comparison)"""
    min_idx = min(range(len(loop)), key=lambda i: loop[i])
    return loop[min_idx:] + loop[:min_idx]


def polygon_area(builder, poly):
    return abs(builder.signed_area(poly))

#fixtures
@pytest.fixture
def prism_builder():
    return PrismBuilder()

@pytest.fixture
def simple_square_mask():
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    #1
    return mask


@pytest.fixture
def filled_block_mask():
    mask = np.ones((2, 2), dtype=bool)
    #11
    #11
    return mask


@pytest.fixture
def donut_mask():
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False
    #11111
    #11111
    #11011
    #11111
    #11111
    return mask


@pytest.fixture
def diagonal_touch_mask():
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    #100
    #010
    #000
    return mask


#extract_edges
def test_prism_extract_edges_single_pixel(prism_builder, simple_square_mask):
    edges = prism_builder.extract_edges(simple_square_mask)

    # A single pixel should produce 4 edges
    assert len(edges) == 4


def test_prism_extract_edges_block(prism_builder, filled_block_mask):
    edges = prism_builder.extract_edges(filled_block_mask)

    # 2x2 block has outer perimeter = 8 edges
    assert len(edges) == 8


#trace_loops
def test_prism_trace_loops_single_square(prism_builder, simple_square_mask):
    edges = prism_builder.extract_edges(simple_square_mask)
    loops = prism_builder.trace_loops(edges)

    assert len(loops) == 1
    assert len(loops[0]) == 4


def test_prism_trace_loops_multiple_components(prism_builder, diagonal_touch_mask):
    edges = prism_builder.extract_edges(diagonal_touch_mask)
    loops = prism_builder.trace_loops(edges)

    # diagonals should NOT connect
    assert len(loops) == 2


def test_prism_trace_loops_broken_raises(prism_builder):
    edges = {((0, 0), (1, 0)), ((1, 0), (2, 0))}  # open chain

    with pytest.raises(RuntimeError):
        prism_builder.trace_loops(edges)


#signed_area
def test_prism_signed_area_orientation(prism_builder):
    square_ccw = [(0,0), (1,0), (1,1), (0,1)]
    square_cw = list(reversed(square_ccw))

    assert prism_builder.signed_area(square_ccw) > 0
    assert abs(prism_builder.signed_area(square_ccw) + prism_builder.signed_area(square_cw)) < 1e-4


#point_in_polygon
def test_prism_point_inside_polygon(prism_builder):
    poly = [(0,0), (4,0), (4,4), (0,4)]
    assert prism_builder.point_in_polygon((2,2), poly)


def test_prism_point_outside_polygon(prism_builder):
    poly = [(0,0), (4,0), (4,4), (0,4)]
    assert not prism_builder.point_in_polygon((5,5), poly)


def test_prism_point_on_edge_behavior(prism_builder):
    poly = [(0,0), (4,0), (4,4), (0,4)]

    # Edge case: algorithm intentionally ignores boundary strictness
    result = prism_builder.point_in_polygon((0,2), poly)
    assert isinstance(result, bool)

def test_prism_point_near_horizontal_edge(prism_builder):
    poly = [(0,0), (4,0), (4,4), (0,4)]

    # tests the "ignore horizontal edges" logic
    assert prism_builder.point_in_polygon((2, 1e-12), poly)


#simplify_colinear
def test_prism_simplify_removes_colinear_points(prism_builder):
    poly = [(0,0), (1,0), (2,0), (2,1), (0,1)]

    simplified = prism_builder.simplify_colinear(poly)

    assert (1,0) not in simplified
    assert len(simplified) < len(poly)


def test_prism_simplify_triangle_unchanged(prism_builder):
    poly = [(0,0), (1,0), (0,1)]
    assert prism_builder.simplify_colinear(poly) == poly


#mask_to_polygons
def test_prism_prism_mask_to_polygons_simple_square(prism_builder, simple_square_mask):
    polys = prism_builder.mask_to_polygons(simple_square_mask)

    assert len(polys) == 1
    assert len(polys[0]["holes"]) == 0


def test_prism_mask_to_polygons_with_hole(prism_builder, donut_mask):
    polys = prism_builder.mask_to_polygons(donut_mask)

    assert len(polys) == 1
    assert len(polys[0]["holes"]) == 1


def test_prism_mask_to_polygons_area_conservation(prism_builder, donut_mask):
    polys = prism_builder.mask_to_polygons(donut_mask)

    outer_area = polygon_area(prism_builder, polys[0]["outer"])
    hole_area = sum(polygon_area(prism_builder, h) for h in polys[0]["holes"])

    # Expected area: 24 (25 total - 1 hole pixel)
    assert pytest.approx(outer_area - hole_area, rel=1e-6) == 24


#Integration-style test
def test_prism_pipeline_consistency(prism_builder, donut_mask):
    edges = prism_builder.extract_edges(donut_mask)
    loops = prism_builder.trace_loops(edges)
    polys = prism_builder.mask_to_polygons(donut_mask)

    # Ensure loop count matches outer + holes
    loop_count = len(loops)
    poly_loops = sum(1 + len(p["holes"]) for p in polys)

    assert loop_count == poly_loops
