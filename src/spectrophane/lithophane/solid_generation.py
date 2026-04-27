from abc import ABC, abstractmethod
from collections.abc import Iterator
import numpy as np
from typing import Iterator, Tuple, Sequence
import numpy as np
from collections import defaultdict

from spectrophane.core.dataclasses import VoxelGeometry, SolidPrimitive, Box, Prism

class SolidBuilder(ABC):
    """Generates solid 3D objects from a VoxelGeometry array. Assumes cartesian coordinates."""
    @abstractmethod
    def solids_for_material(self, geometry: VoxelGeometry, material_id: int) -> Iterator[SolidPrimitive]:
        ...


class PerVoxelBoxBuilder(SolidBuilder):
    def solids_for_material(self, geometry: VoxelGeometry, material_id: int):
        total_x, total_y, total_z = geometry.materials.shape
        cumulative_z = np.concatenate([[0.0], np.cumsum(geometry.layer_thickness)])
        pixel_x, pixel_y = geometry.voxel_size_xy
        for x_idx in range(total_x):
            for y_idx in range(total_y):
                for z_idx in range(total_z):
                    if geometry.materials[x_idx, y_idx, z_idx] != material_id:
                        continue
                    yield Box(x_idx * pixel_x, (x_idx + 1) * pixel_x,
                              y_idx * pixel_y, (y_idx + 1) * pixel_y,
                              cumulative_z[z_idx], cumulative_z[z_idx+1])


class GreedyMeshingBoxBuilder(SolidBuilder):
    def solids_for_material(self, geometry: VoxelGeometry, material_id: int) -> Iterator[SolidPrimitive]:
        materials = geometry.materials
        total_x, total_y, total_z = materials.shape

        pixel_x, pixel_y = geometry.voxel_size_xy
        cumulative_z = np.concatenate([[0.0], np.cumsum(geometry.layer_thickness)])

        # Track which voxels are already consumed
        visited = np.zeros_like(materials, dtype=bool)

        for x0 in range(total_x):
            for y0 in range(total_y):
                for z0 in range(total_z):
                    # Skip if not target material or already merged
                    if visited[x0, y0, z0] or materials[x0, y0, z0] != material_id:
                        continue

                    # fitting startin point, expand in x
                    x1 = x0
                    while x1 < total_x:
                        if visited[x1, y0, z0] or materials[x1, y0, z0] != material_id:
                            break
                        x1 += 1

                    # expand in y
                    y1 = y0
                    while y1 < total_y:
                        valid = True
                        for x in range(x0, x1):
                            if visited[x, y1, z0] or materials[x, y1, z0] != material_id:
                                valid = False
                                break
                        if not valid:
                            break
                        y1 += 1

                    # expand in z
                    z1 = z0
                    while z1 < total_z:
                        valid = True
                        for x in range(x0, x1):
                            for y in range(y0, y1):
                                if visited[x, y, z1] or materials[x, y, z1] != material_id:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                        z1 += 1

                    # mark visited and emit
                    visited[x0:x1, y0:y1, z0:z1] = True
                    yield Box(
                        x0 * pixel_x, x1 * pixel_x,
                        y0 * pixel_y, y1 * pixel_y,
                        cumulative_z[z0], cumulative_z[z1],
                    )

Point = Tuple[int, int]
Edge = Tuple[Point, Point]

class PrismBuilder(SolidBuilder):
    def solids_for_material(self, geometry: VoxelGeometry, material_id: int) -> Iterator[Prism]:
        """Yields prisms with holes for the Voxelmap and the given material_id"""
        materials = geometry.materials
        total_z = materials.shape[2]

        cumulative_z = np.concatenate([[0.0], np.cumsum(geometry.layer_thickness)])
        pixel_size_x, pixel_size_y = geometry.voxel_size_xy
        mask = (materials == material_id)

        for z in range(total_z):
            if not mask.any():
                continue

            polygons = self.mask_to_polygons(mask[z])

            z0 = cumulative_z[z]
            z1 = cumulative_z[z + 1]

            for poly in polygons:
                outer = self.scale_polygon(poly["outer"], pixel_size_x, pixel_size_y)
                holes = [self.scale_polygon(h, pixel_size_x, pixel_size_y) for h in poly["holes"]]

                yield Prism(outer=outer, holes=holes, z0=z0, z1=z1)
    
    def mask_to_polygons(self, layer_mask: np.ndarray) -> list[dict[str, list[Point]]]:
        """Converts a layer mask into a list of polygons with outer and inner loops."""
        edges = self.extract_edges(layer_mask)
        loops = self.trace_loops(edges)

        # classify
        outers = []
        holes = []

        for loop in loops:
            loop = self.simplify_colinear(loop)
            if self.signed_area(loop) > 0:
                outers.append(loop)
            else:
                holes.append(loop)

        # assign holes
        polygons = [{"outer": o, "holes": []} for o in outers]
        for hole in holes:
            p = hole[0]
            for poly in polygons:
                if self.point_in_polygon(p, poly["outer"]):
                    poly["holes"].append(hole)
                    break
            else:
                raise RuntimeError("Hole not assigned")

        return polygons
    

    def extract_edges(self, mask: np.ndarray) -> set[Edge]:
        """Compiles all inside-outside edges of the mask. 
        Left and bottom edges are exported like pixel coordinates, while right and top coordinates are exported as the next pixels coordinates. 
        Inside is always to the left side of the edge."""
        nx, ny = mask.shape
        edges: set[Edge] = set()

        def is_filled(x: int, y: int) -> bool:
            return 0 <= x < nx and 0 <= y < ny and mask[x, y] #will ask for points outside the valid area

        for x in range(nx):
            for y in range(ny):
                if not mask[x, y]:
                    continue

                #left edge
                if not is_filled(x - 1, y):
                    edges.add(((x, y + 1), (x, y)))
                #right edge
                if not is_filled(x + 1, y):
                    edges.add(((x + 1, y), (x + 1, y + 1)))
                #bottom edge
                if not is_filled(x, y - 1):
                    edges.add(((x, y), (x + 1, y)))
                #top edge
                if not is_filled(x, y + 1):
                    edges.add(((x + 1, y + 1), (x, y + 1)))

        return edges
    
    def scale_polygon(self, poly, pixel_size_x, pixel_size_y):
        return [(x * pixel_size_x, y * pixel_size_y) for (x, y) in poly]
    
    def trace_loops(self, edges: Sequence[Edge]) -> list[list[Point]]:
        """Takes a collection of edges and traces loops in those edges. Raises RuntimeError if broken loops are detected."""
        edges = set(edges) #ensure set for performance and to ensure uniqueness
        adj = defaultdict(list) #no assumption of unique starting points
        for e in edges:
            adj[e[0]].append(e)
        loops = []

        while edges:
            start_edge = next(iter(edges))
            start_point = start_edge[0]

            loop = []
            edge = start_edge

            while True:
                try:
                    edges.remove(edge)
                except KeyError:
                    raise RuntimeError("Broken loop")
                v0, v1 = edge
                loop.append(v0)

                if v1 == start_point:
                    break

                for e in adj[v1]:
                    if e in edges:
                        edge = e
                        break
                    else:
                        raise RuntimeError("Broken loop")

            loops.append(loop)

        return loops
    
    def signed_area(self, poly: Sequence[Point]) -> float:
        """calculates signed polygon area with the shoelace formula"""
        area = 0
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            area += x1 * y2 - x2 * y1
        return area / 2

    def point_in_polygon(self, point: Point, poly: Sequence[Point]):
        """Determining if point is inside of the given polygon via the ray casting method with a horizontal ray."""
        x, y = point
        inside = False

        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]

            #no check for point on polygon edge necessary as holes must be inside of the polygon
            #ignore horizontal edges (colinear with ray)
            if y1 == y2:
                continue

            #Ensure that gridpoints are only counted once with open end.
            if y1 <= y < y2 or y2 <= y < y1:
                x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1)

                if x < x_int:
                    inside = not inside

        return inside

    def simplify_colinear(self, poly: Sequence[Point]) -> list[Point]:
        """Shrinks polygon size by filtering out points inside straight lines"""
        if len(poly) < 3:
            return poly

        result = []

        for i in range(len(poly)):
            p0 = poly[i - 1]
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]

            dx1, dy1 = p1[0] - p0[0], p1[1] - p0[1]
            dx2, dy2 = p2[0] - p1[0], p2[1] - p1[1]

            if dx1 * dy2 != dy1 * dx2: #no division here to prevent divide by 0
                result.append(p1)

        return result