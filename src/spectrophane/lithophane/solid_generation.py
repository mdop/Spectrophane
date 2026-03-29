from abc import ABC, abstractmethod
from collections.abc import Iterator
import numpy as np

from spectrophane.core.dataclasses import VoxelGeometry, SolidPrimitive, Box, Prism

class SolidBuilder(ABC):
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