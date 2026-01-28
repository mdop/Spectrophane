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