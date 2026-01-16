from abc import ABC, abstractmethod

from spectrophane.core.dataclasses import VoxelGeometry

class GeometryExporter(ABC):
    @abstractmethod
    def export(self, geometry: VoxelGeometry, path: str) -> None:
        ...
