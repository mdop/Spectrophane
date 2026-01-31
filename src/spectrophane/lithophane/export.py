from abc import ABC, abstractmethod
import numpy as np
import struct
from pathlib import PosixPath, Path

from spectrophane.lithophane.solid_generation import SolidPrimitive, Box, Prism


class SolidBackend(ABC):
    @abstractmethod
    def __init__(self, base_path: str, material_names: list[str]):
        ...
    
    @abstractmethod
    def supports(self, primitive: type) -> bool:
        ...

    @abstractmethod
    def begin(self, material_index: int):
        """Starts writing primitives for a given material_index (index of material names in constructor). May create a new file at this point for the given material."""
        ...

    @abstractmethod
    def add(self, primitive: SolidPrimitive) -> None:
        ...
    
    @abstractmethod
    def end(self) -> list[str]:
        """Finalizes all output files and returns a list of filepaths that were created."""
        ...



class STLTessellationBackend(SolidBackend):
    _box_tesselation_triangles = [
            (0, 1, 2), (0, 2, 3),  # bottom
            (4, 6, 5), (4, 7, 6),  # top
            (0, 4, 5), (0, 5, 1),  # front
            (1, 5, 6), (1, 6, 2),  # right
            (2, 6, 7), (2, 7, 3),  # back
            (3, 7, 4), (3, 4, 0),  # left
        ]

    def __init__(self, base_path: str | PosixPath, material_names: list[str], binary: bool = True):
        """base path is filename with or without stl extension."""
        self._binary = binary
        base_path = str(base_path)
        if base_path.endswith(".stl"):
            base_path = base_path[:-4] #sanitize filepath if stl is provided
        self._base_path = str(base_path)
        self._material_names = material_names
        self._handlers = [None] * len(material_names)
        self._triangle_counts = [0] * len(material_names)
        self._active_handler = None
        self._active_index = None

    def supports(self, primitive):
        return isinstance(primitive, Box)

    def begin(self, material_index: int):
        if self._handlers[material_index] is None:
            material_name = self._material_names[material_index]
            filename = self._base_path + "_" + material_name + ".stl"
            self._handlers[material_index] = open(filename, mode="wt")
            if self._binary:
                self._handlers[material_index] = open(filename, mode="wb")
                header = (f"material_{material_name}" + (" " * 80)).encode("ascii")[:80]
                header = header.ljust(80, b"\0")
                self._handlers[material_index].write(header)
                self._handlers[material_index].write(struct.pack("<I", 0))  # placeholder for triangle count
            else:
                self._handlers[material_index] = open(filename, mode="wt")
                self._handlers[material_index].write(f"solid material_{material_name}\n")

        self._active_handler = self._handlers[material_index]
        self._active_index = material_index

    def add(self, primitive: SolidPrimitive) -> None:
        if isinstance(primitive, Box):
            triangles = self._tessellate_box(primitive)
        else:
            raise TypeError(f"Unsupported geometry primitive {type(primitive)} for STL tessellation")
        
        if self._binary:
            self._write_binary(triangles)
            self._triangle_counts[self._active_index] += len(triangles)
        else:
            self._write_ascii(triangles)

    def end(self) -> list[str]:
        opened_filepaths = []
        for i, file in enumerate(self._handlers):
            if file is None:
                continue
            
            if self._binary:
                file.seek(80)
                file.write(struct.pack("<I", self._triangle_counts[i]))
            else:
                file.write(f"endsolid material_{self._material_names[i]}\n")
            opened_filepaths.append(file.name)
            file.close()
        return opened_filepaths



    def _tessellate_box(self, box: Box) -> np.ndarray:
        vertices = np.array([
            [box.x0, box.y0, box.z0],
            [box.x1, box.y0, box.z0],
            [box.x1, box.y1, box.z0],
            [box.x0, box.y1, box.z0],
            [box.x0, box.y0, box.z1],
            [box.x1, box.y0, box.z1],
            [box.x1, box.y1, box.z1],
            [box.x0, box.y1, box.z1],
        ])

        triangles = np.zeros((len(self._box_tesselation_triangles), 12), dtype=float)
        for i in range(len(self._box_tesselation_triangles)):
            a = vertices[self._box_tesselation_triangles[i][0]]
            b = vertices[self._box_tesselation_triangles[i][1]]
            c = vertices[self._box_tesselation_triangles[i][2]]
            n = np.cross(b - a, c - a)
            n = n / np.linalg.norm(n)
            triangles[i, 0:3 ] = a
            triangles[i, 3:6 ] = b
            triangles[i, 6:9 ] = c
            triangles[i, 9:12] = n

        return triangles
    
    def _write_ascii(self, triangles: np.ndarray):
        out_str = ""
        for i in range(len(triangles)):
            out_str += f" facet normal {triangles[i,9]} {triangles[i,10]} {triangles[i,11]}\n"
            out_str +=  "  outer loop\n"
            out_str += f"   vertex {triangles[i,0]} {triangles[i,1]} {triangles[i,2]}\n"
            out_str += f"   vertex {triangles[i,3]} {triangles[i,4]} {triangles[i,5]}\n"
            out_str += f"   vertex {triangles[i,6]} {triangles[i,7]} {triangles[i,8]}\n"
            out_str +=  "  endloop\n"
            out_str +=  " endfacet\n"
        self._active_handler.write(out_str)
    
    def _write_binary(self, triangles: np.ndarray):
        f = self._active_handler

        for i in range(len(triangles)):
            # normal
            f.write(struct.pack(
                "<3f",
                triangles[i, 9],
                triangles[i,10],
                triangles[i,11],
            ))

            # vertices
            f.write(struct.pack(
                "<9f",
                triangles[i,0], triangles[i,1], triangles[i,2],
                triangles[i,3], triangles[i,4], triangles[i,5],
                triangles[i,6], triangles[i,7], triangles[i,8],
            ))

            # attribute byte count
            f.write(struct.pack("<H", 0))
