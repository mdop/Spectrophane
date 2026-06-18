from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import struct
from pathlib import PosixPath, Path

from spectrophane.core.dataclasses import GridSolidPrimitive, AffineTransform, Box, Prism

@dataclass
class TriangleMesh:
    """Indexed triangle mesh in integer grid coordinates.
    vertices:  (N, 3) (ix, iy, iz) in grid / z-precision units.
    triangles: (M, 3) ivertex indices, CCW winding (right-handrule), outward-facing normals implied by winding order.
 
    Coordinate system assumed throughout:
    Right-handed basis; cross(B−A, C−A) yields outward normal
    """
    vertices:  np.ndarray
    triangles: np.ndarray


class SolidBackend(ABC):
    """Abstract base for all solid-geometry export backends.
 
    Lifecycle per export:
        backend.begin(material_index, transform)   # once per material
        for solid in collection:
            backend.add(solid)
        paths = backend.end()                      # finalise all files
    """

    @abstractmethod
    def __init__(self, base_path: str, material_data: list[dict]):
        ...
    
    @abstractmethod
    def supports(self, primitive: type) -> bool:
        """Return True if this backend can export the given primitive type."""
        ...

    @abstractmethod
    def begin(self, material_index: int, grid_mm_transform: AffineTransform):
        """Starts writing primitives for a given material_index (index of material names in constructor). May create a new file at this point for the given material."""
        ...

    @abstractmethod
    def add(self, grid_primitive: GridSolidPrimitive) -> None:
        """Add one primitive (in grid coordinates) to the current material."""
        ...
    
    @abstractmethod
    def end(self) -> list[str]:
        """Finalizes all output files and returns a list of filepaths that were created."""
        ...


class TriangleTessellatingBackend(SolidBackend, ABC):
    """Intermediate ABC that converts primitives to TriangleMesh objects.
    Subclasses implement add_mesh() to consume the indexed mesh.
 
    Grid-to-mm conversion is to be performed by the subclass if appropriate.
    """
 
    # Box tessellation — CCW winding for right-handed coords
    _BOX_TRIANGLES: list[tuple[int, int, int]] = [
        (0, 2, 1), (0, 3, 2),   # bottom  — outward normal: −z
        (4, 5, 6), (4, 6, 7),   # top     — outward normal: +z
        (0, 1, 5), (0, 5, 4),   # front   — outward normal: −y
        (1, 2, 6), (1, 6, 5),   # right   — outward normal: +x
        (2, 3, 7), (2, 7, 6),   # back    — outward normal: +y
        (3, 0, 4), (3, 4, 7),   # left    — outward normal: −x
    ]

    def supports(self, primitive_type: type) -> bool:
        """Box is supported; Prism tessellation is not yet implemented."""
        return primitive_type is Box
 
    def add(self, primitive: GridSolidPrimitive) -> None:
        if isinstance(primitive, Box):
            self.add_mesh(self._mesh_box(primitive))
        elif isinstance(primitive, Prism):
            raise NotImplementedError("Prism tessellation is not yet implemented.")
        else:
            raise TypeError(f"Unsupported primitive type {type(primitive).__name__!r}.")
 
    @abstractmethod
    def add_mesh(self, mesh: TriangleMesh) -> None:
        """Consume one TriangleMesh in grid coordinates."""
        ...
 
    @staticmethod
    def _mesh_box(box: Box) -> TriangleMesh:
        """Tessellate a grid-space Box into an indexed TriangleMesh. Emits grid coordinates, winding order CCW from outside."""
        vertices = np.array([
            [box.x0, box.y0, box.z0],
            [box.x1, box.y0, box.z0],
            [box.x1, box.y1, box.z0],
            [box.x0, box.y1, box.z0],
            [box.x0, box.y0, box.z1],
            [box.x1, box.y0, box.z1],
            [box.x1, box.y1, box.z1],
            [box.x0, box.y1, box.z1],
        ], dtype=np.int32)
 
        triangles = np.array(
            TriangleTessellatingBackend._BOX_TRIANGLES,
            dtype=np.int32,
        )
 
        return TriangleMesh(vertices=vertices, triangles=triangles)


class STLTessellationBackend(TriangleTessellatingBackend):
    """
    Exports one binary (or ASCII) STL file per material.
    File naming: <base_path>_<material_name>.stl
    material_data dicts need to include the key "name", which is used for file naming
    """

    def __init__(self, base_path: str | PosixPath, material_data: list[dict], binary: bool = True):
        """base path is filename with or without stl extension."""
        self._binary = binary
        base_path = str(base_path)
        if base_path.endswith(".stl"):
            base_path = base_path[:-4] #sanitize filepath if stl is provided
        self._base_path = str(base_path)
        material_names = [mat["name"] for mat in material_data]
        self._material_names = material_names
        self._handlers = [None] * len(material_names)
        self._triangle_counts = [0] * len(material_names)
        self._active_handler = None
        self._active_index = None

    def supports(self, primitive):
        return isinstance(primitive, Box)

    def begin(self, material_index: int, grid_mm_transform: AffineTransform):
        self._grid_mm_transform = grid_mm_transform
        if self._handlers[material_index] is None:
            material_name = self._material_names[material_index]
            filename = self._base_path + "_" + material_name + ".stl"
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

    def add_mesh(self, mesh: TriangleMesh) -> np.ndarray:
        """
        Write all triangles of *mesh* to the active STL file.
        Vertices are converted from grid coordinates to mm using the transform stored by the most recent begin() call.
        Returns triangle array for inspection purposes.
        """
        transform = self._grid_mm_transform
        # Build mm-space vertex array — same shape (N, 3), dtype float32.
        verts_mm = np.empty(mesh.vertices.shape, dtype=np.float32)
        verts_mm[:, 0] = transform.apply_x(mesh.vertices[:, 0])
        verts_mm[:, 1] = transform.apply_y(mesh.vertices[:, 1])
        verts_mm[:, 2] = transform.apply_z(mesh.vertices[:, 2])

        triangles = np.empty((len(mesh.triangles), 12), dtype=np.float32)
        for i, tri in enumerate(mesh.triangles):
            a = verts_mm[tri[0]]
            b = verts_mm[tri[1]]
            c = verts_mm[tri[2]]
            n = np.cross(b - a, c - a)
            norm = np.linalg.norm(n)
            if norm:
                n /= norm
            triangles[i, 0:3 ] = a
            triangles[i, 3:6 ] = b
            triangles[i, 6:9 ] = c
            triangles[i, 9:12] = n
        
        if self._binary:
            self._write_triangles_binary(triangles)
        else:
            self._write_triangles_ascii(triangles)
 
        self._triangle_counts[self._active_index] += len(mesh.triangles)
        return triangles

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
    
    def _write_triangles_ascii(self, triangles: np.ndarray):
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
    
    def _write_triangles_binary(self, triangles: np.ndarray):
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
