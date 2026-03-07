from typing import Tuple, Optional, Sequence, Self
import numpy as np
from numbers import Number
from dataclasses import dataclass, field
import dataclasses

@dataclass
class WavelengthAxis:
    start: float
    step: float
    length: int

    @property
    def end(self) -> float:
        return self.start + self.step * (self.length - 1)

    @property
    def wavelengths(self) -> np.ndarray:
        return self.start + self.step * np.arange(self.length)
    
    @classmethod
    def empty(cls) -> "WavelengthAxis":
        return cls(start=350, step=1.0, length=451)
    
    @classmethod
    def common(cls, axes: Sequence["WavelengthAxis"]) -> "WavelengthAxis | None":
        if not axes:
            return cls.empty()

        com_start = max(ax.start for ax in axes)
        com_step = min(ax.step for ax in axes)
        com_stop = min(ax.end for ax in axes)

        if com_stop < com_start:
            return None

        length = int(round((com_stop - com_start) / com_step, 0)) + 1
        return cls(com_start, com_step, length)

@dataclass(slots=True)
class SpectrumBlock:
    start: float
    step: float
    values: np.ndarray  # shape: (n_spectra, n_wavelengths)

    def __post_init__(self):
        if self.values.ndim < 2:
            raise ValueError("values must be at least 2D (n_spectra, n_wavelengths)")
        if self.step <= 0:
            raise ValueError("step must be positive")

    @property
    def axis(self) -> WavelengthAxis:
        return WavelengthAxis(
            start=self.start,
            step=self.step,
            length=self.values.shape[-1],
        )

    @property
    def n_spectra(self) -> int:
        """returns number of entities in the first axis."""
        return self.values.shape[0]
    
    def resample(self, target_axis: WavelengthAxis) -> "SpectrumBlock":
        """Returns a Spectrum block resampled for the given target_axis"""
        src_axis = self.axis

        new_values = np.apply_along_axis(
            lambda x: np.interp(target_axis.wavelengths, src_axis.wavelengths, x),
            axis=-1,
            arr=self.values
        )

        return SpectrumBlock(start=target_axis.start, step=target_axis.step, values=new_values)
    
    @classmethod
    def merge_resample_spectra(cls, spectra: Sequence["SpectrumBlock"], axis: WavelengthAxis | None = None) -> "SpectrumBlock":
        """Function that merges and interpolates a list of spectra to a common axis into one block. if no wavelength axis is provided the common axis for the given sequence is used"""
        if axis is None:
            axes = [spectrum.axis for spectrum in spectra]
            axis = WavelengthAxis.common(axes)
        if len(spectra) == 0:
            return SpectrumBlock(start=axis.start, step=axis.step, values=np.zeros((0, axis.length)))
        total_entities = sum(spectrum.values.shape[0] for spectrum in spectra)
        shared_shape = spectra[0].values.shape[1:-1]
        output_shape = (total_entities, *shared_shape, axis.length)
        harmonized_values = np.zeros(output_shape)
        start_spectrum_index = 0
        for block in spectra:
            if block.values.shape[1:-1] != shared_shape:
                raise ValueError(f"incompatible shape for merging spectra: got {block.values.shape[1:-1]} vs exected {shared_shape}")
            next_block_index = start_spectrum_index + block.values.shape[0]
            resampled_block = block.resample(axis)
            harmonized_values[start_spectrum_index:next_block_index] = resampled_block.values
            start_spectrum_index = next_block_index
        return SpectrumBlock(start=axis.start, step=axis.step, values=harmonized_values)

@dataclass
class NameSpectraBase:
    names: Tuple[str, ...]
    spectra: Sequence[SpectrumBlock]

    def take_indexes(self, indexes: int | Sequence[int]) -> Self:
        if isinstance(indexes, int):
            indexes = (indexes,)

        new_names = tuple(self.names[i] for i in indexes)
        new_spectra = tuple(self.spectra[i] for i in indexes)

        return type(self)(
            names=new_names,
            spectra=new_spectra,
        )

    def take_names(self, names: str | Sequence[str]) -> Self:
        if isinstance(names, str):
            names = (names,)

        index_map = {name: i for i, name in enumerate(self.names)}
        indexes = [index_map[name] for name in names]

        return self.take_indexes(indexes)

@dataclass
class LightSources(NameSpectraBase):
    """Dataclass representing light sources with their names and spectra. Spectra should have shape (1, wavelengths) to allow correct concatenation when harmonizing.
    Defaults to NumPy arrays. If used with jax, register as a pytree and convert arrays at the boundary."""
    pass

@dataclass
class Observers(NameSpectraBase):
    """Dataclass representing observers with their names and spectra. Spectra should have shape (1, wavelengths) to allow correct concatenation when harmonizing.
    Defaults to NumPy arrays. If used with jax, register as a pytree and convert arrays at the boundary."""
    pass

@dataclass
class StackData:
    """
    Pure backend-neutral dataclass describing a material stack.
    Defaults to NumPy arrays. If used with jax register as a pytree and convert arrays at the boundary.
    """
    material_nums: np.ndarray
    thicknesses: np.ndarray
    def take(self, indices: np.ndarray) -> "StackData":
        return StackData(
            material_nums=self.material_nums[indices],
            thicknesses=self.thicknesses[indices],
        )

@dataclass
class StackCandidates(StackData):
    """
    Dataclass describing StackData with associated color. Score is in range 0..1
    """
    rgb: np.ndarray
    def take(self, indices: np.ndarray) -> "StackData":
        return StackCandidates(
            material_nums=self.material_nums[indices],
            thicknesses=self.thicknesses[indices],
            rgb=self.rgb[indices],
        )

@dataclass
class TrainingRefSpectraData:
    """Dataclass containing reference spectra and corresponding stack data for training.
    Defaults to NumPy arrays. If used with jax, register as a pytree and convert arrays at the boundary."""
    transmission_stacks: StackData
    transmission_spectra: np.ndarray
    reflection_stacks: StackData
    reflection_spectra: np.ndarray
    reflection_background: np.ndarray
    min_wavelength: Number
    step_wavelength: Number

@dataclass
class TrainingRefImageData:
    """Dataclass containing reference image data and corresponding stack information for training.
    Defaults to NumPy arrays. If used with jax, register as a pytree and convert arrays at the boundary."""
    transmission_stacks: StackData
    transmission_xyz: np.ndarray
    transmission_light_source_indexes: np.ndarray


@dataclass
class MaterialParams:
    """
    Backend-neutral material parameter container.
    Defaults to NumPy arrays. If used with jax register as a pytree and convert arrays at the boundary.
    """
    wl_start: Number
    wl_step: Number
    absorption_coeff: Optional[np.ndarray] | None = field(
        default=None,
        metadata={"deserialize": np.array, "filter": True}
    )
    scattering_coeff: Optional[np.ndarray] | None = field(
        default=None,
        metadata={"deserialize": np.array, "filter": True}
    )
    model_type: Optional[str] = None  # "kubelka_munk" (others tbd e.g. "saunderson", "monte_carlo")
    
    def take(self, indexes: np.ndarray) -> "MaterialParams":
        # Create a new instance of MaterialParams
        new_instance = MaterialParams(
            wl_start=self.wl_start,
            wl_step=self.wl_step,
            model_type=self.model_type
        )

        # Iterate over the fields and apply take only to NumPy arrays
        for field in dataclasses.fields(self):
            if field.metadata.get("deserialize", False) and (not getattr(self, field.name) is None):
                setattr(new_instance, field.name, getattr(self, field.name)[indexes])

        return new_instance


@dataclass
class TopologyBlock:
    max_layers_per_material: np.ndarray
    thicknesses: np.ndarray

@dataclass
class StackTopologyRules:
    blocks: list[TopologyBlock]
    ordered: bool #Describes if order of layers matters. Decides max_layers for StackData shapes

    @property
    def material_count(self) -> int:
        """Returns total number of materials. Raises ValueError if inconsistent results are found"""
        count = len(self.blocks[0].max_layers_per_material)
        for block in self.blocks:
            if len(block.max_layers_per_material) != count:
                raise ValueError(f"Inconsistent stack topology found! First block has {count} materials, but {len(block.max_layers_per_material)} were also found!")
        return count
    
    @property
    def layer_thicknesses(self) -> np.ndarray:
        """Returns a combined single array of thicknesses from all blocks"""
        return np.concatenate([block.thicknesses for block in self.blocks])

@dataclass
class VoxelGeometry:
    materials: np.ndarray          # shape (Z, Y, X), dtype=int
    layer_thickness: np.ndarray    # shape (Z,), dtype=float, unit=mm
    voxel_size_xy: tuple[float, float]
    material_names: str


class SolidPrimitive:
    pass

@dataclass(frozen=True, slots=True)
class Box(SolidPrimitive):
    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float

@dataclass(frozen=True, slots=True) #Not used now, but will be interesting for STEP export
class Prism(SolidPrimitive):
    base_xy: np.ndarray  # shape (N, 2), dtype=float64
    z0: float
    z1: float


############################## Pipeline dataclasses ##############################

@dataclass(frozen=True)
class TrainingConfig:
    model: str = "kubelka_munk"
    observer: str = "CIE1931"
    steps: int = 1000
    lr: float = 1e-1
    parameter_plot_filter: Sequence[str] | None = tuple()
    parameter_plot_rows: int = 1
    get_terminal_color_comparison: bool = True


@dataclass(frozen=True)
class EvaluatorSpec:
    theory: str = "kubelka_munk"                # implemented: "kubelka_munk"
    observer: str = "CIE1931"                   # name in the Observers object names list (presumably from config)
    illuminator: str = "D65"                    # name in the LightSources object names list (presumably from config)
    calc_backend: str = "jax"                   # implemented: "jax", "numpy"
    cache_backend: str = "dict"                 # implemented: "dict"
    view_geometry: str = "transmission"         # "transmission" or "reflection"
    background: SpectrumBlock | float = 1.0     # background reflectivity. Only relevant for reflection view geometry, values in range [0.0,1.0]. Single value is interpreted as flat spectrum
    edge_stacks: StackCandidates | None = None  # To expand color gamut extreme stack combinations may be used to renormalize color output. Use if original light source is not visible to viewer (typically in transmission case)
    normalize: bool = True                      # if true evaluator will normalize color space to brightest edge case stack.

@dataclass(frozen=True)
class InverterSpec:
    algorithm: str = "lut"                              # implemented: "lut"
    lut_compression_factor: Optional[int] = None        # relevant for LUT

@dataclass(frozen=True)
class LithophaneConfig:
    resolution: tuple[int,int]
    pixel_xy_dimension: tuple[float,float]
    material_names: Sequence[str]
    builder_algorithm: str = "voxel"                    # implemented: "voxel"
    export_backend_format: str = "stl"               # implemented: "STL"
    export_stl_type: str = "binary"                     # implemented: "binary"/"ASCII"