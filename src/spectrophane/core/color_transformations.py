import numpy as np
import jax.numpy as jnp
from numbers import Number

MATRIX_sRGB_TO_XYZ = np.array([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505],
])
MATRIX_sRGB_TO_XYZ = np.linalg.inv(MATRIX_sRGB_TO_XYZ)

def linrgb_to_xyz(rgb_values: np.ndarray, matrix: str|np.ndarray = "sRGB", clip: bool = True):
    """transforms numpy array of shape (N,3) or (3,) from linear rgb space in [0,1] to xyz space. Matrix argument may be a transformation matrix or a name. Recognized names: "sRGB" (D65 white point)"""
    if isinstance(matrix, str) and matrix == "sRGB":
        transform_matrix = MATRIX_sRGB_TO_XYZ
    elif isinstance(matrix, np.ndarray):
        transform_matrix = matrix
    else:
        raise ValueError(f"Unknown RGB to XYZ conversion matrix name: {matrix}")
    xyz = transform_matrix @ rgb_values.T
    xyz = xyz.T
    if clip:
        xyz = np.clip(xyz, 0, 1)
    return xyz

def spectrum_to_xyz(material_spectrum: np.ndarray | jnp.ndarray, light_source: np.ndarray | jnp.ndarray, observer: np.ndarray | jnp.ndarray, step_wavelength: Number):
    """Takes illuminator spectrum (shape (N,)), relative intensity spectrum (e.g. transmission or reflection spectrum, shape (N,)), and an observer (e.g. CIE1931, shape (3,N)) and integrates to XYZ color space."""
    if isinstance(material_spectrum, np.ndarray):
        MD = np
    else:
        MD = jnp
    detector_spectrum = material_spectrum * light_source
    excitation_spectrum = detector_spectrum[MD.newaxis, :] * observer
    xyz = MD.trapezoid(excitation_spectrum, None, dx=step_wavelength, axis=1).flatten()
    return xyz