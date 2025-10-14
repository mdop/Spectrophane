import numpy as np

MATRIX_sRGB_TO_XYZ = np.array([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505],
])
MATRIX_sRGB_TO_XYZ = np.linalg.inv(MATRIX_sRGB_TO_XYZ)

def linrgb_to_xyz(rgb_values: np.ndarray, matrix: str|np.ndarray = "sRGB", clip: bool = True):
    """transforms numpy array of shape (N,3) or (3,) from linear rgb space in [0,1] to xyz space. Matrix argument may be a transformation matrix or a name. Recognized names: "sRGB" (D65 white point)"""
    if matrix == "sRGB":
        transform_matrix = MATRIX_sRGB_TO_XYZ
    elif isinstance(matrix, np.ndarray):
        transform_matrix = matrix
    else:
        raise ValueError(f"Unknown RGB to XYZ conversion matrix name: {matrix}")
    xyz = transform_matrix @ rgb_values.T
    if clip:
        xyz = np.clip(xyz, 0, 1)
    return xyz