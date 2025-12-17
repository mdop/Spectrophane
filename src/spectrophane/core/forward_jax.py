"""JAX-specific adapters for forward modeling."""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any

from spectrophane.core import dataclasses as dc


def register_with_jax():
    """Register backend-neutral dataclasses with JAX pytrees.

    Additionally, expose a `jaxify` helper (in this module) to convert
    domain dataclasses and containers to JAX arrays. Call this function
    near the JAX boundary (e.g. training/bootstrap code).
    """
    try:
        jax.tree_util.register_dataclass(dc.StackData)
        jax.tree_util.register_dataclass(dc.MaterialParams, data_fields=["absorption_coeff", "scattering_coeff"], meta_fields=["model_type"])
    except Exception:
        pass


def jaxify(obj):
    """Recursively convert NumPy arrays in dataclasses, dicts, lists, or tuples to JAX arrays.

    This mirrors the previous `transformations.jaxify` behaviour but is
    kept in the JAX adapter module so the conversion is explicit at the
    JAX boundary.
    """
    if isinstance(obj, np.ndarray):
        return jnp.asarray(obj)
    # dataclass check without importing dataclasses directly to avoid cycles
    try:
        from dataclasses import is_dataclass, replace, fields
    except Exception:
        is_dataclass = lambda x: False

    if is_dataclass(obj):
        return replace(obj, **{f.name: jaxify(getattr(obj, f.name)) for f in fields(obj)})
    elif isinstance(obj, dict):
        return {k: jaxify(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(jaxify(x) for x in obj)
    else:
        return obj
