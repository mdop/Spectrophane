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

def _backendify(obj, framework):
    """
    Recursively convert arrays in dataclasses, dicts, lists, or tuples to target frameworks arrays.
    """
    if isinstance(obj, np.ndarray) or isinstance(obj, jnp.ndarray):
        return framework.asarray(obj)
    # dataclass check without importing dataclasses directly to avoid cycles
    try:
        from dataclasses import is_dataclass, replace, fields
    except Exception:
        is_dataclass = lambda x: False

    if is_dataclass(obj):
        return replace(obj, **{f.name: _backendify(getattr(obj, f.name), framework) for f in fields(obj)})
    elif isinstance(obj, dict):
        return {k: _backendify(v, framework) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_backendify(x, framework) for x in obj)
    else:
        return obj

def jaxify(obj):
    """
    Recursively convert arrays in dataclasses, dicts, lists, or tuples to JAX arrays.
    """
    return _backendify(obj, jnp)

def numpyify(obj):
    """
    Recursively convert arrays in dataclasses, dicts, lists, or tuples to JAX arrays.
    """
    return _backendify(obj, np)