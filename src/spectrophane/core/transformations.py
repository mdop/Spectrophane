import jax.numpy as jnp
import numpy as np
from dataclasses import is_dataclass, replace, fields

def jaxify(obj):
    """Recursively convert NumPy arrays in dataclasses, dicts, lists, or tuples to JAX arrays."""
    if isinstance(obj, np.ndarray):
        return jnp.asarray(obj)
    elif is_dataclass(obj):
        # Create a new dataclass with jaxified fields
        return replace(obj, **{f.name: jaxify(getattr(obj, f.name)) for f in fields(obj)})
    elif isinstance(obj, dict):
        return {k: jaxify(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(jaxify(x) for x in obj)
    else:
        return obj