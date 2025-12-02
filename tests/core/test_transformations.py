from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np

from spectrophane.core.transformations import jaxify

@dataclass
class mock_dataclass:
    array1: jnp.ndarray
    string: str
    array2: np.ndarray

def test_jaxify():
    input = (
        mock_dataclass(jnp.array([1]), "test", np.array([])),
        {
            "a": jnp.array([]),
            "b": np.array([1,2]),
            "c": 1
        },
        [1,2,3,4],
        1,
        "str",
        np.array([1.0,2.0])
    )
    expected = (
        mock_dataclass(jnp.array([1]), "test", jnp.array([])),
        {
            "a": jnp.array([]),
            "b": jnp.array([1,2]),
            "c": 1
        },
        [1,2,3,4],
        1,
        "str",
        jnp.array([1.0,2.0])
    )
    result = jaxify(expected)
    assert result == expected