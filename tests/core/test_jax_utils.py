from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np

from spectrophane.core.jax_utils import jaxify

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
    result = jaxify(input)
    assert isinstance(result[0].array1, jnp.ndarray)
    assert isinstance(result[0].array2, jnp.ndarray)
    assert isinstance(result[1]["a"], jnp.ndarray)
    assert isinstance(result[1]["b"], jnp.ndarray)
    assert isinstance(result[5], jnp.ndarray)
    assert jnp.all(result[0].array1 == expected[0].array1)
    assert jnp.all(result[0].array2 == expected[0].array2)
    assert jnp.all(result[1]["a"] == expected[1]["a"])
    assert jnp.all(result[1]["b"] == expected[1]["b"])
    assert jnp.all(result[5] == expected[5])
    assert result[0].string == expected[0].string
    assert result[1]["c"] == expected[1]["c"]
    assert result[2] == expected[2]
    assert result[3] == expected[3]
    assert result[4] == expected[4]