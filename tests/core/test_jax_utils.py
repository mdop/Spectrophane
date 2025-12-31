from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np
import pytest

from spectrophane.core.jax_utils import jaxify, numpyify

@dataclass
class mock_dataclass:
    array1: jnp.ndarray
    string: str
    array2: np.ndarray

@pytest.mark.parametrize("func", [jaxify, numpyify])
def test_jaxify(func):
    if func == jaxify:
        framework = jnp
    else:
        framework = np
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
        np.array([1.0,2.0]),
        jnp.array([1.0,2.0])
    )
    expected = (
        mock_dataclass(framework.array([1]), "test", framework.array([])),
        {
            "a": framework.array([]),
            "b": framework.array([1,2]),
            "c": 1
        },
        [1,2,3,4],
        1,
        "str",
        framework.array([1.0,2.0]),
        framework.array([1.0,2.0])
    )
    result = func(input)
    assert isinstance(result[0].array1, framework.ndarray)
    assert isinstance(result[0].array2, framework.ndarray)
    assert isinstance(result[1]["a"], framework.ndarray)
    assert isinstance(result[1]["b"], framework.ndarray)
    assert isinstance(result[5], framework.ndarray)
    assert framework.all(result[0].array1 == expected[0].array1)
    assert framework.all(result[0].array2 == expected[0].array2)
    assert framework.all(result[1]["a"] == expected[1]["a"])
    assert framework.all(result[1]["b"] == expected[1]["b"])
    assert framework.all(result[5] == expected[5])
    assert result[0].string == expected[0].string
    assert result[1]["c"] == expected[1]["c"]
    assert result[2] == expected[2]
    assert result[3] == expected[3]
    assert result[4] == expected[4]
    assert framework.all(result[5] == expected[5])
    assert framework.all(result[6] == expected[6])