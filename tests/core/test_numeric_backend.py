import numpy as np
import jax.numpy as jnp
import pytest

from spectrophane.core.numeric_backend import NumPyBackend, JAXBackend

@pytest.fixture(params=["numpy", "jax"])
def backend(request):
    if(request.param == "jax"):
        return JAXBackend()
    else:
        return NumPyBackend()

@pytest.fixture
def xp(backend):
    return jnp if isinstance(backend, JAXBackend) else np


def test_vmap_basic(backend, xp):
    def f(x):
        return x * 2

    x = xp.arange(5)
    vf = backend.vmap(f, in_axes=0)

    out = vf(x)
    assert out.shape == (5,)
    assert xp.all(out == x * 2)


def test_vmap_multiple_args(backend, xp):
    def f(x, y):
        return x + y

    x = xp.arange(5)
    y = xp.ones(5)

    vf = backend.vmap(f, in_axes=(0, 0))
    out = vf(x, y)

    assert out.shape == (5,)
    assert xp.all(out == x + y)


def test_vmap_none_axis(backend, xp):
    def f(x, y):
        return x + y

    x = xp.arange(5)
    y = 3.0

    vf = backend.vmap(f, in_axes=(0, None))
    out = vf(x, y)

    assert out.shape == (5,)
    assert xp.all(out == x + 3.0)


def test_vmap_negative_in_axes(backend, xp):
    def f(x):
        return x.sum()

    x = xp.ones((4, 3))
    vf = backend.vmap(f, in_axes=-1)

    out = vf(x)
    assert out.shape == (3,)
    assert xp.all(out == 4.0)


def test_vmap_out_axes_last(backend, xp):
    def f(x):
        return x * 2

    x = xp.arange(5)
    vf = backend.vmap(f, in_axes=0, out_axes=-1)

    out = vf(x)
    assert out.shape == (5,)
    assert xp.all(out == x * 2)


def test_vmap_out_axes_middle(backend, xp):
    def f(x):
        return xp.stack([x, x + 1], axis=0)

    x = xp.arange(4)
    vf = backend.vmap(f, in_axes=0, out_axes=1)

    out = vf(x)
    assert out.shape == (2, 4)
    assert xp.all(out[0] == x)
    assert xp.all(out[1] == x + 1)


def test_vmap_axis_mismatch_raises(backend, xp):
    def f(x, y):
        return x + y

    x = xp.ones(3)
    y = xp.ones(4)

    vf = backend.vmap(f, in_axes=(0, 0))
    with pytest.raises(ValueError):
        vf(x, y)


def test_vmap_all_none_raises(backend, xp):
    def f(x, y):
        return x + y

    vf = backend.vmap(f, in_axes=(None, None))
    with pytest.raises(ValueError):
        vf(xp.ones(3), xp.ones(3))



def test_scan_simple_accumulation(backend, xp):
    def fn(acc, x):
        acc = acc + x
        return acc, acc

    xs = xp.array([1, 2, 3, 4])
    init = xp.array(0)

    final_acc, ys = backend.scan(fn, init, xs)

    assert final_acc == 10
    assert xp.all(ys == xp.array([1, 3, 6, 10]))

def test_scan_vector_state(backend, xp):
    def fn(acc, x):
        acc = acc * x
        return acc, acc

    xs = xp.array([2, 3, 4])
    init = xp.array([1, 2])

    final_acc, ys = backend.scan(fn, init, xs)

    expected_ys = xp.array([
        [2, 4],
        [6, 12],
        [24, 48],
    ])

    assert xp.all(final_acc == xp.array([24, 48]))
    assert xp.all(ys == expected_ys)

def test_scan_empty_xs(backend, xp):
    def fn(acc, x):
        return acc + x, acc

    xs = xp.array([])
    init = xp.array(5)

    final_acc, ys = backend.scan(fn, init, xs)

    assert final_acc == 5
    assert ys.shape == (0,)

def test_scan_output_shape(backend, xp):
    def fn(acc, x):
        return acc + x, acc * 2

    xs = xp.ones((5, 3))
    init = xp.zeros((3,))

    final_acc, ys = backend.scan(fn, init, xs)

    assert final_acc.shape == (3,)
    assert ys.shape == (5, 3)


def test_identity_transfer_shape_and_dtype(backend, xp):
    eye_size = 2
    transfer_size = 5
    dtype = xp.float64

    T = backend.identity_transfer(eye_size, transfer_size, dtype)

    assert T.shape == (2, 2, 5)
    assert T.dtype == dtype

def test_identity_transfer_is_identity_per_slice(backend, xp):
    eye_size = 3
    transfer_size = 4

    T = backend.identity_transfer(eye_size, transfer_size, xp.float32)

    for i in range(transfer_size):
        assert xp.all(T[..., i] == xp.eye(eye_size))

def test_identity_transfer_broadcasting(backend, xp):
    T = backend.identity_transfer(2, 3, xp.float32)

    # All slices must be identical
    assert xp.all(T[..., 0] == T[..., 1])
    assert xp.all(T[..., 1] == T[..., 2])

def test_identity_transfer_int_dtype(backend, xp):
    T = backend.identity_transfer(2, 2, xp.int32)

    expected = xp.array(
        [[[1, 1],
          [0, 0]],
         [[0, 0],
          [1, 1]]],
        dtype=xp.int32,
    )

    assert xp.all(T == expected)
