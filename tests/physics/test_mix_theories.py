import jax.numpy as jnp
import numpy as np
import jax
import pytest
import copy

from spectrophane.physics.mix_theories import THEORY_REGISTRY, KubelkaMunk
from spectrophane.core.dataclasses import StackData, MaterialParams

jax.config.update('jax_enable_x64', True)

def test_theory_registry():
    assert "kubelka_munk" in THEORY_REGISTRY
    assert isinstance(THEORY_REGISTRY["kubelka_munk"]("jax"), KubelkaMunk)

@pytest.fixture(params=["jax", "numpy"])
def backend(request):
    return request.param

@pytest.fixture
def xp(backend):
    return jnp if backend == "jax" else np

@pytest.fixture
def KM_instance(backend):
    return KubelkaMunk(backend=backend)

@pytest.fixture
def KM_random_parameter(backend, xp):
    if backend == "jax":
        key = jax.random.key(42)
        K = jax.random.uniform(key, (100,), dtype=jnp.float64) * 10
        key, _ = jax.random.split(key)
        S = jax.random.uniform(key, (100,), dtype=jnp.float64) * 10
        key, _ = jax.random.split(key)
        d = jax.random.uniform(key, (100,), dtype=jnp.float64) / 10
    else:
        rng = np.random.default_rng(42)
        K = rng.random(100) * 10
        S = rng.random(100) * 10
        d = rng.random(100) / 10

    return xp.asarray(K), xp.asarray(S), xp.asarray(d)


def test_KM_layer_shape_determinant(KM_instance: KubelkaMunk, KM_random_parameter, xp):
    K, S, d = KM_random_parameter
    M = KM_instance._single_layer_transfer_matrix(K, S, d)
    det = M[0,0] * M[1,1] - M[1,0] * M[0,1]

    assert M.shape == (2,2,100)
    #determinant of the transfer matrix can be proven to be 1
    assert xp.allclose(det, 1, atol=1e-5)
    assert not xp.isnan(M).any()

def test_KM_layer_zero_thickness_identity(KM_instance: KubelkaMunk, KM_random_parameter, xp):
    K, S, _ = KM_random_parameter
    d = xp.array([0.0]*len(K))

    M = KM_instance._single_layer_transfer_matrix(K, S, d)
    I = xp.array([[[1.0], [0.0]], [[0.0], [1.0]]])

    assert xp.allclose(M, I, atol=1e-5)

def KM_reflectance_from_matrix(M):
    m11, m12 = M[0,0], M[0,1]
    m21, m22 = M[1,0], M[1,1]
    return m12 / m22

def test_KM_layer_semi_infinite_limit(KM_instance: KubelkaMunk, KM_random_parameter, xp):
    K, S, d = KM_random_parameter
    d = (d+1)*10 #very thick, but not too thick for numerical stability

    M = KM_instance._single_layer_transfer_matrix(K, S, d)
    R = KM_reflectance_from_matrix(M)

    R_inf = 1+K/S-xp.sqrt(xp.square(K/S)+2*K/S)
    assert xp.allclose(R, R_inf, atol=1e-3)


def test_KM_chain_transfer_matrizes_shape(KM_instance: KubelkaMunk, xp):
    rng = xp.random.default_rng(42) if xp is not jnp else None

    if xp is jnp:
        key = jax.random.key(42)
        transfer_matrices = jax.random.uniform(
            key, shape=(5, 2, 2, 100), dtype=jnp.float64
        )
    else:
        transfer_matrices = rng.random((5, 2, 2, 100)).astype(xp.float64)
        
    result = KM_instance._chain_transfer_matrizes(transfer_matrices)
    assert result.shape == (2, 2, 100), "Output shape should be (2, 2, 100)"
    assert not xp.isnan(result).any()

def test_KM_chain_transfer_matrizes_identity(KM_instance: KubelkaMunk, xp):
    # Test with identity matrices
    transfer_matrizes = xp.array([[[[1,1,1],[0,0,0]],[[0,0,0],[1,1,1]]]])
    result = KM_instance._chain_transfer_matrizes(transfer_matrizes)
    assert xp.allclose(result, transfer_matrizes), "Identity matrices should return identity matrix"


@pytest.fixture
def mock_stack(xp):
    return StackData(material_nums=xp.array([0, 1]), thicknesses=xp.array([1.0, 2.0]))

@pytest.fixture
def mock_material_params(xp):
    return MaterialParams(wl_start=350,
                          wl_step=10,
                          absorption_coeff=xp.array([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]]), 
                          scattering_coeff=xp.array([[0.5, 0.6, 0.7], [0.7, 0.8, 0.9]]),
                          )

def test_KM_stack_transfer_matrix_multiple_layers(KM_instance: KubelkaMunk, mock_stack, mock_material_params, xp):
    # Mock data for multiple layers
    result = KM_instance._stack_transfer_matrix(mock_stack, mock_material_params)
    assert result.shape == (2, 2, 3), "Transfer matrix should be 2x2"
    assert not xp.isnan(result).any(), "Transfer matrix should not contain NaN values"

def test_KM_stack_transfer_matrix_single_layer(KM_instance: KubelkaMunk, xp, mock_material_params: MaterialParams):
    # Mock data for a single layer
    stack = StackData(material_nums=xp.array([0]), thicknesses=xp.array([1.0]))

    result = KM_instance._stack_transfer_matrix(stack, mock_material_params)
    assert result.shape == (2, 2, 3), "Transfer matrix should be 2x2"
    assert not xp.isnan(result).any(), "Transfer matrix should not contain NaN values"

def test_KM_stack_transfer_matrix_zero_thickness(KM_instance: KubelkaMunk, mock_stack, mock_material_params, xp):
    # Mock data with zero thickness
    mock_stack.thicknesses = xp.array([0,0])
    
    result = KM_instance._stack_transfer_matrix(mock_stack, mock_material_params)
    assert result.shape == (2, 2, 3), "Transfer matrix should be 2x2"
    assert not xp.isnan(result).any(), "Transfer matrix should not contain NaN values"

def test_KM_stack_transfer_matrix_large_number_of_layers(KM_instance: KubelkaMunk, xp, mock_material_params: MaterialParams):
    # Mock data with a large number of layers
    stack = StackData(material_nums=xp.array([0] * 100), thicknesses=xp.array([1.0] * 100))
    
    result = KM_instance._stack_transfer_matrix(stack, mock_material_params)
    assert result.shape == (2, 2, 3), "Transfer matrix should be 2x2"
    assert not xp.isnan(result).any(), "Transfer matrix should not contain NaN values"


@pytest.fixture
def mock_black_top_white_bottom(xp):
    stack = StackData(material_nums=xp.array([0,1]), thicknesses=xp.array([1,1]))
    param = MaterialParams(wl_start= 350, wl_step=10, absorption_coeff=xp.array([[0.001]*100, [3]*100]), scattering_coeff=xp.array([[10]*100, [10]*100]))
    return stack, param

@pytest.fixture
def mock_white_top_black_bottom(mock_black_top_white_bottom, xp):
    data = copy.deepcopy(mock_black_top_white_bottom)
    data[0].material_nums = xp.array([1,0])
    return data

def test_KM_transmission(KM_instance: KubelkaMunk, mock_stack, mock_material_params, xp):
    result = KM_instance.transmission(mock_stack, mock_material_params)
    assert result.shape == (3,)
    assert xp.all(xp.isfinite(result))

def test_KM_transmission_order(KM_instance: KubelkaMunk, mock_black_top_white_bottom, mock_white_top_black_bottom, xp):
    stack_bt, param_bt = mock_black_top_white_bottom
    stack_wt, param_wt = mock_white_top_black_bottom

    result_bt = KM_instance.transmission(stack_bt, param_bt)
    result_wt = KM_instance.transmission(stack_wt, param_wt)
    assert xp.all(xp.isfinite(result_bt))
    assert xp.all(result_bt > 1e-5)
    assert xp.all(result_bt <= 1)
    assert xp.allclose(result_bt, result_wt)

def test_KM_reflection(KM_instance: KubelkaMunk, mock_stack, mock_material_params, xp):
    result = KM_instance.reflection(mock_stack, mock_material_params, xp.array([1.0, 0.5, 0.0]))
    assert result.shape == (3,)
    assert xp.all(xp.isfinite(result))

def test_KM_reflection_order(KM_instance: KubelkaMunk, mock_black_top_white_bottom, mock_white_top_black_bottom, xp):
    stack_bt, param_bt = mock_black_top_white_bottom
    stack_wt, param_wt = mock_white_top_black_bottom

    result_bt = KM_instance.reflection(stack_bt, param_bt, xp.array([1]*len(param_bt.absorption_coeff[0])))
    result_wt = KM_instance.reflection(stack_wt, param_wt, xp.array([1]*len(param_bt.absorption_coeff[0])))
    assert xp.all(xp.isfinite(result_bt))
    assert xp.all(result_bt > 1e-5)
    assert xp.all(result_bt < 1)
    assert xp.all(result_bt < result_wt)
