import jax.numpy as jnp
import jax
import pytest
import copy

from spectrophane.physics.mix_theories import KubelkaMunk
from spectrophane.core.dataclasses import StackData, MaterialParams

jax.config.update('jax_enable_x64', True)

@pytest.fixture
def KM_instance():
    return KubelkaMunk()

@pytest.fixture
def KM_random_parameter():
    key = jax.random.key(42)
    K = jax.random.uniform(key, shape=(100,), dtype=jnp.float64) * 10
    key, _ = jax.random.split(key)
    S = jax.random.uniform(key, shape=(100,), dtype=jnp.float64) * 10
    key, _ = jax.random.split(key)
    d = jax.random.uniform(key, shape=(100,), dtype=jnp.float64) / 10
    key, _ = jax.random.split(key)
    return K, S, d


def test_KM_layer_shape_determinant(KM_instance: KubelkaMunk, KM_random_parameter):
    K, S, d = KM_random_parameter
    M = KM_instance._single_layer_transfer_matrix(K, S, d)
    det = M[0,0] * M[1,1] - M[1,0] * M[0,1]

    assert M.shape == (2,2,100)
    #determinant of the transfer matrix can be proven to be 1
    assert jnp.allclose(det, 1, atol=1e-5)
    assert not jnp.isnan(M).any()

def test_KM_layer_zero_thickness_identity(KM_instance: KubelkaMunk, KM_random_parameter):
    K, S, _ = KM_random_parameter
    d = jnp.array([0.0]*len(K))

    M = KM_instance._single_layer_transfer_matrix(K, S, d)
    I = jnp.array([[[1.0], [0.0]], [[0.0], [1.0]]])

    assert jnp.allclose(M, I, atol=1e-5)

def KM_reflectance_from_matrix(M):
    m11, m12 = M[0,0], M[0,1]
    m21, m22 = M[1,0], M[1,1]
    return m12 / m22

def test_KM_layer_semi_infinite_limit(KM_instance: KubelkaMunk, KM_random_parameter):
    K, S, d = KM_random_parameter
    d = (d+1)*10 #very thick, but not too thick for numerical stability

    M = KM_instance._single_layer_transfer_matrix(K, S, d)
    R = KM_reflectance_from_matrix(M)

    R_inf = 1+K/S-jnp.sqrt(jnp.square(K/S)+2*K/S)
    assert jnp.allclose(R, R_inf, atol=1e-3)


def test_KM_chain_transfer_matrizes_shape(KM_instance: KubelkaMunk):
    key = jax.random.key(42)
    transfer_matrices = jax.random.uniform(key, shape=(5,2,2,100), dtype=jnp.float64)
    result = KM_instance._chain_transfer_matrizes(transfer_matrices)
    assert result.shape == (2, 2, 100), "Output shape should be (2, 2, 100)"
    assert not jnp.isnan(result).any()

def test_KM_chain_transfer_matrizes_identity(KM_instance: KubelkaMunk):
    # Test with identity matrices
    transfer_matrizes = jnp.array([[[[1,1,1],[0,0,0]],[[0,0,0],[1,1,1]]]])
    result = KM_instance._chain_transfer_matrizes(transfer_matrizes)
    assert jnp.allclose(result, transfer_matrizes), "Identity matrices should return identity matrix"


@pytest.fixture
def mock_stack():
    return StackData(material_nums=jnp.array([0, 1]), thicknesses=jnp.array([1.0, 2.0]), stack_counts=jnp.array([2]))

@pytest.fixture
def mock_material_params():
    return MaterialParams(
        absorption_coeff=jnp.array([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]]), 
        scattering_coeff=jnp.array([[0.5, 0.6, 0.7], [0.7, 0.8, 0.9]])
    )

def test_KM_stack_transfer_matrix_multiple_layers(KM_instance: KubelkaMunk, mock_stack, mock_material_params):
    # Mock data for multiple layers
    result = KM_instance._stack_transfer_matrix(mock_stack, mock_material_params)
    assert result.shape == (2, 2, 3), "Transfer matrix should be 2x2"
    assert not jnp.isnan(result).any(), "Transfer matrix should not contain NaN values"

def test_KM_stack_transfer_matrix_single_layer(KM_instance: KubelkaMunk):
    # Mock data for a single layer
    stack = StackData(material_nums=jnp.array([0]), thicknesses=jnp.array([1.0]), stack_counts=jnp.array([1]))
    params = MaterialParams(
        absorption_coeff=jnp.array([[0.1, 0.2, 0.3]]), 
        scattering_coeff=jnp.array([[0.5, 0.6, 0.7]])
    )
    
    result = KM_instance._stack_transfer_matrix(stack, params)
    assert result.shape == (2, 2, 3), "Transfer matrix should be 2x2"
    assert not jnp.isnan(result).any(), "Transfer matrix should not contain NaN values"

def test_KM_stack_transfer_matrix_zero_thickness(KM_instance: KubelkaMunk, mock_stack, mock_material_params):
    # Mock data with zero thickness
    mock_stack.thicknesses = jnp.array([0,0])
    
    result = KM_instance._stack_transfer_matrix(mock_stack, mock_material_params)
    assert result.shape == (2, 2, 3), "Transfer matrix should be 2x2"
    assert not jnp.isnan(result).any(), "Transfer matrix should not contain NaN values"

def test_KM_stack_transfer_matrix_large_number_of_layers(KM_instance: KubelkaMunk):
    # Mock data with a large number of layers
    stack = StackData(material_nums=jnp.array([0] * 100), thicknesses=jnp.array([1.0] * 100), stack_counts=jnp.array([100]))
    params = MaterialParams(
        absorption_coeff=jnp.array([[0.1, 0.2, 0.3]]), 
        scattering_coeff=jnp.array([[0.5, 0.6, 0.7]])
    )
    
    result = KM_instance._stack_transfer_matrix(stack, params)
    assert result.shape == (2, 2, 3), "Transfer matrix should be 2x2"
    assert not jnp.isnan(result).any(), "Transfer matrix should not contain NaN values"


@pytest.fixture
def mock_black_top_white_bottom():
    stack = StackData(material_nums=jnp.array([0,1]), thicknesses=jnp.array([1,1]), stack_counts=jnp.array([2]))
    param = MaterialParams(absorption_coeff=jnp.array([[0.001]*100, [3]*100]), scattering_coeff=jnp.array([[10]*100, [10]*100]))
    return stack, param

@pytest.fixture
def mock_white_top_black_bottom(mock_black_top_white_bottom):
    data = copy.deepcopy(mock_black_top_white_bottom)
    data[0].material_nums = jnp.array([1,0])
    return data

def test_KM_transmission(KM_instance: KubelkaMunk, mock_stack, mock_material_params):
    result = KM_instance.transmission(mock_stack, mock_material_params)
    assert result.shape == (3,)
    assert jnp.all(jnp.isfinite(result))

def test_KM_transmission_order(KM_instance: KubelkaMunk, mock_black_top_white_bottom, mock_white_top_black_bottom):
    stack_bt, param_bt = mock_black_top_white_bottom
    stack_wt, param_wt = mock_white_top_black_bottom

    result_bt = KM_instance.transmission(stack_bt, param_bt)
    result_wt = KM_instance.transmission(stack_wt, param_wt)
    assert jnp.all(jnp.isfinite(result_bt))
    assert jnp.all(result_bt > 1e-5)
    assert jnp.all(result_bt <= 1)
    assert jnp.allclose(result_bt, result_wt)

def test_KM_reflection(KM_instance: KubelkaMunk, mock_stack, mock_material_params):
    result = KM_instance.reflection(mock_stack, mock_material_params, jnp.array([1.0, 0.5, 0.0]))
    assert result.shape == (3,)
    assert jnp.all(jnp.isfinite(result))

def test_KM_reflection_order(KM_instance: KubelkaMunk, mock_black_top_white_bottom, mock_white_top_black_bottom):
    stack_bt, param_bt = mock_black_top_white_bottom
    stack_wt, param_wt = mock_white_top_black_bottom

    result_bt = KM_instance.reflection(stack_bt, param_bt, jnp.array([1]*len(param_bt.absorption_coeff[0])))
    result_wt = KM_instance.reflection(stack_wt, param_wt, jnp.array([1]*len(param_bt.absorption_coeff[0])))
    assert jnp.all(jnp.isfinite(result_bt))
    assert jnp.all(result_bt > 1e-5)
    assert jnp.all(result_bt < 1)
    assert jnp.all(result_bt < result_wt)
