# tests/test_parameter_training.py
import pytest
import jax.numpy as jnp
import jax

from spectrophane.material_characterization.parameter_training import compute_loss, train_parameter, initialize_parameter
from spectrophane.material_characterization.mix_theories import BaseTheory
from spectrophane.core.dataclasses import TrainingRefImageData, TrainingRefSpectraData, StackData, MaterialParams

# Mock classes for testing
class MockTheory(BaseTheory):
    def initial_guess(self, material_count, min_wavelength, step_wavelength, spectrum_length):
        """Return an OpticalParams object as initial values."""
        return MaterialParams(jnp.ones((material_count, spectrum_length)),jnp.ones((material_count, spectrum_length)), "mock_model")
    
    def transmission(self, stacks, parameter):
        #return jnp.ones((10,))
        return parameter.absorption_coeff[0]+parameter.scattering_coeff[0]

    def reflection(self, stacks, parameter, background):
        return jnp.ones((10,))

class MockRefImageData(TrainingRefImageData):
    def __init__(self):
        self.transmission_stacks = StackData(material_nums=jnp.array([[0,1],[1,-1]]), thicknesses=jnp.ones((2,2)), stack_counts=jnp.array([2,1]))
        self.transmission_light_source_indexes = jnp.array([0,0])
        self.transmission_xyz = jnp.ones((2, 3))

class MockRefSpectraData(TrainingRefSpectraData):
    def __init__(self):
        self.transmission_stacks   = StackData(material_nums=jnp.array([[0,1],[1,-1]]), thicknesses=jnp.ones((2,2)), stack_counts=jnp.array([2,1]))
        self.reflection_stacks     = StackData(material_nums=jnp.array([[0,1],[1,-1]]), thicknesses=jnp.ones((2,2)), stack_counts=jnp.array([2,1]))
        self.transmission_spectra  = jnp.ones((2, 10))
        self.reflection_spectra    = jnp.ones((2, 10))
        self.reflection_background = jnp.ones((2, 10))

class MockLightSources:
    def __init__(self):
        self.data = jnp.ones((1, 10))

class MockCIE1931:
    def __init__(self):
        self.data = jnp.ones((3,10))

# Test fixtures
@pytest.fixture
def mock_model():
    return MockTheory()

@pytest.fixture
def mock_ref_image_data():
    return MockRefImageData()

@pytest.fixture
def mock_ref_spectrum_data():
    return MockRefSpectraData()

@pytest.fixture
def mock_light_sources():
    return MockLightSources()

@pytest.fixture
def mock_cie1931():
    return MockCIE1931()

# Test cases
def test_compute_loss(mock_model, mock_ref_image_data, mock_ref_spectrum_data, mock_light_sources, mock_cie1931):
    parameter = initialize_parameter("mock_model", 2, 400, 40, 10)
    light_sources = mock_light_sources.data
    CIE1931 = mock_cie1931.data

    loss = compute_loss(
        model=mock_model,
        parameter=parameter,
        ref_image_data=mock_ref_image_data,
        ref_spectrum_data=mock_ref_spectrum_data,
        light_sources=light_sources,
        CIE1931=CIE1931
    )

    assert jnp.ndim(loss) == 0, "loss must be a scalar"
    assert loss >= 0, "loss must be positive or 0"

def test_train_parameter_initialization(mocker, mock_model, mock_ref_image_data, mock_ref_spectrum_data, mock_light_sources, mock_cie1931):
    """Test that the training process initializes correctly."""
    light_sources = mock_light_sources.data
    CIE1931 = mock_cie1931.data

    mocker.patch.dict("spectrophane.material_characterization.mix_theories.THEORY_REGISTRY", {"mock_model": mock_model})
    parameter = train_parameter("mock_model", 2, 400, 40, 10, mock_ref_image_data, mock_ref_spectrum_data, light_sources, CIE1931)
    assert parameter is not None, "Training should return a parameter"

def test_train_parameter_parameters_update(mocker, mock_model, mock_ref_image_data, mock_ref_spectrum_data, mock_light_sources, mock_cie1931):
    """Test that the parameters are updated during training steps."""
    light_sources = mock_light_sources.data
    CIE1931 = mock_cie1931.data

    mocker.patch.dict("spectrophane.material_characterization.mix_theories.THEORY_REGISTRY", {"mock_model": mock_model})
    start_params = initialize_parameter("mock_model", 2, 400, 40, 10)

    # Mock the compute_loss and gradient functions to ensure parameters change
    #mock_grad_fn = mocker.patch("jax.value_and_grad")
    #mock_grad_fn.return_value = mocker.MagicMock(side_effect=lambda params: (1.0, jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), start_params)))
    def fake_grad_fn(fn):
        def wrapper(params):
            grads = jax.tree_util.tree_map(
                lambda x: jnp.ones_like(x),   # return ONES instead of zeros
                params,
            )
            return 1.0, grads
        return wrapper

    mock_grad = mocker.patch("jax.value_and_grad", side_effect=lambda fn: fake_grad_fn(fn))

    parameter1, _  = train_parameter("mock_model", 2, 400, 40, 10, mock_ref_image_data, mock_ref_spectrum_data, light_sources, CIE1931, 1)
    parameter10, _ = train_parameter("mock_model", 2, 400, 40, 10, mock_ref_image_data, mock_ref_spectrum_data, light_sources, CIE1931, 10)

    # Verify that the parameters are not the same as the initial ones
    assert not jnp.allclose(start_params.absorption_coeff, parameter1.absorption_coeff)
    assert not jnp.allclose(parameter10.absorption_coeff, parameter1.absorption_coeff)
    assert not jnp.allclose(start_params.absorption_coeff, parameter1.absorption_coeff)
    assert not jnp.allclose(parameter10.absorption_coeff, parameter1.absorption_coeff)
