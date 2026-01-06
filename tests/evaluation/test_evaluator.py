import pytest
import numpy as np
from spectrophane.evaluation.evaluator import Evaluator
from spectrophane.core.dataclasses import MaterialParams, StackData
from spectrophane.core.jax_utils import jaxify
from spectrophane.evaluation.cache import ForwardCache
from spectrophane.physics.mix_theories import THEORY_REGISTRY

@pytest.fixture
def config():
    res = {}
    res["theory"] = "kubelka_munk"
    res["view_geometry"] = "reflection"
    res["cache"] = ForwardCache("dict")
    res["material_parameters"] = MaterialParams(absorption_coeff=np.random.random((3,10)), scattering_coeff=np.random.random((3,10)), model_type="kubelka_munk")
    res["illuminator"] = np.ones(10)
    res["observer"] = np.ones((3,10))
    res["step_wavelength"] = 1
    res["backing"] = np.ones(10)
    res["calc_backend"] = "jax"
    res["edge_stacks"] = StackData(material_nums=np.array([[0,1],[1,0],[2,1]]), thicknesses=np.array([[0.1,0.1],[0.2,0.1],[0.3,0.2]]))
    return res

def test_evaluator_initialization(config, mocker):
    #mocker.patch("spectrophane.evaluation.evaluator.Evaluator.set_renormalizer", return_value = None)
    evaluator = Evaluator(**config)

    assert evaluator._cache == config["cache"]
    assert evaluator._calc_backend_str == config["calc_backend"]
    assert isinstance(evaluator._model, THEORY_REGISTRY[config["theory"]])
    assert evaluator._view_geometry == config["view_geometry"]
    assert evaluator._backing.shape[0] == 10

def test_set_renormalizer(config):
    calibration_stack = config["edge_stacks"]
    config["edge_stacks"] = None
    evaluator = Evaluator(**config)

    # Mocking evaluate to return dummy colors
    dummy_colors = np.random.rand(5, 3)
    evaluator.evaluate = lambda stacks, normalize: dummy_colors

    assert evaluator._renormalizer._scale_xyz == 1
    evaluator.set_renormalizer(calibration_stack)
    assert evaluator._renormalizer._scale_xyz != 1

def test_evaluate_renormalize_caching(config):
    calibration_stack = config["edge_stacks"]
    config["edge_stacks"] = None
    evaluator = Evaluator(**config)

    eval_stack1 = StackData(material_nums=np.array([[1,0],[0,1],[1,2]]), thicknesses=np.array([[0.1,0.1],[0.2,0.1],[0.3,0.2]]))
    eval_stack2 = StackData(material_nums=np.array([[1,0],[0,1],[1,2]]), thicknesses=np.array([[0.2,0.1],[0.2,0.2],[0.3,0.1]]))

    #show change in return after calibration
    stack_reflection_result = np.random.rand(10,) * 0.9
    evaluator._model.reflection = lambda stacks, params, backing: stack_reflection_result
    stack1_initial_xyz = evaluator.evaluate(eval_stack1)
    evaluator._model.reflection = lambda stacks, params, backing: stack_reflection_result*0.9 #change return value to proof values were taken from cache
    stack1_initial_xyz_cached = evaluator.evaluate(eval_stack1)
    assert np.all(stack1_initial_xyz == stack1_initial_xyz_cached)

    evaluator._model.reflection = lambda stacks, params, backing: np.array([0.9]*10)
    evaluator.set_renormalizer(calibration_stack)
    evaluator._model.reflection = lambda stacks, params, backing: stack_reflection_result
    stack1_xyz = evaluator.evaluate(eval_stack1)
    assert np.all(stack1_xyz > stack1_initial_xyz)

    stack2_xyz = evaluator.evaluate(eval_stack2)
    evaluator._model.reflection = lambda stacks, params, backing: stack_reflection_result*0.9
    stack2_xyz_cached = evaluator.evaluate(eval_stack2)
    assert np.all(stack2_xyz == stack2_xyz_cached)


def test_evaluate_KM_integration(config):
    test_stacks = config["edge_stacks"]
    config["edge_stacks"] = None
    evaluator = Evaluator(**config)
    result = evaluator.evaluate(test_stacks)

    assert np.all(result <= 1)
