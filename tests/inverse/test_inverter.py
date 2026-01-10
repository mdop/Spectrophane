import numpy as np
from math import ceil
import pytest
from spectrophane.inverse.inverter import LUTInverter
from spectrophane.core.dataclasses import StackCandidates
from spectrophane.color.conversions import decode_rgb, linrgb_to_xyz

@pytest.fixture
def mock_stack_generator():
    class MockStackGenerator:
        def generate(self, mode):
            if mode == "complete":
                return StackCandidates(material_nums=   np.array([[1, 2, 3],[1, 2, 3],[1, 2, 3]]),
                                       thicknesses=     np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]]),
                                       request_id=      np.array([1, 2, 3]),
                                       score=           np.array([1, 2, 3]))
            else:
                return []
    return MockStackGenerator()

@pytest.fixture
def mock_evaluator():
    class MockEvaluator:
        def evaluate(self, stacks):
            return np.array([np.array([0.1, 0.2, 0.3])])
    return MockEvaluator()

@pytest.mark.parametrize("compression_factor, steps", [
    (1, 256),
    (2, 128),
    (7, 37),
])
def test_lut_inverter_initialization(mock_stack_generator, mock_evaluator, compression_factor, steps):
    inverter = LUTInverter(lut_compression_factor=compression_factor, stack_generator=mock_stack_generator, evaluator=mock_evaluator)
    assert inverter._stack_generator is mock_stack_generator
    assert inverter._eval is mock_evaluator
    assert inverter._steps == steps

def test_lut_generate_xyz_space(mock_stack_generator, mock_evaluator):
    compression = 3
    count = ceil(256/compression)
    inverter = LUTInverter(lut_compression_factor=compression, stack_generator=mock_stack_generator, evaluator=mock_evaluator)
    xyz_space = inverter._generate_xyz_space()
    assert xyz_space.shape == (count, count, count, 3)

    #test correct array assembly by random sampling (other than potential last clipped entry)
    test_points = np.random.randint(0,255//compression,size=(100,3))
    test_xyz = linrgb_to_xyz(decode_rgb(np.clip((test_points+0.5)*compression/255.0, 0, 1)))
    result_sample_xyz = xyz_space[test_points[:,0], test_points[:,1], test_points[:,2]]
    assert np.allclose(test_xyz, result_sample_xyz)



def test_generate_lut(mock_stack_generator, mock_evaluator):
    inverter = LUTInverter(lut_compression_factor=4, stack_generator=mock_stack_generator, evaluator=mock_evaluator)
    inverter._generate_lut()
    assert inverter._lut is not None
    assert inverter._lut.shape == (64, 64, 64)

def test_lut_invert_rgb(mock_stack_generator, mock_evaluator):
    inverter = LUTInverter(lut_compression_factor=1, stack_generator=mock_stack_generator, evaluator=mock_evaluator)
    rgb = np.array([0.5, 0.5, 0.5])
    result = inverter.invert_rgb(rgb)
    assert isinstance(result, StackCandidates)