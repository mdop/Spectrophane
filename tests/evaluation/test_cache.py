import pytest
import numpy as np
from copy import deepcopy
from spectrophane.evaluation.cache import ForwardCache, DictCacheBackend
from spectrophane.core.dataclasses import StackData

@pytest.fixture
def stack_data():
    material_nums = np.array([[1, 2, 3],[2, 2, 3],[1, 2, 1],[1, 1, 3]])
    thicknesses = np.array([[0.1, 0.3, 0.3],[0.1, 0.2, 0.4],[0.2, 0.2, 0.3],[0.1, 0.2, 0.5]])
    return StackData(material_nums=material_nums, thicknesses=thicknesses)

@pytest.fixture
def values():
    return np.array([[10.0]*3, [20.0]*3, [30.0]*3, [40.0]*3])

def test_forward_cache_batch_contains(stack_data, values):
    # Create a ForwardCache with DictCacheBackend
    cache = ForwardCache(cache_backend="dict")
    cache.batch_set(stack_data, values)

    # Check if all values are present
    contains_mask = cache.batch_contains(stack_data)
    assert np.all(contains_mask), "All values should be present in the cache"

def test_forward_cache_batch_set_and_get(stack_data, values):
    cache = ForwardCache(cache_backend="dict")
    cache.batch_set(stack_data, values)

    # Batch get values
    found, result = cache.batch_get(stack_data)

    # Check if all values are found and match
    assert np.all(found), "All values should be found in the cache"
    assert np.array_equal(result, values), "Retrieved values should match the set values"

def test_forward_cache_partial_sets(stack_data, values):
    # Create two different datasets
    stack_data2 = deepcopy(stack_data)
    stack_data2.material_nums = stack_data2.material_nums + 1
    stack_data2.thicknesses = stack_data2.thicknesses + 1
    stack_data_basesubset = stack_data.take(np.array([0,1]))

    cache = ForwardCache(cache_backend="dict")
    cache.batch_set(stack_data, values)

    # Check if values for stack_data1 are present
    contains_mask = cache.batch_contains(stack_data)
    assert np.all(contains_mask), "Values for stack_data1 should be present in the cache"
    contains_partial_mask, values_partial = cache.batch_get(stack_data_basesubset)
    assert np.all(contains_partial_mask)
    assert np.array_equal(values_partial, values[0:2])
    contains_mask2 = cache.batch_contains(stack_data2)
    assert not np.any(contains_mask2), "Values for stack_data2 should not be present in the cache"