import numpy as np

from spectrophane.core.dataclasses import StackData
from spectrophane.stacks.generator import RuleBasedGenerator
from spectrophane.evaluation.cache import InMemoryForwardCache
from spectrophane.inverse.inverse import StackCandidate


def test_rule_based_generator_and_cache_roundtrip():
    gen = RuleBasedGenerator()
    params = {"material_nums": [1, 2], "thicknesses": [0.1, 0.2], "stack_counts": [1, 1]}
    stack = gen.generate(params)

    assert isinstance(stack, StackData)
    assert np.all(stack.material_nums == np.array([1, 2]))

    cache = InMemoryForwardCache()
    result = {"spectrum": np.linspace(0.0, 1.0, 10)}
    cache.set(stack, result)
    retrieved = cache.get(stack)
    assert retrieved is result


def test_stack_candidate_container():
    stack = StackData(material_nums=np.array([0]), thicknesses=np.array([0.1]), stack_counts=np.array([1]))
    cand = StackCandidate(stack=stack, score=0.5, meta={"note": "test"})
    assert cand.score == 0.5
    assert cand.meta["note"] == "test"
