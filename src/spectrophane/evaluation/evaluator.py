from spectrophane.evaluation.cache import ForwardCache

class Evaluator:
    """Wrapper class for physics model forward calculation including caching to speed up calculation and post-processing of resulting colors."""
    def __init__(self, cache: ForwardCache, mode: str, renormalizers=None):
        self._cache = cache
        self._renormalizers = renormalizers or {}

    def evaluate(self, theory, stacks, mode, normalize=True):
        #jax border is somewhere in here
        pass
