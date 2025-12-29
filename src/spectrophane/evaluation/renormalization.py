

class Renormalizer:
    def __init__(self, scale_xyz):
        self._scale_xyz = scale_xyz

    def normalize(self, xyz):
        return xyz / self._scale_xyz



class RenormalizerFactory:

    def build(self, *, theory, constraints, evaluator, wavelengths, mode, backend):
        pass
        """# 1. Determine brightest admissible stack (placeholder)
        brightest_stack = constraints.brightest_stack()

        # 2. Evaluate once
        raw_xyz = evaluator.evaluate(theory, brightest_stack, wavelengths, mode, backend, normalize=False)

        # 3. Compute scale
        scale_xyz = raw_xyz.max(axis=0)

        return Renormalizer(scale_xyz)"""
