import numpy as np
from numbers import Number

from spectrophane.evaluation.cache import ForwardCache
from spectrophane.core.dataclasses import MaterialParams, StackData
from spectrophane.core.jax_utils import jaxify, numpyify, register_with_jax
import spectrophane.physics.mix_theories as physics
from spectrophane.evaluation.renormalization import Renormalizer
from spectrophane.core.numeric_backend import JAXBackend, NumPyBackend
from spectrophane.color.conversions import spectrum_to_xyz, compute_spectrum_xyz_normalization_factor

class Evaluator:
    """Wrapper class for physics model forward calculation including caching to speed up calculation and post-processing of resulting colors."""
    def __init__(self, theory: str, view_geometry: str, cache: ForwardCache, material_parameters: MaterialParams, illuminator: np.ndarray, observer: np.ndarray, step_wavelength: np.ndarray,
                 backing: np.ndarray = None, calc_backend: str= "jax", edge_stacks: StackData = None):
        """Base settings for the evaluator. Theory is the string for which the theory is registered, e.g. kubelka_munk. illuminator and observer are assumed to be in the correct spectrum format."""
        self._cache = cache
        self._calc_backend_str = calc_backend
        self._model = physics.THEORY_REGISTRY[theory](calc_backend)
        self._view_geometry = view_geometry
        self._renormalizer = Renormalizer()
        self._step_wavelength = step_wavelength
        if not backing is None:
            self._backing = backing
        else:
            self._backing = np.ones(self._parameters.shape[1], dtype=np.float64)
        if self._calc_backend_str == "jax":
            self._calc_backend = JAXBackend()
            register_with_jax()
            self._parameters = jaxify(material_parameters)
            self._backing = jaxify(backing)
            self._illuminator = jaxify(illuminator)
            self._observer = jaxify(observer)
        else:
            self._calc_backend = NumPyBackend()
            self._parameters = numpyify(material_parameters)
            self._backing = numpyify(backing)
            self._illuminator = numpyify(illuminator)
            self._observer = numpyify(observer)
        self._spectrum_xyz_renormalization_factor = compute_spectrum_xyz_normalization_factor(self._illuminator, self._observer, step_wavelength)
        if edge_stacks:
            self.set_renormalizer(edge_stacks)
    
    def set_renormalizer(self, edge_stacks: StackData):
        """Finds a color corrector to account for different maximum brightness. Pass edge case stacks, e.g. thinnest stacks of every material, to correct colors in evaluate."""
        colors = self.evaluate(edge_stacks, normalize=False)
        self._renormalizer.find_scaling_factor(colors)

    def evaluate(self, stacks: StackData, normalize=True) -> np.ndarray:
        """Returns colors for a batch of requested stacks. Utilizes caching if set and optional post-processing."""
        #get cached values and filter down stacks to calculate
        found, cached = self._cache.batch_get(stacks)
        calc_indexes = np.nonzero(~found)
        calc_stacks = stacks.take(calc_indexes)

        #calculate missing, JAX-Entry
        if self._calc_backend_str == "jax":
            calc_stacks = jaxify(calc_stacks)

        if self._view_geometry == "transmission":
            calc_spectrum = self._model.transmission_batch(calc_stacks, self._parameters)
        else:
            calc_backings = self._backing[calc_indexes]
            calc_spectrum = self._model.reflection_batch(calc_stacks, self._parameters, calc_backings)
        calc_xyz = self._calc_backend.vmap(spectrum_to_xyz, in_axes=[0,None,None, None, None])(calc_spectrum, self._illuminator, self._observer, self._step_wavelength, self._spectrum_xyz_renormalization_factor)
        
        if self._calc_backend_str == "jax":
            calc_xyz = numpyify(calc_xyz)
        #JAX-Exit

        #reassemble result for complete request and cache calculation
        cached[calc_indexes] = calc_xyz
        self._cache.batch_set(calc_stacks, calc_xyz)
        if normalize:
            cached = self._renormalizer.normalize(cached)
        return cached