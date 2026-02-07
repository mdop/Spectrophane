import jax.numpy as jnp
import numpy as np

from spectrophane.core.dataclasses import StackData, MaterialParams
from spectrophane.core.numeric_backend import Backend, NumPyBackend, JAXBackend

class BaseTheory:
    def __init__(self, backend: str):
        if(backend == "jax"):
            self.bn = JAXBackend()
        else:
            self.bn = NumPyBackend()
        self.xp = self.bn.xp

    def transmission_single(self, stack: StackData, params: MaterialParams) -> jnp.ndarray:
        """Returns transmission spectrum for the given stack. Only pass one stack to the function!"""
        raise NotImplementedError
    
    def transmission_batch(self, stacks: StackData, params: MaterialParams) -> jnp.ndarray:
        """Returns transmission spectrum for a given batch of stacks."""
        return self.bn.vmap(self.transmission_single, in_axes=(0, None))(stacks, params)
    
    def reflection_single(self, stack: StackData, params: MaterialParams, backing: jnp.ndarray):
        """Returns reflection spectrum for a given stack. Only pass one stack to the function!"""
        raise NotImplementedError
    
    def reflection_batch(self, stacks: StackData, params: MaterialParams, backing: jnp.ndarray) -> jnp.ndarray:
        """Returns transmission spectrum for a given batch of stacks."""
        return self.bn.vmap(self.reflection_single, in_axes=(0, None, 0))(stacks, params, backing)



THEORY_REGISTRY: dict[str, BaseTheory] = {}

def register_theory(name):
    """Decorates theory classes to register then as physics model that predicts reflection/transmission spectrum based on a stack and material parameters.
    Requires the following functions in the class:
    transmission(params: MaterialParams, stack: StackData)
    reflection(params: MaterialParams, stack: StackData, background: str)"""
    def decorator(cls):
        THEORY_REGISTRY[name] = cls
        return cls
    return decorator


@register_theory("kubelka_munk")
class KubelkaMunk(BaseTheory):
    def __init__(self, backend: str):
        super().__init__(backend)
    
    def _single_layer_transfer_matrix(self, K: jnp.ndarray | np.ndarray, S: jnp.ndarray | np.ndarray, d: jnp.ndarray | np.ndarray) -> jnp.ndarray | np.ndarray:
        """Returns the transfer matrix of a single layer. Arrays should be of float64 to prevent numerical instability."""
        a=(K+S)/S
        b=self.bn.sqrt(self.bn.square(a)-1)
        sinh_factor = self.bn.sinh(b*S*d)
        cosh_factor = self.bn.cosh(b*S*d)
        m11 = -a*sinh_factor + b*cosh_factor
        m22 =  a*sinh_factor + b*cosh_factor
        m12 = sinh_factor

        M = self.bn.stack([
            self.bn.stack([m11, m12], axis=0),
            self.bn.stack([-m12, m22], axis=0),
        ], axis=0) / b  # shape (2,2,wavelength)

        is_zero = d == 0
        I = self.bn.identity_transfer(2, K.shape[0], M.dtype)
        return self.bn.where(is_zero, I, M)
    
    def _chain_transfer_matrizes(self, transfer_matrizes: jnp.ndarray | np.ndarray, top_to_bottom: bool = True) -> jnp.ndarray | np.ndarray:
        """Multiplies transfer matrices for a stack to get a global transfer matrix. If top_to_bottom multiplication will stack for (I+(k), I-(k)) = M (I+(0), I-(0)). Matrix shape (matrizes, 2, 2, wavelengths)"""
        if top_to_bottom:
            transfer_matrizes = transfer_matrizes[::-1]

        # Chaining single wavelength
        def chain_one_wavelength(matrices):
            def body(acc, M):
                return self.bn.matmul(acc, M), None

            init = self.bn.eye(2, dtype=matrices.dtype)
            acc, _ = self.bn.scan(body, init, matrices)
            return acc

        # Apply the function to each wavelength independently
        M_total = self.bn.vmap(chain_one_wavelength, in_axes=-1, out_axes=2)(transfer_matrizes) # shape (2,2,wavelength)
        return M_total
    
    def _stack_transfer_matrix(self, stack: StackData, params: MaterialParams) -> jnp.ndarray | np.ndarray:
        """Calculates transfer matrix for a given stack (characterized by index and stack) and fundamental material parameters. The transfer matrix can be used to calculate reflection and transmission spectra."""
        material_ids = stack.material_nums
        thicknesses = stack.thicknesses

        Ks = params.absorption_coeff[material_ids]
        Ss = params.scattering_coeff[material_ids]
        ds = thicknesses

        layer_fn = self.bn.vmap(self._single_layer_transfer_matrix, in_axes=(0, 0, 0))
        Ms = layer_fn(Ks, Ss, ds)

        return self._chain_transfer_matrizes(Ms)
    
    def transmission_single(self, stack: StackData, params: MaterialParams):
        """Calculates transmission spectrum of a material stack. Returns a spectrum array in the shape (wavelengths,). Only pass one stack to the function!"""
        #TODO: Raise Warning if arrays are not float64?
        M = self._stack_transfer_matrix(stack, params)
        T = M[0,0] - M[0,1]*M[1,0]/M[1,1]
        return T
    
    def reflection_single(self, stack: StackData, params: MaterialParams, backing: jnp.ndarray | np.ndarray):
        """Calculates reflection spectrum of a material stack. Returns a spectrum array in the shape (wavelengths,). Only pass one stack to the function!"""
        M = self._stack_transfer_matrix(stack, params)
        R = (backing*M[0,0] + M[0,1]) / (backing*M[1,0] + M[1,1])
        return R