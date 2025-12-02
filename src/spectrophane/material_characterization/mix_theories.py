import jax.numpy as jnp
import jax

from spectrophane.core.dataclasses import StackData, MaterialParams

THEORY_REGISTRY = {}

def register_theory(name):
    """Decorates theory classes to register then as physics model that predicts reflection/transmission spectrum based on a stack and material parameters.
    Requires the following functions in the class:
    transmission(params: MaterialParams, stack: StackData)
    reflection(params: MaterialParams, stack: StackData, background: str)"""
    def decorator(cls):
        THEORY_REGISTRY[name] = cls()
        return cls
    return decorator


class BaseTheory:
    def transmission(self, stack: StackData, params: MaterialParams) -> jnp.ndarray:
        """Returns transmission spectrum for the given stack."""
        raise NotImplementedError
    
    def reflection(self, stack: StackData, params: MaterialParams, backing: jnp.ndarray):
        """Returns reflection spectrum for a given stack. Assumes black backing."""
        raise NotImplementedError

    def initial_guess(self, material_count, min_wavelength, step_wavelength, spectrum_length):
        """Return an OpticalParams object as initial values."""
        raise NotImplementedError



@register_theory("kubelka_munk")
class KubelkaMunk(BaseTheory):
    def _single_layer_transfer_matrix(self, K: jnp.ndarray, S: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
        """Returns the transfer matrix of a single layer. Arrays should be of float64 to prevent numerical instability."""
        a=(K+S)/S
        b=jnp.sqrt(jnp.square(a)-1)
        sinh_factor = jnp.sinh(b*S*d)
        cosh_factor = jnp.cosh(b*S*d)
        m11 = -a*sinh_factor + b*cosh_factor
        m22 =  a*sinh_factor + b*cosh_factor
        m12 = sinh_factor

        M = jnp.stack([
            jnp.stack([m11, m12], axis=0),
            jnp.stack([-m12, m22], axis=0),
        ], axis=0) / b  # shape (2,2,wavelength)
        return M
    
    def _chain_transfer_matrizes(self, transfer_matrizes: jnp.ndarray, top_to_bottom: bool = True) -> jnp.ndarray:
        """Multiplies transfer matrices for a stack to get a global transfer matrix. If top_to_bottom multiplication will stack for (I+(k), I-(k)) = M (I+(0), I-(0)). Matrix shape (matrizes, 2, 2, wavelengths)"""
        if top_to_bottom:
            transfer_matrizes = transfer_matrizes[::-1]

        # Chaining single wavelength
        def chain_one_wavelength(matrices):
            def body(acc, M):
                return acc @ M, None

            init = jnp.eye(2, dtype=matrices.dtype)
            acc, _ = jax.lax.scan(body, init, matrices)
            return acc

        # Apply the function to each wavelength independently
        M_total = jax.vmap(chain_one_wavelength, in_axes=-1)(transfer_matrizes)
        M_total = jnp.moveaxis(M_total, 0, -1)
        return M_total
    
    def _stack_transfer_matrix(self, stack: StackData, params: MaterialParams) -> jnp.ndarray:
        """Calculates transfer matrix for a given stack (characterized by index and stack) and fundamental material parameters. The transfer matrix can be used to calculate reflection and transmission spectra."""
        layer_material_ids = jax.lax.dynamic_slice(stack.material_nums, (0,), (stack.stack_counts.squeeze() ,))
        layer_thicknesses  = jax.lax.dynamic_slice(stack.thicknesses, (0,), (stack.stack_counts.squeeze() ,))
        Ks = params.absorption_coeff[layer_material_ids, :]
        Ss = params.scattering_coeff[layer_material_ids, :]
        transfer_matrizes = jax.vmap(self._single_layer_transfer_matrix, in_axes=(0, 0, 0))(Ks, Ss, layer_thicknesses)
        M_total = self._chain_transfer_matrizes(transfer_matrizes)
        return M_total
    
    def transmission(self, stack: StackData, params: MaterialParams):
        """Calculates transmission spectrum of a material stack. Returns a spectrum array in the shape (wavelengths,)"""
        #TODO: Raise Warning if arrays are not float64?
        M = self._stack_transfer_matrix(stack, params)
        T = M[0,0] - M[0,1]*M[1,0]/M[1,1]
        return T
    
    def reflection(self, stack: StackData, params: MaterialParams, backing: jnp.ndarray):
        """Calculates reflection spectrum of a material stack. Returns a spectrum array in the shape (wavelengths,)"""
        M = self._stack_transfer_matrix(stack, params)
        R = (backing*M[0,0] + M[0,1]) / (backing*M[1,0] + M[1,1])
        return R