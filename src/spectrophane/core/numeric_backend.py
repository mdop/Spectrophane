from abc import ABC, abstractmethod

class Backend(ABC):
    """Minimal numerical backend abstraction."""

    @property
    @abstractmethod
    def xp(self):
        """Array namespace (np or jnp)."""

    @abstractmethod
    def sqrt(self, x): ...
    @abstractmethod
    def square(self, x): ...
    @abstractmethod
    def sinh(self, x): ...
    @abstractmethod
    def cosh(self, x): ...
    @abstractmethod
    def stack(self, xs, axis=0): ...
    @abstractmethod
    def eye(self, n, dtype): ...
    @abstractmethod
    def matmul(self, a, b): ...
    @abstractmethod
    def where(self, cond, x, y): ...
    @abstractmethod
    def vmap(self, fn, in_axes): ...
    @abstractmethod
    def scan(self, fn, in_axes): ...
    @abstractmethod
    def moveaxis(self, fn, in_axes): ...
    @abstractmethod
    def identity_transfer(self, wavelengths, dtype):
        """Return identity transfer matrix of shape (2, 2, wavelengths)."""
        ...


class NumPyBackend(Backend):
    def __init__(self):
        import numpy as np
        self._np = np

    @property
    def xp(self):
        return self._np

    def sqrt(self, x): return self._np.sqrt(x)
    def square(self, x): return self._np.square(x)
    def sinh(self, x): return self._np.sinh(x)
    def cosh(self, x): return self._np.cosh(x)
    def stack(self, xs, axis=0): return self._np.stack(xs, axis=axis)
    def eye(self, n, dtype): return self._np.eye(n, dtype=dtype)
    def matmul(self, a, b): return self._np.matmul(a, b)
    def where(self, cond, x, y): return self._np.where(cond, x, y)

    def vmap(self, fn, in_axes=0, out_axes=0):
        if isinstance(in_axes, (int, type(None))):
            in_axes = (in_axes,)

        def mapped(*args):
            xp = self.xp

            # Broadcast in_axes if needed
            if len(in_axes) == 1 and len(args) > 1:
                axes = in_axes * len(args)
            elif len(in_axes) != len(args):
                raise ValueError("in_axes must match number of arguments")
            else:
                axes = in_axes

            # Determine batch size
            batch_size = None
            for a, ax in zip(args, axes):
                if ax is not None:
                    axis = ax if ax >= 0 else a.ndim + ax
                    if axis < 0 or axis >= a.ndim:
                        raise ValueError("Invalid axis")
                    size = a.shape[axis]
                    if batch_size is None:
                        batch_size = size
                    elif size != batch_size:
                        raise ValueError("Mapped axes must have same size")

            if batch_size is None:
                raise ValueError("At least one in_axes entry must be non-None")

            outputs = []
            for i in range(batch_size):
                sliced_args = []
                for a, ax in zip(args, axes):
                    if ax is None:
                        sliced_args.append(a)
                    else:
                        axis = ax if ax >= 0 else a.ndim + ax
                        sliced_args.append(xp.take(a, i, axis=axis))
                outputs.append(fn(*sliced_args))

            # JAX semantics: new leading batch axis
            return xp.stack(outputs, axis=out_axes)

        return mapped
    
    def scan(self, fn, init, xs):
        acc = init
        outs = []
        for x in xs:
            acc, y = fn(acc, x)
            outs.append(y)
        if(xs.size == 0):
            out = self.xp.array([])
        else:
            out = self.xp.stack(outs)
        return acc, out
    def moveaxis(self, x, src, dst): return self.xp.moveaxis(x, src, dst)
    def identity_transfer(self, eye_size, transfer_size, dtype):
        """Return identity transfer matrix of shape (eye_size, eye_size, transfer_size) by broadcasting the identity matrix over the last axis."""
        I = self.xp.eye(eye_size, dtype=dtype)          #(2, 2)
        return self.xp.broadcast_to(I[..., None], (eye_size, eye_size, transfer_size))

    


class JAXBackend(Backend):
    def __init__(self):
        import jax.numpy as jnp
        import jax
        self._jnp = jnp
        self._jax = jax
        jax.config.update('jax_enable_x64', True)

    @property
    def xp(self):
        return self._jnp

    def sqrt(self, x): return self._jnp.sqrt(x)
    def square(self, x): return self._jnp.square(x)
    def sinh(self, x): return self._jnp.sinh(x)
    def cosh(self, x): return self._jnp.cosh(x)
    def stack(self, xs, axis=0): return self._jnp.stack(xs, axis=axis)
    def eye(self, n, dtype): return self._jnp.eye(n, dtype=dtype)
    def matmul(self, a, b): return a @ b
    def where(self, cond, x, y): return self._jnp.where(cond, x, y)

    def vmap(self, fn, in_axes=0, out_axes=0):
        return self._jax.vmap(fn, in_axes=in_axes, out_axes=out_axes)
    def scan(self, fn, init, xs): return self._jax.lax.scan(fn, init, xs)
    def moveaxis(self, x, src, dst): return self.xp.moveaxis(x, src, dst)
    def identity_transfer(self, eye_size, transfer_size, dtype):
        """Return identity transfer matrix of shape (eye_size, eye_size, transfer_size) by broadcasting the identity matrix over the last axis."""
        I = self.xp.eye(eye_size, dtype=dtype)          #(2, 2)
        return self.xp.broadcast_to(I[..., None], (eye_size, eye_size, transfer_size))