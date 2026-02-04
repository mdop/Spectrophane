from typing import Tuple

import jax.numpy as jnp
import jax
from jax import jit
import optax

from spectrophane.core.dataclasses import MaterialParams, WavelengthAxis, LightSources, Observers
from spectrophane.color.conversions import spectrum_to_xyz
from spectrophane.core.jax_utils import jaxify
from spectrophane.io.resources import get_json_resource
from spectrophane.training.ingest_spectra import TrainingRefSpectraData, prepare_spectrum_data
from spectrophane.training.ingest_images import TrainingRefImageData, parse_image_data
from spectrophane.color.spectral_helper import parse_light_sources, parse_observers


from spectrophane.physics.mix_theories import BaseTheory, THEORY_REGISTRY

#jax prevents 64 bit arrays if not explicitly specified
jax.config.update('jax_enable_x64', True)

# Register backend-neutral dataclasses with JAX at this JAX boundary.
from spectrophane.core.jax_utils import register_with_jax
register_with_jax()


def import_test_data(filename: str) -> Tuple[TrainingRefSpectraData, TrainingRefImageData]:
    """Imports data for training spectra from the specified file."""
    input_data = get_json_resource("material_data/"+filename)
    image_ref_data = parse_image_data(input_data)
    spectrum_ref_data = prepare_spectrum_data(input_data)
    target_axis = WavelengthAxis(start=spectrum_ref_data.min_wavelength, step=spectrum_ref_data.step_wavelength, length=spectrum_ref_data.transmission_spectra.shape[1])
    light_sources = parse_light_sources(input_data, target_axis)
    observer = parse_observers(input_data, target_axis)
    return input_data["materials"], jaxify(image_ref_data), jaxify(spectrum_ref_data), jaxify(light_sources), jaxify(observer)

def initialize_parameter(model: BaseTheory, material_count, wavelength_axis: WavelengthAxis):
    """Initialize parameters for training. Prefers theory specific initialization. If not implemented defaults to set all parameters as 1"""
    if hasattr(model, "initial_guess"):
        return model.initial_guess(material_count, wavelength_axis)
    else:
        return MaterialParams(wl_start=wavelength_axis.start,
                              wl_step=wavelength_axis.step,
                              absorption_coeff=jnp.ones((material_count, wavelength_axis.length), dtype=jnp.float64),
                              scattering_coeff=jnp.ones((material_count, wavelength_axis.length), dtype=jnp.float64))


def compute_loss(model: BaseTheory, parameter: jnp.ndarray, ref_image_data: TrainingRefImageData, ref_spectrum_data: TrainingRefSpectraData, light_sources: jnp.ndarray, CIE1931: jnp.ndarray):
    """Calculates loss for spectrum and image data."""
    #calculate predicted spectra batched and take mean difference of Reflection/Transmission as loss
    pred_spectrum_transmission = jax.vmap(model.transmission, in_axes=(0, None))(ref_spectrum_data.transmission_stacks, parameter)
    pred_spectrum_reflection = jax.vmap(model.reflection, in_axes=(0, None, 0))(ref_spectrum_data.reflection_stacks, parameter, ref_spectrum_data.reflection_background)
    spectrum_transmission_loss = jnp.mean(jnp.abs(pred_spectrum_transmission-ref_spectrum_data.transmission_spectra), axis=0)
    spectrum_reflection_loss = jnp.mean(jnp.abs(pred_spectrum_reflection-ref_spectrum_data.reflection_spectra), axis=0)
    spectrum_loss = jnp.mean(jnp.stack((spectrum_reflection_loss, spectrum_transmission_loss), axis=0), axis = None)

    #calculate predicted xyz. Use CIE 1931 observer to adhere to sRGB definition, for human perception later CIE 1964 is better suited
    pred_transmission_image_spectrum = jax.vmap(model.transmission, in_axes=(0, None))(ref_image_data.transmission_stacks, parameter)
    light_spectra = light_sources[ref_image_data.transmission_light_source_indexes]
    pred_transmission_image_xyz = jax.vmap(spectrum_to_xyz, in_axes=[0,0,None, None])(pred_transmission_image_spectrum, light_spectra, CIE1931, parameter.absorption_coeff.shape[1])
    image_loss = jnp.mean(jnp.abs(pred_transmission_image_xyz - ref_image_data.transmission_xyz),None)

    #combine losses to total loss
    #TODO: Improve total loss calculation from manual weights, e.g. Uncertainty-based weighting (Kendall et al., CVPR 2018) or GradNorm?
    total_loss = 0.7 * spectrum_loss + 0.3 * image_loss
    return total_loss


def train_parameter(model_name: str, material_count: int, 
                    wavelength_axis: WavelengthAxis, image_ref: TrainingRefImageData, spectra_ref: TrainingRefSpectraData, 
                    light_sources: LightSources, CIE1931: Observers, 
                    num_steps=10, lr=1e-1):
    losses = [0.0]*num_steps
    model = THEORY_REGISTRY[model_name]("jax")
    parameter = initialize_parameter(model, material_count, wavelength_axis)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(parameter)

    loss_fn = lambda p: compute_loss(model, p, image_ref, spectra_ref, light_sources.spectra.values, CIE1931.spectra.values)
    grad_fn = jax.value_and_grad(loss_fn)

    @jit
    def train_step(params, opt_state):
        loss, grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for step in range(num_steps):
        parameter, opt_state, loss = train_step(parameter, opt_state)
        losses[step] = loss

    return parameter, losses
