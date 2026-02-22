import json
from pathlib import Path
from typing import Sequence
import jax.numpy as jnp

from spectrophane.training.trainer import import_test_data, train_parameter
from spectrophane.training.material_parameter import serialize_parameter, plot_parameter, terminal_color_comparison_string, plot_loss_series, extract_spectral_plot_series
from spectrophane.io.resources import get_resource_path
from spectrophane.core.dataclasses import MaterialParams, TrainingRefImageData, TrainingConfig
from spectrophane.color.conversions import xyz_to_linrgb, encode_rgb


def resolve_training_paths(calibration_path: str, output_path: str | None, calibration_local: bool, output_local: bool) -> tuple[Path, Path | None]:
    calib = Path(get_resource_path(calibration_path)) if calibration_local else Path(calibration_path)
    out = None
    if output_path is not None:
        out = Path(get_resource_path(output_path)) if output_local else Path(output_path)
    return calib, out

def load_training_references(calibration_file: Path):
    with calibration_file.open("r") as f:
        calibration_data = json.load(f)

    return import_test_data(calibration_data)

def emit_training_outputs(material_data: Sequence[dict], 
                          params: MaterialParams, 
                          loss_series: Sequence[float] | None, 
                          image_ref: TrainingRefImageData, 
                          output_path: Path | None, 
                          calc_colors: jnp.ndarray | None,
                          config: TrainingConfig) -> dict:
    output = {}

    #parameter to disk
    metadata = {"model": config.model,
                "observer": config.observer,
                "steps": config.steps,
                "lr": config.lr}
    if loss_series:
        metadata["losses"] = loss_series
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        serialized_output = serialize_parameter(material_data=material_data, parameter=params, metadata=metadata)
        with open(output_path, "w") as file:
            json.dump(serialized_output, file, indent=4)

    #parameter plots
    if config.parameter_plot_filter is not None:
        plot_lines = extract_spectral_plot_series(material_data, params, config.parameter_plot_filter)
        output["parameter_plot"] = plot_parameter(plot_lines, rows=config.parameter_plot_rows)
    
    #losses
    if loss_series is not None:
        output["loss_plot"] = plot_loss_series(loss_series)
    
    #terminal color comparison output
    if config.get_terminal_color_comparison:
        calc_rgb = round(encode_rgb(xyz_to_linrgb(calc_colors))*255)
        actual_rgb = round(encode_rgb(xyz_to_linrgb(image_ref.transmission_xyz))*255)
        output["terminal_color_str"] = terminal_color_comparison_string(calculated_colors=calc_rgb, actual_color=actual_rgb)

    return output


def parameter_training_pipeline(calibration_filepath: str, output_path: str | None, calibration_local: bool = True, output_local: bool = True, config: TrainingConfig = TrainingConfig()):
    """Process pipeline for parameter training and output. 
    Calibration and output path may be resource paths or absolute paths, depending on _local switches.
    Model is described by its registration string (see physics model for keys)
    Observer CIE1931 is downloaded with the installation script.
    """
    calib_path, out_path = resolve_training_paths(calibration_filepath,
                                                  output_path,
                                                  calibration_local=calibration_local,
                                                  output_local=output_local)

    material_list, image_ref, spectra_ref, light_sources, observers, wavelength_axis = load_training_references(calib_path)

    parameter, loss_series, calc_ref_image_xyz = train_parameter(model_name=config.model,
                                                                material_count=len(material_list),
                                                                wavelength_axis=wavelength_axis,
                                                                image_ref=image_ref,
                                                                spectra_ref=spectra_ref,
                                                                light_sources=light_sources,
                                                                single_observer=observers.take_names(config.observer),
                                                                num_steps=config.training_steps,
                                                                lr=config.lr)

    output = emit_training_outputs(material_data=material_list,
                                   params=parameter,
                                   loss_series=loss_series,
                                   image_ref=image_ref,
                                   output_path=out_path,
                                   calc_colors=calc_ref_image_xyz,
                                   config=config)

    return output