# src/spectrophane/cli/lithophane.py

import typer
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image

from spectrophane.pipeline.lithophane_pipeline import (
    file_to_parameter,
    file_to_spectral_helper,
    parameter_to_inverter,
    image_to_lithophane,
)

from spectrophane.pipeline.lithophane_factories import (
    generate_stack_rules_bottom_color_top_blocks,
)

from spectrophane.core.dataclasses import (
    EvaluatorSpec,
    InverterSpec,
    LithophaneConfig,
)

from spectrophane.inverse.stack_generation import StackGenerator


app = typer.Typer()


def lithophane_command(
    # Paths
    parameter_file: Path = typer.Option(..., exists=True),
    spectral_config: Path = typer.Option(..., exists=True),
    image_path: Path = typer.Option(..., exists=True),
    output_base: Path = typer.Option(...),

    # Stack topology
    layer_count: int = typer.Option(..., help="Number of layers"),
    layer_thickness: float = typer.Option(..., help="Thickness per layer (in mm)"),
    bottom_thickness: float = typer.Option(0.2, help="Thickness of monochrome bottom layer"),
    bottom_material: str = typer.Option("", help="Material of the bottom layer (default first material)"),
    top_thickness: float = typer.Option(0.0, help="Thickness of monochrome top layer"),
    top_material: str = typer.Option("", help="Material of the top layer (default first material)"),
    ordered: bool = typer.Option(False, help="Enforce ordered stacking (mainly for reflection)"),

    # Materials
    material_names: Optional[List[str]] = typer.Option(None, help="Subset of material names to use"),

    # Lithophane geometry
    resolution: Tuple[int, int] = typer.Option(..., help="Output resolution (width height)"),
    pixel_size: Tuple[float, float] = typer.Option(..., help="Pixel size in mm (x y)"),

    # Evaluator
    observer: str = typer.Option("CIE1931"),
    illuminator: str = typer.Option("D65"),
    view_geometry: str = typer.Option("transmission"),
    calc_backend: str = typer.Option("jax"),

    # Inverter
    inverter_algorithm: str = typer.Option("lut"),
    lut_compression_factor: Optional[int] = typer.Option(None),
):
    """
    Generate a color lithophane from an RGB image.
    """

    # -------------------------
    # Load parameter file
    # -------------------------
    materials, material_parameter, metadata = file_to_parameter(path=parameter_file, local_path=False, material_filter=material_names)

    available_names = [m["name"] for m in materials]
    if material_names is None:
        material_names = available_names
    material_count = len(material_names)
    if bottom_material in available_names:
        bottom_material_index = available_names.index(bottom_material)
    elif bottom_material == "":
        bottom_material_index = 0
    else:
        typer.secho(f"Invalid bottom material names: {bottom_material}. Materials found: {available_names}", fg=typer.colors.RED)
        raise typer.Exit(1)
    
    if top_material in available_names:
        top_material_index = available_names.index(top_material)
    elif top_material == "":
        top_material_index = 0
    else:
        typer.secho(f"Invalid bottom material names: {top_material}. Materials found: {available_names}", fg=typer.colors.RED)
        raise typer.Exit(1)
    
    if len(material_names) != len(available_names):
        invalid = [m for m in material_names if m not in available_names]
        if invalid:
            typer.secho(
                f"Invalid material names: {invalid}. "
                f"Materials found: {available_names}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)

    # -------------------------
    # Stack topology (MVP: single homogeneous block)
    # -------------------------
    stack_rules = generate_stack_rules_bottom_color_top_blocks(
        color_layer_thickness=layer_thickness,
        color_layer_count=layer_count,
        material_count=material_count,
        bottom_thickness=bottom_thickness,
        bottom_layer_material=bottom_material_index,
        top_thickness=top_thickness,
        top_layer_material=top_material_index,
        ordered=ordered,
    )
    stack_generator = StackGenerator(rules=stack_rules)

    # -------------------------
    # Spectral configuration
    # -------------------------
    if spectral_config is None or (not spectral_config.exists()):
        spectral_config = None #only CIE data
    light_sources, observers = file_to_spectral_helper(spectral_config, local_path=False)

    if illuminator not in light_sources.names:
        typer.secho(
            f"Illuminator '{illuminator}' not found. "
            f"Available: {light_sources.names}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    if observer not in observers.names:
        typer.secho(
            f"Observer '{observer}' not found. "
            f"Available: {observers.names}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    # -------------------------
    # Build evaluator + inverter
    # -------------------------
    evaluator_spec = EvaluatorSpec(
        observer=observer,
        illuminator=illuminator,
        view_geometry=view_geometry,
        calc_backend=calc_backend,
    )

    inverter_spec = InverterSpec(
        algorithm=inverter_algorithm,
        lut_compression_factor=lut_compression_factor,
    )

    inverter = parameter_to_inverter(
        material_parameter=material_parameter,
        illuminators=light_sources,
        observers=observers,
        stack_generator=stack_generator,
        evaluator_config=evaluator_spec,
        inverter_config=inverter_spec,
    )

    # -------------------------
    # Load image
    # -------------------------
    image = Image.open(image_path).convert("RGB")

    # -------------------------
    # Lithophane config
    # -------------------------
    litho_config = LithophaneConfig(
        resolution=resolution,
        pixel_xy_dimension=pixel_size,
        material_names=material_names,
    )

    # -------------------------
    # Run pipeline
    # -------------------------
    output_paths, _, _ = image_to_lithophane(
        image=image,
        output_base_path=output_base,
        inverter=inverter,
        stack_rules=stack_rules,
        config=litho_config,
    )

    typer.secho("Lithophane generation complete.", fg=typer.colors.GREEN)
    for p in output_paths:
        typer.echo(f"  -> {p}")
