import typer
from pathlib import Path
from typing import List, Optional
import plotly.graph_objects as go

from spectrophane.pipeline.training_pipeline import parameter_training_pipeline
from spectrophane.core.dataclasses import TrainingConfig


def training_command(
    calibration_file: str = typer.Option(..., exists=True),
    output_path: Optional[Path] = typer.Option(None),

    model: str = typer.Option("kubelka_munk"),
    observer: str = typer.Option("CIE1931"),
    training_steps: int = typer.Option(1000),
    lr: float = typer.Option(1e-1),

    parameter_plot_filter: Optional[List[str]] = typer.Option(None),
    parameter_plot_rows: int = typer.Option(1),
    show_color_comparison: bool = typer.Option(False),
):
    """
    Train material parameters from calibration data.
    """

    config = TrainingConfig(
        model=model,
        observer=observer,
        steps=training_steps,
        lr=lr,
        parameter_plot_filter=parameter_plot_filter,
        parameter_plot_rows=parameter_plot_rows,
        get_terminal_color_comparison=show_color_comparison,
    )

    try:
        output = parameter_training_pipeline(
            calibration_filepath=str(calibration_file),
            output_path=str(output_path) if output_path else None,
            calibration_local=True,
            output_local=False,
            config=config,
        )
    except Exception as e:
        typer.secho(f"Training failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho("Training completed successfully.", fg=typer.colors.GREEN)

    if "terminal_color_str" in output:
        typer.echo(output["terminal_color_str"])
    if "parameter_plot" in output:
        output["parameter_plot"].show()
    if "loss_plot" in output:
        output["loss_plot"].show()
