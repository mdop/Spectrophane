import plotly.graph_objects as go
import numpy as np
from pathlib import Path

from spectrophane.pipeline.lithophane_pipeline import (
    file_to_parameter,
    file_to_spectral_helper,
    parameter_to_inverter
)
from spectrophane.core.dataclasses import (
    EvaluatorSpec,
    InverterSpec,
)
from spectrophane.pipeline.lithophane_factories import (
    generate_stack_rules_bottom_color_top_blocks,
)
from spectrophane.inverse.stack_generation import StackGenerator

def show_gamut(parameter_file: str,
               layer_thickness: float,
               layer_count: int,
               outer_layer_index: int = 0,
               lut_compression_factor = 4):
    materials, material_parameter, metadata = file_to_parameter(path=parameter_file, local_path=False)
    material_names = [m["name"] for m in materials]
    material_count = len(material_names)

    stack_rules = generate_stack_rules_bottom_color_top_blocks(
        color_layer_thickness=layer_thickness,
        color_layer_count=layer_count,
        material_count=material_count,
        bottom_thickness=0.2,
        bottom_layer_material=outer_layer_index,
        top_thickness=0.2,
        top_layer_material=outer_layer_index,
        ordered=False,
    )
    stack_generator = StackGenerator(rules=stack_rules)

    light_sources, observers = file_to_spectral_helper(None, local_path=False)

    evaluator_spec = EvaluatorSpec(
        observer="CIE1931",
        illuminator="D65",
        view_geometry="transmission",
        calc_backend="jax",
    )
    inverter_spec = InverterSpec(
        algorithm="lut",
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
    rgb_gamut = inverter._stacks.rgb

    rgb = np.asarray(rgb_gamut)

    colors = [
        f"rgb({int(r)},{int(g)},{int(b)})"
        for r, g, b in rgb
    ]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=rgb[:,0].tolist(),
                y=rgb[:,1].tolist(),
                z=rgb[:,2].tolist(),
                mode="markers",
                marker=dict(
                    size=2,
                    color=colors
                )
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="R",
            yaxis_title="G",
            zaxis_title="B"
        ),
        title="RGB Gamut in 3D"
    )

    fig.show()



if __name__ == "__main__":
    param_file = Path.home() / "Spectrophane" / "material_parameter" / "default.json"
    show_gamut(parameter_file=str(param_file),
               layer_thickness=0.03,
               layer_count=12)