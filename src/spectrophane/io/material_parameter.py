import jax.numpy as jnp
import numpy as np
import shutil
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dataclasses

from spectrophane.core.dataclasses import MaterialParams, SpectralBlock
from spectrophane.io.data_io import write_json_resource


def color_str(rgb: jnp.ndarray):
    """
    Returns a coloured square that can be printed to the terminal. Input is an array of shape (3,) in the intensity interval [0,1]
    """
    a= [0xFF, 0x0F, 0x00]
    return f"\033[48;2;{int(rgb[0]*255)};{int(rgb[1]*255)};{int(rgb[2]*255)}m \033[0m"

def print_color_comparison(calculated_colors: jnp.ndarray, actual_colors: jnp.ndarray):
    """Shows comparison of learned and measured colors"""
    learn_start = "learned: "
    learn = learn_start
    real_start = "measured:"
    real = real_start
    
    # Get terminal width to break line at the right time
    terminal_width, _ = shutil.get_terminal_size()
    
    # Iterate over the arrays and build the color strings
    for i in range(calculated_colors.shape[0]):
        learn += color_str(calculated_colors[i])
        real += color_str(actual_colors[i])

        if i % (terminal_width-len(learn_start)-1) == 0 and i != 0:
            print(learn)
            print(real)
            learn = learn_start
            real =  real_start
    if learn != learn_start:
        print(learn)
        print(real)


def extract_spectral_blocks(params, metadata, wavelengths):
    blocks = []

    for p, meta in zip(params, metadata):
        # Iterate through all fields of the params object
        for field_name, field_value in p.__dict__.items():
            if field_value is not None:
                blocks.append(
                    SpectralBlock(
                        wavelengths=wavelengths,
                        values=np.asarray(field_value),
                        material_id=meta["id"],
                        material_name=meta["name"],
                        plotcolor=meta["plotcolor"],
                        parameter=field_name,
                    )
                )

    return blocks

def plot_parameter(blocks: list[SpectralBlock], *, x_label: str = "Wavelength (nm)", y_label: str = "Value"):
    # Determine facet order
    parameters = sorted({b.parameter for b in blocks})
    
    fig = make_subplots(rows=len(parameters), cols=1, shared_xaxes=True, subplot_titles=parameters)
    row_index = {p: i + 1 for i, p in enumerate(parameters)}

    for b in blocks:
        fig.add_trace(
            go.Scatter(x=b.wavelengths, y=b.values, mode="lines", name=b.material_name, legendgroup=b.material_id, line=dict(color=f"#{b.plotcolor}")),
            row=row_index[b.parameter],
            col=1,
        )

    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label, template="plotly_white", legend_title="Material")

    return fig


def save_parameter(resource_path: str, material_data: list, parameter: MaterialParams, no_overwrite: bool = True):
    """Serializes trained parameter set to json and saves to file. Filepath is relative to [resources]/material_data/"""
    result = {}
    result["materials"] = material_data
    result["parameter"] = {}
    param_dict = result["parameter"]
    for field in dataclasses.fields(parameter):
        value = getattr(parameter, field.name)
        #arrays to list, otherwise use value directly
        if isinstance(value, jnp.ndarray):
            param_dict[field.name] = value.tolist()
        else:
            param_dict[field.name] = value
    #################TODO: add wavelength data!
    write_json_resource(resource_path, result, no_overwrite)
    

def load_parameter():
    pass