import jax.numpy as jnp
import numpy as np
import shutil
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dataclasses
from numbers import Number
from typing import Sequence
from math import ceil

from spectrophane.core.dataclasses import MaterialParams, WavelengthAxis
from spectrophane.io.resources import write_json_resource, get_json_resource


@dataclasses.dataclass
class SpectrumPlotLineData:
    wavelengths: np.ndarray
    values: np.ndarray
    material_id: Number
    material_name: str
    plotcolor: str #HTML hex color
    parameter: str #What the series description should be


def color_str(rgb: jnp.ndarray):
    """
    Returns a coloured square that can be printed to the terminal. Input is an array of shape (3,) in the intensity interval [0,1]
    """
    a= [0xFF, 0x0F, 0x00]
    return f"\033[48;2;{int(rgb[0]*255)};{int(rgb[1]*255)};{int(rgb[2]*255)}m \033[0m"

def terminal_color_comparison_string(calculated_colors: jnp.ndarray | np.ndarray, actual_colors: jnp.ndarray | np.ndarray):
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


def extract_spectral_plot_series(params:MaterialParams, metadata: list[dict[str]], field_filter: list[str] = []) -> list[SpectrumPlotLineData]:
    """Parses material parameter and their metadata to create a list of plot data of all fields. If field filter is empty all fields are accepted, otherwise only field names contained in the list are appended."""
    if len(metadata) != params.absorption_coeff.shape[0]:
        raise ValueError("Cannot zip metadata and material parameter, they have different lengths!")
    blocks = []
    wavelength_axis = WavelengthAxis(start=params.wl_start, step=params.wl_step, length=params.absorption_coeff.shape[1])
    wavelengths = wavelength_axis.wavelengths

    for i in range(len(metadata)):
        for field_name, field_value in params.__dict__.items():
            if (len(field_filter) == 0 or field_name in field_filter) and isinstance(field_value, (np.ndarray, jnp.ndarray)):
                meta = metadata[i]
                blocks.append(SpectrumPlotLineData( wavelengths=wavelengths,
                                                    values=np.asarray(field_value[i]),
                                                    material_id=meta["id"],
                                                    material_name=meta["name"],
                                                    plotcolor=meta["plotcolor"],
                                                    parameter=field_name))
    return blocks

def plot_parameter(series: list[SpectrumPlotLineData], *, rows: int = 0, x_label: str = "Wavelength (nm)", y_label: str = "Value") -> go.Figure:
    """Generates a plotly plot with subplots for each parameter"""
    # Determine facet order
    parameters = sorted({b.parameter for b in series})
    
    if rows <= 0:
        rows = len(parameters)
    cols = ceil(len(parameters) / rows)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, subplot_titles=parameters)
    row_index = {p: i + 1 for i, p in enumerate(parameters)}

    for b in series:
        fig.add_trace(
            go.Scatter(x=b.wavelengths, y=b.values, mode="lines", name=b.material_name, legendgroup=b.material_id, line=dict(color=f"#{b.plotcolor}")),
            row=row_index[b.parameter],
            col=1,
        )

    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label, template="plotly_white", legend_title="Material")

    return fig

def plot_loss_series(losses: Sequence[float]):
    """Takes loss series of training and returns a scatter plot of training step vs loss."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
                             x=list(range(len(losses))),
                             y=losses,
                             mode='lines+markers',
                             name='Loss',
                             line=dict(color='blue'),
                             marker=dict(size=6)
                            ))

    fig.update_layout(
                      title='Training Loss Series',
                      xaxis_title='Epoch',
                      yaxis_title='Loss',
                      showlegend=True
                     )

    return fig

def serialize_parameter(material_data: list, parameter: MaterialParams) -> dict:
    """Serializes trained parameter set to json and saves to file. Filepath is relative to [resources]/material_data/"""
    param_dict = {}
    param_dict["materials"] = material_data
    param_dict["wl_start"] = parameter.wl_start
    param_dict["wl_step"] = parameter.wl_step
    param_dict["parameter"] = {}
    for field in dataclasses.fields(parameter):
        value = getattr(parameter, field.name)
        #arrays to list, otherwise use value directly
        if isinstance(value, (jnp.ndarray, np.ndarray)):
            param_dict["parameter"][field.name] = value.tolist()
        else:
            param_dict["parameter"][field.name] = value
    return param_dict
    

def deserialize_parameter(param_dict: dict) -> MaterialParams:
    """Deserializes saved training parameter data."""
    result = MaterialParams(wl_start=param_dict["parameter"]["wl_start"],
                            wl_step=param_dict["parameter"]["wl_step"])
    for field in dataclasses.fields(result):
        if field.name not in param_dict["parameter"]:
            setattr(result, field.name, None)
            continue

        value = param_dict["parameter"][field.name]

        deserializer = field.metadata.get("deserialize")
        if deserializer is not None and value is not None:
            value = deserializer(value)

        setattr(result, field.name, value)
    return result