"""Plotting functions that might be useful. Based on plotly"""

from typing import Literal

from plotly import graph_objects as go
from plotly import io as pio
from plotly import subplots
from polars import DataFrame


def set_plotly_template():
    plot_temp = pio.templates["plotly_dark"]
    plot_temp.layout.width = 400
    plot_temp.layout.height = 300
    plot_temp.layout.autosize = False
    pio.templates.default = plot_temp


def marginal_plots(
    marginals: dict[str, DataFrame],
    plot_agg: list[Literal["mean", "median", "min", "max"]] = ["min"],
    n_cols: int = 3,
) -> list[go.Figure]:
    """Plot all marginal distributions"""

    stat_cols = ["mean", "std", "median", "min", "max"]

    n = len(marginals)

    n_cols = min(n_cols, n)
    n_rows = (n - 1) // n_cols + 1

    fig = subplots.make_subplots(
        n_rows,
        n_cols,
        subplot_titles=list(marginals.keys()),
    )

    return fig
