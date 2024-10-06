"""Plotting functions that might be useful. Based on plotly"""

from plotly import io as pio


def set_plotly_template():
    plot_temp = pio.templates["plotly_dark"]
    plot_temp.layout.width = 400
    plot_temp.layout.height = 300
    plot_temp.layout.autosize = False
    pio.templates.default = plot_temp
