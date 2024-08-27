# %%
from importlib import reload
from time import sleep

import numpy as np
import polars as pl
from plotly import express as px, graph_objects as go
import sys

sys.path.append("src")

import src.cool_search as cool
import src.cool_search_models as cm
import src.utility_functions as util

util.set_plotly_template()


# %%
def f(x):
    """1D curve, with a clear minimum
    - slightly slow
    """
    sleep(0.1)
    return np.sqrt((x - 2) ** 2 + 1) + np.sin(x / 2)


X = np.linspace(-10, 10, 300)
Y = f(X)
fig = px.line(x=X, y=Y)
fig.show()

# %%

lowres = pl.DataFrame({"x": np.linspace(-10, 10, 5)})
lowres = lowres.with_columns(
    pl.col("x").map_elements(f, return_dtype=pl.Float64).alias("f"),
)

display(lowres)  # type: ignore # noqa: F821


model = cm.PolynomialModel(
    samples=lowres,
    features=["x"],
    degree=1,
    target="f",
)
model.fit()

lowres = lowres.with_columns(f_pred=model.predict())

# %%
fig = go.Figure()
fig.add_traces(
    go.Scatter(x=lowres["x"], y=lowres["f"], mode="markers"),
)
