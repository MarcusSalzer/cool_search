"""
Some things we learned.
"""

import math
from typing import Literal

import numpy as np
import sympy as sp
import polars as pl
from plotly import io as pio
from time import sleep


def set_plotly_template():
    plot_temp = pio.templates["plotly_dark"]
    plot_temp.layout.width = 400
    plot_temp.layout.height = 300
    plot_temp.layout.autosize = False
    pio.templates.default = plot_temp


def monomials(N, d):
    """Create sympy-symbols of terms up to a degree.
    ## parameters
    - N (int): number of variables
    - d (int): degree
    """
    print(f"should be {math.comb(N + d, d)} terms:")

    combinations = np.stack([x.ravel()
                            for x in np.meshgrid(*[range(d + 1)] * N)]).T

    assert N <= 4, "can only visualize up to 4 vars"

    vars = "xyzw"
    monomials = ["1"]

    for pows in combinations:
        terms = []
        if 1 <= sum(pows) <= d:
            for v, p in zip(vars, pows):
                if p == 1:
                    terms.append(v)
                elif p > 1:
                    terms.append(v + "^{" + str(p) + "}")
            monomials.append(" ".join(terms))

    assert len(monomials) == math.comb(N + d, d), "Total count"

    return [sp.Symbol(p) for p in monomials]


def test_function_01(x, delay=0):
    """1D curve, with a clear minimum"""
    if delay > 0:
        sleep(delay)

    return np.sqrt((x - 2) ** 2 + 1) + np.sin(x / 2)


class Scaler:
    def __init__(self, axis=0) -> None:
        self.axis = axis

    def standardize(self, X: np.ndarray | pl.DataFrame):
        self.mu = X.mean(axis=self.axis, keepdims=True)
        self.std = X.mean(axis=self.axis, keepdims=True)

        X_standard = (X - self.mu) / self.std

        return X_standard

    def transform(self):
        pass

    def inverse(self, X_standard: np.ndarray | pl.DataFrame):
        return X_standard * self.std + self.mu
