"""
Some things we learned.
"""

import math

import numpy as np
import sympy as sp
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

    combinations = np.stack([x.ravel() for x in np.meshgrid(*[range(d + 1)] * N)]).T

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


def polynomial_features(X, d, verbose=True):
    """Create polynomial features of terms up to a degree (including constant).
    ## parameters
    - X (ndarray): original feature matrix, shape (N, n_feat)
    - d (int): maximum degree
    ## returns
    - X_new (ndarray): extended feature matrix, shape (N, comb(n_feat+d, d))

    Note: output feature order is consistent, but not lexicographic/intuitive.
    """

    N, n_feat = X.shape
    n_feat_new = math.comb(n_feat + d, d)
    X_new = np.empty((N, n_feat_new), X.dtype)
    X_new[:, 0] = 1  # constant term

    if verbose:
        print(f"{n_feat} -> {n_feat_new} features:")

    combinations = np.stack(
        [x.ravel() for x in np.meshgrid(*[range(d + 1)] * n_feat)]
    ).T

    column = 1
    for pows in combinations:
        if 1 <= sum(pows) <= d:
            f_new = (X**pows).prod(axis=1)
            X_new[:, column] = f_new
            column += 1

    return X_new


def test_function_01(x, delay=0):
    """1D curve, with a clear minimum"""
    if delay > 0:
        sleep(delay)

    return np.sqrt((x - 2) ** 2 + 1) + np.sin(x / 2)
