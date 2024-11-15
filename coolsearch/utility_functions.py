"""
Utilities

- grid
- monomial
- a test function
- a feature scaler
"""

import math
from time import sleep
from typing import Literal

import numpy as np
import polars as pl

from coolsearch.Param import Param


def get_grid(
    steps: int | dict[str, int],
    param_range: dict[str, tuple],
    param_types: dict[str, Literal["float", "int"]],
) -> pl.DataFrame:
    """Get a grid for parameters.

    ## Parameters
    - steps (int | dict[str, int]): Number of steps/points per parameter
        - Note: integer-parameters might get fewer steps to avoid duplicates.

    ## Returns
    - grid (DataFrame): points in parameter space.
    """

    if isinstance(steps, int):
        steps = dict.fromkeys(param_range.keys(), steps)

    # validate step counts

    for param, s in steps.items():
        if s < 2:
            print(f"WARNING: {s} < 2 steps for {param}, gives only minimum value")

    grid_points = []
    for param, r in param_range.items():
        param_type = param_types[param]
        if param_type == "int":
            grid = np.unique(np.linspace(r[0], r[1], steps[param], dtype=int))
        elif param_type == "float":
            grid = np.linspace(r[0], r[1], steps[param], dtype=float)
        else:
            raise ValueError(f"Unsupported parameter type ({param_type})")

        grid_points.append(grid)

    # Create the grid by meshing and flattening the arrays
    mesh = np.meshgrid(*grid_points, indexing="ij")
    grid = [
        dict(zip(param_range.keys(), point))
        for point in zip(*(np.ravel(m) for m in mesh))
    ]

    return pl.DataFrame(
        grid,
        schema=param_range.keys(),
        orient="row",
    )


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

    return monomials


def test_function_01(x: float, delay: float = 0) -> float:
    """1D curve, with a clear minimum"""
    if delay > 0:
        sleep(delay)

    return np.sqrt((x - 2) ** 2 + 1) + np.sin(x / 2)


def rastrigin_slow(x, y, iterations=10**7) -> float:
    """A function with the minimum at (0,0).

    Note: slow because of actual computations.

    ## parameters
    - x, y (float): for evaluating function
    - iterations (int): Create slowdown.
        - 10**7 gives ca 3.7 seconds

    """

    # Artificial slowness loop
    total = 0
    for _ in range(iterations):
        total += (x**2 - 10 * math.cos(2 * math.pi * x)) + (
            y**2 - 10 * math.cos(2 * math.pi * y)
        )
        total -= math.sqrt(x**2 + y**2)

        # some useless trigonometry
        total = math.sin(total) ** 2 + math.cos(total) ** 2

    # The actual Rastrigin calculation, which is the target function
    result = (
        20 + x**2 + y**2 - 10 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y))
    )

    return result


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


def make_mesh(params: dict[str, Param] | list[Param], steps: int | dict[str, int] = 3):
    # standardixe argument types
    if isinstance(params, list):
        params = {p.name: p for p in params}

    if isinstance(steps, int):
        steps = dict.fromkeys(params.keys(), steps)

    if steps.keys() != params.keys():
        raise ValueError("inconsistent parameter names")

    # Create the grid by meshing and flattening the arrays
    arrays = [p.grid(steps[k]) for k, p in params.items()]
    schema = {k: p.pl_dtype for k, p in params.items()}

    mesh = np.meshgrid(*arrays, indexing="ij")

    points = zip(*(np.ravel(m) for m in mesh))
    param_points = [dict(zip(schema.keys(), point)) for point in points]

    return pl.DataFrame(
        param_points,
        schema=schema,
        orient="row",
    )
