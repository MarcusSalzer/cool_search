import math
from typing import Literal
import numpy as np
import polars as pl

import coolsearch.utility_functions as util


class PolynomialModel:
    """Polynomial regression"""

    __slots__ = [
        "beta",
        "degree",
        "features",
        "param_range",
        "residuals",
        "X_poly",
        "y",
    ]

    def __init__(
        self,
        samples: pl.DataFrame,
        features: list[str] | None = None,
        degree: int = 1,
        param_range: dict | None = None,
        interaction: bool = True,
        target: str = "score",
    ) -> None:
        if not interaction:
            raise NotImplementedError("only with interactions now")

        self.degree = degree

        # initialize ranges if missing
        if param_range is None:
            param_range = dict.fromkeys(features)
            for k in features:
                param_range[k] = (
                    samples[k].min(),
                    samples[k].max(),
                )
        self.param_range = param_range

        if param_range:
            self.features = list(map(str, param_range.keys()))
        else:
            self.features = features

        self.y = samples[target].to_numpy()
        X = samples.select(self.features).to_numpy()
        self.X_poly = polynomial_features(X, self.degree, verbose=False)

        if self.y.shape[0] != self.X_poly.shape[0]:
            raise ValueError("Inconsistent sample count")

    def __str__(self) -> str:
        lines = [
            f"Polynomial model (degree: {self.degree})",
            f"{self.X_poly.shape[0]} samples",
        ]
        return "\n".join(lines)

    def fit(self, verbose=True):
        """Fit model by least squares."""
        X = self.X_poly
        y = self.y
        N, M = X.shape

        if verbose:
            print(f"{N} samples\n{M} poly features")

            if N < M:
                print("Note: under-determined system")

        beta, res, _, _ = np.linalg.lstsq(X, y, rcond=None)

        self.beta = beta
        self.residuals = res

        if verbose:
            print(f"coefficients: {beta}, residuals: {res}")

    @property
    def polynomial(self):
        if len(self.features) != 1:
            raise NotImplementedError

        return np.polynomial.Polynomial(self.beta)

    @property
    def yhat(self):
        """Prediction of training data"""

        return self.X_poly @ self.beta

    def predict(
        self,
        samples: pl.DataFrame,
    ):
        """Predict new samples."""
        X = samples.select(self.features).to_numpy()
        X_poly = polynomial_features(X, self.degree)

        y_pred = X_poly @ self.beta

        return samples.with_columns(y_pred=y_pred)

    def minimum_numeric(self, steps=10):
        """Evaluate polynomial on a (float type) grid, and find minimum"""

        high_res = util.get_grid(
            steps,
            self.param_range,
            dict.fromkeys(self.features, "float"),
        )
        # add predictions
        high_res = high_res.with_columns(value=self.predict(samples=high_res))

        minimum = high_res.filter(pl.col("value") == pl.col("value").min())
        return minimum, high_res

    def poly_marginals(self, steps=10):
        """TODO"""
        polyval = self.predict(
            util.get_grid(
                100,
                self.param_range,
                dict.fromkeys(self.features, "float"),
            )
        )

        marginals = {}
        for k in self.features:
            marginals[k] = (
                polyval.group_by(k)
                .agg(
                    pl.col("y_pred").mean().alias("mean"),
                    pl.col("y_pred").std().alias("std"),
                    pl.col("y_pred").median().alias("median"),
                )
                .sort(k)
            )

        return marginals


def polynomial_features(X: np.ndarray, d: int, verbose=True):
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
