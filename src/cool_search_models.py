import numpy as np
import polars as pl

from utility_functions import polynomial_features


class PolynomialModel:
    """Polynomial regression"""

    def __init__(
        self,
        samples: pl.DataFrame,
        features: list[str],
        degree: int = 1,
        interaction: bool = True,
        target: str = "score",
    ) -> None:
        if not interaction:
            raise NotImplementedError("only with interactions now")

        self.degree = degree
        self.features = features

        self.y = samples[target].to_numpy()
        X = samples.select(features).to_numpy()
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

        self.beta, self.residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)

        if verbose:
            print(f"coefficients: {self.beta}")

    @property
    def polynomial(self):
        if len(self.features) != 1:
            raise NotImplementedError

        return np.polynomial.Polynomial(self.beta)

    def predict(
        self,
        samples: pl.DataFrame | None = None,
    ):
        """Predict new samples, or samples used for fit if omitted."""
        if samples is None:
            X_poly = self.X_poly
        else:
            X = samples.select(self.features).to_numpy()
            X_poly = polynomial_features(X, self.degree)

        y_pred = X_poly @ self.beta

        return y_pred
