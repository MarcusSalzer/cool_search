from timeit import default_timer
from typing import Literal

import numpy as np
import polars as pl

from sklearn.neighbors import KernelDensity
from sklearn.gaussian_process import GaussianProcessRegressor, kernels as gpk


# idea: a convenient and smooth hyperparameter search


class search_one_dim:
    """Optimization of a single parameter, optimizing a single metric"""

    __slots__ = [
        "estimator",
        "fixed_params",
        "metric",
        "metric_name",
        "param_name",
        "param_range",
        "samples",
        "X_train",
        "X_val",
        "Y_train",
        "Y_val",
    ]

    @staticmethod
    def MSE_score(y_true: np.ndarray, y_pred: np.ndarray):
        if y_true.shape != y_pred.shape:
            raise ValueError("Need arrays of same shape")

        return np.average((y_true - y_pred) ** 2).item()

    @staticmethod
    def MAE_score(y_true: np.ndarray, y_pred: np.ndarray):
        if y_true.shape != y_pred.shape:
            raise ValueError("Need arrays of same shape")

        return np.average(abs(y_true - y_pred)).item()

    @staticmethod
    def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray):
        if y_true.shape != y_pred.shape:
            raise ValueError("Need arrays of same shape")

        return np.average(y_true == y_pred)

    METRICS = {
        "MSE": MSE_score,
        "MAE": MAE_score,
        "accuracy": accuracy_score,
    }

    def __init__(
        self,
        estimator,
        param_name: str,
        param_range: tuple,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        metric: Literal["MSE", "MAE", "accuracy"] = "MSE",
        fixed_params={},
    ) -> None:
        if X_val is None or Y_val is None:
            raise NotImplementedError("requires validation data (for now).")

        self.estimator = estimator
        self.fixed_params = fixed_params
        self.param_name = param_name
        self.param_range = param_range
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.metric_name = metric
        self.metric = self.METRICS[metric]
        self.samples = pl.DataFrame(
            schema={param_name: pl.Float64, metric: pl.Float64, "runtime": pl.Float64}
        )

    def objective(self, param_value) -> float:
        """Fit estimator with `param_value` and evaluate and score using `self.metric`."""

        t_start = default_timer()

        model = self.estimator(**self.fixed_params, **{self.param_name: param_value})
        model.fit(self.X_train, self.Y_train)
        pred_val = model.predict(self.X_val).reshape(self.Y_val.shape)
        t_fitpredict = default_timer() - t_start

        score = self.metric(self.Y_val, pred_val)
        return score, t_fitpredict

    def __str__(self) -> str:
        lines = [
            "Cool search:",
            f"  - {self.estimator}",
            f"  - training data   {self.X_train.shape}, {self.Y_train.shape}",
            f"  - validation data {self.X_val.shape}, {self.Y_val.shape}",
            f"  - parameter       {self.param_name} {self.param_range}",
            f"  - samples         {len(self.samples)}",
        ]

        return "\n".join(lines)

    def sample_grid(self, resolution=10, verbose=False):
        pp = np.linspace(
            self.param_range[0],
            self.param_range[1],
            resolution,
        )

        new_samples = []
        for p in pp:
            if p in self.samples[self.param_name]:
                continue

            if verbose:
                print(f"{self.param_name} = {p}")

            score, time = self.objective(p)

            new_samples.append((p, score, time))

        print(f"computed {len(new_samples)} new samples")

        self.samples.extend(pl.DataFrame(new_samples, self.samples.schema))

        return self.samples

    def sample_point(self, param_value):
        if param_value in self.samples[self.param_name]:
            print(f"already sampled {param_value}")
            return self.samples.row(
                by_predicate=pl.col(self.param_name) == param_value
            )[1]
        else:
            score, time = self.objective(param_value)
            self.samples.extend(
                pl.DataFrame([(param_value, score, time)], self.samples.schema)
            )
            return score

    def sample_points_KDE(
        self,
        kernel="gaussian",
        bandwidth=0.1,
        resolution=100,
    ):
        X = self.samples.select(pl.col(self.param_name)).to_numpy()

        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(X)
        log_likelihood = kde.score_samples(
            np.linspace(*self.param_range, resolution).reshape(-1, 1)
        )
        return np.exp(log_likelihood)

    def estimate_score_GP(self, resolution=100):
        X = self.samples.select(pl.col(self.param_name)).to_numpy()
        Y = self.samples[self.metric_name].to_numpy()

        # Define kernel (example: Matern kernel + noise term)
        kernel = gpk.Matern(length_scale=1.0, nu=2.5) + gpk.WhiteKernel(
            noise_level=1e-1
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=10
        )
        gp.fit(X, Y)
        pred_mean, pred_std = gp.predict(
            np.linspace(*self.param_range, resolution).reshape(-1, 1), return_std=True
        )
        return pred_mean, pred_std

    def binary_search():
        pass

    # idea: approx gradient search
    # idea: genetic search
    # idea: bayesian search
