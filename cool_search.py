from timeit import default_timer
from typing import Literal

import numpy as np
import polars as pl

# idea: a convenient and smooth hyperparameter search


class search_one_dim:
    """Optimization of a single parameter, optimizing a single metric"""

    __slots__ = [
        "estimator",
        "fixed_params",
        "metric",
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

        return np.average((y_true - y_pred) ** 2)

    @staticmethod
    def MAE_score(y_true: np.ndarray, y_pred: np.ndarray):
        if y_true.shape != y_pred.shape:
            raise ValueError("Need arrays of same shape")

        return np.average(abs(y_true - y_pred))

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
        self.metric = self.METRICS[metric]
        self.samples = pl.DataFrame(schema={param_name: pl.Float64, metric: pl.Float64})

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

    def grid(self, steps=10, verbose=False):
        pp = np.linspace(
            self.param_range[0],
            self.param_range[1],
            steps,
        )

        new_samples = []
        for p in pp:
            if p in self.samples[self.param_name]:
                continue

            if verbose:
                print(f"{self.param_name} = {p}")
            model = self.estimator(**self.fixed_params, **{self.param_name: p})
            model.fit(self.X_train, self.Y_train)

            pred_val = model.predict(self.X_val).reshape(self.Y_val.shape)
            score = self.metric(self.Y_val, pred_val)
            new_samples.append((p, score))

        print(f"computed {len(new_samples)} new samples")

        self.samples.extend(pl.DataFrame(new_samples, self.samples.schema)).sort()

        return self.samples

    def binary_search():
        pass

    # idea: approx gradient search
    # idea: genetic search
    # idea: bayesian search
