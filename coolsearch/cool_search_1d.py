from timeit import default_timer
from typing import Literal

import numpy as np
import polars as pl
from plotly import graph_objects as go
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as gpk
from sklearn.neighbors import KernelDensity


class search_one_dim:
    """Optimization of a single parameter, optimizing a single metric.

    The idea is that the function is somewhat smooth?"""

    __slots__ = [
        "estimator",
        "fixed_params",
        "metric",
        "metric_name",
        "param_name",
        "param_range",
        "samples",
        "score_gp",
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
            schema={param_name: pl.Float64,
                    metric: pl.Float64, "runtime": pl.Float64}
        )

    def objective(self, param_value) -> float:
        """Fit estimator with `param_value` and evaluate and score using `self.metric`."""

        t_start = default_timer()

        model = self.estimator(**self.fixed_params, **
                               {self.param_name: param_value})
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

    def sample_grid(
        self,
        resolution=10,
        verbose=False,
        update=True,
    ):
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

        new_df = pl.DataFrame(new_samples, self.samples.schema)
        if update:
            self.samples.extend(new_df)

        return new_df

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

    def fit_score_GP(self, predict_resolution: int | None = None):
        X = self.samples.select(pl.col(self.param_name)).to_numpy()
        Y = self.samples[self.metric_name].to_numpy()

        kernel = gpk.Matern(length_scale=1.0, nu=2.5)
        self.score_gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=10
        )
        self.score_gp.fit(X, Y)

        if predict_resolution:
            return self.score_gp.predict(
                np.linspace(*self.param_range,
                            predict_resolution).reshape(-1, 1),
                return_std=True,
            )

    def sample_next_GP(
        self,
        explore_ratio=0.5,
        eval_resolution=100,
        verbose=True,
        confint=0.95,
    ):
        """Pick and sample one point based on the score GP estimate."""

        # if len(self.samples) < 5:
        #     raise ValueError("Needs more initial samples")

        t_start = default_timer()

        self.fit_score_GP()
        if verbose:
            print(f"Fitted GP ({default_timer()-t_start:.4f} seconds)")

        xx = np.linspace(*self.param_range, eval_resolution).reshape(-1, 1)

        pred_mean, pred_std = self.score_gp.predict(
            xx,
            return_std=True,
        )
        z = stats.norm.ppf(1 - (1 - confint) / 2)

        LCI = pred_mean - z * pred_std

        criterion = (1 - explore_ratio) * (LCI / LCI.max())
        criterion -= explore_ratio * (pred_std / pred_std.max())

        next_val = xx[np.argmin(criterion)][0]

        print(f"sampling {next_val}")

        t_start = default_timer()
        self.sample_point(next_val)

        if verbose:
            print(f"sampled point ({default_timer()-t_start:.4f} seconds)")
        return criterion

    def plot_score_GP(
        self,
        confidence=0.95,
        resolution=100,
        high_res_reference: int | None = None,
    ):
        try:
            gp = self.score_gp
        except AttributeError:
            raise AttributeError("fit GP first")

        pred_mean, pred_std = gp.predict(
            np.linspace(*self.param_range, resolution).reshape(-1, 1), return_std=True
        )

        samples_low = self.samples.clone()
        if high_res_reference:
            samples_high = self.sample_grid(high_res_reference, update=False)

        z = stats.norm.ppf(1 - (1 - confidence) / 2)

        fig = go.Figure().add_traces(
            [
                go.Scatter(
                    x=np.linspace(*self.param_range, resolution),
                    y=pred_mean - z * pred_std,
                    name="lower CI",
                    line_color="orange",
                ),
                go.Scatter(
                    x=np.linspace(*self.param_range, resolution),
                    y=pred_mean + z * pred_std,
                    name="upper CI",
                    fill="tonexty",
                    line_color="orange",
                ),
                go.Scatter(
                    x=samples_low[self.param_name],
                    y=samples_low[self.metric_name],
                    name=f"{len(samples_low)} samples",
                    mode="markers",
                    marker_size=8,
                ),
            ]
        )
        if high_res_reference:
            fig.add_trace(
                go.Scatter(
                    x=samples_high[self.param_name],
                    y=samples_high[self.metric_name],
                    name=f"{len(samples_high)} samples",
                    mode="lines",
                    line_dash="dash",
                ),
            )
        fig.update_xaxes(title="parameter value")
        fig.update_yaxes(title=self.metric_name)
        fig.update_layout(title="Gaussian process model")
        return fig
