from timeit import default_timer
from typing import Literal

import numpy as np
import polars as pl
from plotly import graph_objects as go
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as gpk
from sklearn.neighbors import KernelDensity


class CoolSearch:
    """Minimization of a black box function."""

    __slots__ = [
        "objective",
        "dtype",
        "fixed_params",
        "ndim",
        "param_range",
        "param_types",
        "samples",
    ]

    def __init__(
        self,
        objective,
        param_range: dict,
        param_types: dict[str, str] | None = None,
        fixed_params={},
        dtype=pl.Float64,
    ) -> None:
        """Tools for minimizing a function.

        ## parameters
        - objective (callable):

        """

        # set constants
        self.objective = objective
        self.param_range = param_range
        self.fixed_params = fixed_params

        if param_types is None:
            param_types = dict.fromkeys(param_range.keys(), dtype)
        self.param_types = param_types

        self.ndim = len(self.param_range)
        self.dtype = dtype

        schema = {param: self.dtype for param in param_range.keys()}
        schema["score"] = self.dtype
        schema["runtime"] = self.dtype

        self.samples = pl.DataFrame(schema=schema)

    def __str__(self):
        return "\n".join(
            [
                f"{self.ndim} dimensional search",
                f"  - {len(self.samples)} samples",
            ],
        )

    @property
    def param_names(self):
        return list(self.param_range.keys())

    # def get_param_values(self, factors: pl.DataFrame | np.ndarray):
    #     """Compute actual parameter values from factor in [0,1]."""

    #     try:
    #         names = factors.columns
    #     except AttributeError:
    #         names = self.param_names

    #     factors = factors.reshape(-1, self.ndim)
    #     N = factors.shape[0]

    #     pr = self.param_range

    #     scale = np.vstack((np.array([pr[k][1] - pr[k][0] for k in names]).T,) * N)
    #     offset = np.vstack((np.array([pr[k][0] for k in names]).T,) * N)

    #     return pl.DataFrame(
    #         factors * scale + offset, schema={k: self.param_types[k] for k in names}
    #     )

    def get_grid(self, steps):
        """Get an evenly spaced grid for all parameters.

        ## Parameters
        - steps (int): TODO

        ## Returns
        - grid (ndarray): array of points in parameter space
        """

        # TODO: handle datatypes

        mesh = np.meshgrid(*[np.linspace(*r, steps) for r in self.param_range.values()])
        return np.vstack(list(map(np.ravel, mesh))).T

    def make_factor_grid(self, steps):
        mesh = np.meshgrid(*[np.linspace(0, 1, steps)] * self.ndim)
        return np.vstack(list(map(np.ravel, mesh))).T

    def grid_search(
        self,
        steps=10,
        target_runtime=None,
        verbose: Literal[0, 1, 2] = 1,
    ):
        """Evaluate objective on an evenly spaced grid"""

        if target_runtime:
            if self.samples.is_empty():
                if verbose >= 1:
                    print("No previous samples. Running 1 initial evaluation")
                self.random_search(1)

            # choose steps to approximately run for ´target_runtime´ seconds
            n_samples = target_runtime / self.samples["runtime"].mean()
            steps = min(int(round(n_samples ** (1 / self.ndim))), 1)
            if verbose >= 1:
                print(
                    "\n".join(
                        [
                            f"choose {steps} steps",
                            f"  -> maximum {steps**self.ndim} samples",
                        ]
                    )
                )

        t_start = default_timer()

        param_names = self.param_names

        # TODO fixa det här
        grid = pl.DataFrame(self.get_grid(steps), schema=param_names)

        # avoid previously sampled points
        grid_new = grid.join(
            self.samples.select(param_names),
            on=param_names,
            how="anti",
        )

        if verbose >= 1:
            print(f"searching {len(grid_new)} new parameter points")
            if not self.samples.is_empty():
                est_runtime = len(grid_new) * self.samples["runtime"].mean()
                print(f"estimated runtime: {est_runtime:.4f} s.")

        # grid_new = grid_new.with_columns(
        #     pl.struct(param_names)
        #     .map_elements(lambda row: self.objective(**row), return_dtype=self.dtype)
        #     .alias("score")
        # )
        scores_new = []
        runtimes_new = []
        for row in grid_new.iter_rows(named=True):
            t_run = default_timer()
            scores_new.append(self.objective(**row, **self.fixed_params))
            runtimes_new.append(default_timer() - t_run)

        grid_new = grid_new.with_columns(
            score=pl.Series(scores_new),
            runtime=pl.Series(runtimes_new),
        )
        self.samples = pl.concat([self.samples, grid_new])

        runtime_sum = sum(runtimes_new)
        t_overhead = default_timer() - t_start - runtime_sum
        if verbose >= 1:
            print(f"total runtime: {runtime_sum:.4f} s + overhead: {t_overhead:.4f} s.")

        return grid_new

    def random_search(self, N, seed=None, verbose: Literal[0, 1, 2] = 1):
        """Sample N random points."""
        rng = np.random.default_rng(seed)

        samples = rng.uniform(0, 1, (N, self.ndim))
        if verbose >= 1:
            # print(f"searching {len(grid_new)} new parameter points")
            if not self.samples.is_empty():
                pass
                # est_runtime = len(grid_new) * self.samples["runtime"].mean()
                # print(f"estimated runtime: {est_runtime:.4f} s.")

    def run_sequence():
        """Specify a 'macro-program' to run multiple searches."""
        # idea. run sequentially, /and parallel?
        # allow for conditions to choose next step
