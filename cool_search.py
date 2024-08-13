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
        "_objective",
        "dtype",
        "_fixed_params",
        "_ndim",
        "_param_range",
        "_param_types",
        "samples",
    ]

    def __init__(
        self,
        objective,
        param_range: dict,
        param_types: dict[str, Literal["int", "float"]] | None = None,
        # min_delta? max_steps?
        fixed_params={},
    ) -> None:
        """Tools for minimizing a function.

        ## parameters
        - objective (callable):
        - param_range (dict): ranges for all parameters
        - param_types (dict): specify int or float types. (Defaults to float for all parameters)
        - fixed_params (dict): fixed kwargs provided to `objective`.

        """

        # set constants
        self._objective = objective
        self._param_range = param_range
        self._fixed_params = fixed_params

        # set parameter types, default to float
        if param_types is None:
            param_types = dict.fromkeys(param_range.keys(), "float")

        # Validate the parameter types
        for param, p_type in param_types.items():
            if p_type not in {"int", "float"}:
                raise ValueError(
                    f"Unsupported parameter type: '{p_type}' for parameter '{param}'."
                )

        self._param_types = param_types
        self._ndim = len(self._param_range)

        # schema for parameters, score and runtime
        schema = {
            param: (pl.Float32 if dtype == "float" else pl.Int32)
            for param, dtype in param_types.items()
        }
        schema["score"] = pl.Float64
        schema["runtime"] = pl.Float64

        # init empty samples-frame
        self.samples = pl.DataFrame(schema=schema)

    def __str__(self):
        return "\n".join(
            [
                f"{self._ndim} dimensional search",
                f"  - has {len(self.samples)} samples",
            ],
        )

    @property
    def param_names(self):
        return list(self._param_range.keys())

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
        - steps (int): Number of steps/points per parameter
            - Note: integer-parameters might get fewer steps to avoid duplicates.

        ## Returns
        - grid (DataFrame): points in parameter space.
            - each column has the apropriate data type.
        """

        grid_points = []
        for param, r in self._param_range.items():
            param_type = self._param_types[param]
            if param_type == "int":
                # Generate evenly spaced integer points within the range
                grid = np.linspace(r[0], r[1], steps, dtype=int)
                grid_points.append(np.unique(grid))  # Ensure uniqueness
            elif param_type == "float":
                # Generate evenly spaced floating points within the range
                grid_points.append(np.linspace(r[0], r[1], steps, dtype=float))

        # Create the grid by meshing and flattening the arrays
        mesh = np.meshgrid(*grid_points, indexing="ij")
        grid = [
            dict(zip(self._param_range.keys(), point))
            for point in zip(*(np.ravel(m) for m in mesh))
        ]

        return pl.DataFrame(
            grid,
            schema=self.samples.select(self.param_names).schema,
            orient="row",
        )

    def make_factor_grid(self, steps):
        mesh = np.meshgrid(*[np.linspace(0, 1, steps)] * self._ndim)
        return np.vstack(list(map(np.ravel, mesh))).T

    def grid_search(
        self,
        steps=10,
        # TODO: Specify dimensions to grid, and strategy for others?
        target_runtime=None,
        verbose: Literal[0, 1, 2] = 1,
        print_eta: bool = True,
    ):
        """Sample objective on an evenly spaced grid."""

        if target_runtime:
            if self.samples.is_empty():
                if verbose >= 1:
                    print("No previous samples. Running 1 initial evaluation")
                self.random_search(1)
                # TODO: Replace with coarse grid (or 1 point central in grid)?

            # choose steps to approximately run for ´target_runtime´ seconds
            n_samples = target_runtime / self.samples["runtime"].mean()
            steps = min(int(round(n_samples ** (1 / self._ndim))), 1)
            if verbose >= 1:
                print(
                    "\n".join(
                        [
                            f"choose {steps} steps",
                            f"  -> maximum {steps**self._ndim} samples",
                        ]
                    )
                )

        t_start = default_timer()

        param_names = self.param_names

        grid = self.get_grid(steps)

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
            scores_new.append(self._objective(**row, **self._fixed_params))
            runtimes_new.append(default_timer() - t_run)
            # TODO: update time remaining estimate

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
        """Sample objective at N randomly chosen points."""
        rng = np.random.default_rng(seed)

        samples = rng.uniform(0, 1, (N, self._ndim))  # TODO: fix proper random samples.
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
