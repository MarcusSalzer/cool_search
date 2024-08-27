from timeit import default_timer
from typing import Literal

import numpy as np
import polars as pl


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
        # Validate dict keys
        if set(param_range.keys()) != set(param_types.keys()):
            raise ValueError("Inconsistent parameter names")

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

    @classmethod
    def from_classifier():
        pass  # ??

    def get_grid(
        self,
        steps,
        distribution: Literal["even", "random"] = "even",  # deprecated?
        seed=None,
    ):
        """Get a grid for all parameters.

        ## Parameters
        - steps (int): Number of steps/points per parameter
            - Note: integer-parameters might get fewer steps to avoid duplicates.

        ## Returns
        - grid (DataFrame): points in parameter space.
        """

        # TODO THIS ISNT RANDOM IN A SENSE!

        if distribution == "random":
            rng = np.random.default_rng(seed)

        grid_points = []
        for param, r in self._param_range.items():
            param_type = self._param_types[param]
            if param_type == "int":
                if distribution == "even":
                    grid = np.unique(np.linspace(r[0], r[1], steps, dtype=int))
                elif distribution == "random":
                    grid = np.unique(rng.integers(r[0], r[1], steps))
            elif param_type == "float":
                if distribution == "even":
                    grid = np.linspace(r[0], r[1], steps, dtype=float)
                elif distribution == "random":
                    grid = rng.uniform(r[0], r[1], steps)
            else:
                raise ValueError(f"Unsupported parameter type ({param_type})")

            grid_points.append(grid)

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

    def get_random_samples(
        self,
        N,
        seed=None,
    ):
        rng = np.random.default_rng(seed)

        grid = []
        for _ in range(N):
            param_values = []
            for param, r in self._param_range.items():
                param_type = self._param_types[param]

                if param_type == "int":
                    param_values.append(rng.integers(r[0], r[1]))
                elif param_type == "float":
                    param_values.append(rng.uniform(r[0], r[1]))

            grid.append(param_values)

        # TODO: only keep unique

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
        steps=10,  # TODO: allow dict of steps per param!
        # TODO: Specify dimensions to grid, and strategy for others? (include in above dict)
        target_runtime=None,
        verbose: Literal[0, 1, 2] = 2,
        etr_update_step: int = 1,
    ):
        """Sample objective on an evenly spaced grid.

        ## Parameters
        - steps (TODO)
        - target_runtime (float): target time (seconds) to estimate number of steps.
        - verbose (int): amount of status information printed
            - 0: no prints
            - 1: setup, summary
            - 2: setup, summary, progress.
        - etr_update_step (int): if `verbose>=2`, how often to print updates.


        ## Returns
        - grid_new (DataFrame): new sampled points

        Note: to get all sampled points, use `self.samples`.

        """

        if target_runtime:
            if self.samples.is_empty():
                if verbose >= 1:
                    print("No previous samples. Running 1 initial evaluation")
                self.grid_search(steps=1, verbose=0)
            # choose steps to approximately run for ´target_runtime´ seconds
            n_samples = target_runtime / self.samples["runtime"].mean()
            steps = max(int(round(n_samples ** (1 / self._ndim))), 1)
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
            print(f"Searching {len(grid_new)} new parameter points")
            if not self.samples.is_empty():
                est_runtime = len(grid_new) * self.samples["runtime"].mean()
                print(f"Estimated runtime: {est_runtime:.4f} s.")

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

            if verbose >= 2 and ((len(runtimes_new) - 1) % etr_update_step == 0):
                # est time remaining based on old and new samples
                total_runtime = self.samples["runtime"].sum() + sum(runtimes_new)
                mean_runtime = total_runtime / (len(self.samples) + len(runtimes_new))
                etr = (len(grid_new) - len(runtimes_new)) * mean_runtime
                print(f"Estimated time remaining: {etr:.1f}...", end="\r")

        grid_new = grid_new.with_columns(
            score=pl.Series(scores_new),
            runtime=pl.Series(runtimes_new),
        )
        self.samples = pl.concat([self.samples, grid_new])

        runtime_sum = sum(runtimes_new)
        t_overhead = default_timer() - t_start - runtime_sum
        if verbose >= 1:
            print(f"Total runtime: {runtime_sum:.4f} s + overhead: {t_overhead:.4f} s.")

        return grid_new

    def random_search(
        self,
        N,
        target_runtime=None,  # TODO make a function for est this
        seed=None,
        verbose: Literal[0, 1, 2] = 1,
    ):
        """Sample objective at N randomly chosen points."""
        # TODO remove this? have parametr on gridserach`?`
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

    def marginals(self):
        """Aggregate over unique parameter values each parameter.

        ## Returns
        - marginals (dict[str,DataFrame]): aggregated scores for each parameter.
            - columns: parameter, mean, std, median
        """
        marginals = {}
        for param in self._param_range.keys():
            marginals[param] = (
                self.samples.group_by(param)
                .agg(
                    pl.col("score").mean().alias("mean"),
                    pl.col("score").std().alias("std"),
                    pl.col("score").median().alias("median"),
                )
                .sort(param)
            )

        return marginals

    def model_poly():
        """Polynomial model of function"""

    def model_GP():
        """Gaussian process model of function"""
