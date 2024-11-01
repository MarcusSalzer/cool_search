from timeit import default_timer
from typing import Callable, Literal

import numpy as np
import polars as pl
from tqdm import tqdm

import coolsearch.utility_functions as util
from coolsearch.models import PolynomialModel

try:
    import joblib
except ImportError:
    joblib = None


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
        "njobs",
    ]

    def __init__(
        self,
        objective,
        param_range: dict,
        param_types: dict[str, Literal["int", "float"]] | None = None,
        # min_delta? max_steps?
        fixed_params={},
        n_jobs=-1,
    ) -> None:
        """Tools for minimizing a function.

        ## parameters
        - objective (callable):
        - param_range (dict): ranges for all parameters
        - param_types (dict): specify int or float types. (Defaults to float for all parameters)
        - fixed_params (dict): fixed kwargs provided to `objective`.
        - n_jobs (int): number of parallel jobs. Default: -1 uses cpu_count.

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

        ## handle number of parallel jobs
        if joblib is None:
            self.n_jobs = 1
            print("Note: joblib unavailable, using single-threaded mode.")
        else:
            cpu_count = joblib.cpu_count()
            if n_jobs == -1:
                self.n_jobs = cpu_count
            if n_jobs > cpu_count:
                self.n_jobs = cpu_count
                print(f"Note: setting n_jobs to cpu_count = {cpu_count}")
            else:
                self.n_jobs = n_jobs

        # schema for parameters, score and runtime
        schema = {
            param: (pl.Float64 if dtype == "float" else pl.Int64)
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
    def model_validate(
        cls,
        model,
        param_range: dict[str, tuple],
        param_types: dict[str, Literal["float", "int"]] | None,
        data: tuple,
        loss_fn: Callable[[np.ndarray, np.ndarray], float],
        invert: bool = False,
        fixed_params={},
    ):
        """
        Create a `CoolSearch` for tuning a classifier/regressor.

        ## parameters
        - model: A classifier/regressor that implements
            - set_params()
            - fit()
            - predict()
        - param_range (dict[str, tuple]): ranges for parameters to tune
        - param_types (dict[str, str]): types for parameters to tune
        - data (tuple): data for training and validation
            - X_train, X_val, Y_train, Y_val
        - loss_fn (Callable[[arraylike, arraylike], float]): loss function to minimize
        - invert (bool): maximize loss function instead

        ## returns
        - search (CoolSearch)
        """

        if len(data) == 4:
            X_train, X_val, Y_train, Y_val = data

            if X_train.shape[1] != X_val.shape[1]:
                raise ValueError("inconsistent column count (X)")
            if X_train.shape[0] != Y_train.shape[0]:
                raise ValueError("inconsistent row count (train)")
            if X_val.shape[0] != Y_val.shape[0]:
                raise ValueError("inconsistent row count (val)")

        else:
            raise ValueError(f"Unsupported number of data arrays ({len(data)})")

        def objective(**kwargs):
            model.set_params(**kwargs, **fixed_params)
            model.fit(X_train, Y_train)

            pred_val = model.predict(X_val)

            if invert:
                return -loss_fn(Y_val, pred_val)
            else:
                return loss_fn(Y_val, pred_val)

        return cls(objective, param_range, param_types, fixed_params)

    def get_grid(self, steps: int | dict[str, int]):
        return util.get_grid(steps, self._param_range, self._param_types)

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

        return pl.DataFrame(
            grid,
            schema=self.samples.select(self.param_names).schema,
            orient="row",
        ).unique()

    def make_factor_grid(self, steps):
        mesh = np.meshgrid(*[np.linspace(0, 1, steps)] * self._ndim)
        return np.vstack(list(map(np.ravel, mesh))).T

    def grid_search(
        self,
        steps: int | dict[str, int] | None = 10,
        target_runtime=None,
        verbose: Literal[0, 1, 2] = 2,
        etr_update_step: int = 1,
    ):
        """Sample objective on an evenly spaced grid.

        ## Parameters
        - steps (int | dict[str, int]): grid resolution, either:
            - same for all parameters (int)
            - specify per parameter (dict).
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
                self.grid_search(steps=1, verbose=0)  # TODO "POINTSEARCH?"

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
        grid = self.get_grid(steps)
        return self._eval_samples(grid, verbose, etr_update_step)

    def random_search(
        self,
        N,
        target_runtime=None,  # TODO make a function for est this
        seed=None,
        verbose: Literal[0, 1, 2] = 1,
        etr_update_step: int = 1,
    ):
        """Sample objective at N randomly chosen points."""

        if target_runtime is not None:
            raise NotImplementedError("coming soon")

        grid = self.get_random_samples(N, seed)
        return self._eval_samples(grid, verbose, etr_update_step)

    def run_sequence():
        """Specify a 'macro-program' to run multiple searches."""
        # idea. run sequentially, /and parallel?
        # allow for conditions to choose next step

    def marginals(self) -> dict[str, pl.DataFrame]:
        """Aggregate score values over unique parameter values.

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

    def model_poly(self, degree: int = 1, verbose=True):
        """Polynomial model of function.

        ## parameters
        - degree (int): maximum degree of polynomial

        ## returns
        - model_poly (PolynomialModel): model fitted to known samples.

        """

        if len(self.samples) < degree + 1:
            raise ValueError(f"Needs more samples: {len(self.samples)}<{degree} + 1")

        model = PolynomialModel(
            self.samples,
            self.param_names,
            degree,
            interaction=True,
            target="score",
        )
        model.fit(verbose)
        return model

    def model_GP():
        """Gaussian process model of function"""

    def _eval_samples(
        self,
        grid_new: pl.DataFrame,
        verbose: int,
        etr_update_step: int,
    ) -> pl.DataFrame:
        """Evaluate a grid of samples, append new values and runtimes to self.samples."""

        # measure total runtime to compute overhead
        t_start = default_timer()

        # avoid previously sampled points
        grid_new = grid_new.join(
            self.samples.select(self.param_names),
            on=self.param_names,
            how="anti",
        )

        if verbose >= 1:
            print(f"Searching {len(grid_new)} new parameter points")
            if not self.samples.is_empty():
                est_runtime = (
                    len(grid_new) * self.samples["runtime"].mean() / self.njobs
                )
                print(f"Estimated runtime: {est_runtime:.2f} s.")

        param_dicts = grid_new.iter_rows(named=True)

        p = joblib.Parallel(n_jobs=self.njobs)

        output = p(
            joblib.delayed(self._eval_obj)(params) for params in tqdm(param_dicts)
        )
        values = []
        runtimes = []
        for val, rt in output:
            values.append(val)
            runtimes.append(rt)

        grid_new = grid_new.with_columns(
            score=pl.Series(values),
            runtime=pl.Series(runtimes),
        )
        self.samples = pl.concat([self.samples, grid_new])

        rt_sum = sum(runtimes)
        rt_total = default_timer() - t_start
        p_fac = rt_sum / rt_total
        if verbose >= 1:
            print(f"Sum of runtime: {rt_sum:.2f}. Elapsed time {rt_total:.2f} s.")

            print(f"paralellness: {p_fac:.2f} " + ":)" if p_fac > 1 else ":/")

        return grid_new

    def _eval_obj(self, params: dict):
        """Compute the objective function for a parameter point
        ## parameters
        - params (dict): named parameters to objective
        ## returns
        - objective value (float)
        - runtime (float): runtime in seconds
        """
        ts = default_timer()
        return self._objective(**params), default_timer() - ts
