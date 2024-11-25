from enum import Enum, auto
from timeit import default_timer
from typing import Callable, Literal
import os

import numpy as np
import polars as pl
from tqdm import tqdm

from coolsearch.models import PolynomialModel

try:
    import joblib
except ImportError:
    joblib = None


class ParamMode(Enum):
    RANGE = auto()
    CONSTANT = auto()
    OPTIONS = auto()


class CoolSearch:
    """Minimization of a black box function."""

    # SLOTS?

    def __init__(
        self,
        objective: Callable,
        parameters: dict[str, int | float | str | bool | tuple | list],
        values: list[str] = "value",  # like this?
        n_jobs: int = -1,
        samples_file: str | None = None,
    ) -> None:
        """Tools for minimizing a function.

        ## parameters
        - objective (callable):
        - parameters: TODO
        - n_jobs (int): number of parallel jobs. Default: -1 uses cpu_count.
        - samples_file(str|None): file for loading and storing files on disk.
            - supported formats: `.parquet`, `.csv`, `.json`.
        """

        # set constants
        self._objective = objective

        # schema for parameters, value and runtime
        # this is permanent to allow serialization
        schema = {}

        # Modes for parameters
        # can be change to change the focus of searh
        param_modes = {}

        ## Argument validation and schema inference
        # TODO Cleanup and use Param-class
        for p_name, value in parameters.items():
            if not isinstance(p_name, str):
                raise ValueError(f"{p_name} is not a string")

            # Determine schema type and mode based on `value`
            if isinstance(value, int):
                schema[p_name] = pl.Int32
                param_modes[p_name] = ParamMode.CONSTANT
                print(p_name, ": fixed int param")

            elif isinstance(value, float):
                schema[p_name] = pl.Float64
                param_modes[p_name] = ParamMode.CONSTANT
                print(p_name, ": fixed float param")

            elif isinstance(value, str):
                schema[p_name] = pl.String
                param_modes[p_name] = ParamMode.CONSTANT
                print(p_name, ": fixed string param")

            elif isinstance(value, bool):
                schema[p_name] = pl.Boolean
                param_modes[p_name] = ParamMode.CONSTANT
                print(p_name, ": fixed bool param")

            elif isinstance(value, tuple):
                if len(value) != 2:
                    raise ValueError(f"cannot parse {value} as (min, max)")

                t1, t2 = type(value[0]), type(value[1])
                if t1 != t2:
                    raise ValueError(f"inconsistent types in range ({t1}, {t2})")

                if t1 is int:
                    schema[p_name] = pl.Int32
                    param_modes[p_name] = ParamMode.RANGE
                    print(p_name, ": ranged int param")
                elif t1 is float:
                    schema[p_name] = pl.Float64
                    param_modes[p_name] = ParamMode.RANGE
                    print(p_name, ": ranged float param")
                else:
                    raise ValueError(f"Cannot make range for {p_name} ({t1})")

            elif isinstance(value, list):
                if not value:
                    raise ValueError(f"List for {p_name} cannot be empty")

                first_type = type(value[0])
                if first_type is int:
                    schema[p_name] = pl.Int32
                elif first_type is float:
                    schema[p_name] = pl.Float64
                elif first_type is str:
                    schema[p_name] = pl.String
                else:
                    raise ValueError(
                        f"Unsupported type in list for {p_name} ({first_type})"
                    )

                # Ensure homogeneity within list
                if not all(isinstance(v, first_type) for v in value):
                    raise ValueError(f"Inconsistent types in list for {p_name}")

                param_modes[p_name] = ParamMode.OPTIONS
                print(p_name, ": fixed list of options")

            else:
                raise ValueError(f"Unsupported type for {p_name} ({type(value)})")

        ## handle number of parallel jobs
        if joblib is None:
            self.n_jobs = 1
            print("Note: joblib unavailable, using single-threaded mode.")
        else:
            cpu_count = joblib.cpu_count()
            if n_jobs == -1:
                self.n_jobs = cpu_count
            elif n_jobs > cpu_count:
                self.n_jobs = cpu_count
                print(f"Note: setting n_jobs to cpu_count = {cpu_count}")
            elif 0 < n_jobs:
                self.n_jobs = n_jobs
            else:
                raise ValueError(f"Invalid number of jobs: {n_jobs} ({type(n_jobs)})")

        schema["value"] = pl.Float64
        schema["runtime"] = pl.Float64

        # init empty samples-frame
        self.samples = pl.DataFrame(schema=schema)

        ## handle save file
        self.samples_file = samples_file
        if samples_file:
            if os.path.isfile(samples_file) and os.path.getsize(samples_file) > 0:
                self._load_samples()
            else:
                # Save (empty) samples if not exists
                self._save_samples()

    def __str__(self):
        return "\n".join(
            [
                f"{self._ndim} dimensional search",
                f"  - has {len(self.samples)} samples",
            ],
        )

    @property
    def schema(self):
        return self.samples.schema

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
        n_jobs: int = -1,
        samples_file: str | None = None,
    ):
        """
        Create a `CoolSearch` for tuning a classifier/regressor.

        ## parameters
        - model: A classifier/regressor that implements
        - params
        - data (tuple): data for training and validation
            - X_train, X_val, Y_train, Y_val
        - loss_fn (Callable[[arraylike, arraylike], float]): loss function to minimize
        - invert (bool): maximize loss function instead
        - n_jobs (int): number of parallel jobs. Default: -1 uses cpu_count.
        - samples_file(str|None): file for loading and storing files on disk.
            - supported formats: `.parquet`, `.csv`, `.json`.

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

        return cls()  # TODO

    def get_grid(self, steps: int | dict[str, int]):
        raise NotImplementedError("TODO")

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
    ):
        """Sample objective on an evenly spaced grid.

        ## Parameters
        - steps (int | dict[str, int]): grid resolution, either:
            - same for all parameters (int)
            - specify per parameter (dict).
        - target_runtime (float): target time (seconds) to estimate number of steps.
        - verbose (int): amount of status information printed

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
                    f"choose {steps} steps\n",
                    f"  -> maximum {steps**self._ndim} samples",
                )
        grid = self.get_grid(steps)
        return self._eval_samples(grid, verbose)

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
        """Aggregate values over unique parameter values.

        ## Returns
        - marginals (dict[str,DataFrame]): aggregated values for each parameter.
            - columns: parameter, mean, std, median
        """
        marginals = {}
        for param in self._param_range.keys():
            marginals[param] = (
                self.samples.group_by(param)
                .agg(
                    pl.col("value").mean().alias("mean"),
                    pl.col("value").std().alias("std"),
                    pl.col("value").median().alias("median"),
                    pl.col("value").min().alias("min"),
                    pl.col("value").max().alias("max"),
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
            target="value",
        )
        model._fit(verbose)
        return model

    def model_GP():
        """Gaussian process model of function"""

    def _eval_samples(
        self,
        grid_new: pl.DataFrame,
        verbose: int,
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
                    len(grid_new) * self.samples["runtime"].mean() / self.n_jobs
                )
                print(f"Estimated runtime: {est_runtime:.2f} s.")

        param_dicts = grid_new.iter_rows(named=True)

        if verbose >= 1:
            param_dicts = tqdm(param_dicts)

        values = []
        runtimes = []
        if self.n_jobs > 1:
            para = joblib.Parallel(n_jobs=self.n_jobs)
            output = para(joblib.delayed(self._eval_obj)(p) for p in param_dicts)
            for val, rt in output:
                values.append(val)
                runtimes.append(rt)
        else:
            # single-threaded mode
            for p in param_dicts:
                val, rt = self._eval_obj(p)
                values.append(val)
                runtimes.append(rt)

        grid_new = grid_new.with_columns(
            value=pl.Series(values),
            runtime=pl.Series(runtimes),
        )

        # update samples and save if file provided
        self.samples = pl.concat([self.samples, grid_new])
        if self.samples_file:
            self._save_samples()

        rt_sum = sum(runtimes)
        rt_total = default_timer() - t_start
        p_fac = rt_sum / rt_total
        if verbose >= 1:
            print(f"Sum of runtime: {rt_sum:.2f} s. Elapsed time {rt_total:.2f} s.")
            print(f"Overhead: {rt_total-rt_sum:.4f} s.")
            if self.n_jobs > 1:
                print(f"paralellness: {p_fac:.2f} " + ":)" if p_fac > 1 else ":/")

    def _eval_obj(self, params: dict) -> tuple[float, float]:
        """Compute the objective function for a parameter point
        ## parameters
        - params (dict): named parameters to objective
        ## returns
        - objective value (float)
        - runtime (float): runtime in seconds
        """
        ts = default_timer()
        return self._objective(**params), default_timer() - ts

    def _load_samples(self):
        """Load samples from file"""

        fp = self.samples_file
        if fp is None:
            raise ValueError("No filepath provided")
        sample_schema = self.samples.schema
        _, ext = os.path.splitext(fp)

        # load file based on extension
        if ext == ".parquet":
            loaded = pl.read_parquet(fp)
            if loaded.schema != sample_schema:
                raise ValueError(f"Incorrect file schema: {loaded.schema}")
        elif ext == ".csv":
            loaded = pl.read_csv(fp)
        elif ext == ".json":
            loaded = pl.read_json(fp, schema=sample_schema)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        self.samples = loaded

    def _save_samples(self):
        """Save samples to file"""
        fp = self.samples_file
        if fp is None:
            raise ValueError("No filepath provided")
        _, ext = os.path.splitext(fp)

        # save file based on extension
        if ext == ".parquet":
            self.samples.write_parquet(fp)
        elif ext == ".csv":
            self.samples.write_csv(fp)
        elif ext == ".json":
            self.samples.write_json(fp)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
