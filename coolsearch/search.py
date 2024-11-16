from timeit import default_timer
from typing import Callable, Literal
import os

import polars as pl
from tqdm import tqdm

from coolsearch.Param import Param
from coolsearch.models import PolynomialModel
from coolsearch.utility_functions import make_mesh

try:
    import joblib
except ImportError:
    joblib = None


class CoolSearch:
    """Minimization of a black box function.

    Note: we assume that
    - the function takes a number of parameters
        - these are of type int, float, str, bool
    - the function returns some values
        - these are floats
    """

    # SLOTS?

    def __init__(
        self,
        objective: Callable[..., float | dict[str, float]],
        parameters: dict[str, int | float | str | bool | tuple | list],
        n_jobs: int = -1,
        samples_file: str | None = None,
    ) -> None:
        """Tools for minimizing a function.

        ## parameters
        - objective (callable):
            - Multiple returns as dict??
        - parameters: TODO
        - n_jobs (int): number of parallel jobs. Default: -1 uses cpu_count.
        - samples_file(str|None): file for loading and storing files on disk.
            - supported formats: `.parquet`, `.csv`, `.json`.
        """

        # set constants
        self._objective = objective

        ## Argument validation and schema inference
        self.params: dict[str, Param] = {}
        for p_name, val in parameters.items():
            if isinstance(val, (int, float, str, bool)):
                self.params[p_name] = Param(searchable=False, default=val)
            elif isinstance(val, list):
                self.params[p_name] = Param(searchable=True, options=val)
            elif isinstance(val, tuple):
                self.params[p_name] = Param(searchable=True, param_range=val)
            else:
                raise ValueError(f"Unsupported parameter value: {val}")

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

        ## schema for parameters, value and runtime
        # this is permanent to allow serialization
        schema = {p_name: p.pl_dtype for p_name, p in self.params.items()}

        # Add the things we're measuring to schema.
        schema["value"] = pl.Float64

        # We will also measure runtime
        if "runtime" in schema.keys():
            raise ValueError("Sorry, `runtime` is reserved for tracking runtime.")
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
                f"{self.ndim} dimensional search",
                f"  - has {len(self.samples)} samples",
            ],
        )

    @property
    def schema(self):
        return self.samples.schema

    @property
    def ndim(self):
        return len(self.params.keys())

    @classmethod
    def model_validate(
        cls,
        model,
        parameters: dict[str, int | float | str | bool | tuple | list],
        data: tuple,
        metrics: dict[str, Callable] | Callable,
        n_jobs: int = 1,
        samples_file: str | None = None,
    ):
        """
        Create a `CoolSearch` for tuning a classifier/regressor.

        ## parameters
        - model: A classifier/regressor that implements
        - params TODO!
        - data (tuple): data for training and validation
            - X_train, X_val, Y_train, Y_val
        - metrics: TODO
        - n_jobs (int): number of parallel jobs. Default: 1
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

        if not isinstance(metrics, dict):
            metrics = {"value": metrics}

        def objective(**kwargs):
            model.set_params(**kwargs)
            model.fit(X_train, Y_train)

            pred_val = model.predict(X_val)

            return {k: m(Y_val, pred_val) for k, m in metrics.items()}

        return cls(
            objective,
            parameters,
            n_jobs=n_jobs,
            samples_file=samples_file,
        )

    @classmethod
    def model_cv():
        raise NotImplementedError("is this separate from model_val?")

    def get_grid(self, steps: int | dict[str, int]):
        grid = make_mesh(self.params, steps)
        return grid

    def get_random_samples(
        self,
        N,
        seed=None,
    ):
        # rng = np.random.default_rng(seed)

        grid = []
        raise NotImplementedError("randsearch new impl?")
        for _ in range(N):
            grid.append(None)

        return pl.DataFrame(
            grid,
            schema=self.samples.select(self.param.keys()).schema,
            orient="row",
        ).unique()

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
            steps = max(int(round(n_samples ** (1 / self.ndim))), 1)
            if verbose >= 1:
                print(
                    f"choose {steps} steps\n",
                    f"  -> maximum {steps**self.ndim} samples",
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
        for param in self.params.keys():
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
            self.params.keys(),
            degree,
            interaction=True,
            target="value",
        )
        model.fit(verbose)
        return model

    def model_GP():
        """Gaussian process model of function"""
        raise NotImplementedError("TODO: Lose coupling? senf samples to modeling?")

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
            self.samples.select(self.params.keys()),
            on=self.params.keys(),
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
            # multi-thread mode
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

        # insert new runtimes
        grid_new = grid_new.with_columns(
            runtime=pl.Series(runtimes),
        )
        if all(isinstance(v, dict) for v in values):
            keys = values[0].keys()
            # insert multiple columns
            grid_new = grid_new.with_columns(
                (pl.Series(k, [v[k] for v in values]) for k in keys)
            )
        else:
            # insert single value
            grid_new = grid_new.with_columns(
                pl.Series("value", values),
            )

        # allow adding columns
        new_columns = set(grid_new.columns) - set(self.samples.columns)
        self.samples = self.samples.with_columns(
            (pl.Series(col, [None] * len(self.samples)) for col in new_columns)
        )

        # ensure columns are sorted
        cols = sorted(self.samples.columns)
        self.samples = self.samples.select(cols)
        grid_new = grid_new.select(cols)

        # include new samples and save if file provided
        self.samples = pl.concat([self.samples, grid_new], how="vertical")
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

    def _eval_obj(self, params: dict) -> tuple[float | dict[str | float], float]:
        """Compute the objective function for a parameter point
        ## parameters
        - params (dict): named parameters to objective
        ## returns
        - objective value (or dict of values)
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
            loaded = pl.read_csv(fp, schema=sample_schema)
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
