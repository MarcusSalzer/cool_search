import numpy as np
import polars as pl


class Param:
    """A parameter that can generate ranges

    Note: allow changing searchability, range and default value, but not datatype"""

    _counter = 0

    def __init__(
        self,
        name: str | None = None,
        searchable: bool = False,
        default: int | float | str | bool | None = None,
        param_range: tuple[int | float, int | float] | None = None,
        options: list[int | float | str] | None = None,
    ) -> None:
        if name is None:
            name = f"p_{Param._counter:02}"
            Param._counter += 1

        # infer default
        if default is None:
            if param_range is not None:
                # infer default from range
                mean = (param_range[0] + param_range[1]) / 2
                if isinstance(param_range[0], int):
                    default = int(mean)
                else:
                    default = mean
            elif options is not None:
                # just take first option
                default = options[0]
            else:
                raise ValueError("Requires at least one of default, range, options")

        if param_range is not None:
            if len(param_range) != 2:
                raise ValueError(f"Expects range as (min,max), not {param_range}")

        self.name = name
        self.pl_dtype = self.infer_polars_dtype(default)
        self.searchable = searchable
        self.param_range = param_range
        self.default = default
        self.options = options

    def __str__(self) -> str:
        stat = f"({self.default})"
        if self.param_range is not None:
            stat += f" {self.param_range[0]} <-> {self.param_range[1]}"

        if self.searchable:
            stat += " SEARCHABLE"
        else:
            stat += " FIXED"

        return f"{self.name}: {stat}"

    def grid(self, steps: int) -> np.ndarray:
        if (not self.searchable) or steps == 1:
            return np.array([self.default])

        ## if not fixed
        # prioritize: 1. options 2. range
        if self.options is not None:
            return np.array(self.options[:steps])
        if self.param_range is not None:
            if isinstance(self.param_range[0], float):
                return np.linspace(*self.param_range, steps)
            elif isinstance(self.param_range[0], int):
                return np.unique(np.linspace(*self.param_range, steps, dtype=int))

    def infer_polars_dtype(self, value):
        if type(value) is int:
            return pl.Int32
        elif type(value) is bool:
            return pl.Boolean
        elif type(value) is float:
            return pl.Float64
        elif type(value) is str:
            return pl.String
        else:
            raise ValueError("Unsupported type for polars dtype inference.")
