from typing import Literal
import polars as pl

stat_type = Literal["all", "mean", "std", "median", "min", "max"]


def marginals(
    samples: pl.DataFrame,
    param: str | list[str],
    target: str,
    stat: stat_type | list[stat_type] = "all",
) -> pl.DataFrame | dict[str, pl.DataFrame]:
    """Aggregate values over unique parameter values.

    ## Returns
    - marginals (dict[str,DataFrame]): aggregated values for each parameter.
        - columns: parameter, mean, std, median
    """

    if stat == "all":
        stat = ["mean", "std", "median", "min", "max"]
    elif isinstance(stat, str):
        stat = [stat]

    aggs = {
        "mean": pl.col(target).mean().alias("mean"),
        "std": pl.col(target).std().alias("std"),
        "median": pl.col(target).median().alias("median"),
        "min": pl.col(target).min().alias("min"),
        "max": pl.col(target).max().alias("max"),
    }

    do_aggs = [aggs[k] for k in stat]

    def compute_marg(p: str):
        return samples.group_by(p).agg(*do_aggs).sort(p)

    # return single df
    if isinstance(param, str):
        return compute_marg(param)

    # return dict of dfs
    marginals = {}
    for p in param:
        marginals[p] = compute_marg(p)

    return marginals
