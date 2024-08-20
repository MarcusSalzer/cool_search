from timeit import default_timer
from typing import Literal

import numpy as np
import polars as pl


class PolynomialModel:
    """Polynomial regression"""

    def __init__(
        self,
        samples: pl.DataFrame,
        order: int = 1,
        interaction: bool = True,
        target: str = "score",
    ) -> None:
        y = samples[target].to_numpy()
