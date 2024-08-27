import os
import sys
import unittest

import numpy as np
import polars as pl

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../src"),
    ),
)

from cool_search_models import PolynomialModel, polynomial_features
import utility_functions as util

# decimals to compare for numerical accuracy
DECIMALS = 10


class TestPolyModel1D(unittest.TestCase):
    def setUp(self) -> None:
        self.f = lambda x: util.test_function_01(x, 0.01)
        self.samples = pl.DataFrame({"x": np.linspace(-10, 10, 5)})
        self.samples = self.samples.with_columns(
            pl.col("x").map_elements(self.f, return_dtype=pl.Float64).alias("f"),
        )

    def test_degree2(self):
        model = PolynomialModel(self.samples, ["x"], 2, target="f")
        with self.assertRaises(AttributeError):
            model.beta

        model.fit(verbose=False)
        beta = model.beta

        f_pred = model.predict()
        f_poly = model.polynomial(self.samples["x"].to_numpy())

        self.assertEqual(2 + 1, len(beta), "Incorrect number of coefs.")

        self.assertListEqual(
            f_poly.round(DECIMALS).tolist(),
            f_pred.round(DECIMALS).tolist(),
            "predict not consistent with polynomial",
        )

    def test_exact_degree(self):
        model = PolynomialModel(self.samples, ["x"], len(self.samples) - 1, target="f")
        model.fit(verbose=False)

        f_true = self.samples["f"].to_numpy()
        f_pred = model.predict()
        f_poly = model.polynomial(self.samples["x"].to_numpy())

        self.assertListEqual(
            f_true.round(DECIMALS).tolist(),
            f_pred.round(DECIMALS).tolist(),
            "not exact for order N-1",
        )

        self.assertListEqual(
            f_poly.tolist(),
            f_pred.tolist(),
            "predict not consistent with polynomial",
        )


class TestPolyFeat(unittest.TestCase):
    def test_poly_1d(self):
        X = np.array([[1, 2, 3]]).T
        X_correct = np.array([[1, 1, 1], [1, 2, 3], [1, 4, 9], [1, 8, 27]]).T
        X_new = polynomial_features(X, 3, verbose=False)
        self.assertEqual(X_new.shape, X_correct.shape, "incorrect shape")
        self.assertTrue((X_new == X_correct).all(), "incorrect array values")


if __name__ == "__main__":
    unittest.main()
