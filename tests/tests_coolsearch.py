import unittest
import polars as pl


from coolsearch.search import CoolSearch


class TestCoolSearch1D(unittest.TestCase):
    def setUp(self):
        self.objective = lambda x: x**2
        self.search = CoolSearch(
            self.objective,
            param_range={"x": (-1, 1)},
        )

    def test_grid(self):
        grid = self.search.get_grid(3)

        self.assertEqual(grid.columns, ["x"], "wrong columns")
        self.assertEqual(len(grid), 3, "wrong length")
        self.assertEqual(grid["x"].to_list(), [-1, 0, 1], "wrong elements")

    def test_grid_unique_ints(self):
        search = CoolSearch(
            self.objective,
            param_range={"x": (-2, 2)},
            param_types={"x": "int"},
        )
        grid = search.get_grid(100)
        self.assertEqual(len(grid), 5, "wrong length")

    def test_invalid_dtype(self):
        with self.assertRaises(ValueError):
            CoolSearch(
                self.objective, param_range={"x": (1, 4)}, param_types={"x": "complex"}
            )

    def test_inconsistent_param_names(self):
        with self.assertRaises(ValueError) as cm:
            CoolSearch(
                self.objective, param_range={"x": (1, 4)}, param_types={"y": "float"}
            )

        self.assertEqual(
            str(cm.exception),
            "Inconsistent parameter names",
            "wrong exception msg",
        )


class TestCoolSearch2Dint(unittest.TestCase):
    def setUp(self) -> None:
        def f(x, y):
            return x**2 + (y - 1) ** 2

        self.objective = f
        self.search = CoolSearch(
            self.objective,
            param_range={
                "x": (0, 2),
                "y": (0, 2),
            },
            param_types={
                "x": "int",
                "y": "int",
            },
        )

    def test_random_samples(self):
        grid = self.search.get_random_samples(100)

        ok_types = {pl.Int32, pl.Int64}
        self.assertListEqual(
            [t in ok_types for t in grid.dtypes],
            [True, True],
            "incorrect datatypes",
        )
        self.assertLessEqual(len(grid), 9, "too many points")


class TestCoolSearch3D(unittest.TestCase):
    def setUp(self) -> None:
        def f(x, y, z):
            return x**2 + (y - 1) ** 2 + (z + 1) ** 2

        self.objective = f
        self.search = CoolSearch(
            self.objective,
            param_range={
                "x": (-3, 3),
                "y": (-4, 3),
                "z": (-5, 2),
            },
        )

    def test_random_samples(self):
        grid = self.search.get_random_samples(50)
        self.assertEqual(grid.columns, ["x", "y", "z"], "wrong columns")
        self.assertEqual(len(grid), 50, "wrong length")

    def test_grid(self):
        grid = self.search.get_grid(4)
        self.assertEqual(grid.columns, ["x", "y", "z"], "wrong columns")
        self.assertEqual(len(grid), 4**3, "wrong length")

    def test_dict_step_grid(self):
        grid = self.search.get_grid({"x": 7, "y": 4, "z": 3})
        self.assertEqual(grid.columns, ["x", "y", "z"], "wrong columns")
        self.assertEqual(len(grid), 7 * 4 * 3, "wrong length")


class TestPolymodel(unittest.TestCase):
    def setUp(self) -> None:
        def f(x, y):
            return x**2 + (y - 1) ** 2

        self.search = CoolSearch(f, {"x": (-5, 5), "y": (-5, 5)})

    def test_error(self):
        with self.assertRaises(ValueError):
            self.search.model_poly(2)


if __name__ == "__main__":
    unittest.main()
