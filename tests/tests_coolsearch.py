import unittest
from coolsearch.search import CoolSearch


class TestCoolSearch1D(unittest.TestCase):
    def setUp(self):
        self.objective = lambda x: x**2
        self.search = CoolSearch(
            self.objective,
            parameters={"x": (-1, 1)},
        )

    def test_grid(self):
        grid = self.search.get_grid(3)

        self.assertEqual(grid.columns, ["x"], "wrong columns")
        self.assertEqual(len(grid), 3, "wrong length")
        self.assertEqual(grid["x"].to_list(), [-1, 0, 1], "wrong elements")

    def test_grid_unique_ints(self):
        search = CoolSearch(
            self.objective,
            parameters={"x": (-2, 2)},
        )
        grid = search.get_grid(100)
        self.assertEqual(len(grid), 5, "wrong length")


class TestCoolSearch2Dint(unittest.TestCase):
    def setUp(self) -> None:
        def f(x, y):
            return x**2 + (y - 1) ** 2

        self.objective = f
        self.search = CoolSearch(
            self.objective,
            parameters={
                "x": (0, 2),
                "y": (0, 2),
            },
        )


class TestCoolSearch3D(unittest.TestCase):
    def setUp(self) -> None:
        def f(x, y, z):
            return x**2 + (y - 1) ** 2 + (z + 1) ** 2

        self.objective = f
        self.search = CoolSearch(
            self.objective,
            parameters={
                "x": (-3.0, 3.0),
                "y": (-4.0, 3.0),
                "z": (-5.0, 2.0),
            },
        )

    # def test_random_samples(self):
    #     grid = self.search.get_random_samples(50)
    #     self.assertEqual(grid.columns, ["x", "y", "z"], "wrong columns")
    #     self.assertEqual(len(grid), 50, "wrong length")

    def test_grid(self):
        grid = self.search.get_grid(4)
        self.assertEqual(grid.columns, ["x", "y", "z"], "wrong columns")
        self.assertEqual(len(grid), 4**3, "wrong length")

    def test_dict_step_grid(self):
        grid = self.search.get_grid({"x": 7, "y": 4, "z": 3})
        self.assertEqual(grid.columns, ["x", "y", "z"], "wrong columns")
        self.assertEqual(len(grid), 7 * 4 * 3, "wrong length")


class TestMultiOutput(unittest.TestCase):
    def setUp(self) -> None:
        def f(x, y):
            val = x**2 + (y - 1) ** 2
            val2 = 2 * val
            return {"val_a": val, "val_b": val2}

        self.objective = f
        self.search = CoolSearch(
            self.objective,
            parameters={
                "x": (-3.0, 3.0),
                "y": (-4.0, 3.0),
            },
            n_jobs=1,
        )

    def test_m(self):
        self.search.grid_search(2, verbose=0)


if __name__ == "__main__":
    unittest.main()
