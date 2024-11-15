import unittest

from coolsearch.Param import Param
from coolsearch.utility_functions import make_mesh
import numpy as np
import polars as pl


class TestParam(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_fixed_int_default(self):
        p = Param(searchable=False, default=7)

        g = p.grid(33)
        t = np.array([7])
        self.assertEqual(g.dtype, t.dtype, "Mismatched grid dtypes")
        self.assertListEqual(g.tolist(), t.tolist())
        self.assertEqual(p.pl_dtype, pl.Int32, "Mismatched polars dtype")

    def test_fixed_int_range(self):
        p = Param(searchable=False, param_range=(0, 3))

        self.assertEqual(p.default, 1, "Incorrect default inference")

        g = p.grid(3)
        t = np.array([1])
        self.assertEqual(g.dtype, t.dtype, "Mismatched dtypes")
        self.assertListEqual(g.tolist(), t.tolist())
        self.assertEqual(p.pl_dtype, pl.Int32, "Mismatched polars dtype")

    def test_fixed_str_options(self):
        p = Param(searchable=False, options=["low", "medium", "high"])

        self.assertEqual(p.default, "low", "Incorrect default inference")

        g = p.grid(2)
        t = np.array(["low"])
        self.assertEqual(g.dtype, t.dtype, "Nonmatching dtypes")
        self.assertListEqual(g.tolist(), t.tolist())
        self.assertEqual(p.pl_dtype, pl.String, "Mismatched polars dtype")

    def test_bool_options(self):
        p = Param(searchable=True, options=[True, False])

        self.assertEqual(p.default, True, "Incorrect default inference")

        self.assertListEqual(
            p.grid(1).tolist(),
            np.array([True]).tolist(),
        )
        self.assertListEqual(
            p.grid(2).tolist(),
            np.array([True, False]).tolist(),
        )

        self.assertEqual(p.pl_dtype, pl.Boolean, "Mismatched polars dtype")


class TestMesh(unittest.TestCase):
    def test_1str(self):
        p1 = Param(searchable=True, options=["A", "B", "C"])

        m = make_mesh([p1], 40)

        self.assertEqual(m.shape, (3, 1))
        self.assertListEqual(m[p1.name].to_list(), ["A", "B", "C"])
        self.assertEqual(m[p1.name].dtype, pl.String)

    def test_2str(self):
        p1 = Param(searchable=True, options=["A", "B", "C"])
        p2 = Param(searchable=True, options=["X", "Y"])
        m = make_mesh([p1, p2], 2)

        self.assertEqual(m.shape, (4, 2))
        self.assertListEqual(m[p1.name].to_list(), ["A", "A", "B", "B"])
        self.assertListEqual(m[p2.name].to_list(), ["X", "Y", "X", "Y"])
        self.assertEqual(m[p1.name].dtype, pl.String)

    def test_int_float(self):
        p1 = Param(searchable=True, param_range=(1, 5))
        p2 = Param(searchable=True, param_range=(7.0, 8.0))
        p3 = Param(searchable=False, param_range=(7, 9))
        p4 = Param(searchable=False, options=["z", "y"])

        m = make_mesh([p3, p1, p2, p4])
        # print(m)

        self.assertEqual(m.shape, (9, 4))
        self.assertListEqual(sorted(m[p1.name].to_list()), [1, 1, 1, 3, 3, 3, 5, 5, 5])
        self.assertListEqual(
            sorted(m[p2.name].to_list()),
            [7.0, 7.0, 7.0, 7.5, 7.5, 7.5, 8.0, 8.0, 8.0],
        )
        self.assertListEqual(sorted(m[p3.name].to_list()), [8] * 9)
        self.assertListEqual(sorted(m[p4.name].to_list()), ["z"] * 9)

        self.assertEqual(m[p1.name].dtype, pl.Int32)
        self.assertEqual(m[p2.name].dtype, pl.Float64)
        self.assertEqual(m[p3.name].dtype, pl.Int32)
        self.assertEqual(m[p4.name].dtype, pl.String)

    def test_str_float(self):
        p1 = Param(searchable=True, param_range=(0.0, 1.0))
        p2 = Param(searchable=True, options=["A", "B", "C"])
        m = make_mesh([p1, p2], 20)

        self.assertEqual(m.shape, (60, 2))
        self.assertListEqual(m[p2.name].to_list(), ["A", "B", "C"] * 20)
        self.assertEqual(m[p1.name].dtype, pl.Float64)
        self.assertEqual(m[p2.name].dtype, pl.String)
