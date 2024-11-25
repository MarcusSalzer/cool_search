import unittest
import tempfile
import polars as pl
from coolsearch.search import CoolSearch


def f(x, y):
    return (x + 1) ** 2 + (y - 3) ** 2


PARAMS = {"x": (-10, 1.0), "y": (-5.0, 10.0)}


class TestSaveLoad(unittest.TestCase):
    def test_csv(self):
        self.save_load(".csv")

    def test_parquet(self):
        self.save_load(".parquet")

    def test_json(self):
        self.save_load(".json")

    def test_unsupported(self):
        with self.assertRaises(ValueError):
            self.save_load(".abc")

    def save_load(self, ext: str):
        with tempfile.NamedTemporaryFile(suffix=ext) as tf:
            search1 = CoolSearch(f, PARAMS, samples_file=tf.name)
            search1.grid_search(3, verbose=0)
            search2 = CoolSearch(f, PARAMS, samples_file=tf.name)

            self.assertListEqual(
                search1.samples["value"].to_list(),
                search2.samples["value"].to_list(),
                "wrong `value` column",
            )
            self.assertDictEqual(
                search1.samples.schema,
                search2.samples.schema,
                "wrong `samples.schema`",
            )
