import unittest
import tempfile

from coolsearch.search import CoolSearch


def f(x, y):
    return (x + 1) ** 2 + (y - 3) ** 2


PARAMS = {"x": (-10.0, 1.0), "y": (-5.0, 10.0)}


class TestSaveLoad(unittest.TestCase):
    # def test_csv(self):
    #     self.save_load(".csv")

    def test_parquet(self):
        self.save_load(".parquet")

    # def test_json(self):
    #     self.save_load(".json")

    def test_unsupported(self):
        with self.assertRaises(ValueError):
            self.save_load(".abc")

    def save_load(self, ext: str):
        with tempfile.NamedTemporaryFile(suffix=ext) as tf:
            search1 = CoolSearch(f, PARAMS, samples_file=tf.name)
            search1.grid_search(3, verbose=0)
            search2 = CoolSearch(f, PARAMS, samples_file=tf.name)

            val1 = search1.samples["value"].to_list()
            val2 = search2.samples["value"].to_list()

            self.assertListEqual(val1, val2)
