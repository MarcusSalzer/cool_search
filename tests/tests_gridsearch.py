import unittest

from coolsearch.search import CoolSearch
from coolsearch import utility_functions as util
import numpy as np


def test1d(x, delay=0.01):
    return util.test_function_01(x, delay)


class TestGridsearch1D(unittest.TestCase):
    def setUp(self) -> None:
        self.steps = 12
        self.range = (-5, 10)
        xx = np.linspace(*self.range, self.steps)
        self.correct = [test1d(x, delay=0) for x in xx]

    def test1Thread(self):
        search = CoolSearch(test1d, {"x": (-5, 10)}, n_jobs=1)
        search.grid_search(self.steps, verbose=0)
        result = search.samples.sort("x")["value"].to_list()
        self.assertListEqual(self.correct, result)

    def test6Thread(self):
        search = CoolSearch(test1d, {"x": (-5, 10)}, n_jobs=6)
        search.grid_search(self.steps, verbose=0)
        result = search.samples.sort("x")["value"].to_list()
        self.assertListEqual(self.correct, result)
