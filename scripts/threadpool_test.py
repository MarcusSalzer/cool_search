import os
from multiprocessing import pool
from coolsearch.utility_functions import rastrigin_slow
from timeit import default_timer
import numpy as np

# cpu_count = multiprocessing.cpu_count()
cpu_count = os.cpu_count()
print(f"You seem to have {cpu_count} cpu cores")

ITERATIONS = 10**5
NTHREADS = 7


def test_1d(x):
    return rastrigin_slow(x, 0, ITERATIONS)


def test_2d(param):
    """min at (0, 1)"""
    x, y = param
    t_run = default_timer()
    value = rastrigin_slow(x, y - 1, ITERATIONS)
    rt = default_timer() - t_run
    return value, rt


xx = np.linspace(-10, 10, 60)
yy = np.linspace(-1, 3, 60)

params = list(zip(xx, yy))
print("computing with starmap")
ts = default_timer()
with pool.Pool(NTHREADS) as p:
    results = p.map(test_2d, params)

print(f"time taken: {default_timer()-ts}\n\n")
# print(results)

print("computing with imap_unordered")
ts = default_timer()

with pool.Pool(NTHREADS) as p:
    results = []
    for i, res in enumerate(p.imap_unordered(test_2d, params)):
        results.append(res)
        print(f"completed {len(results)}/{len(params)}")
print(f"time taken: {default_timer()-ts}\n\n")
