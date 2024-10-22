import os
from multiprocessing import pool
from coolsearch.utility_functions import rastrigin_slow
from timeit import default_timer
import numpy as np


ITERATIONS = 10**5
n_processes = 8
N = 100
etr_update_step = 10


def test_1d(x):
    return rastrigin_slow(x, 0, ITERATIONS)


def test_2d(param):
    """min at (0, 1)"""
    x, y = param
    t_run = default_timer()
    value = rastrigin_slow(x, y - 1, ITERATIONS)
    rt = default_timer() - t_run
    return value, rt


if __name__ == "__main__":
    cpu_count = os.cpu_count()
    print(f"You seem to have {cpu_count} cpu cores")

    xx = np.linspace(-10, 10, N)
    yy = np.linspace(-1, 3, N)

    params = list(zip(xx, yy))
    print("computing with starmap")
    ts = default_timer()
    with pool.Pool(n_processes) as p:
        results = p.map(test_2d, params)

    print(f"time taken: {default_timer()-ts}\n\n")
    # print(results)

    print("computing with imap_unordered")
    ts = default_timer()

    with pool.Pool(n_processes) as p:
        results = []
        for i, res in enumerate(p.imap_unordered(test_2d, params)):
            results.append(res)
            if i % etr_update_step == 0:
                print(f"completed {len(results)}/{len(params)}")
    print(f"time taken: {default_timer()-ts}\n\n")
