import os
from multiprocessing import pool
from timeit import default_timer
from typing import Callable
from coolsearch.utility_functions import rastrigin_slow

# cpu_count = multiprocessing.cpu_count()


NTHREADS = 8
N = 100
etr_update_step = 3
ITERATIONS = 10**5


def objective(x, y, z):
    return rastrigin_slow(x, y + z, ITERATIONS)


# Helper function must be at the top level for pickling
def eval_objective(objective_and_params: tuple[Callable, dict]):
    """Compute the objective function for a parameter point
    ## parameters
    - objective_and_params (tuple): both things
    ## returns
    - objective value
    - runtime (float): runtime in seconds
    """
    ts = default_timer()
    objective, params = objective_and_params
    return objective(**params), default_timer() - ts


class MyClass:
    def __init__(self, data, objective):
        self.data = data  # Data is a list of dict-like argument sets
        self.objective = objective  # Objective function passed in constructor

    def process_data(self):
        ts = default_timer()
        with pool.Pool(NTHREADS) as p:
            # Create pairs of (objective function, params) for each item in self.data
            args_for_pool = [(self.objective, params) for params in self.data]
            results = []
            runtimes = []
            for i, (res, rt) in enumerate(
                p.imap_unordered(
                    eval_objective,
                    args_for_pool,
                )
            ):
                if i - 1 % etr_update_step == 0:
                    print(f"completed {i}/{len(self.data)}")
                results.append(res)
                runtimes.append(rt)

        return results, runtimes, default_timer() - ts


def main():
    data = [
        dict(x=1, y=2, z=3),
        dict(x=4, y=2, z=3),
        dict(x=7, y=9, z=3),
        dict(x=7, y=9, z=6),
        dict(x=7, y=2, z=3),
        dict(x=7, y=80, z=3),
    ] * int(N / 6)

    my_obj = MyClass(data, objective)
    results, runtimes, total_time = my_obj.process_data()

    print(results)
    s_rt = sum(runtimes)
    p_fac = s_rt / total_time
    print(f"sum of runtimes: { s_rt:.2f}. total runtime: {total_time:.2f}")
    print(f"paralellness: {p_fac:.2f}")
    if p_fac > 1:
        print(":)")
    else:
        print(":/")


if __name__ == "__main__":
    main()
