from timeit import default_timer

from joblib import Parallel, delayed
import joblib

from coolsearch.utility_functions import rastrigin_slow
from tqdm import tqdm


NTHREADS = joblib.cpu_count()
print(f"NTHREADS = {NTHREADS}")
N = 50
etr_update_step = 10
ITERATIONS = 10**6


def objective(x, y, z):
    return rastrigin_slow(x, y + z, ITERATIONS)


class MyClass:
    def __init__(self, data, objective):
        self.data = data
        self.objective = objective

    def eval_objective(self, params: dict):
        ts = default_timer()

        return objective(**params), default_timer() - ts

    def process_data(self):
        t_init = default_timer()

        p = Parallel(n_jobs=NTHREADS)

        output = p(delayed(self.eval_objective)(params) for params in tqdm(self.data))
        results = []
        runtimes = []
        t_start_loop = default_timer()
        print("starting computation")
        for i, (res, rt) in enumerate(output):
            n_completed = i + 1
            if (n_completed == len(self.data)) or (n_completed % etr_update_step) == 0:
                seq_avg = (default_timer() - t_start_loop) / n_completed
                etr = seq_avg * (len(self.data) - n_completed)
                print(
                    f"completed {n_completed}/{len(self.data)}. Estimated time left: {int(etr)} s"
                )
            results.append(res)
            runtimes.append(rt)

        return results, runtimes, default_timer() - t_init


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

    print("\nresults:")
    print(results)
    print("---\n")
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
