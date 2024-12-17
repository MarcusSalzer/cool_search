# from timeit import default_timer
# import multiprocessing as mp

# from multiprocessing import get_context
# import joblib


# def par_eval(n_jobs, objective, param_dicts):
#     """Evaluate using joblib"""
#     mp.set_start_method("spawn", force=True)
#     print(mp.get_start_method())

#     values = []
#     runtimes = []
#     with get_context("spawn").Pool() as pool:
#         output = pool.map(_eval_obj, param_dicts)

#     # para = joblib.Parallel(n_jobs=n_jobs, backend="loky")
#     # output = para(joblib.delayed(_eval_obj)(p) for p in param_dicts)

#     for val, rt in output:
#         values.append(val)
#         runtimes.append(rt)

#     return values, runtimes


# def _eval_obj(params: dict) -> tuple[float | dict[str | float], float]:
#     ts = default_timer()
#     return 1.0, default_timer() - ts
