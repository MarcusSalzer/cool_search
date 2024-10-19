from timeit import default_timer

from coolsearch.utility_functions import rastrigin_slow

print("start computation")
ts = default_timer()
res = rastrigin_slow(1, 1.7)
print(f"result: {res}")
print(f"time taken: {default_timer()-ts}")
