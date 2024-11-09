from coolsearch.search import CoolSearch


def f(x, y, z):
    return (x - 1) ** 2 + y**2 * z


params = {
    "x": (-3, 9),
    "y": 4.0,
    "z": [0.1, 2.0, 3.33, 7.999],
}

cs = CoolSearch(f, params, n_jobs=1, samples_file=None)

print(cs.samples)

print(cs.schema)
