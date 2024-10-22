import numpy as np

import coolsearch.plotting as cplt
import coolsearch.search as csearch


def objective(x, y):
    trash = 0
    for i in range(20000):
        trash += np.cos(np.sin(np.sqrt(np.abs(i + np.cos(x**2))) ** 9))
    return x**2 + y - x * y


def main():
    cplt.set_plotly_template()

    search = csearch.CoolSearch(
        objective,
        {"x": (-10, 10), "y": (-10, 10)},
    )
    print(search)
    search.grid_search(5, etr_update_step=5)


if __name__ == "__main__":
    main()
