"""Quick start: minimise a function with RIME and MRIME-CD under the same evaluation budget.

Run with ``python examples/quickstart.py`` (a few seconds, no data files needed).
"""

import numpy as np

from mrime import minimize, mrime_cd
from mrime.benchmarks import get_problem


def my_objective(x: np.ndarray) -> float:
    """Any function of a 1-D array works; this one is a shifted, ill-conditioned bowl."""
    scales = 10.0 ** np.linspace(0.0, 3.0, x.size)
    return float(np.sum(scales * (x - 3.0) ** 2))


def main() -> None:
    dim, pop_size, max_evals = 10, 70, 30_000

    result = mrime_cd(my_objective, lower=-100.0, upper=100.0, dim=dim,
                      pop_size=pop_size, max_evals=max_evals, seed=0)
    print(f"custom objective : f = {result.fun:.3e} after {result.n_evals} evaluations")
    print(f"                   x = {np.round(result.x, 3)}")

    problem = get_problem("classical", "ellipsoid", dim)  # rotated quadratic, optimum known
    for algorithm in ("rime", "rime-g", "mrime-cd"):
        finals = [
            minimize(problem, problem.lower, problem.upper, dim, pop_size, max_evals,
                     seed=seed, algorithm=algorithm).fun
            for seed in range(5)
        ]
        print(f"{algorithm:<9} on the rotated ellipsoid: median f = {np.median(finals):.3e}")


if __name__ == "__main__":
    main()
