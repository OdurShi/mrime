"""MRIME-CD against RIME under equal budgets on the built-in benchmark, with fixed seeds.

The run is deterministic (fixed seeds), and the thresholds are far from the measured margins:
on sphere, schwefel_2_22, ackley and zakharov the *worst* MRIME-CD run is better than the
*best* RIME run. Rastrigin is the known exception at this budget (RIME is better there), which
is why the suite-level criterion is "at least 6 of 8 functions" rather than all of them.
"""

import numpy as np
import pytest

from mrime import minimize
from mrime.benchmarks import CLASSICAL_NAMES, get_problem

DIM, POP, BUDGET, SEEDS = 10, 30, 8000, range(8)


@pytest.fixture(scope="module")
def finals() -> dict[str, dict[str, np.ndarray]]:
    table: dict[str, dict[str, np.ndarray]] = {}
    for name in CLASSICAL_NAMES:
        problem = get_problem("classical", name, DIM)
        table[name] = {}
        for algorithm in ("rime", "mrime-cd"):
            results = [
                minimize(problem, problem.lower, problem.upper, DIM, POP, BUDGET, seed=seed,
                         algorithm=algorithm)
                for seed in SEEDS
            ]
            assert all(r.n_evals == BUDGET for r in results)  # equal budgets, strictly
            table[name][algorithm] = np.array([r.fun for r in results])
    return table


def test_mrime_cd_is_not_worse_than_rime_on_the_benchmark(finals):
    better = [
        name for name in CLASSICAL_NAMES
        if np.median(finals[name]["mrime-cd"]) <= np.median(finals[name]["rime"])
    ]
    assert len(better) >= 6, f"MRIME-CD has the better median only on {better}"


@pytest.mark.parametrize("name", ["sphere", "schwefel_2_22", "ackley", "zakharov"])
def test_mrime_cd_wins_every_paired_run_on_smooth_functions(finals, name):
    # same seed -> same initial population for both algorithms
    assert np.all(finals[name]["mrime-cd"] < finals[name]["rime"])
    assert np.median(finals[name]["mrime-cd"]) < 0.2 * np.median(finals[name]["rime"])


def test_mrime_cd_wins_most_paired_runs_overall(finals):
    wins = sum(int(np.sum(finals[n]["mrime-cd"] < finals[n]["rime"])) for n in CLASSICAL_NAMES)
    assert wins >= 0.75 * len(CLASSICAL_NAMES) * len(SEEDS)
