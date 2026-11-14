"""The optimiser loop: evaluation accounting, bounds, determinism, convergence, switches."""

import numpy as np
import pytest

from mrime import (
    MATLAB,
    PAPER,
    VARIANTS,
    MRIMEConfig,
    make_config,
    minimize,
    mrime_cd,
    parse_algorithm,
    rime,
    run,
)
from mrime.benchmarks import get_problem
from mrime.problem import BudgetExceededError, Evaluator, as_bounds


class Recorder:
    """Objective that records every point it is asked to evaluate."""

    def __init__(self, shift: float = 1.5) -> None:
        self.points: list[np.ndarray] = []
        self.shift = shift

    def __call__(self, x: np.ndarray) -> float:
        self.points.append(np.array(x))
        return float(np.sum((x - self.shift) ** 2))


@pytest.mark.parametrize("profile", ["paper", "matlab"])
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_strict_evaluation_accounting_and_bounds(variant, profile):
    objective = Recorder()
    dim, pop, budget = 4, 17, 1234  # the budget is deliberately not a multiple of the population
    lower, upper = np.array([-5.0, -1.0, 0.0, -3.0]), np.array([5.0, 4.0, 2.0, 3.0])
    result = minimize(objective, lower, upper, dim, pop, budget, seed=3,
                      algorithm=f"{variant}@{profile}")
    points = np.array(objective.points)
    assert len(points) == budget == result.n_evals == result.max_evals
    assert result.history_evals[0] == pop and result.history_evals[-1] == budget
    assert np.all(points >= lower) and np.all(points <= upper)
    values = np.sum((points - objective.shift) ** 2, axis=1)
    assert result.fun == pytest.approx(values.min())
    assert objective(result.x) == pytest.approx(result.fun)
    assert np.all(np.diff(result.history_best) <= 0)
    assert np.all(np.diff(result.history_evals) > 0)
    assert np.array_equal(result.history_evals, result.history_counter)
    assert result.history_best[-1] == result.fun
    assert len(result.history_best) == result.n_generations + 1


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_operator_usage_follows_the_strategy_switches(variant):
    gcls, _, spdm = VARIANTS[variant]
    problem = get_problem("classical", "sphere", 5)
    result = minimize(problem, problem.lower, problem.upper, 5, 20, 6000, seed=0, algorithm=variant)
    assert (result.extras["gcls_candidates"] > 0) == gcls
    assert (result.extras["spdm_candidates"] > 0) == spdm
    assert result.extras["spdm_accepted"] <= result.extras["spdm_candidates"]
    assert result.n_evals == 6000


def test_runs_are_deterministic_given_the_seed():
    problem = get_problem("classical", "rosenbrock", 6)
    args = (problem, problem.lower, problem.upper, 6, 24, 3000)
    first = mrime_cd(*args, seed=11)
    second = mrime_cd(*args, seed=11)
    other = mrime_cd(*args, seed=12)
    assert np.array_equal(first.x, second.x) and first.fun == second.fun
    assert np.array_equal(first.history_best, second.history_best)
    assert first.fun != other.fun


def test_plain_callable_and_batch_problem_give_the_same_run():
    problem = get_problem("classical", "ackley", 5)

    def plain(x: np.ndarray) -> float:
        return problem(x)

    a = mrime_cd(problem, problem.lower, problem.upper, 5, 20, 2500, seed=4)
    b = mrime_cd(plain, problem.lower, problem.upper, 5, 20, 2500, seed=4)
    assert a.fun == b.fun and np.array_equal(a.x, b.x)


def test_named_entry_points_agree():
    problem = get_problem("classical", "sphere", 5)
    args = (problem, -100.0, 100.0, 5, 20, 2000)
    assert rime(*args, seed=1).fun == minimize(*args, seed=1, algorithm="rime").fun
    assert mrime_cd(*args, seed=1).fun == minimize(*args, seed=1, algorithm="mrime-cd").fun
    assert (
        mrime_cd(*args, seed=1, config=MATLAB).fun
        == minimize(*args, seed=1, algorithm="mrime-cd@matlab").fun
    )
    assert rime(*args, seed=1).algorithm == "RIME"
    assert mrime_cd(*args, seed=1).algorithm == "MRIME-CD"
    assert run(*args, seed=1, config=make_config("rime-ga")).algorithm == "RIME-GA"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mrime_cd_converges_on_simple_functions(seed):
    sphere = get_problem("classical", "sphere", 5)
    result = mrime_cd(sphere, sphere.lower, sphere.upper, 5, 30, 15000, seed=seed)
    assert result.fun < 1e-6
    assert np.allclose(result.x, 0.0, atol=1e-2)
    ellipsoid = get_problem("classical", "ellipsoid", 5)
    result = mrime_cd(ellipsoid, ellipsoid.lower, ellipsoid.upper, 5, 30, 15000, seed=seed)
    assert result.fun < 1e-2
    assert np.allclose(result.x, ellipsoid.x_opt, atol=0.5)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rime_makes_progress_on_the_sphere(seed):
    sphere = get_problem("classical", "sphere", 5)
    result = rime(sphere, sphere.lower, sphere.upper, 5, 30, 15000, seed=seed)
    assert result.fun < 1e-3 * result.history_best[0]


@pytest.mark.parametrize("pop_size", [2, 3, 4])
@pytest.mark.parametrize("variant", ["rime", "rime-a", "rime-s", "mrime-cd"])
def test_tiny_populations_do_not_break(pop_size, variant):
    problem = get_problem("classical", "sphere", 3)
    result = minimize(problem, -100, 100, 3, pop_size, 200, seed=0, algorithm=variant)
    assert result.n_evals == 200 and np.isfinite(result.fun)


def test_nan_objective_values_are_never_selected():
    def objective(x: np.ndarray) -> float:
        return float("nan") if x[0] > 0 else float(np.sum(x**2))

    result = mrime_cd(objective, -10, 10, 3, 12, 1500, seed=0)
    assert np.isfinite(result.fun) and result.x[0] <= 0


def test_invalid_arguments():
    def objective(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    with pytest.raises(ValueError, match="pop_size"):
        rime(objective, -1, 1, 3, 1, 100)
    with pytest.raises(ValueError, match="max_evals"):
        rime(objective, -1, 1, 3, 10, 5)
    with pytest.raises(ValueError, match="lower"):
        rime(objective, 1, -1, 3, 10, 100)
    with pytest.raises(ValueError):
        rime(objective, [-1, -1], [1, 1], 3, 10, 100)  # wrong bound length
    with pytest.raises(ValueError, match="unknown variant"):
        minimize(objective, -1, 1, 3, 10, 100, algorithm="pso")
    with pytest.raises(ValueError, match="unknown profile"):
        minimize(objective, -1, 1, 3, 10, 100, algorithm="rime@octave")


def test_config_validation_and_profiles():
    assert PAPER.theta_mode == "paper" and PAPER.weight_offset == 1.0 and PAPER.cov_ddof == 0
    assert PAPER.abs_reference == "best" and PAPER.abs_gate == "shared"
    assert PAPER.nvol_threshold == 0.01 and PAPER.accounting == "strict"
    assert MATLAB.theta_mode == "matlab" and MATLAB.weight_offset == 0.5 and MATLAB.cov_ddof == 1
    assert MATLAB.weight_order == "distance" and MATLAB.abs_reference == "roulette"
    assert MATLAB.abs_gate == "independent" and MATLAB.nvol_threshold == 1e-3
    assert MATLAB.accounting == "strict"
    assert parse_algorithm("MRIME-CD") == ("mrime-cd", "paper")
    assert parse_algorithm("rime-ga@matlab") == ("rime-ga", "matlab")
    config = make_config("rime-gs", nvol_threshold=0.1)
    assert (config.use_gcls, config.use_abs, config.use_spdm) == (True, False, True)
    assert config.nvol_threshold == 0.1
    for bad in ({"w": 0}, {"theta_mode": "x"}, {"cov_ddof": 2}, {"elite_fraction": 0.0},
                {"weight_order": "x"}, {"abs_reference": "x"}, {"abs_gate": "x"},
                {"accounting": "x"}, {"weight_offset": 0.0}):
        with pytest.raises(ValueError):
            MRIMEConfig(**bad)
    with pytest.raises(ValueError, match="only defined for the full MRIME-CD"):
        make_config("rime-ga", "matlab-as-shipped")


def test_matlab_as_shipped_accounting_overruns_the_budget_and_says_so():
    objective = Recorder()
    budget = 1500
    result = minimize(objective, -10, 10, 3, 20, budget, seed=0,
                      algorithm="mrime-cd@matlab-as-shipped")
    assert result.extras["accounting"] == "matlab-as-shipped"
    assert result.n_evals == len(objective.points)
    # the counter advances with probability 0.1 per evaluation: about ten times the budget
    assert result.n_evals > 5 * budget
    assert result.history_counter[-1] >= budget
    assert result.history_counter[0] == 0  # the initial population is not counted
    assert result.max_evals == budget


def test_strict_accounting_never_overruns():
    for algorithm in ("rime", "mrime-cd", "mrime-cd@matlab", "rime-s"):
        objective = Recorder()
        result = minimize(objective, -10, 10, 3, 20, 777, seed=5, algorithm=algorithm)
        assert len(objective.points) == 777 == result.n_evals


def test_best_at_is_a_step_function_of_the_counter():
    problem = get_problem("classical", "sphere", 4)
    result = rime(problem, -100, 100, 4, 10, 100, seed=0)
    assert result.best_at(0)[0] == result.history_best[0]  # before the first record
    assert result.best_at(10)[0] == result.history_best[0]
    assert result.best_at(19.9)[0] == result.history_best[0]
    assert result.best_at(20)[0] == result.history_best[1]
    assert result.best_at(1e9)[0] == result.fun
    grid = result.best_at(np.linspace(0, 100, 11))
    assert grid.shape == (11,) and np.all(np.diff(grid) <= 0)


def test_evaluator_enforces_the_budget():
    evaluator = Evaluator(lambda x: float(x.sum()), max_evals=5)
    assert evaluator.evaluate(np.ones((3, 2))).tolist() == [2.0, 2.0, 2.0]
    assert evaluator.remaining == 2
    with pytest.raises(BudgetExceededError):
        evaluator.evaluate(np.ones((3, 2)))
    assert evaluator.n_evals == 3


def test_as_bounds_broadcasts_and_validates():
    lower, upper = as_bounds(-1.0, [1.0, 2.0, 3.0], 3)
    assert lower.tolist() == [-1.0, -1.0, -1.0] and upper.tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        as_bounds(0.0, np.inf, 2)
    with pytest.raises(ValueError):
        as_bounds(0.0, 1.0, 0)
