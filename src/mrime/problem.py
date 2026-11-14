"""Bounds handling, strict evaluation accounting and the optimiser result type."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Objective = Callable[[np.ndarray], float]
ArrayLike = float | Sequence[float] | np.ndarray


class BudgetExceededError(RuntimeError):
    """Raised when an optimiser asks for more objective evaluations than its budget allows."""


def as_bounds(lower: ArrayLike, upper: ArrayLike, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast scalar or per-dimension bounds to two float arrays of shape ``(dim,)``.

    Raises:
        ValueError: if ``dim`` is not positive, a bound has the wrong length, a bound is not
            finite, or ``lower >= upper`` in any dimension.
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    lo = np.array(np.broadcast_to(np.asarray(lower, dtype=float), (dim,)))
    hi = np.array(np.broadcast_to(np.asarray(upper, dtype=float), (dim,)))
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("bounds must be finite")
    if np.any(lo >= hi):
        raise ValueError("lower bound must be strictly below upper bound in every dimension")
    return lo, hi


class Evaluator:
    """Wraps an objective, counts every call and enforces a hard evaluation budget.

    The objective is called with a 1-D array and must return a scalar. If the objective object
    exposes ``evaluate_batch(X)`` (as the built-in benchmark problems do) it is used to evaluate
    a whole ``(m, dim)`` block in one call; the count still increases by ``m``.

    NaN objective values are mapped to ``+inf`` so that greedy selection never accepts them.

    Args:
        objective: function to minimise.
        max_evals: hard budget; ``None`` disables the check (used only by the
            ``matlab-as-shipped`` accounting mode, which is documented to overrun).
    """

    def __init__(self, objective: Objective, max_evals: int | None) -> None:
        self._objective = objective
        self._batch = getattr(objective, "evaluate_batch", None)
        self.max_evals = max_evals
        self.n_evals = 0

    @property
    def remaining(self) -> int:
        """Evaluations left in the budget (a large number when the budget is disabled)."""
        if self.max_evals is None:
            return np.iinfo(np.int64).max
        return self.max_evals - self.n_evals

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the rows of ``points`` (shape ``(m, dim)``) and return ``m`` values."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        m = points.shape[0]
        if m > self.remaining:
            raise BudgetExceededError(
                f"requested {m} evaluations with only {self.remaining} left of {self.max_evals}"
            )
        if self._batch is not None:
            values = np.asarray(self._batch(points), dtype=float).reshape(-1)
            if values.shape[0] != m:
                raise ValueError("evaluate_batch must return one value per row")
        else:
            values = np.array([float(self._objective(row)) for row in points], dtype=float)
        values = np.where(np.isnan(values), np.inf, values)
        self.n_evals += m
        return values


@dataclass
class OptimizeResult:
    """Outcome of one optimiser run.

    Attributes:
        x: best position found, shape ``(dim,)``.
        fun: objective value at ``x``.
        n_evals: true number of objective calls performed.
        max_evals: the budget that was requested.
        history_evals: true evaluation count after the initial population and after every
            generation.
        history_counter: value of the schedule/termination counter at the same instants. It
            equals ``history_evals`` under strict accounting and differs only in the
            ``matlab-as-shipped`` mode.
        history_best: best objective value found so far at the same instants (non-increasing).
        n_generations: number of completed generations.
        algorithm: label of the algorithm variant.
        seed: seed the run was started with.
        extras: operator usage counters (e.g. number of GCLS and SPDM candidates evaluated).
    """

    x: np.ndarray
    fun: float
    n_evals: int
    max_evals: int
    history_evals: np.ndarray
    history_counter: np.ndarray
    history_best: np.ndarray
    n_generations: int
    algorithm: str
    seed: int | None
    extras: dict[str, Any] = field(default_factory=dict)

    def best_at(self, counter_values: ArrayLike) -> np.ndarray:
        """Best-so-far value at the given schedule-counter values (step interpolation).

        Counter values earlier than the first record return the first record.
        """
        query = np.atleast_1d(np.asarray(counter_values, dtype=float))
        idx = np.searchsorted(self.history_counter, query, side="right") - 1
        idx = np.clip(idx, 0, len(self.history_best) - 1)
        return self.history_best[idx]
