"""The benchmark problem type shared by all suites."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Problem:
    """A box-constrained minimisation problem.

    Calling the problem with a 1-D array returns a float; :meth:`evaluate_batch` evaluates the
    rows of a 2-D array (the optimisers use it when present).

    Attributes:
        name: identifier, e.g. ``"cec2022-F3-D10"``.
        dim: number of variables.
        lower: lower bounds, shape ``(dim,)``.
        upper: upper bounds, shape ``(dim,)``.
        f_opt: objective value at the global optimum, if known.
        x_opt: location of the global optimum, if known.
        batch_function: maps ``(m, dim)`` to ``(m,)``.
        info: free-form metadata (suite, function id, backend, ...).
    """

    name: str
    dim: int
    lower: np.ndarray
    upper: np.ndarray
    f_opt: float | None
    x_opt: np.ndarray | None
    batch_function: Callable[[np.ndarray], np.ndarray] = field(repr=False)
    info: dict = field(default_factory=dict, repr=False)

    def evaluate_batch(self, points: np.ndarray) -> np.ndarray:
        """Objective values of the rows of ``points``."""
        return np.asarray(self.batch_function(np.atleast_2d(points)), dtype=float)

    def __call__(self, x: np.ndarray) -> float:
        return float(self.evaluate_batch(np.asarray(x, dtype=float)[None, :])[0])
