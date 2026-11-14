"""Benchmark suites: built-in classical functions and the CEC2017 / CEC2022 suites."""

from __future__ import annotations

from mrime.benchmarks.base import Problem
from mrime.benchmarks.cec import CEC2017_DIMS, CEC2022_DIMS, cec_f_opt, cec_problem
from mrime.benchmarks.classical import CLASSICAL_NAMES, classical_problem

SUITES: tuple[str, ...] = ("classical", "cec2017", "cec2022")


def suite_functions(suite: str) -> list[int]:
    """Function ids of a suite: 1..8 (classical), 1..29 (CEC2017), 1..12 (CEC2022)."""
    if suite == "classical":
        return list(range(1, len(CLASSICAL_NAMES) + 1))
    if suite == "cec2017":
        return list(range(1, 30))
    if suite == "cec2022":
        return list(range(1, 13))
    raise ValueError(f"unknown suite {suite!r}; choose from {SUITES}")


def get_problem(
    suite: str,
    function: int | str,
    dim: int,
    backend: str = "native",
    matlab_bounds: bool = False,
) -> Problem:
    """Build problem ``function`` of ``suite`` in dimension ``dim``.

    ``function`` is the paper's function id (an integer); for the classical suite a name from
    :data:`mrime.benchmarks.classical.CLASSICAL_NAMES` is accepted as well. ``backend`` and
    ``matlab_bounds`` only apply to the CEC suites, see :func:`mrime.benchmarks.cec.cec_problem`.
    """
    if suite == "classical":
        if isinstance(function, str):
            name = function
        else:
            if not 1 <= function <= len(CLASSICAL_NAMES):
                raise ValueError(f"classical function id must be in 1..{len(CLASSICAL_NAMES)}")
            name = CLASSICAL_NAMES[function - 1]
        return classical_problem(name, dim)
    if suite in ("cec2017", "cec2022"):
        return cec_problem(suite, int(function), dim, backend, matlab_bounds)
    raise ValueError(f"unknown suite {suite!r}; choose from {SUITES}")


__all__ = [
    "CEC2017_DIMS",
    "CEC2022_DIMS",
    "CLASSICAL_NAMES",
    "SUITES",
    "Problem",
    "cec_f_opt",
    "cec_problem",
    "classical_problem",
    "get_problem",
    "suite_functions",
]
