"""Checks of the CEC benchmark backends against the reference C code in ``matlab/``."""

from __future__ import annotations

import numpy as np

from mrime.analysis import markdown_table
from mrime.benchmarks import suite_functions
from mrime.benchmarks.cec import CEC2017_DIMS, CEC2022_DIMS, cec_problem


def native_optimum_gaps() -> dict[tuple[str, int], float]:
    """Largest ``|f(x*) - f*|`` of the native backend per (suite, dimension).

    ``x*`` is the documented optimum location (the shift vector; for CEC2017 paper-F8 the
    point ``o + M^-1 1``) and ``f*`` the documented optimum value (``100 * official number``
    for CEC2017, paper Table 4 for CEC2022).
    """
    gaps = {}
    for suite, dims in (("cec2017", CEC2017_DIMS), ("cec2022", CEC2022_DIMS)):
        for dim in dims:
            worst = 0.0
            for function in suite_functions(suite):
                problem = cec_problem(suite, function, dim, backend="native")
                worst = max(worst, abs(problem(problem.x_opt) - problem.f_opt))
            gaps[(suite, dim)] = worst
    return gaps


def opfunu_rows(points: int = 5, seed: int = 0, tolerance: float = 1e-9) -> list[list[object]]:
    """Compare the ``opfunu`` backend with the native backend on random points.

    For every function the two backends are evaluated at ``points`` uniform random points of
    ``[-100, 100]^D``; the row reports the largest relative difference, whether it is below
    ``tolerance``, and the reference value at the point opfunu declares to be the optimum.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for suite, dim in (("cec2017", 10), ("cec2017", 30), ("cec2022", 10), ("cec2022", 20)):
        for function in suite_functions(suite):
            native = cec_problem(suite, function, dim, backend="native")
            other = cec_problem(suite, function, dim, backend="opfunu")
            sample = rng.uniform(-100.0, 100.0, size=(points, dim))
            a, b = native.evaluate_batch(sample), other.evaluate_batch(sample)
            difference = float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(a))))
            gap = native(other.x_opt) - native.f_opt
            rows.append([
                suite, dim, f"F{function}", native.info["title"], f"{difference:.1e}",
                "yes" if difference < tolerance else "no", f"{gap:.3g}",
            ])
    return rows


def verify_backends(points: int = 5, seed: int = 0) -> str:
    """Markdown report of both checks (the opfunu part is skipped if it is not installed)."""
    lines = ["Native backend: largest |f(x*) - f*| per suite and dimension", ""]
    lines.append(markdown_table(
        ["suite", "D", "max abs gap"],
        [[suite, dim, f"{gap:.3g}"] for (suite, dim), gap in native_optimum_gaps().items()],
    ))
    try:
        rows = opfunu_rows(points, seed)
    except ImportError as error:
        lines.append(f"opfunu comparison skipped: {error}")
        return "\n".join(lines)
    matching = sum(row[5] == "yes" for row in rows)
    lines.append(
        f"opfunu vs native on {points} random points per function: {matching} of {len(rows)} "
        "(suite, D, function) cases agree to 1e-9 relative"
    )
    lines.append("")
    lines.append(markdown_table(
        ["suite", "D", "function", "name", "max rel. difference", "identical",
         "reference f - f* at opfunu's optimum"],
        rows,
    ))
    return "\n".join(lines)
