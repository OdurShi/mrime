"""CEC2017 and CEC2022 benchmark suites as used in the paper (Tables 3 and 4).

Two backends are available:

``native`` (default)
    The reference C code shipped in ``matlab/`` (the code behind the MEX functions the paper's
    experiments call), compiled on demand; see :mod:`mrime.benchmarks.native`.

``opfunu``
    The pure-Python ``opfunu`` package (optional dependency). It is *not* numerically identical
    to the reference code: see ``docs/implementation_notes.md`` ("Benchmark backends") and
    ``scripts/verify_cec_backends.py``. Its CEC2017 biases are shifted here so that the optimum
    values equal the official ones.

Function identifiers follow the paper: CEC2017 ``F1..F29`` (the official suite without the
withdrawn ``F2``, i.e. paper ``Fk`` is official function ``k + 1`` for ``k >= 2``) and CEC2022
``F1..F12``.
"""

from __future__ import annotations

import warnings

import numpy as np

from mrime.benchmarks.base import Problem

CEC2017_DIMS: tuple[int, ...] = (10, 30, 50, 100)
CEC2022_DIMS: tuple[int, ...] = (10, 20)

#: Optimum values of CEC2022 F1..F12 (paper Table 4).
CEC2022_F_OPT: tuple[float, ...] = (
    300.0, 400.0, 600.0, 800.0, 900.0, 1800.0, 2000.0, 2200.0, 2300.0, 2400.0, 2600.0, 2700.0,
)

_CEC2017_TITLES: tuple[str, ...] = (
    "Shifted and Rotated Bent Cigar",
    "Shifted and Rotated Zakharov",
    "Shifted and Rotated Rosenbrock",
    "Shifted and Rotated Rastrigin",
    "Shifted and Rotated Expanded Schaffer F6",
    "Shifted and Rotated Lunacek Bi-Rastrigin",
    "Shifted and Rotated Non-Continuous Rastrigin",
    "Shifted and Rotated Levy",
    "Shifted and Rotated Schwefel",
    *(f"Hybrid Function {i}" for i in range(1, 11)),
    *(f"Composition Function {i}" for i in range(1, 11)),
)

_CEC2022_TITLES: tuple[str, ...] = (
    "Shifted and full Rotated Zakharov",
    "Shifted and full Rotated Rosenbrock",
    "Shifted and full Rotated Expanded Schaffer F6",
    "Shifted and full Rotated Non-Continuous Rastrigin",
    "Shifted and full Rotated Levy",
    "Hybrid Function 1",
    "Hybrid Function 2",
    "Hybrid Function 3",
    "Composition Function 1",
    "Composition Function 2",
    "Composition Function 3",
    "Composition Function 4",
)


def cec2017_official_number(function: int) -> int:
    """Official CEC2017 function number of paper function ``F<function>`` (1..29)."""
    if not 1 <= function <= 29:
        raise ValueError(f"CEC2017 function id must be in 1..29, got {function}")
    return 1 if function == 1 else function + 1


def cec_f_opt(suite: str, function: int) -> float:
    """Optimum value ``f*``: ``100 * official number`` for CEC2017, Table 4 for CEC2022."""
    if suite == "cec2017":
        return 100.0 * cec2017_official_number(function)
    if suite == "cec2022":
        if not 1 <= function <= 12:
            raise ValueError(f"CEC2022 function id must be in 1..12, got {function}")
        return CEC2022_F_OPT[function - 1]
    raise ValueError(f"unknown CEC suite {suite!r}")


def _needs_shuffle(suite: str, official: int) -> bool:
    if suite == "cec2017":
        return 11 <= official <= 20 or official in (29, 30)
    return official in (6, 7, 8)


def _read_shift(path, dim: int) -> np.ndarray:
    """First ``dim`` numbers of a CEC shift file (the optimum of the first component)."""
    with open(path, encoding="ascii") as handle:
        tokens = handle.readline().split()
    return np.array([float(token) for token in tokens[:dim]])


def _native_problem(suite: str, function: int, dim: int, lower: float, upper: float) -> Problem:
    from mrime.benchmarks.native import load_suite

    official = cec2017_official_number(function) if suite == "cec2017" else function
    library = load_suite(suite)
    if not library.has_data(official, dim, _needs_shuffle(suite, official)):
        raise ValueError(
            f"{suite} F{function} is not defined for D={dim} "
            f"(no data files in {library.data_dir})"
        )

    def batch(points: np.ndarray) -> np.ndarray:
        return library.evaluate(points, official)

    x_opt = _read_shift(library.data_dir / f"shift_data_{official}.txt", dim)
    if suite == "cec2017" and official == 9:
        # The reference Levy function uses w = 1 + (z - 1)/4, so its minimum is at z = 1,
        # i.e. x* = o + M^-1 * 1, not at the shift vector o.
        rotation = np.loadtxt(library.data_dir / f"M_{official}_D{dim}.txt").reshape(dim, dim)
        x_opt = x_opt + np.linalg.solve(rotation, np.ones(dim))

    titles = _CEC2017_TITLES if suite == "cec2017" else _CEC2022_TITLES
    return Problem(
        name=f"{suite}-F{function}-D{dim}",
        dim=dim,
        lower=np.full(dim, lower),
        upper=np.full(dim, upper),
        f_opt=cec_f_opt(suite, function),
        x_opt=x_opt,
        batch_function=batch,
        info={
            "suite": suite,
            "function": function,
            "official_number": official,
            "title": titles[function - 1],
            "backend": "native",
        },
    )


def _opfunu_problem(suite: str, function: int, dim: int, lower: float, upper: float) -> Problem:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from opfunu.cec_based import cec2017, cec2022
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the 'opfunu' backend needs the optional dependency: pip install 'mrime[cec]' "
            "(opfunu 1.0.4 also needs setuptools<81 for pkg_resources)"
        ) from error

    module, year = (cec2017, 2017) if suite == "cec2017" else (cec2022, 2022)
    official = cec2017_official_number(function) if suite == "cec2017" else function
    instance = getattr(module, f"F{function}{year}")(ndim=dim)
    f_opt = cec_f_opt(suite, function)
    # opfunu numbers CEC2017 F1..F29 with bias 100*k; the official optimum is 100*(k+1), k>=2.
    bias_correction = f_opt - float(instance.f_bias)

    def batch(points: np.ndarray) -> np.ndarray:
        return np.array([instance.evaluate(row) for row in points]) + bias_correction

    titles = _CEC2017_TITLES if suite == "cec2017" else _CEC2022_TITLES
    return Problem(
        name=f"{suite}-F{function}-D{dim}",
        dim=dim,
        lower=np.full(dim, lower),
        upper=np.full(dim, upper),
        f_opt=f_opt,
        x_opt=np.asarray(instance.x_global, dtype=float)[:dim],
        batch_function=batch,
        info={
            "suite": suite,
            "function": function,
            "official_number": official,
            "title": titles[function - 1],
            "backend": "opfunu",
        },
    )


def cec_problem(
    suite: str,
    function: int,
    dim: int,
    backend: str = "native",
    matlab_bounds: bool = False,
) -> Problem:
    """Build CEC2017 ``F1..F29`` or CEC2022 ``F1..F12`` in the paper's numbering.

    Args:
        suite: ``"cec2017"`` or ``"cec2022"``.
        function: paper function id.
        dim: problem dimension (paper: 10/30/50/100 for CEC2017, 10/20 for CEC2022).
        backend: ``"native"`` or ``"opfunu"``.
        matlab_bounds: the search range is ``[-100, 100]^D`` (paper Tables 3-4). With
            ``True``, CEC2022 F11 uses ``[-600, 600]^D`` as in
            ``matlab/Get_Functions_cec2022.m``.
    """
    cec_f_opt(suite, function)  # validates suite and function id
    lower, upper = -100.0, 100.0
    if matlab_bounds and suite == "cec2022" and function == 11:
        lower, upper = -600.0, 600.0
    if backend == "native":
        return _native_problem(suite, function, dim, lower, upper)
    if backend == "opfunu":
        return _opfunu_problem(suite, function, dim, lower, upper)
    raise ValueError(f"unknown backend {backend!r}; choose 'native' or 'opfunu'")
