"""Classical test functions, built in so that the package and its tests need no data files.

All functions are vectorised over the rows of an ``(m, dim)`` array. The ``ellipsoid`` problem
is a shifted, rotated, ill-conditioned quadratic whose rotation and shift are generated
deterministically, so the eigen-directions of its Hessian are known; it is used to test the
covariance-learning operator.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from mrime.benchmarks.base import Problem

BatchFunction = Callable[[np.ndarray], np.ndarray]


def sphere(x: np.ndarray) -> np.ndarray:
    """``sum_j x_j^2``; minimum 0 at the origin."""
    return np.sum(x**2, axis=1)


def schwefel_2_22(x: np.ndarray) -> np.ndarray:
    """``sum_j |x_j| + prod_j |x_j|``; minimum 0 at the origin."""
    a = np.abs(x)
    return np.sum(a, axis=1) + np.prod(a, axis=1)


def rosenbrock(x: np.ndarray) -> np.ndarray:
    """``sum_j 100 (x_{j+1} - x_j^2)^2 + (x_j - 1)^2``; minimum 0 at ``(1, ..., 1)``."""
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1.0) ** 2, axis=1)


def rastrigin(x: np.ndarray) -> np.ndarray:
    """``sum_j x_j^2 - 10 cos(2 pi x_j) + 10``; minimum 0 at the origin."""
    return np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0, axis=1)


def ackley(x: np.ndarray) -> np.ndarray:
    """Ackley's function; minimum 0 at the origin."""
    d = x.shape[1]
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2, axis=1) / d))
    term2 = -np.exp(np.sum(np.cos(2.0 * np.pi * x), axis=1) / d)
    return term1 + term2 + 20.0 + np.e


def griewank(x: np.ndarray) -> np.ndarray:
    """Griewank's function; minimum 0 at the origin."""
    j = np.arange(1, x.shape[1] + 1)
    return np.sum(x**2, axis=1) / 4000.0 - np.prod(np.cos(x / np.sqrt(j)), axis=1) + 1.0


def zakharov(x: np.ndarray) -> np.ndarray:
    """``sum x_j^2 + (sum 0.5 j x_j)^2 + (sum 0.5 j x_j)^4``; minimum 0 at the origin."""
    j = np.arange(1, x.shape[1] + 1)
    s = np.sum(0.5 * j * x, axis=1)
    return np.sum(x**2, axis=1) + s**2 + s**4


def ellipsoid_parameters(
    dim: int, condition: float = 1e4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic rotation ``R``, eigenvalues and shift of the ``ellipsoid`` problem.

    Returns:
        ``(rotation, eigenvalues, shift)`` with ``rotation`` orthonormal (columns are the
        Hessian eigenvectors), ``eigenvalues[k] = condition ** (k / (dim - 1))`` and
        ``shift`` in ``[-40, 40]^dim``.
    """
    rng = np.random.default_rng(20250903 + dim)
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    rotation = q * np.sign(np.diag(r))
    exponents = np.arange(dim) / max(dim - 1, 1)
    eigenvalues = condition**exponents
    shift = rng.uniform(-40.0, 40.0, size=dim)
    return rotation, eigenvalues, shift


def _ellipsoid(dim: int) -> tuple[BatchFunction, np.ndarray]:
    rotation, eigenvalues, shift = ellipsoid_parameters(dim)

    def function(x: np.ndarray) -> np.ndarray:
        z = (x - shift) @ rotation
        return np.sum(eigenvalues * z**2, axis=1)

    return function, shift


# name -> (function, lower bound, upper bound, optimum location as a callable of dim)
_CLASSICAL: dict[str, tuple[BatchFunction, float, float, Callable[[int], np.ndarray]]] = {
    "sphere": (sphere, -100.0, 100.0, np.zeros),
    "schwefel_2_22": (schwefel_2_22, -10.0, 10.0, np.zeros),
    "rosenbrock": (rosenbrock, -30.0, 30.0, np.ones),
    "rastrigin": (rastrigin, -5.12, 5.12, np.zeros),
    "ackley": (ackley, -32.0, 32.0, np.zeros),
    "griewank": (griewank, -600.0, 600.0, np.zeros),
    "zakharov": (zakharov, -5.0, 10.0, np.zeros),
}

#: Names of the built-in problems, in suite order (``F1`` .. ``F8`` of the ``classical`` suite).
CLASSICAL_NAMES: tuple[str, ...] = (*_CLASSICAL, "ellipsoid")


def classical_problem(name: str, dim: int) -> Problem:
    """Build one of the built-in problems (see :data:`CLASSICAL_NAMES`)."""
    if dim < 2:
        raise ValueError("classical problems need dim >= 2")
    if name == "ellipsoid":
        function, shift = _ellipsoid(dim)
        lower, upper, x_opt = -100.0, 100.0, shift
    elif name in _CLASSICAL:
        function, lower, upper, optimum = _CLASSICAL[name]
        x_opt = optimum(dim)
    else:
        raise ValueError(f"unknown classical problem {name!r}; choose from {CLASSICAL_NAMES}")
    return Problem(
        name=f"classical-{name}-D{dim}",
        dim=dim,
        lower=np.full(dim, lower),
        upper=np.full(dim, upper),
        f_opt=0.0,
        x_opt=np.asarray(x_opt, dtype=float),
        batch_function=function,
        info={
            "suite": "classical",
            "function": CLASSICAL_NAMES.index(name) + 1,
            "title": name,
            "backend": "builtin",
        },
    )
