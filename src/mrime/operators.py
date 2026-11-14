"""Equation-level operators of RIME and MRIME-CD.

Every function is pure (apart from drawing from the generator it is given) and cites the
equation of the paper it implements. Equation numbers refer to

    S. Shi, L. Zhang, Y. Yin, X. Yang, H. Lee, "A modified RIME algorithm with covariance
    learning and diversity enhancement for numerical optimization", Cluster Computing 28:658
    (2025), https://doi.org/10.1007/s10586-025-05348-3

Arrays follow one convention: a population is ``(n, dim)``, one row per rime agent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------------------------
# Base RIME, Section 2
# --------------------------------------------------------------------------------------------


def round_half_away(value: float) -> float:
    """Round to the nearest integer with halves away from zero (the MATLAB ``round``)."""
    return math.copysign(math.floor(abs(value) + 0.5), value)


def exploration_probability(progress: float) -> float:
    """Eq. (7): ``E = sqrt(FEs / FEsmax)``, with ``progress = FEs / FEsmax`` clipped to [0, 1]."""
    return math.sqrt(min(max(progress, 0.0), 1.0))


def soft_rime_angle(progress: float, theta_mode: str = "paper") -> float:
    """Eq. (5): ``theta = pi * FEs / (10 * FEsmax)``.

    ``theta_mode="matlab"`` returns ``10 * pi * FEs / FEsmax`` instead, which is what the
    expression ``cos(pi*it/(FesMax/10))`` in the MATLAB files evaluates.
    """
    if theta_mode == "paper":
        return math.pi * progress / 10.0
    if theta_mode == "matlab":
        return 10.0 * math.pi * progress
    raise ValueError(f"unknown theta_mode {theta_mode!r}")


def soft_rime_step(progress: float, w: int = 5) -> float:
    """Eq. (6): ``beta = 1 - round(w * FEs / FEsmax) / w`` (a staircase from 1 down to 0)."""
    return 1.0 - round_half_away(w * progress) / w


def rime_factor(alpha: float, progress: float, w: int = 5, theta_mode: str = "paper") -> float:
    """The scalar ``alpha * cos(theta) * beta`` that multiplies the random step in Eq. (4).

    Args:
        alpha: random number in (-1, 1), drawn once per generation.
        progress: ``FEs / FEsmax``.
        w: number of steps of ``beta`` (Table 2: ``w = 5``).
        theta_mode: see :func:`soft_rime_angle`.
    """
    return alpha * math.cos(soft_rime_angle(progress, theta_mode)) * soft_rime_step(progress, w)


def soft_rime_positions(
    best: np.ndarray,
    factor: float,
    lower: np.ndarray,
    upper: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Eq. (4), right-hand side, for ``n`` agents and every dimension.

    Returns ``X_best,j + factor * (rand * (UB_j - LB_j) + LB_j)`` with an independent ``rand``
    per agent and dimension. The caller decides (``r1 < E``) which entries are used.
    """
    h = rng.random((n, best.shape[0]))
    return best + factor * (h * (upper - lower) + lower)


def normalize_fitness(fitness: np.ndarray) -> np.ndarray:
    """``nrom(Fit)`` of Eq. (8): the fitness vector divided by its Euclidean norm.

    This is what MATLAB's ``normr`` does to the row vector of fitness values in ``RIME.m``;
    a zero (or non-finite) norm yields zeros, so that no puncture happens.
    """
    norm = float(np.linalg.norm(fitness))
    if norm == 0.0 or not math.isfinite(norm):
        return np.zeros_like(fitness, dtype=float)
    return fitness / norm


def hard_rime_mask(
    normalized_fitness: np.ndarray, dim: int, rng: np.random.Generator
) -> np.ndarray:
    """Condition of Eq. (8): ``rand < nrom(Fit_i)``, drawn per agent and dimension."""
    return rng.random((normalized_fitness.shape[0], dim)) < normalized_fitness[:, None]


def clip_to_bounds(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Boundary absorption: every coordinate outside ``[LB_j, UB_j]`` is set to the bound."""
    return np.clip(points, lower, upper)


# --------------------------------------------------------------------------------------------
# Dominant group, Section 3.1
# --------------------------------------------------------------------------------------------


def rfdb_select(positions: np.ndarray, fitness: np.ndarray, rng: np.random.Generator) -> int:
    """Roulette fitness-distance balance selection of ``X_r`` (Section 3.1; ``matlab/RFDB.m``).

    Each candidate gets the score ``(1 - normalised fitness) + normalised L1 distance to the
    best candidate`` and one index is drawn with probability proportional to its score. When
    all fitness values are equal, or a value is not finite, a uniformly random index is
    returned, as in the MATLAB code.
    """
    n = fitness.shape[0]
    best = int(np.argmin(fitness))
    f_min, f_max = float(fitness.min()), float(fitness.max())
    degenerate = (
        f_min == f_max
        or not math.isfinite(float(fitness.sum()))
        or not math.isfinite(float(positions[best].sum()))
    )
    if degenerate:
        return int(rng.integers(n))
    distances = np.abs(positions - positions[best]).sum(axis=1)
    d_min, d_max = float(distances.min()), float(distances.max())
    norm_fitness = 1.0 - (fitness - f_min) / (f_max - f_min)
    norm_distance = (distances - d_min) / (d_max - d_min) if d_max > d_min else np.zeros(n)
    cumulative = np.cumsum(norm_fitness + norm_distance)
    threshold = rng.random() * cumulative[-1]
    return int(min(np.searchsorted(cumulative, threshold, side="left"), n - 1))


class DominantArchive:
    """First-in-first-out store of elite individuals retained over generations (Section 3.1).

    Every generation the best ``round(elite_fraction * N)`` agents of the current population
    are appended; when more than ``capacity`` entries are held, the oldest are discarded.
    """

    def __init__(self, capacity: int, dim: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.positions = np.empty((0, dim), dtype=float)
        self.fitness = np.empty((0,), dtype=float)

    def __len__(self) -> int:
        return self.fitness.shape[0]

    def push_elite(self, positions: np.ndarray, fitness: np.ndarray, fraction: float) -> None:
        """Append the best ``round(fraction * n)`` rows (at least one) and trim the oldest."""
        n_elite = max(1, int(round_half_away(fraction * fitness.shape[0])))
        order = np.argsort(fitness, kind="stable")[:n_elite]
        self.positions = np.concatenate([self.positions, positions[order]], axis=0)
        self.fitness = np.concatenate([self.fitness, fitness[order]])
        overflow = len(self) - self.capacity
        if overflow > 0:
            self.positions = self.positions[overflow:]
            self.fitness = self.fitness[overflow:]


@dataclass(frozen=True)
class DominantGroup:
    """The dominant group ``S`` around the roulette-selected reference ``X_r``.

    Attributes:
        positions: members of ``S``, nearest to ``X_r`` first, shape ``(|S|, dim)``.
        fitness: their objective values.
        reference: ``X_r``.
    """

    positions: np.ndarray
    fitness: np.ndarray
    reference: np.ndarray


def select_dominant_group(
    positions: np.ndarray,
    fitness: np.ndarray,
    rng: np.random.Generator,
    neighbour_fraction: float = 0.5,
) -> DominantGroup:
    """Roulette domain selection (Section 3.1).

    ``X_r`` is drawn with :func:`rfdb_select`; the ``floor(neighbour_fraction * n)`` candidates
    (at least one) with the smallest Euclidean distance to ``X_r`` form the dominant group.
    """
    reference = positions[rfdb_select(positions, fitness, rng)]
    distances = np.linalg.norm(positions - reference, axis=1)
    size = max(1, math.floor(neighbour_fraction * fitness.shape[0]))
    nearest = np.argsort(distances, kind="stable")[:size]
    return DominantGroup(positions[nearest], fitness[nearest], reference.copy())


# --------------------------------------------------------------------------------------------
# Gaussian model, Eqs. (9)-(12)
# --------------------------------------------------------------------------------------------


def rank_weights(size: int, offset: float = 1.0) -> np.ndarray:
    """Eq. (10): ``w_i = (ln(|S| + c) - ln(i)) / sum_k (ln(|S| + c) - ln(k))``, ``i = 1..|S|``.

    The paper prints ``c = 1`` and omits ``- ln(i)`` in the numerator, which would make all
    weights equal and not sum to one; the form above is the only reading under which "better
    individuals have a greater impact". The MATLAB code uses ``c = 0.5``.
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    raw = math.log(size + offset) - np.log(np.arange(1, size + 1))
    return raw / raw.sum()


def weighted_mean(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Eq. (9): ``X_mean = sum_i w_i X_i``."""
    return weights @ points


def covariance(points: np.ndarray, mean: np.ndarray, ddof: int = 0) -> np.ndarray:
    """Eq. (11): ``C = 1/|S| * sum_i (X_i - X_mean)(X_i - X_mean)^T``.

    ``ddof=1`` gives the ``1/(|S|-1)`` normalisation of the MATLAB code. A group that is too
    small for the chosen normalisation yields the zero matrix.
    """
    denominator = points.shape[0] - ddof
    dim = points.shape[1]
    if denominator <= 0:
        return np.zeros((dim, dim))
    centred = points - mean
    return centred.T @ centred / denominator


@dataclass(frozen=True)
class GaussianModel:
    """``Gaussian(X_mean, C)`` of Eq. (12) in eigen-decomposed form.

    Attributes:
        mean: ``X_mean``, shape ``(dim,)``.
        covariance: ``C`` of Eq. (11).
        basis: orthonormal eigenvectors of ``C`` in columns.
        scale: square roots of the (floored) eigenvalues; a sample is
            ``mean + basis @ (scale * z)`` with ``z ~ N(0, I)``.
    """

    mean: np.ndarray
    covariance: np.ndarray
    basis: np.ndarray
    scale: np.ndarray

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` points, shape ``(n, dim)``."""
        z = rng.standard_normal((n, self.mean.shape[0]))
        return self.mean + (z * self.scale) @ self.basis.T


def fit_gaussian_model(
    group: DominantGroup,
    *,
    weight_order: str = "fitness",
    weight_offset: float = 1.0,
    ddof: int = 0,
    eig_floor: float = 1e-7,
) -> GaussianModel:
    """Maximum-likelihood style Gaussian model of the dominant group, Eqs. (9)-(11).

    Args:
        group: output of :func:`select_dominant_group` (rows ordered by distance to ``X_r``).
        weight_order: ``"fitness"`` assigns the largest weight to the fittest member (paper);
            ``"distance"`` assigns it to the member nearest to ``X_r`` (MATLAB code).
        weight_offset: ``c`` of :func:`rank_weights`.
        ddof: see :func:`covariance`.
        eig_floor: added to every eigenvalue of ``C`` before the square root, as in the MATLAB
            code (``sqrt(diag(D + 1e-7*eye))``); keeps sampling possible when ``C`` is singular.
    """
    if weight_order == "fitness":
        order = np.argsort(group.fitness, kind="stable")
        points = group.positions[order]
    elif weight_order == "distance":
        points = group.positions
    else:
        raise ValueError(f"unknown weight_order {weight_order!r}")
    mean = weighted_mean(points, rank_weights(points.shape[0], weight_offset))
    cov = covariance(points, mean, ddof)
    cov = (cov + cov.T) / 2.0
    eigenvalues, basis = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    largest, smallest = float(eigenvalues[-1]), float(eigenvalues[0])
    if largest > 1e20 * smallest:
        # Condition-number guard of the MATLAB code (C <- C + tmp * I).
        eigenvalues = eigenvalues + (largest / 1e20 - smallest)
    return GaussianModel(mean, cov, basis, np.sqrt(eigenvalues + eig_floor))


def gcls_candidates(
    agents: np.ndarray, model: GaussianModel, rng: np.random.Generator
) -> np.ndarray:
    """Eq. (12): ``X_i_new = Gaussian(X_mean, C) + rand * (X_mean - X_i)``.

    One Gaussian sample and one scalar ``rand`` are drawn per agent.
    """
    n = agents.shape[0]
    return model.sample(n, rng) + rng.random((n, 1)) * (model.mean - agents)


# --------------------------------------------------------------------------------------------
# Average bootstrapping strategy, Eq. (14)
# --------------------------------------------------------------------------------------------


def abs_puncture(
    candidates: np.ndarray,
    hard_mask: np.ndarray,
    late_gate: np.ndarray,
    reference: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    """Eq. (14): hard-rime puncture with average bootstrapping.

    Where ``hard_mask`` (``rand < nrom(Fit_i)``) holds, coordinate ``j`` becomes
    ``reference_j`` if ``late_gate`` (``r1 < E``) holds and ``(reference_j + X_mean,j) / 2``
    otherwise. ``reference`` is ``X_best`` in the paper.

    Args:
        candidates: positions after the soft-rime / GCLS step, shape ``(n, dim)``.
        hard_mask: boolean ``(n, dim)``.
        late_gate: boolean, broadcastable to ``(n, dim)``.
        reference: shape ``(dim,)``.
        mean: ``X_mean``, shape ``(dim,)``.
    """
    late = hard_mask & late_gate
    early = hard_mask & ~late_gate
    out = np.where(late, reference, candidates)
    return np.where(early, (reference + mean) / 2.0, out)


# --------------------------------------------------------------------------------------------
# Stagnation indicator and SPDM, Eqs. (15)-(18)
# --------------------------------------------------------------------------------------------


def normalized_volume(population: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Eqs. (15)-(17): ``nVOL = sqrt(V_pop / V_lim)``.

    ``V_lim = sqrt(prod_j |UB_j - LB_j|)`` and
    ``V_pop = sqrt(prod_j |max_i X_ij - min_i X_ij| / 2)``.
    The products are evaluated in log space, which is identical in exact arithmetic and avoids
    overflow/underflow in high dimension; a population that is degenerate in any dimension
    gives ``nVOL = 0``.
    """
    span = population.max(axis=0) - population.min(axis=0)
    with np.errstate(divide="ignore"):
        log_ratio = np.sum(np.log(span / 2.0) - np.log(upper - lower))
    return float(np.exp(0.25 * log_ratio))


def spdm_candidates(
    population: np.ndarray,
    indices: np.ndarray,
    model: GaussianModel,
    rng: np.random.Generator,
) -> np.ndarray:
    """Eq. (18) for the stagnant agents ``indices``.

    ``X_i_new = Gaussian(X_mean, C) + rand * (X_r1 - X_i) + rand * (X_r2 - X_i)`` where ``X_r1``
    and ``X_r2`` are two different individuals drawn at random from the population (they may
    coincide with ``X_i``, as in the MATLAB code) and the two ``rand`` are independent scalars.
    """
    n = population.shape[0]
    m = indices.shape[0]
    r1 = rng.integers(n, size=m)
    r2 = (r1 + rng.integers(1, n, size=m)) % n if n > 1 else r1
    agents = population[indices]
    return (
        model.sample(m, rng)
        + rng.random((m, 1)) * (population[r1] - agents)
        + rng.random((m, 1)) * (population[r2] - agents)
    )
