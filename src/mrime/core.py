"""The RIME / MRIME-CD optimiser (paper Algorithm 1) with the ablation switches of Table 5.

One generation performs, in this order (Algorithm 1 and Fig. 2 of the paper):

1. ``E`` (Eq. 7) and the soft-rime factor (Eqs. 5-6) from the evaluation counter;
2. the dominant group, ``X_mean`` and ``C`` (Eqs. 9-11)                       [GCLS/ABS/SPDM];
3. for every agent: soft-rime search (Eq. 4) if ``r1 < E``, else the Gaussian covariance
   learning move (Eq. 12)                                                              [GCLS];
4. hard-rime puncture (Eq. 8), replaced by average bootstrapping (Eq. 14)               [ABS];
5. boundary absorption, evaluation, positive greedy selection, stagnation counters;
6. if ``nVOL`` (Eqs. 15-17) is below the threshold, stagnant agents are re-sampled with
   Eq. (18) and accepted greedily                                                      [SPDM].

Function-evaluation accounting is strict by default: the initial population, every candidate
and every SPDM candidate count against ``max_evals`` and the optimiser never exceeds it.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np

from mrime import operators as ops
from mrime.config import PAPER, MRIMEConfig, make_config, parse_algorithm
from mrime.problem import ArrayLike, Evaluator, Objective, OptimizeResult, as_bounds

logger = logging.getLogger(__name__)


def _label(config: MRIMEConfig) -> str:
    """Human-readable name of the strategy combination (paper Table 5)."""
    flags = (config.use_gcls, config.use_abs, config.use_spdm)
    if flags == (False, False, False):
        return "RIME"
    if flags == (True, True, True):
        return "MRIME-CD"
    return "RIME-" + "".join(c for c, on in zip("GAS", flags, strict=True) if on)


def run(
    objective: Objective,
    lower: ArrayLike,
    upper: ArrayLike,
    dim: int,
    pop_size: int,
    max_evals: int,
    seed: int | None = None,
    *,
    config: MRIMEConfig = PAPER,
) -> OptimizeResult:
    """Minimise ``objective`` inside the box ``[lower, upper]`` with a member of the RIME family.

    Args:
        objective: maps a 1-D array of length ``dim`` to a scalar. Objects that also provide
            ``evaluate_batch(X)`` are evaluated block-wise (same result, faster).
        lower: lower bound(s), scalar or length ``dim``.
        upper: upper bound(s), scalar or length ``dim``.
        dim: number of decision variables ``D``.
        pop_size: number of rime agents ``NP`` (the paper does not state it; the author's
            scripts use ``ceil(15 * D**(2/3))`` or ``ceil(30 * D**(2/3))``).
        max_evals: maximum number of objective evaluations ``FEsmax`` (paper: ``3000 * D``).
        seed: seed of the NumPy ``Generator``; the run is deterministic given the seed.
        config: strategy switches and constants, see :class:`mrime.config.MRIMEConfig`.

    Returns:
        :class:`mrime.problem.OptimizeResult`.

    Raises:
        ValueError: on inconsistent arguments.
    """
    if pop_size < 2:
        raise ValueError(f"pop_size must be >= 2, got {pop_size}")
    if max_evals < pop_size:
        raise ValueError(
            f"max_evals ({max_evals}) must cover at least the initial population ({pop_size})"
        )
    lo, hi = as_bounds(lower, upper, dim)
    rng = np.random.default_rng(seed)
    strict = config.accounting == "strict"
    evaluator = Evaluator(objective, max_evals if strict else None)
    n = pop_size

    # Eq. (1): uniform random initial population; Eq. (3): its fitness.
    population = lo + rng.random((n, dim)) * (hi - lo)
    fitness = evaluator.evaluate(population)
    # Schedule/termination counter. Strict mode: all evaluations so far. As-shipped mode: the
    # counter of matlab/MRIME_RL_Cov.m, which does not include the initial population.
    counter = n if strict else 0

    best_index = int(np.argmin(fitness))
    best_x = population[best_index].copy()
    best_f = float(fitness[best_index])

    stagnation = np.zeros(n, dtype=np.int64)
    archive = (
        ops.DominantArchive(config.archive_capacity_per_dim * dim, dim)
        if config.needs_dominant_group
        else None
    )
    usage = {"gcls_candidates": 0, "spdm_candidates": 0, "spdm_accepted": 0, "spdm_generations": 0}
    history_evals = [evaluator.n_evals]
    history_counter = [counter]
    history_best = [best_f]
    generations = 0

    while counter < max_evals:
        progress = counter / max_evals
        explore = ops.exploration_probability(progress)  # Eq. (7)
        factor = ops.rime_factor(  # Eqs. (5)-(6)
            rng.uniform(-1.0, 1.0), progress, config.w, config.theta_mode
        )
        norm_fit = ops.normalize_fitness(fitness)

        model = None
        reference = best_x
        if archive is not None:
            archive.push_elite(population, fitness, config.elite_fraction)
            group = ops.select_dominant_group(
                archive.positions, archive.fitness, rng, config.neighbour_fraction
            )
            model = ops.fit_gaussian_model(  # Eqs. (9)-(11)
                group,
                weight_order=config.weight_order,
                weight_offset=config.weight_offset,
                ddof=config.cov_ddof,
                eig_floor=config.eig_floor,
            )
            if config.abs_reference == "roulette":
                reference = group.reference

        # --- candidate generation: soft-rime search or GCLS, Eq. (13) ---
        soft = ops.soft_rime_positions(best_x, factor, lo, hi, n, rng)  # Eq. (4)
        if config.use_gcls:
            agent_soft = rng.random(n) < explore
            dim_mask = (rng.random((n, dim)) < explore) & agent_soft[:, None]
            candidates = np.where(dim_mask, soft, population)
            learners = ~agent_soft
            if learners.any():
                candidates[learners] = ops.gcls_candidates(  # Eq. (12)
                    population[learners], model, rng
                )
                usage["gcls_candidates"] += int(learners.sum())
            shared_gate = agent_soft[:, None]
        else:
            dim_mask = rng.random((n, dim)) < explore
            candidates = np.where(dim_mask, soft, population)
            shared_gate = dim_mask

        # --- hard-rime puncture, Eq. (8), or average bootstrapping, Eq. (14) ---
        hard = ops.hard_rime_mask(norm_fit, dim, rng)
        if config.use_abs:
            if config.abs_gate == "shared":
                late_gate = shared_gate
            else:
                late_gate = (rng.random(n) < explore)[:, None]
            candidates = ops.abs_puncture(candidates, hard, late_gate, reference, model.mean)
        else:
            candidates = np.where(hard, best_x, candidates)
        candidates = ops.clip_to_bounds(candidates, lo, hi)

        # --- evaluation and positive greedy selection ---
        if strict:
            n_eval = min(n, max_evals - counter)
            counter_gain = n_eval
        else:
            ticks = np.cumsum(rng.random(n) < config.shipped_counter_probability)
            reached = np.nonzero(counter + ticks >= max_evals)[0]
            n_eval = int(reached[0]) + 1 if reached.size else n
            counter_gain = int(ticks[n_eval - 1])
        new_fitness = evaluator.evaluate(candidates[:n_eval])
        improved = np.zeros(n, dtype=bool)
        improved[:n_eval] = new_fitness < fitness[:n_eval]
        population[improved] = candidates[improved]
        fitness[improved] = new_fitness[improved[:n_eval]]
        evaluated = np.zeros(n, dtype=bool)
        evaluated[:n_eval] = True
        stagnation[evaluated & ~improved] += 1  # Count_i of Algorithm 1
        counter += counter_gain

        # --- SPDM, Eqs. (15)-(18) ---
        if config.use_spdm and (evaluator.remaining > 0):
            nvol = ops.normalized_volume(population, lo, hi)
            if nvol < config.nvol_threshold:
                stagnant = stagnation > config.stagnation_factor * dim
                if config.spdm_skip_best:
                    stagnant[int(np.argmin(fitness))] = False
                indices = np.nonzero(stagnant)[0][: evaluator.remaining]
                if indices.size:
                    usage["spdm_generations"] += 1
                    usage["spdm_candidates"] += int(indices.size)
                    resampled = ops.clip_to_bounds(
                        ops.spdm_candidates(population, indices, model, rng), lo, hi
                    )
                    resampled_fitness = evaluator.evaluate(resampled)
                    better = resampled_fitness < fitness[indices]
                    population[indices[better]] = resampled[better]
                    fitness[indices[better]] = resampled_fitness[better]
                    stagnation[indices[better]] = 0
                    stagnation[indices[~better]] += 1
                    usage["spdm_accepted"] += int(better.sum())
                    if strict:
                        counter += int(indices.size)

        best_index = int(np.argmin(fitness))
        if fitness[best_index] < best_f:
            best_f = float(fitness[best_index])
            best_x = population[best_index].copy()
        generations += 1
        history_evals.append(evaluator.n_evals)
        history_counter.append(counter)
        history_best.append(best_f)

    label = _label(config)
    logger.debug("%s finished: f=%.6g after %d evaluations", label, best_f, evaluator.n_evals)
    return OptimizeResult(
        x=best_x,
        fun=best_f,
        n_evals=evaluator.n_evals,
        max_evals=max_evals,
        history_evals=np.asarray(history_evals, dtype=np.int64),
        history_counter=np.asarray(history_counter, dtype=np.int64),
        history_best=np.asarray(history_best, dtype=float),
        n_generations=generations,
        algorithm=label,
        seed=seed,
        extras={**usage, "accounting": config.accounting},
    )


def rime(
    objective: Objective,
    lower: ArrayLike,
    upper: ArrayLike,
    dim: int,
    pop_size: int,
    max_evals: int,
    seed: int | None = None,
    *,
    w: int = 5,
    theta_mode: str = "paper",
) -> OptimizeResult:
    """Base RIME algorithm (paper Section 2, Eqs. 1-8).

    Soft-rime search (Eq. 4) per dimension with probability ``E`` (Eq. 7), hard-rime puncture
    (Eq. 8) per dimension with probability ``nrom(Fit_i)``, boundary absorption and positive
    greedy selection. See :func:`run` for the arguments.
    """
    config = replace(make_config("rime"), w=w, theta_mode=theta_mode)
    return run(objective, lower, upper, dim, pop_size, max_evals, seed, config=config)


def mrime_cd(
    objective: Objective,
    lower: ArrayLike,
    upper: ArrayLike,
    dim: int,
    pop_size: int,
    max_evals: int,
    seed: int | None = None,
    *,
    config: MRIMEConfig = PAPER,
) -> OptimizeResult:
    """MRIME-CD (paper Section 3, Algorithm 1): RIME with GCLS, ABS and SPDM.

    ``config`` defaults to the paper profile with all three strategies enabled; pass
    :data:`mrime.config.MATLAB` to reproduce the operator-level behaviour of the author's
    MATLAB implementation. See :func:`run` for the other arguments.
    """
    return run(objective, lower, upper, dim, pop_size, max_evals, seed, config=config)


def minimize(
    objective: Objective,
    lower: ArrayLike,
    upper: ArrayLike,
    dim: int,
    pop_size: int,
    max_evals: int,
    seed: int | None = None,
    *,
    algorithm: str = "mrime-cd",
) -> OptimizeResult:
    """Run a named algorithm: ``"<variant>[@<profile>]"``.

    Variants are the rows of paper Table 5 (``rime``, ``rime-g``, ``rime-a``, ``rime-s``,
    ``rime-ga``, ``rime-gs``, ``rime-as``, ``mrime-cd``); profiles are ``paper`` (default),
    ``matlab`` and ``matlab-as-shipped``. The returned ``algorithm`` label repeats the spec.
    """
    variant, profile = parse_algorithm(algorithm)
    result = run(
        objective, lower, upper, dim, pop_size, max_evals, seed,
        config=make_config(variant, profile),
    )
    result.algorithm = algorithm
    return result
