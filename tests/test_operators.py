"""Each equation-level operator in isolation."""

import math

import numpy as np
import pytest

from mrime import operators as ops
from mrime.benchmarks.classical import classical_problem, ellipsoid_parameters

# ------------------------------------------------------------------ base RIME, Eqs. (4)-(8)


@pytest.mark.parametrize(
    ("value", "expected"), [(0.5, 1.0), (1.5, 2.0), (2.5, 3.0), (0.49, 0.0), (-0.5, -1.0)]
)
def test_round_half_away_matches_matlab_round(value, expected):
    assert ops.round_half_away(value) == expected


def test_exploration_probability_is_sqrt_of_progress_eq7():
    assert ops.exploration_probability(0.25) == pytest.approx(0.5)
    assert ops.exploration_probability(0.0) == 0.0
    assert ops.exploration_probability(1.7) == 1.0  # clipped


@pytest.mark.parametrize(
    ("progress", "expected"),
    [(0.0, 1.0), (0.09, 1.0), (0.11, 0.8), (0.29, 0.8), (0.31, 0.6), (0.69, 0.4), (0.89, 0.2),
     (0.91, 0.0), (1.0, 0.0)],
)
def test_soft_rime_step_is_a_staircase_eq6(progress, expected):
    assert ops.soft_rime_step(progress, w=5) == pytest.approx(expected)


def test_soft_rime_angle_paper_and_matlab_eq5():
    assert ops.soft_rime_angle(0.5, "paper") == pytest.approx(math.pi * 0.5 / 10)
    assert ops.soft_rime_angle(0.5, "matlab") == pytest.approx(10 * math.pi * 0.5)
    with pytest.raises(ValueError):
        ops.soft_rime_angle(0.5, "other")


def test_rime_factor_combines_alpha_cos_beta():
    progress = 0.42
    expected = -0.7 * math.cos(math.pi * progress / 10) * 0.6
    assert ops.rime_factor(-0.7, progress, w=5, theta_mode="paper") == pytest.approx(expected)
    # In the last 10 % of the budget beta = 0, so the soft-rime step vanishes.
    assert ops.rime_factor(0.9, 0.95) == 0.0


def test_soft_rime_positions_shape_and_range_eq4():
    rng = np.random.default_rng(0)
    best = np.array([1.0, -2.0, 3.0])
    lower, upper = np.full(3, -10.0), np.full(3, 10.0)
    out = ops.soft_rime_positions(best, 0.5, lower, upper, 200, rng)
    assert out.shape == (200, 3)
    # best + 0.5 * U[-10, 10]
    assert np.all(out >= best - 5.0) and np.all(out <= best + 5.0)
    assert np.ptp(out, axis=0).min() > 8.0  # the whole interval is used


def test_normalize_fitness_is_unit_euclidean_norm():
    fitness = np.array([3.0, 4.0])
    assert np.allclose(ops.normalize_fitness(fitness), [0.6, 0.8])
    assert np.allclose(ops.normalize_fitness(np.zeros(4)), 0.0)
    assert np.allclose(ops.normalize_fitness(np.array([1.0, np.inf])), 0.0)


def test_hard_rime_mask_probability_follows_normalised_fitness_eq8():
    rng = np.random.default_rng(1)
    mask = ops.hard_rime_mask(np.array([0.0, 1.0, 0.25]), 4000, rng)
    assert mask.shape == (3, 4000)
    assert not mask[0].any() and mask[1].all()
    assert mask[2].mean() == pytest.approx(0.25, abs=0.03)


def test_clip_to_bounds_absorbs_at_the_boundary():
    lower, upper = np.array([-1.0, 0.0]), np.array([1.0, 2.0])
    points = np.array([[-5.0, 1.0], [0.5, 9.0]])
    assert np.array_equal(ops.clip_to_bounds(points, lower, upper), [[-1.0, 1.0], [0.5, 2.0]])


# ------------------------------------------------------------------ dominant group, Sec. 3.1


def test_rfdb_roulette_probabilities():
    # scores: (1 - norm fitness) + norm L1 distance to the best = [1, 0, 1.5]
    positions = np.array([[0.0, 0.0], [0.0, 0.0], [4.0, 0.0]])
    fitness = np.array([0.0, 10.0, 5.0])
    rng = np.random.default_rng(2)
    picks = np.array([ops.rfdb_select(positions, fitness, rng) for _ in range(4000)])
    assert not np.any(picks == 1)
    assert np.mean(picks == 2) == pytest.approx(0.6, abs=0.03)


def test_rfdb_falls_back_to_uniform_choice_when_degenerate():
    rng = np.random.default_rng(3)
    positions = rng.normal(size=(6, 2))
    picks = {ops.rfdb_select(positions, np.ones(6), rng) for _ in range(300)}
    assert picks == set(range(6))
    assert 0 <= ops.rfdb_select(positions, np.array([1, 2, np.inf, 4, 5, 6.0]), rng) < 6


def test_dominant_archive_keeps_elite_and_is_first_in_first_out():
    archive = ops.DominantArchive(capacity=5, dim=1)
    first = np.arange(10.0)[:, None]
    archive.push_elite(first, fitness=np.arange(10.0), fraction=0.3)  # round(3.0) = 3 best
    assert archive.fitness.tolist() == [0.0, 1.0, 2.0]
    second = 100.0 + np.arange(10.0)[:, None]
    archive.push_elite(second, fitness=np.arange(10.0)[::-1].copy(), fraction=0.3)
    # six entries were pushed in total; the oldest one (fitness 0) is dropped
    assert len(archive) == 5
    assert archive.positions[:, 0].tolist() == [1.0, 2.0, 109.0, 108.0, 107.0]


def test_dominant_archive_elite_count_rounds_half_away_and_is_at_least_one():
    archive = ops.DominantArchive(capacity=50, dim=1)
    archive.push_elite(np.zeros((5, 1)), np.arange(5.0), fraction=0.3)  # round(1.5) = 2
    assert len(archive) == 2
    archive.push_elite(np.zeros((2, 1)), np.arange(2.0), fraction=0.1)  # round(0.2) = 0 -> 1
    assert len(archive) == 3


def test_select_dominant_group_takes_the_nearest_half():
    rng = np.random.default_rng(4)
    positions = rng.uniform(-1, 1, size=(11, 3))
    fitness = rng.uniform(size=11)
    group = ops.select_dominant_group(positions, fitness, rng, neighbour_fraction=0.5)
    assert group.positions.shape == (5, 3)  # floor(0.5 * 11)
    distances = np.linalg.norm(group.positions - group.reference, axis=1)
    assert distances[0] == 0.0  # X_r itself is the nearest member
    assert np.all(np.diff(distances) >= 0)
    all_distances = np.sort(np.linalg.norm(positions - group.reference, axis=1))
    assert np.allclose(distances, all_distances[:5])


# ------------------------------------------------------------------ Gaussian model, Eqs. (9)-(12)


@pytest.mark.parametrize("offset", [1.0, 0.5])
def test_rank_weights_eq10(offset):
    weights = ops.rank_weights(6, offset)
    raw = np.log(6 + offset) - np.log(np.arange(1, 7))
    assert np.allclose(weights, raw / raw.sum())
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(np.diff(weights) < 0)  # earlier ranks weigh more
    assert np.allclose(ops.rank_weights(1, offset), [1.0])
    with pytest.raises(ValueError):
        ops.rank_weights(0)


def test_weighted_mean_eq9():
    points = np.array([[0.0, 0.0], [2.0, 4.0]])
    assert np.allclose(ops.weighted_mean(points, np.array([0.75, 0.25])), [0.5, 1.0])


def test_covariance_eq11_normalisation():
    rng = np.random.default_rng(5)
    points = rng.normal(size=(40, 3))
    mean = points.mean(axis=0)
    assert np.allclose(ops.covariance(points, mean, ddof=0), np.cov(points.T, bias=True))
    assert np.allclose(ops.covariance(points, mean, ddof=1), np.cov(points.T))
    # about an arbitrary centre it is the plain average of outer products
    centre = np.array([0.3, -0.2, 1.0])
    manual = sum(np.outer(p - centre, p - centre) for p in points) / 40
    assert np.allclose(ops.covariance(points, centre), manual)
    assert np.array_equal(ops.covariance(points[:1], points[0], ddof=1), np.zeros((3, 3)))


def test_fit_gaussian_model_weight_order():
    positions = np.array([[0.0], [1.0], [2.0]])  # ordered by distance to X_r
    fitness = np.array([5.0, 1.0, 3.0])
    group = ops.DominantGroup(positions, fitness, positions[0])
    weights = ops.rank_weights(3, 1.0)
    by_distance = ops.fit_gaussian_model(group, weight_order="distance")
    by_fitness = ops.fit_gaussian_model(group, weight_order="fitness")
    assert by_distance.mean[0] == pytest.approx(weights @ np.array([0.0, 1.0, 2.0]))
    assert by_fitness.mean[0] == pytest.approx(weights @ np.array([1.0, 2.0, 0.0]))
    with pytest.raises(ValueError):
        ops.fit_gaussian_model(group, weight_order="random")


def test_gaussian_model_recovers_a_known_covariance():
    rng = np.random.default_rng(6)
    dim = 5
    rotation, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    eigenvalues = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    true_cov = (rotation * eigenvalues) @ rotation.T
    centre = rng.uniform(-3, 3, size=dim)
    points = centre + rng.standard_normal((6000, dim)) @ (rotation * np.sqrt(eigenvalues)).T
    group = ops.DominantGroup(points, np.zeros(6000), points[0])
    model = ops.fit_gaussian_model(group, weight_order="distance", eig_floor=0.0)
    assert np.allclose(model.scale**2, eigenvalues, rtol=0.1)  # eigh returns ascending order
    for k in range(dim):
        assert abs(model.basis[:, k] @ rotation[:, k]) > 0.99
    assert np.allclose(model.covariance, true_cov, atol=0.08 * eigenvalues.max())
    # sampling reproduces the fitted covariance
    samples = model.sample(40000, rng)
    assert samples.shape == (40000, dim)
    assert np.allclose(np.cov(samples.T), model.covariance, atol=0.06 * eigenvalues.max())


def test_covariance_learning_finds_the_eigen_directions_of_a_quadratic():
    """On f(x) = sum_k lambda_k z_k^2, z = R^T (x - o), the best points fill an ellipsoid whose
    axes are the Hessian eigenvectors R[:, k] with variances proportional to 1 / lambda_k; the
    learnt covariance must have exactly these eigen-directions and variance ratios."""
    dim = 4
    rotation, hessian_eigenvalues, shift = ellipsoid_parameters(dim, condition=100.0)
    rng = np.random.default_rng(7)
    population = shift + rng.uniform(-50.0, 50.0, size=(100_000, dim))
    fitness = np.sum(hessian_eigenvalues * ((population - shift) @ rotation) ** 2, axis=1)
    elite = np.argsort(fitness)[:200]  # a level set that lies inside the sampled box
    group = ops.DominantGroup(population[elite], fitness[elite], population[elite[0]])
    model = ops.fit_gaussian_model(group, weight_order="fitness", eig_floor=0.0)
    assert np.allclose(model.mean, shift, atol=3.0)
    # eigh sorts the variances ascending, the Hessian eigenvalues are ascending as well, so the
    # k-th largest variance belongs to the k-th smallest Hessian eigenvalue.
    for k in range(dim):
        assert abs(model.basis[:, dim - 1 - k] @ rotation[:, k]) > 0.97
    variances = (model.scale**2)[::-1]
    ratios = variances * hessian_eigenvalues  # constant if variance_k ~ 1 / lambda_k
    assert ratios.max() / ratios.min() < 1.6


def test_ellipsoid_problem_matches_its_documented_parameters():
    problem = classical_problem("ellipsoid", 4)
    rotation, eigenvalues, shift = ellipsoid_parameters(4)
    assert np.allclose(rotation.T @ rotation, np.eye(4))
    assert problem(shift) == 0.0
    for k in range(4):  # moving one unit along eigenvector k costs lambda_k
        assert problem(shift + rotation[:, k]) == pytest.approx(eigenvalues[k])


def test_fit_gaussian_model_survives_a_degenerate_group():
    point = np.array([[1.0, 2.0, 3.0]])
    group = ops.DominantGroup(point, np.array([0.5]), point[0])
    for ddof in (0, 1):
        model = ops.fit_gaussian_model(group, ddof=ddof, eig_floor=1e-7)
        assert np.allclose(model.mean, point[0])
        assert np.allclose(model.scale, math.sqrt(1e-7))
        assert np.all(np.isfinite(model.sample(5, np.random.default_rng(0))))


def test_gcls_candidates_eq12_with_zero_covariance():
    # C = 0 and no eigenvalue floor: X_new = X_mean + rand * (X_mean - X_i), rand in [0, 1)
    mean = np.array([1.0, -1.0])
    group = ops.DominantGroup(np.tile(mean, (4, 1)), np.zeros(4), mean)
    model = ops.fit_gaussian_model(group, eig_floor=0.0)
    agents = np.array([[3.0, 3.0], [-2.0, 0.0], [2.0, 5.0]])
    out = ops.gcls_candidates(agents, model, np.random.default_rng(8))
    assert out.shape == agents.shape
    ratio = (out - mean) / (mean - agents)
    assert np.allclose(ratio[:, 0], ratio[:, 1])  # one scalar rand per agent
    assert np.all((ratio >= 0) & (ratio < 1))


def test_gcls_candidates_are_centred_between_mean_and_reflection():
    rng = np.random.default_rng(9)
    points = rng.normal(size=(500, 3))
    model = ops.fit_gaussian_model(ops.DominantGroup(points, np.zeros(500), points[0]))
    agents = np.tile(np.array([4.0, 0.0, -4.0]), (20000, 1))
    out = ops.gcls_candidates(agents, model, rng)
    expected = model.mean + 0.5 * (model.mean - agents[0])  # E[rand] = 0.5
    assert np.allclose(out.mean(axis=0), expected, atol=0.06)


# ------------------------------------------------------------------ ABS, Eq. (14)


def test_abs_puncture_eq14():
    candidates = np.zeros((2, 3))
    hard = np.array([[True, False, True], [True, True, False]])
    late = np.array([[True], [False]])  # agent 0: r1 < E, agent 1: r1 >= E
    reference = np.array([10.0, 20.0, 30.0])
    mean = np.array([0.0, 40.0, 0.0])
    out = ops.abs_puncture(candidates, hard, late, reference, mean)
    assert np.array_equal(out[0], [10.0, 0.0, 30.0])  # X_best,j where punctured
    assert np.array_equal(out[1], [5.0, 30.0, 0.0])  # (X_best,j + X_mean,j) / 2


def test_abs_puncture_accepts_a_per_dimension_gate():
    hard = np.ones((1, 2), dtype=bool)
    late = np.array([[True, False]])
    out = ops.abs_puncture(np.zeros((1, 2)), hard, late, np.array([2.0, 2.0]), np.array([4.0, 4.0]))
    assert np.array_equal(out, [[2.0, 3.0]])


# ------------------------------------------------------------------ SPDM, Eqs. (15)-(18)


@pytest.mark.parametrize("dim", [2, 10, 100])
def test_normalized_volume_closed_form_eqs15_17(dim):
    lower, upper = np.full(dim, -100.0), np.full(dim, 100.0)
    fraction = 0.3  # the population spans 30 % of the range in every dimension
    population = np.vstack([np.full(dim, -30.0), np.full(dim, 30.0), np.zeros(dim)])
    expected = (fraction / 2.0) ** (dim / 4.0)
    assert ops.normalized_volume(population, lower, upper) == pytest.approx(expected, rel=1e-9)


def test_normalized_volume_matches_the_literal_formula():
    rng = np.random.default_rng(10)
    lower, upper = np.array([-5.0, 0.0, -1.0]), np.array([5.0, 2.0, 3.0])
    population = lower + rng.random((7, 3)) * (upper - lower)
    v_lim = math.sqrt(np.prod(np.abs(upper - lower)))
    v_pop = math.sqrt(np.prod(np.abs(population.max(0) - population.min(0)) / 2.0))
    nvol = ops.normalized_volume(population, lower, upper)
    assert nvol == pytest.approx(math.sqrt(v_pop / v_lim))


def test_normalized_volume_is_zero_for_a_collapsed_dimension():
    population = np.array([[0.0, 1.0], [0.0, 2.0]])
    assert ops.normalized_volume(population, np.full(2, -1.0), np.full(2, 3.0)) == 0.0


def test_spdm_candidates_eq18():
    mean = np.array([1.0, 2.0])
    model = ops.fit_gaussian_model(
        ops.DominantGroup(np.tile(mean, (3, 1)), np.zeros(3), mean), eig_floor=0.0
    )
    rng = np.random.default_rng(11)
    # identical agents: both difference terms vanish and the candidate is X_mean exactly
    population = np.tile(np.array([7.0, 7.0]), (6, 1))
    out = ops.spdm_candidates(population, np.array([0, 3, 5]), model, rng)
    assert out.shape == (3, 2)
    assert np.allclose(out, mean)
    # distinct agents: X_new - X_mean lies in the span of (X_r1 - X_i) and (X_r2 - X_i)
    population = np.array([[0.0, 0.0], [1.0, 0.0]])
    out = ops.spdm_candidates(population, np.array([0]), model, rng)
    assert out[0, 1] == pytest.approx(mean[1])  # no component along the second axis
    assert 0.0 <= out[0, 0] - mean[0] < 1.0  # r1 != r2, so exactly one of them is agent 1
