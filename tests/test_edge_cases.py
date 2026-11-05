"""Edge-case tests: identical marginals, eps extremes, path time endpoints.

These sit at the boundary of the regimes the main suites exercise: the
degenerate case where source and target coincide, the two eps limits of
the static bridge, and the exact endpoint behaviour of sampled paths.
"""

import math
import random
import statistics

import pytest

from bridge_lab.gaussian import (
    gaussian_bridge_marginal,
    optimal_cross_covariance,
    sample_coupling,
)
from bridge_lab.simulate import sample_at_time, sample_paths
from bridge_lab.sinkhorn import discretize_gaussian, sinkhorn_bridge

GRID = [-2.0, -1.0, 0.0, 1.0, 2.0]


# --- Identical marginals -------------------------------------------------


def test_identical_marginals_sinkhorn_is_symmetric():
    # With mu == nu and a symmetric kernel, the unique static bridge is
    # symmetric: pi^T solves the same problem, so pi^T == pi.
    mu = discretize_gaussian(GRID, 0.0, 1.0)
    result = sinkhorn_bridge(GRID, mu, mu, eps=0.5)
    for i in range(len(GRID)):
        for j in range(len(GRID)):
            assert result.pi[i][j] == pytest.approx(result.pi[j][i], abs=1e-6)


def test_identical_marginals_gaussian_midpoint():
    # Mean stays put; variance = var + 2 t (1-t) (c* - var) + eps t (1-t)
    # with c* < var, so the bridge still bulges above var at mid-time.
    mean, var = gaussian_bridge_marginal(1.0, 2.0, 1.0, 2.0, 0.5, 0.5)
    assert mean == pytest.approx(1.0, abs=1e-12)
    c_star = optimal_cross_covariance(2.0, 2.0, 0.5)
    assert var == pytest.approx(2.0 + 0.5 * (c_star - 2.0) + 0.125, abs=1e-12)
    assert var > 2.0


def test_identical_marginals_path_endpoints_share_distribution():
    rng = random.Random(5)
    paths = sample_paths(1.0, 2.0, 1.0, 2.0, 0.5, [0.0, 0.5, 1.0], n=8_000, rng=rng)
    starts = [p[0] for p in paths]
    ends = [p[-1] for p in paths]
    assert statistics.fmean(starts) == pytest.approx(1.0, abs=0.08)
    assert statistics.fmean(ends) == pytest.approx(1.0, abs=0.08)
    assert statistics.variance(starts) == pytest.approx(2.0, abs=0.15)
    assert statistics.variance(ends) == pytest.approx(2.0, abs=0.15)


# --- eps extremes --------------------------------------------------------


def test_eps_zero_gaussian_bridge_is_comonotone():
    # eps = 0: c* = sqrt(var0 var1), noise term vanishes, and X_t is the
    # comonotone interpolation (1-t) X + t Y.
    c_star = optimal_cross_covariance(1.0, 1.5, 0.0)
    assert c_star == pytest.approx(math.sqrt(1.5), abs=1e-12)
    rng = random.Random(9)
    samples = sample_at_time(0.0, 1.0, 2.0, 1.5, 0.0, t=0.5, n=20_000, rng=rng)
    _, var = gaussian_bridge_marginal(0.0, 1.0, 2.0, 1.5, 0.0, 0.5)
    assert var == pytest.approx(0.25 + 0.375 + 0.5 * math.sqrt(1.5), abs=1e-12)
    assert statistics.fmean(samples) == pytest.approx(1.0, abs=0.05)
    assert statistics.variance(samples) == pytest.approx(var, abs=0.1)


def test_tiny_eps_sinkhorn_is_near_diagonal():
    # Kernel exp(-d^2 / eps) with eps = 0.01 is ~identity on a unit grid,
    # so the bridge barely moves mass off the diagonal.
    mu = discretize_gaussian(GRID, 0.0, 1.0)
    result = sinkhorn_bridge(GRID, mu, mu, eps=0.01)
    off_diag = sum(
        result.pi[i][j] for i in range(len(GRID)) for j in range(len(GRID)) if i != j
    )
    assert off_diag < 1e-6
    for i in range(len(GRID)):
        assert result.pi[i][i] == pytest.approx(mu[i], abs=1e-6)


def test_huge_eps_sinkhorn_is_near_product():
    # Kernel -> all ones, so the bridge -> the independent product mu x nu.
    mu = discretize_gaussian(GRID, 0.0, 1.0)
    nu = discretize_gaussian(GRID, 0.5, 1.5)
    result = sinkhorn_bridge(GRID, mu, nu, eps=1e4)
    for i in range(len(GRID)):
        for j in range(len(GRID)):
            assert result.pi[i][j] == pytest.approx(mu[i] * nu[j], abs=1e-3)


# --- Path time endpoints ---------------------------------------------------


def test_paths_endpoints_are_exactly_the_coupling_draws():
    # On the two-point grid [0, 1] the Brownian bridge vanishes at both
    # endpoints, so path[0] and path[-1] must equal the coupling draws
    # from the same seed — exactly, not just in distribution.
    rng_paths = random.Random(123)
    paths = sample_paths(0.0, 1.0, 2.0, 1.5, 0.5, [0.0, 1.0], n=50, rng=rng_paths)
    rng_pairs = random.Random(123)
    pairs = sample_coupling(0.0, 1.0, 2.0, 1.5, 0.5, 50, rng_pairs)
    assert [p[0] for p in paths] == [x for x, _ in pairs]
    assert [p[-1] for p in paths] == [y for _, y in pairs]


def test_paths_endpoints_exact_on_fine_grid():
    # The same exactness holds whatever the interior grid is: the bridge
    # term w_t - t w_1 is 0 at t = 0 and t = 1 by construction.
    times = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0]
    rng_paths = random.Random(77)
    paths = sample_paths(0.0, 1.0, 2.0, 1.5, 0.5, times, n=50, rng=rng_paths)
    rng_pairs = random.Random(77)
    pairs = sample_coupling(0.0, 1.0, 2.0, 1.5, 0.5, 50, rng_pairs)
    assert [p[0] for p in paths] == [x for x, _ in pairs]
    assert [p[-1] for p in paths] == [y for _, y in pairs]
    assert all(len(p) == len(times) for p in paths)


def test_paths_endpoints_correlation_matches_closed_form():
    # With only [0, 1], each path is one coupling pair: the start/end
    # correlation must recover c* (seeded, ~4 standard errors).
    rng = random.Random(31)
    paths = sample_paths(0.0, 1.0, 2.0, 1.5, 0.5, [0.0, 1.0], n=20_000, rng=rng)
    starts = [p[0] for p in paths]
    ends = [p[-1] for p in paths]
    cov = statistics.fmean(
        [(x - 0.0) * (y - 2.0) for x, y in zip(starts, ends)]
    )
    assert cov == pytest.approx(optimal_cross_covariance(1.0, 1.5, 0.5), abs=0.05)


def test_sample_at_time_rejects_bad_t():
    rng = random.Random(0)
    with pytest.raises(ValueError):
        sample_at_time(0.0, 1.0, 2.0, 1.5, 0.5, t=-0.1, n=10, rng=rng)
    with pytest.raises(ValueError):
        sample_at_time(0.0, 1.0, 2.0, 1.5, 0.5, t=1.1, n=10, rng=rng)
