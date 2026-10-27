"""Tests for the discrete static bridge (Sinkhorn)."""

import math

import pytest

from bridge_lab.gaussian import optimal_cross_covariance
from bridge_lab.sinkhorn import coupling_moments, discretize_gaussian, sinkhorn_bridge

M0, VAR0 = 0.0, 1.0
M1, VAR1 = 2.0, 1.5
EPS = 0.5  # bridge parameter; the grid kernel uses eps_k = 2 * EPS (see DESIGN.md)

GRID = [-6.0 + 0.25 * i for i in range(57)]


@pytest.fixture(scope="module")
def solved():
    mu = discretize_gaussian(GRID, M0, VAR0)
    nu = discretize_gaussian(GRID, M1, VAR1)
    return mu, nu, sinkhorn_bridge(GRID, mu, nu, eps=2.0 * EPS)


def test_marginals_match(solved):
    mu, nu, result = solved
    n = len(GRID)
    for i in range(n):
        assert abs(sum(result.pi[i]) - mu[i]) < 1e-8
    for j in range(n):
        assert abs(sum(result.pi[i][j] for i in range(n)) - nu[j]) < 1e-8
    assert result.row_error < 1e-8
    assert result.col_error < 1e-8


def test_coupling_is_a_probability(solved):
    _, _, result = solved
    total = sum(sum(row) for row in result.pi)
    assert total == pytest.approx(1.0, abs=1e-9)
    assert all(v >= 0.0 for row in result.pi for v in row)


def test_discrete_covariance_vs_closed_form(solved):
    """Empirical coupling covariance vs the closed-form c*.

    Grid-resolution tolerance: the discrete problem replaces the Gaussian
    marginals by normalized densities on a 0.25-spaced grid truncated to
    [-6, 8], so its covariance differs from the continuum c* by a small
    discretization error. We assert agreement to 0.05, and record the
    convention gap separately: against c* evaluated at the kernel's own
    parameter (eps_k = 1.0 instead of eps_k / 2 = 0.5) the discrepancy
    would be ~0.16 — the factor of 2 documented in docs/DESIGN.md.
    """
    _, _, result = solved
    _, _, cov = coupling_moments(GRID, result.pi)
    c_star = optimal_cross_covariance(VAR0, VAR1, EPS)
    assert abs(cov - c_star) < 0.05
    c_star_kernel_param = optimal_cross_covariance(VAR0, VAR1, 2.0 * EPS)
    assert abs(cov - c_star_kernel_param) > 0.1  # the documented factor-of-2 gap


def test_discretize_gaussian():
    p = discretize_gaussian(GRID, M0, VAR0)
    assert sum(p) == pytest.approx(1.0)
    mean = sum(pi * x for pi, x in zip(p, GRID))
    assert mean == pytest.approx(M0, abs=0.01)


def test_rejects_bad_marginal():
    mu = discretize_gaussian(GRID, M0, VAR0)
    bad = [0.5] + [0.0] * (len(GRID) - 1)  # sums to 0.5
    with pytest.raises(ValueError):
        sinkhorn_bridge(GRID, bad, mu, eps=1.0)
    with pytest.raises(ValueError):
        sinkhorn_bridge(GRID, mu, mu, eps=0.0)
    assert math.isfinite(mu[0])
