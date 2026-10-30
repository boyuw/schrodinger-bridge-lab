"""Tests for Schrödinger potentials and the h-function (dynamic bridge, part 1)."""

import math

import pytest

from bridge_lab.potentials import h_values, log_potentials, reconstruct_coupling
from bridge_lab.sinkhorn import discretize_gaussian, kernel_matrix, sinkhorn_bridge

M0, VAR0 = 0.0, 1.0
M1, VAR1 = 2.0, 1.5
EPS = 0.5  # bridge parameter; the grid kernel uses eps_k = 2 * EPS
EPS_K = 2.0 * EPS

GRID = [-6.0 + 0.25 * i for i in range(57)]


@pytest.fixture(scope="module")
def solved():
    mu = discretize_gaussian(GRID, M0, VAR0)
    nu = discretize_gaussian(GRID, M1, VAR1)
    result = sinkhorn_bridge(GRID, mu, nu, eps=EPS_K)
    return mu, nu, result


def test_log_potentials_reconstruct_coupling(solved):
    _, _, result = solved
    log_a, log_b = log_potentials(result)
    rebuilt = reconstruct_coupling(GRID, EPS_K, log_a, log_b)
    for i in range(len(GRID)):
        for j in range(len(GRID)):
            assert rebuilt[i][j] == pytest.approx(result.pi[i][j], rel=1e-9, abs=1e-15)


def test_h_at_terminal_time_is_b(solved):
    _, _, result = solved
    h = h_values(GRID, result.b, EPS_K, t=1.0)
    assert h == pytest.approx(result.b)


def test_h_at_time_zero_is_mu_over_a(solved):
    """h(0, x_i) = (K b)_i = mu_i / a_i, the Sinkhorn update for a."""
    mu, _, result = solved
    h = h_values(GRID, result.b, EPS_K, t=0.0)
    for i, (hi, mui, ai) in enumerate(zip(h, mu, result.a)):
        assert hi == pytest.approx(mui / ai, rel=1e-9)


def test_h_at_time_zero_matches_kernel_expectation(solved):
    """Independent check of h(0): direct (K b) matrix-vector product."""
    _, _, result = solved
    K = kernel_matrix(GRID, EPS_K)
    h = h_values(GRID, result.b, EPS_K, t=0.0)
    for i in range(len(GRID)):
        direct = sum(K[i][j] * result.b[j] for j in range(len(GRID)))
        assert h[i] == pytest.approx(direct, rel=1e-12)


def test_h_is_positive_and_finite_at_mid_bridge(solved):
    _, _, result = solved
    h = h_values(GRID, result.b, EPS_K, t=0.5)
    assert all(math.isfinite(v) and v > 0.0 for v in h)
    # h(0.5, .) is a smoothed version of b: it must not equal b, and it
    # must vary across the grid (the terminal potential is not constant).
    assert h != pytest.approx(result.b)
    assert max(h) > min(h)


def test_h_rejects_bad_inputs(solved):
    _, _, result = solved
    with pytest.raises(ValueError):
        h_values(GRID, result.b, EPS_K, t=1.5)
    with pytest.raises(ValueError):
        h_values(GRID, result.b, 0.0, t=0.5)
    with pytest.raises(ValueError):
        h_values(GRID, result.b[:-1], EPS_K, t=0.5)
