"""Tests for the Doob h-transform drift (dynamic bridge, part 2).

The anchor is exact: for a point-mass terminal potential the log
h-function is quadratic in x, so the central-difference gradient on a
uniform grid is exact and the drift must reduce to the Brownian-bridge
pull -(x - x_j) / (1 - t). The remaining tests check the t = 1 endpoint
against an independent gradient computation, the sign of the drift
around a peaked potential, and input validation.
"""

import math

import pytest

from bridge_lab.drift import drift_values, gradient_log_h, log_h_values
from bridge_lab.potentials import h_values
from bridge_lab.sinkhorn import discretize_gaussian, sinkhorn_bridge

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
    return result


def test_point_mass_drift_is_bridge_pull():
    """With b a point mass at x_j, drift = -(x - x_j) / (1 - t).

    Here h(t, x) = exp(-(x - x_j)^2 / (eps_k (1 - t))), so
    log h is quadratic, the uniform-grid central difference is exact,
    and (eps_k / 2) * grad log h = -(x - x_j) / (1 - t). Only interior
    points are checked: the one-sided end differences are not exact
    for a quadratic.
    """
    grid = [-2.0 + 0.25 * i for i in range(17)]  # -2 .. 2, step 0.25
    target = 0.5
    j = grid.index(target)
    b = [1.0 if k == j else 0.0 for k in range(len(grid))]
    eps_k, t = 1.0, 0.5
    drift = drift_values(grid, b, eps_k, t)
    for i in range(1, len(grid) - 1):
        assert drift[i] == pytest.approx(
            -(grid[i] - target) / (1.0 - t), rel=1e-9, abs=1e-12
        )


def test_drift_at_terminal_time_is_grad_log_b(solved):
    """At t = 1, drift = (eps_k / 2) * grad log b, recomputed by hand."""
    log_b = [math.log(v) for v in solved.b]
    drift = drift_values(GRID, solved.b, EPS_K, t=1.0)
    expected = gradient_log_h(GRID, log_b)
    for d, e in zip(drift, expected):
        assert d == pytest.approx((EPS_K / 2.0) * e, rel=1e-12)


def test_drift_points_toward_peaked_potential():
    """A terminal potential peaked at x = 1 pulls particles toward 1.

    b is a Gaussian bump centred at 1; its kernel smoothing h at t = 0.5
    stays peaked near 1, so the drift is positive well left of the peak
    and negative well right of it.
    """
    grid = [-3.0 + 0.25 * i for i in range(25)]  # -3 .. 3
    b = [math.exp(-((x - 1.0) ** 2)) for x in grid]
    drift = drift_values(grid, b, 1.0, t=0.5)
    for x, d in zip(grid, drift):
        if x <= -1.0:
            assert d > 0.0
        if x >= 2.5:
            assert d < 0.0


def test_drift_on_solved_bridge_is_finite(solved):
    for t in [0.0, 0.25, 0.5, 0.9, 1.0]:
        drift = drift_values(GRID, solved.b, EPS_K, t=t)
        assert len(drift) == len(GRID)
        assert all(math.isfinite(d) for d in drift)


def test_log_h_matches_log_of_h(solved):
    log_h = log_h_values(GRID, solved.b, EPS_K, t=0.5)
    h = h_values(GRID, solved.b, EPS_K, t=0.5)
    for lh, hv in zip(log_h, h):
        assert lh == pytest.approx(math.log(hv), rel=1e-12)


def test_drift_rejects_bad_inputs(solved):
    with pytest.raises(ValueError):
        drift_values(GRID, solved.b, EPS_K, t=1.5)
    with pytest.raises(ValueError):
        drift_values(GRID, solved.b, 0.0, t=0.5)
    with pytest.raises(ValueError):
        drift_values(GRID, solved.b[:-1], EPS_K, t=0.5)
    with pytest.raises(ValueError):
        # All-zero terminal potential: h is identically zero, log h undefined.
        drift_values(GRID, [0.0] * len(GRID), EPS_K, t=0.5)
    with pytest.raises(ValueError):
        gradient_log_h([0.0], [0.0])
    with pytest.raises(ValueError):
        gradient_log_h([0.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        gradient_log_h([0.0, 1.0], [0.0])
