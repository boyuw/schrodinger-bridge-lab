"""Tests for the forward SDE simulation (dynamic bridge, part 3).

The anchors are exact where the theory is exact:

* ``drift_at`` agrees with ``drift_values`` on grid points and clamps
  outside the grid;
* with a point-mass terminal potential the SDE is a Brownian bridge to
  the target point, and the last Euler step makes the endpoint spread
  ``sqrt(eps * dt)`` (drift ``-(x - x_j) / (1 - t)`` at ``t = 1 - dt``
  pulls the particle exactly onto ``x_j`` before the final noise
  increment);
* on the solved demo bridge the simulated endpoints recover the target
  marginal ``N(2, 1.5)``. Observed with seed 11, 2000 particles, 50
  steps: mean 2.019, var 1.439; the tolerances below are ~5x the
  Monte Carlo error, so they test the SDE, not the seed.

The mid-time marginals are NOT asserted here: comparing the simulated
``t = 0.5`` marginal against the closed form is part 4 of the backlog
item (the Euler discretization visibly biases it — observed var 1.21
against closed-form 1.25 — and that bias deserves its own check).
"""

import math
import random
import statistics

import pytest

from bridge_lab.drift import drift_values
from bridge_lab.sde import drift_at, simulate_sde
from bridge_lab.sinkhorn import discretize_gaussian, sinkhorn_bridge

M0, VAR0 = 0.0, 1.0
M1, VAR1 = 2.0, 1.5
EPS = 0.5  # bridge parameter; the grid kernel uses eps_k = 2 * EPS
EPS_K = 2.0 * EPS

GRID = [-6.0 + 0.25 * i for i in range(57)]
TIMES_50 = [i / 50 for i in range(51)]


@pytest.fixture(scope="module")
def solved():
    mu = discretize_gaussian(GRID, M0, VAR0)
    nu = discretize_gaussian(GRID, M1, VAR1)
    result = sinkhorn_bridge(GRID, mu, nu, eps=EPS_K)
    return result, mu


def test_drift_at_matches_drift_values_on_grid(solved):
    result, _ = solved
    drift = drift_values(GRID, result.b, EPS_K, t=0.5)
    for x, d in zip(GRID, drift):
        assert drift_at(GRID, result.b, EPS_K, 0.5, x) == pytest.approx(d, rel=1e-12)


def test_drift_at_clamps_outside_grid(solved):
    result, _ = solved
    drift = drift_values(GRID, result.b, EPS_K, t=0.5)
    assert drift_at(GRID, result.b, EPS_K, 0.5, GRID[0] - 3.0) == pytest.approx(drift[0])
    assert drift_at(GRID, result.b, EPS_K, 0.5, GRID[-1] + 3.0) == pytest.approx(drift[-1])


def test_sde_starts_at_x0_and_is_deterministic(solved):
    result, _ = solved
    x0 = [0.0, 1.0, -1.5]
    times = [0.0, 0.5, 1.0]
    first = simulate_sde(GRID, result.b, EPS_K, times, 3, random.Random(5), x0)
    second = simulate_sde(GRID, result.b, EPS_K, times, 3, random.Random(5), x0)
    assert first == second
    for path, start in zip(first, x0):
        assert path[0] == start
        assert len(path) == 3


def test_point_mass_sde_is_brownian_bridge():
    """b a point mass at 0.5: endpoints concentrate there, sd ~ sqrt(eps*dt).

    The last step starts at t = 1 - dt with drift -(x - 0.5)/dt, so the
    drift increment lands exactly on 0.5 and only the final noise
    increment of sd sqrt((eps_k/2) * dt) = sqrt(0.5 * 0.02) = 0.1
    remains.
    """
    grid = [-2.0 + 0.25 * i for i in range(17)]  # -2 .. 2, step 0.25
    target = 0.5
    j = grid.index(target)
    b = [1.0 if k == j else 0.0 for k in range(len(grid))]
    paths = simulate_sde(grid, b, 1.0, TIMES_50, 2000, random.Random(3), [0.0] * 2000)
    endpoints = [p[-1] for p in paths]
    assert statistics.fmean(endpoints) == pytest.approx(target, abs=0.05)
    assert 0.07 < statistics.pstdev(endpoints) < 0.14


def test_sde_endpoints_recover_target_marginal(solved):
    result, mu = solved
    rng = random.Random(11)
    x0 = rng.choices(GRID, weights=mu, k=2000)
    paths = simulate_sde(GRID, result.b, EPS_K, TIMES_50, 2000, rng, x0)
    endpoints = [p[-1] for p in paths]
    assert statistics.fmean(endpoints) == pytest.approx(M1, abs=0.15)
    assert statistics.variance(endpoints) == pytest.approx(VAR1, abs=0.25)


def test_sde_rejects_bad_inputs(solved):
    result, _ = solved
    with pytest.raises(ValueError):
        simulate_sde(GRID, result.b, EPS_K, [0.0, 1.0], 0, random.Random(1), [])
    with pytest.raises(ValueError):
        simulate_sde(GRID, result.b, EPS_K, [0.0, 0.5, 1.0], 2, random.Random(1), [0.0])
    with pytest.raises(ValueError):
        simulate_sde(GRID, result.b, EPS_K, [0.5, 1.0], 1, random.Random(1), [0.0])
    with pytest.raises(ValueError):
        simulate_sde(GRID, result.b, EPS_K, [0.0, 1.0, 0.5], 1, random.Random(1), [0.0])
    with pytest.raises(ValueError):
        simulate_sde(GRID, result.b, 0.0, [0.0, 1.0], 1, random.Random(1), [0.0])
    with pytest.raises(ValueError):
        # All-zero terminal potential: h is identically zero, log h undefined.
        simulate_sde(GRID, [0.0] * len(GRID), EPS_K, [0.0, 1.0], 1, random.Random(1), [0.0])
