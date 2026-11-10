"""Tests for the marginal check (dynamic bridge, part 4).

The anchors are exact where the theory is exact:

* on synthetic paths the rows are hand-computed from the closed form
  (``c*`` evaluated by hand for ``var0=2, var1=4, eps=0.5``);
* on the solved demo bridge the SDE marginals agree with the closed
  form within Monte Carlo error at every time — worst |z| 0.95 (mean)
  and 1.35 (variance) with seed 11, 2000 particles, 50 steps;
* the check has teeth: doubling the diffusion parameter blows the
  worst mean z-score to ~74, so a wrong SDE cannot slip through;
* the SDE mid-time variance agrees with the independent
  mixture-representation sampler at ``t = 0.5``.

The ``t = 0.5`` variance error with seed 11 (-0.044 against a
Monte Carlo error of ~0.04) is pinned below, not hidden: it is the
number the old ``test_sde`` note misread as Euler bias. It is one
standard error of noise — at 20000 particles the same quantity is
-0.001, +0.021, +0.013 across seeds 8, 9, 10 — so the test asserts
the z-score, not the raw error.
"""

import math
import random
import statistics

import pytest

from bridge_lab.gaussian import gaussian_bridge_marginal
from bridge_lab.marginals import marginal_rows, worst_mean_z, worst_var_z
from bridge_lab.sde import simulate_sde
from bridge_lab.simulate import sample_at_time
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


def test_rows_match_hand_computed_values():
    """Synthetic paths: every number checked against a hand calculation.

    Paths [[0,1,2],[2,3,4]] on times [0, .5, 1] against the bridge
    N(1, 2) -> N(3, 4), eps = 0.5. c* = sqrt(8 + 0.25^2) - 0.25
    = 2.58945, so the closed-form variance at t = 0.5 is
    0.25*2 + 0.25*4 + 2*0.5*0.5*c* + 0.5*0.5*0.5 = 2.91973.
    """
    rows = marginal_rows(
        [[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]],
        [0.0, 0.5, 1.0],
        m0=1.0, var0=2.0, m1=3.0, var1=4.0, eps=0.5,
    )
    assert len(rows) == 3
    r0, r_half, r1 = rows
    assert (r0.sample_mean, r0.sample_var) == (1.0, 2.0)
    assert (r0.closed_mean, r0.closed_var) == (1.0, 2.0)
    assert r0.mean_z == 0.0 and r0.var_z == 0.0
    assert r_half.sample_mean == 2.0 and r_half.sample_var == 2.0
    assert r_half.closed_mean == 2.0
    assert r_half.closed_var == pytest.approx(2.91973, abs=0.001)
    assert r_half.mean_z == 0.0
    assert r_half.var_z == pytest.approx(-0.2227, abs=0.001)
    assert (r1.sample_mean, r1.sample_var) == (3.0, 2.0)
    assert (r1.closed_mean, r1.closed_var) == (3.0, 4.0)
    assert r1.var_z == pytest.approx(-2.0 / (4.0 * math.sqrt(2.0)), abs=1e-9)


def test_rows_reject_bad_inputs():
    with pytest.raises(ValueError):
        marginal_rows([], [0.0, 1.0], 0.0, 1.0, 2.0, 1.5, 0.5)
    with pytest.raises(ValueError):
        marginal_rows([[0.0, 1.0]], [0.0, 0.5, 1.0], 0.0, 1.0, 2.0, 1.5, 0.5)
    with pytest.raises(ValueError):
        marginal_rows([[0.0, 1.0]], [0.0, 1.0], 0.0, 1.0, 2.0, 1.5, 0.5)
    with pytest.raises(ValueError):
        marginal_rows([[0.0, 1.0], [2.0, 3.0]], [0.0, 1.0], 0.0, 1.0, 2.0, 1.5, -0.5)
    with pytest.raises(ValueError):
        worst_mean_z([])
    with pytest.raises(ValueError):
        worst_var_z([])


def test_sde_marginals_match_closed_form(solved):
    """Seed 11, 2000 particles, 50 steps: every |z| is small.

    Worst |mean z| 0.95, worst |variance z| 1.35 across all 51 times.
    The t = 0.5 variance error is -0.044 — the number once misread as
    Euler bias — which is 1.1 Monte Carlo standard errors, not a bias.
    """
    result, mu = solved
    rng = random.Random(11)
    x0 = rng.choices(GRID, weights=mu, k=2000)
    paths = simulate_sde(GRID, result.b, EPS_K, TIMES_50, 2000, rng, x0)
    rows = marginal_rows(paths, TIMES_50, M0, VAR0, M1, VAR1, EPS)
    assert worst_mean_z(rows) < 2.0
    assert worst_var_z(rows) < 2.0
    mid = rows[25]
    assert mid.t == pytest.approx(0.5)
    assert mid.var_err == pytest.approx(-0.0443, abs=0.005)
    assert abs(mid.var_z) < 2.0
    end = rows[-1]
    assert end.closed_mean == pytest.approx(M1)
    assert end.closed_var == pytest.approx(VAR1)
    assert abs(end.mean_z) < 2.0 and abs(end.var_z) < 2.0


def test_check_catches_wrong_diffusion(solved):
    """Doubling eps_k (wrong SDE) must blow the z-scores up."""
    result, mu = solved
    rng = random.Random(11)
    x0 = rng.choices(GRID, weights=mu, k=2000)
    paths = simulate_sde(GRID, result.b, 2.0 * EPS_K, TIMES_50, 2000, rng, x0)
    rows = marginal_rows(paths, TIMES_50, M0, VAR0, M1, VAR1, EPS)
    assert worst_mean_z(rows) > 10.0
    assert worst_var_z(rows) > 5.0


def test_sde_agrees_with_mixture_representation(solved):
    """Independent code path: SDE vs bridge-mixture sampler at t = 0.5.

    Two 2000-particle samples of the same marginal; their variances
    differ by 0.072, inside ~1.3 combined Monte Carlo errors.
    """
    result, mu = solved
    rng = random.Random(11)
    x0 = rng.choices(GRID, weights=mu, k=2000)
    paths = simulate_sde(GRID, result.b, EPS_K, TIMES_50, 2000, rng, x0)
    sde_var = statistics.variance([p[25] for p in paths])
    mixture = sample_at_time(M0, VAR0, M1, VAR1, EPS, t=0.5, n=2000, rng=random.Random(21))
    assert abs(sde_var - statistics.variance(mixture)) < 0.15
    _, closed_var = gaussian_bridge_marginal(M0, VAR0, M1, VAR1, EPS, 0.5)
    assert abs(statistics.variance(mixture) - closed_var) < 0.15
