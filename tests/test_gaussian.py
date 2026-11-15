"""Tests for the Gaussian closed form."""

import math
import random
import statistics

import pytest

from bridge_lab.gaussian import (
    gaussian_bridge_marginal,
    optimal_cross_covariance,
    sample_coupling,
)

M0, VAR0 = 0.0, 1.0
M1, VAR1 = 2.0, 1.5
EPS = 0.5


def test_cross_covariance_value():
    # sqrt(1 * 1.5 + 0.25^2) - 0.25 = sqrt(1.5625) - 0.25 = 1.25 - 0.25 = 1
    assert optimal_cross_covariance(VAR0, VAR1, EPS) == pytest.approx(1.0, abs=1e-12)
    # c* is the positive root of c^2 + eps c = var0 var1.
    c = optimal_cross_covariance(0.7, 2.3, 0.9)
    assert c * c + 0.9 * c == pytest.approx(0.7 * 2.3, abs=1e-12)


def _literal_cross_covariance(var0: float, var1: float, eps: float) -> float:
    """The literal form ``sqrt(var0*var1 + (eps/2)^2) - eps/2``.

    Algebraically identical to :func:`optimal_cross_covariance`, but it
    subtracts two nearly equal numbers at large ``eps``. Kept here only
    as the reference the rationalized implementation is tested against.
    """
    return math.sqrt(var0 * var1 + (eps / 2.0) ** 2) - eps / 2.0


def test_cross_covariance_matches_literal_form():
    # At moderate eps the two forms agree to near machine precision.
    for var0, var1, eps in [(1.0, 1.5, 0.5), (0.7, 2.3, 0.9), (2.0, 0.3, 3.0),
                            (1.0, 1.0, 1e-3), (4.0, 9.0, 10.0)]:
        c = optimal_cross_covariance(var0, var1, eps)
        assert c == pytest.approx(_literal_cross_covariance(var0, var1, eps), rel=1e-12)


def test_cross_covariance_limits():
    ot_limit = math.sqrt(VAR0 * VAR1)
    # eps -> 0: comonotone optimal-transport coupling.
    assert optimal_cross_covariance(VAR0, VAR1, 1e-12) == pytest.approx(ot_limit, abs=1e-9)
    # eps -> inf: independent endpoints, c* ~ var0 var1 / eps -> 0.
    # The rationalized form keeps full relative precision here: at
    # eps = 1e12 the asymptotic value is matched to machine precision,
    # where the literal form returns exactly 0.0 (see the test below).
    for eps in (1e6, 1e12, 1e15):
        c_big = optimal_cross_covariance(VAR0, VAR1, eps)
        assert c_big == pytest.approx(VAR0 * VAR1 / eps, rel=1e-12)
        assert 0.0 < c_big


def test_cross_covariance_large_eps_stable():
    # The literal form cancels catastrophically at large eps while the
    # rationalized form still satisfies the defining quadratic
    # c^2 + eps c = var0 var1 to machine precision.
    for var0, var1, eps in [(1.0, 1.5, 1e6), (1.0, 1.5, 1e12), (0.7, 2.3, 1e12)]:
        c = optimal_cross_covariance(var0, var1, eps)
        assert c * c + eps * c == pytest.approx(var0 * var1, rel=1e-12)
        literal = _literal_cross_covariance(var0, var1, eps)
        # The literal form is already wrong at the 1e-5 relative level
        # at eps = 1e6 and returns exactly 0.0 at eps = 1e12.
        assert abs(literal * literal + eps * literal - var0 * var1) / (var0 * var1) > 1e-6


def test_bridge_marginal_endpoints_exact():
    assert gaussian_bridge_marginal(M0, VAR0, M1, VAR1, EPS, 0.0) == (M0, VAR0)
    assert gaussian_bridge_marginal(M0, VAR0, M1, VAR1, EPS, 1.0) == (M1, VAR1)


def test_bridge_marginal_midpoint():
    mean, var = gaussian_bridge_marginal(M0, VAR0, M1, VAR1, EPS, 0.5)
    assert mean == pytest.approx(1.0, abs=1e-12)
    # 0.25*1 + 0.25*1.5 + 2*0.25*c*(=1) + 0.5*0.25 = 0.25+0.375+0.5+0.125
    assert var == pytest.approx(1.25, abs=1e-12)
    # Full formula at a non-symmetric time.
    t = 0.3
    c = optimal_cross_covariance(VAR0, VAR1, EPS)
    expected = (1 - t) ** 2 * VAR0 + t**2 * VAR1 + 2 * t * (1 - t) * c + EPS * t * (1 - t)
    _, var_t = gaussian_bridge_marginal(M0, VAR0, M1, VAR1, EPS, t)
    assert var_t == pytest.approx(expected, abs=1e-12)


def test_bridge_marginal_rejects_bad_t():
    with pytest.raises(ValueError):
        gaussian_bridge_marginal(M0, VAR0, M1, VAR1, EPS, 1.5)


def test_coupling_sampler_moments():
    rng = random.Random(2026)
    pairs = sample_coupling(M0, VAR0, M1, VAR1, EPS, 20_000, rng)
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    assert statistics.fmean(xs) == pytest.approx(M0, abs=0.05)
    assert statistics.fmean(ys) == pytest.approx(M1, abs=0.05)
    cov = statistics.fmean([(x - M0) * (y - M1) for x, y in pairs])
    assert cov == pytest.approx(optimal_cross_covariance(VAR0, VAR1, EPS), abs=0.05)
