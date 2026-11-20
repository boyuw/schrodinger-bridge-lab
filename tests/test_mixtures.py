"""Tests for the non-Gaussian (mixture) endpoints in bridge_lab.mixtures.

No closed form exists for the mixture coupling, so the bridge checks
here are labelled sanity checks: discretization and moment agreement,
small Sinkhorn marginal errors, positive coupling covariance,
mode-matched mass near 1 at small eps and eroding as eps grows, and
the eps -> 0 / eps -> inf limits as inequalities.
"""

import math
import random
import statistics

import pytest

from bridge_lab.mixtures import (
    MixtureComponent,
    coupling_same_side_mass,
    discretize_mixture,
    mixture_moments,
    sample_mixture_at_time,
)
from bridge_lab.sinkhorn import coupling_moments, sinkhorn_bridge

GRID = [-6.0 + 0.25 * i for i in range(57)]  # -6.00, -5.75, ..., 8.00
# Symmetric bimodal source, mean 0, var 4.5; shifted bimodal target,
# mean 1, var 4.5. Modes at -2/+2 (source) and -1/+3 (target); the
# mode split points are 0.0 and 1.0.
MU_COMPONENTS = [
    MixtureComponent(0.5, -2.0, 0.5),
    MixtureComponent(0.5, 2.0, 0.5),
]
NU_COMPONENTS = [
    MixtureComponent(0.5, -1.0, 0.5),
    MixtureComponent(0.5, 3.0, 0.5),
]


def grid_moments(p):
    ex = sum(pi * x for pi, x in zip(p, GRID))
    exx = sum(pi * x * x for pi, x in zip(p, GRID))
    return ex, exx - ex * ex


def test_discretize_mixture_validation():
    with pytest.raises(ValueError):
        discretize_mixture(GRID, [])
    with pytest.raises(ValueError):
        discretize_mixture(
            GRID, [MixtureComponent(0.6, 0.0, 1.0), MixtureComponent(0.6, 1.0, 1.0)]
        )
    with pytest.raises(ValueError):
        discretize_mixture(
            GRID, [MixtureComponent(-0.5, 0.0, 1.0), MixtureComponent(1.5, 1.0, 1.0)]
        )
    with pytest.raises(ValueError):
        discretize_mixture(GRID, [MixtureComponent(1.0, 0.0, 0.0)])
    with pytest.raises(ValueError):
        discretize_mixture(GRID, [MixtureComponent(1.0, 0.0, -1.0)])


def test_discretize_mixture_sums_to_one_and_is_bimodal():
    mu = discretize_mixture(GRID, MU_COMPONENTS)
    assert abs(sum(mu) - 1.0) < 1e-12
    assert all(w >= 0.0 for w in mu)
    # Grid indices: x = -2 at 16, x = 0 at 24, x = 2 at 32.
    assert mu[16] > 20 * mu[24]
    assert mu[32] > 20 * mu[24]
    assert abs(mu[16] - mu[32]) < 1e-9  # symmetric modes


def test_mixture_moments_analytic():
    assert mixture_moments(MU_COMPONENTS) == pytest.approx((0.0, 4.5))
    assert mixture_moments(NU_COMPONENTS) == pytest.approx((1.0, 4.5))
    # Single component reduces to its own mean/var.
    assert mixture_moments([MixtureComponent(1.0, 3.0, 2.0)]) == pytest.approx(
        (3.0, 2.0)
    )


def test_grid_moments_match_analytic():
    mu = discretize_mixture(GRID, MU_COMPONENTS)
    nu = discretize_mixture(GRID, NU_COMPONENTS)
    assert grid_moments(mu) == pytest.approx((0.0, 4.5), abs=1e-6)
    assert grid_moments(nu) == pytest.approx((1.0, 4.5), abs=1e-6)


def test_same_side_mass_of_independent_product_is_half():
    mu = discretize_mixture(GRID, MU_COMPONENTS)
    nu = discretize_mixture(GRID, NU_COMPONENTS)
    product = [[mu[i] * nu[j] for j in range(len(GRID))] for i in range(len(GRID))]
    # Balanced modes on both sides: P(X<0)=P(Y<1)=1/2, so the
    # independent product puts exactly half its mass same-side.
    assert coupling_same_side_mass(GRID, product, 0.0, 1.0) == pytest.approx(
        0.5, abs=0.01
    )


def test_bridge_between_mixtures_keeps_modes_matched():
    mu = discretize_mixture(GRID, MU_COMPONENTS)
    nu = discretize_mixture(GRID, NU_COMPONENTS)
    result = sinkhorn_bridge(GRID, mu, nu, eps=1.0)
    assert result.row_error < 1e-9
    assert result.col_error < 1e-9
    _, _, cov = coupling_moments(GRID, result.pi)
    assert cov > 4.0  # near the comonotone ceiling 4.5, far from 0
    same_side = coupling_same_side_mass(GRID, result.pi, 0.0, 1.0)
    assert same_side > 0.98  # left mode -> left mode, right -> right


def test_mode_matching_erodes_with_eps():
    mu = discretize_mixture(GRID, MU_COMPONENTS)
    nu = discretize_mixture(GRID, NU_COMPONENTS)
    small = sinkhorn_bridge(GRID, mu, nu, eps=1.0)
    large = sinkhorn_bridge(GRID, mu, nu, eps=20.0)
    _, _, cov_small = coupling_moments(GRID, small.pi)
    _, _, cov_large = coupling_moments(GRID, large.pi)
    assert cov_large < cov_small
    side_small = coupling_same_side_mass(GRID, small.pi, 0.0, 1.0)
    side_large = coupling_same_side_mass(GRID, large.pi, 0.0, 1.0)
    assert 0.5 < side_large < side_small


def test_identical_mixture_limits():
    mu = discretize_mixture(GRID, MU_COMPONENTS)
    # eps -> 0: the coupling tends to the identity, Cov(X, Y) -> Var(X).
    near_identity = sinkhorn_bridge(GRID, mu, mu, eps=0.5)
    _, _, cov_small = coupling_moments(GRID, near_identity.pi)
    assert cov_small / 4.5 > 0.97
    # eps -> inf: the coupling tends to the independent product, Cov -> 0.
    near_independent = sinkhorn_bridge(GRID, mu, mu, eps=1e4)
    _, _, cov_large = coupling_moments(GRID, near_independent.pi)
    assert cov_large < 0.05


def test_sample_mixture_at_time_validation():
    mu = discretize_mixture(GRID, MU_COMPONENTS)
    pi = [[mu[i] * mu[j] for j in range(len(GRID))] for i in range(len(GRID))]
    rng = random.Random(0)
    with pytest.raises(ValueError):
        sample_mixture_at_time(GRID, pi, 0.5, -0.1, 10, rng)
    with pytest.raises(ValueError):
        sample_mixture_at_time(GRID, pi, 0.5, 1.1, 10, rng)
    with pytest.raises(ValueError):
        sample_mixture_at_time(GRID, pi, -0.5, 0.5, 10, rng)
    with pytest.raises(ValueError):
        sample_mixture_at_time(GRID, pi, 0.5, 0.5, 0, rng)
    with pytest.raises(ValueError):
        sample_mixture_at_time(GRID, [[0.5]], 0.5, 0.5, 10, rng)


def test_sample_mixture_at_time_matches_coupling_moments():
    mu = discretize_mixture(GRID, MU_COMPONENTS)
    nu = discretize_mixture(GRID, NU_COMPONENTS)
    result = sinkhorn_bridge(GRID, mu, nu, eps=1.0)
    _, _, cov = coupling_moments(GRID, result.pi)
    rng = random.Random(7)
    n = 20_000
    # t = 0 and t = 1 recover the two marginals.
    at_0 = sample_mixture_at_time(GRID, result.pi, 0.5, 0.0, n, rng)
    assert statistics.fmean(at_0) == pytest.approx(0.0, abs=0.15)
    assert statistics.variance(at_0) == pytest.approx(4.5, abs=0.2)
    at_1 = sample_mixture_at_time(GRID, result.pi, 0.5, 1.0, n, rng)
    assert statistics.fmean(at_1) == pytest.approx(1.0, abs=0.15)
    assert statistics.variance(at_1) == pytest.approx(4.5, abs=0.2)
    # t = 0.5 against the mixture formula with the coupling covariance:
    # mean = 0.5*0 + 0.5*1, var = 0.25*4.5 + 0.25*4.5 + 0.5*cov + 0.5*0.25.
    at_half = sample_mixture_at_time(GRID, result.pi, 0.5, 0.5, n, rng)
    assert statistics.fmean(at_half) == pytest.approx(0.5, abs=0.15)
    assert statistics.variance(at_half) == pytest.approx(
        2.25 + 0.5 * cov + 0.125, abs=0.25
    )
