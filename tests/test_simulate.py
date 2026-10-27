"""Tests for the bridge-mixture particle simulation."""

import random
import statistics

import pytest

from bridge_lab.gaussian import gaussian_bridge_marginal
from bridge_lab.simulate import sample_at_time, sample_paths

M0, VAR0 = 0.0, 1.0
M1, VAR1 = 2.0, 1.5
EPS = 0.5


def test_endpoint_samples_match_target():
    rng = random.Random(7)
    samples = sample_at_time(M0, VAR0, M1, VAR1, EPS, t=1.0, n=20_000, rng=rng)
    # Loose seeded tolerances: ~4 standard errors of the sample statistics.
    assert statistics.fmean(samples) == pytest.approx(M1, abs=0.05)
    assert statistics.variance(samples) == pytest.approx(VAR1, abs=0.1)


def test_startpoint_samples_match_source():
    rng = random.Random(7)
    samples = sample_at_time(M0, VAR0, M1, VAR1, EPS, t=0.0, n=20_000, rng=rng)
    assert statistics.fmean(samples) == pytest.approx(M0, abs=0.05)
    assert statistics.variance(samples) == pytest.approx(VAR0, abs=0.1)


def test_midtime_samples_match_closed_form():
    rng = random.Random(11)
    samples = sample_at_time(M0, VAR0, M1, VAR1, EPS, t=0.5, n=20_000, rng=rng)
    mean, var = gaussian_bridge_marginal(M0, VAR0, M1, VAR1, EPS, 0.5)
    assert statistics.fmean(samples) == pytest.approx(mean, abs=0.05)
    assert statistics.variance(samples) == pytest.approx(var, abs=0.1)


def test_paths_endpoints_and_shape():
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    rng = random.Random(3)
    paths = sample_paths(M0, VAR0, M1, VAR1, EPS, times, n=4_000, rng=rng)
    assert len(paths) == 4_000
    assert all(len(p) == len(times) for p in paths)
    starts = [p[0] for p in paths]
    ends = [p[-1] for p in paths]
    # At t=0 and t=1 the Brownian bridge vanishes: path values are exactly
    # the coupling draws, hence distributed as the endpoint marginals.
    assert statistics.fmean(starts) == pytest.approx(M0, abs=0.08)
    assert statistics.fmean(ends) == pytest.approx(M1, abs=0.08)


def test_determinism_under_fixed_seed():
    rng1, rng2 = random.Random(42), random.Random(42)
    out1 = sample_at_time(M0, VAR0, M1, VAR1, EPS, t=0.5, n=500, rng=rng1)
    out2 = sample_at_time(M0, VAR0, M1, VAR1, EPS, t=0.5, n=500, rng=rng2)
    assert out1 == out2

    rng1, rng2 = random.Random(42), random.Random(42)
    times = [0.0, 0.5, 1.0]
    paths1 = sample_paths(M0, VAR0, M1, VAR1, EPS, times, n=50, rng=rng1)
    paths2 = sample_paths(M0, VAR0, M1, VAR1, EPS, times, n=50, rng=rng2)
    assert paths1 == paths2


def test_paths_rejects_bad_times():
    rng = random.Random(0)
    with pytest.raises(ValueError):
        sample_paths(M0, VAR0, M1, VAR1, EPS, [0.1, 1.0], n=10, rng=rng)
    with pytest.raises(ValueError):
        sample_paths(M0, VAR0, M1, VAR1, EPS, [0.0, 0.5, 0.4, 1.0], n=10, rng=rng)
