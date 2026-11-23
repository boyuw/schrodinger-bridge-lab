"""Tests for bridge_lab.convergence: covariance error vs grid spacing.

All numbers below were measured from the implementation (see the
probe runs recorded in the increment log); the tests pin them with
slack, they do not re-derive them.
"""

import math

import pytest

from bridge_lab.convergence import (
    ConvergenceRow,
    grid_convergence_rows,
    observed_order,
    theta_scale,
)
from bridge_lab.gaussian import optimal_cross_covariance

M0, VAR0, M1, VAR1, EPS = 0.0, 1.0, 2.0, 1.5, 0.5
SPACINGS = (1.0, 0.5, 0.25, 0.125, 0.0625)


def demo_rows():
    return grid_convergence_rows(M0, VAR0, M1, VAR1, EPS, SPACINGS, -6.0, 8.0)


def test_demo_range_coarse_grid_error():
    rows = demo_rows()
    # h = 1.0 is coarser than the kernel width sqrt(eps) ~ 0.71, so the
    # theta-function corrections surface: measured 3.95e-04.
    assert 1e-4 < rows[0].abs_error < 1e-3
    assert rows[0].n_points == 15
    assert rows[0].iterations > 0
    assert rows[0].row_error < 1e-9 and rows[0].col_error < 1e-9


def test_demo_range_halving_collapses_error_then_plateaus():
    rows = demo_rows()
    # 1.0 -> 0.5 drops the error ~170x (measured 3.95e-04 -> 2.32e-06).
    assert rows[1].abs_error < rows[0].abs_error / 100
    # Below h = 0.5 the error stops shrinking: the ~1e-6 plateau is
    # the tail-truncation floor of the [-6, 8] range, not spacing.
    for r in rows[1:]:
        assert r.abs_error < 1e-5
    assert rows[-1].abs_error > rows[1].abs_error  # floor, not monotone
    assert [r.n_points for r in rows] == [15, 29, 57, 113, 225]


def test_wide_range_removes_the_floor():
    # Same spacings on [-10, 12]: at h = 0.5 the error is ~1e-15
    # (measured 6.7e-16), proving the demo-range plateau is truncation.
    rows = grid_convergence_rows(M0, VAR0, M1, VAR1, EPS, (1.0, 0.5), -10.0, 12.0)
    assert rows[1].abs_error < 1e-12
    # ...while the coarse-grid spacing effect survives the wider range.
    assert rows[0].abs_error > 1e-4


def test_smaller_eps_needs_finer_grid():
    # eps = 0.25 narrows the kernel, so the theta corrections grow at
    # fixed h: measured 1.2e-09 at h = 0.5 and 5.0e-03 at h = 1.0
    # on [-10, 12] (vs 6.7e-16 / 3.96e-04 at eps = 0.5).
    rows = grid_convergence_rows(M0, VAR0, M1, VAR1, 0.25, (1.0, 0.5), -10.0, 12.0)
    assert rows[1].abs_error < 1e-8
    assert rows[0].abs_error > 1e-3


def test_cov_matches_closed_form_within_reported_error():
    c_star = optimal_cross_covariance(VAR0, VAR1, EPS)
    for r in demo_rows():
        assert abs(r.cov - c_star) == pytest.approx(r.abs_error)
        assert r.rel_error == pytest.approx(r.abs_error / abs(c_star))


def test_theta_scale_tracks_observed_errors():
    # exp(-2 pi^2 eps / h^2): at (eps=0.5, h=1.0) the scale is 5.2e-05
    # against an observed 3.95e-04; at (eps=0.25, h=1.0) it is 7.2e-03
    # against an observed 5.0e-03. Both within a factor of 10.
    assert theta_scale(0.5, 1.0) == pytest.approx(math.exp(-2 * math.pi**2 * 0.5), rel=1e-12)
    assert 0.1 < 3.951e-04 / theta_scale(0.5, 1.0) < 10.0
    assert 0.1 < 5.011e-03 / theta_scale(0.25, 1.0) < 10.0
    # Monotone in the right directions: finer h or larger eps shrinks it.
    assert theta_scale(0.5, 0.5) < theta_scale(0.5, 1.0)
    assert theta_scale(0.25, 0.5) > theta_scale(0.5, 0.5)


def test_observed_order():
    # log2(3.951e-04 / 2.315e-06) ~ 7.4: the 1.0 -> 0.5 drop is far
    # steeper than any polynomial order — a theta-function cliff, not
    # a power law.
    assert 7.0 < observed_order(3.951e-04, 2.315e-06) < 8.0
    assert observed_order(1e-3, 2.5e-4) == pytest.approx(2.0)
    assert observed_order(1e-6, 0.0) == math.inf
    assert observed_order(0.0, 0.0) == 0.0
    assert observed_order(0.0, 1e-6) == -math.inf
    # A negative order is the floor signature: error grew on refinement.
    assert observed_order(2.315e-06, 4.287e-06) < 0.0
    with pytest.raises(ValueError):
        observed_order(-1e-6, 1e-6)


def test_validation():
    with pytest.raises(ValueError):
        grid_convergence_rows(M0, VAR0, M1, VAR1, EPS, (), -6.0, 8.0)
    with pytest.raises(ValueError):
        grid_convergence_rows(M0, VAR0, M1, VAR1, EPS, (0.0,), -6.0, 8.0)
    with pytest.raises(ValueError):
        grid_convergence_rows(M0, VAR0, M1, VAR1, EPS, (-0.5,), -6.0, 8.0)
    # 0.3 does not divide the 14-wide range exactly.
    with pytest.raises(ValueError):
        grid_convergence_rows(M0, VAR0, M1, VAR1, EPS, (0.3,), -6.0, 8.0)
    with pytest.raises(ValueError):
        grid_convergence_rows(M0, VAR0, M1, VAR1, EPS, (0.5,), 8.0, -6.0)
    with pytest.raises(ValueError):
        grid_convergence_rows(M0, VAR0, M1, VAR1, -0.5, (0.5,), -6.0, 8.0)
    with pytest.raises(ValueError):
        theta_scale(0.0, 0.5)
    with pytest.raises(ValueError):
        theta_scale(0.5, 0.0)


def test_row_is_a_dataclass():
    r = ConvergenceRow(0.5, 29, 1.0, 2e-6, 2e-6, 65, 1e-12, 1e-12)
    assert r.spacing == 0.5 and r.n_points == 29
