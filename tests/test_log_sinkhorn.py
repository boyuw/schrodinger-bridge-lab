"""Tests for the log-domain Sinkhorn path.

The probability-space path iterates on the scalings ``a`` and ``b``
directly; the log-domain path iterates on ``f = log a`` and ``g = log b``
with logsumexp updates. The two solve the same problem, so they must
agree wherever both run — and the log path must keep working at small
``eps`` where the scalings overflow float range (on the demo grid the
probability path already reaches ``a ~ 1e150`` at ``eps_k = 0.05`` and
produces NaN marginal errors at ``eps_k = 0.02``).
"""

import math

import pytest

from bridge_lab.gaussian import optimal_cross_covariance
from bridge_lab.potentials import log_potentials, reconstruct_coupling
from bridge_lab.sinkhorn import (
    coupling_moments,
    discretize_gaussian,
    sinkhorn_bridge,
)

M0, VAR0 = 0.0, 1.0
M1, VAR1 = 2.0, 1.5

GRID = [-6.0 + 0.25 * i for i in range(57)]
EPS_K_MODERATE = 1.0  # demo kernel parameter (bridge eps = eps_k / 2)
EPS_K_SMALL = 0.02  # probability path overflows here; log path must not


@pytest.fixture(scope="module")
def demo_marginals():
    return discretize_gaussian(GRID, M0, VAR0), discretize_gaussian(GRID, M1, VAR1)


@pytest.fixture(scope="module")
def solved_pair(demo_marginals):
    mu, nu = demo_marginals
    prob = sinkhorn_bridge(GRID, mu, nu, eps=EPS_K_MODERATE)
    log = sinkhorn_bridge(GRID, mu, nu, eps=EPS_K_MODERATE, domain="log")
    return mu, nu, prob, log


@pytest.fixture(scope="module")
def small_eps_log(demo_marginals):
    mu, nu = demo_marginals
    return mu, nu, sinkhorn_bridge(
        GRID, mu, nu, eps=EPS_K_SMALL, domain="log", tol=1e-9, max_iter=10_000
    )


def test_log_and_probability_agree_at_moderate_eps(solved_pair):
    # Same mathematical iteration, different arithmetic: the couplings
    # must agree to near machine precision, with the same iteration count.
    _, _, prob, log = solved_pair
    assert log.domain == "log"
    assert log.iterations == prob.iterations
    for i in range(len(GRID)):
        for j in range(len(GRID)):
            assert log.pi[i][j] == pytest.approx(prob.pi[i][j], abs=1e-12)
    assert log.row_error < 1e-12
    assert log.col_error < 1e-12


def test_log_potentials_match_probability_scalings(solved_pair):
    # Where both paths run, log_a / log_b are just log a / log b.
    _, _, prob, log = solved_pair
    assert log.log_a is not None and log.log_b is not None
    for la, a in zip(log.log_a, prob.a):
        assert la == pytest.approx(math.log(a), abs=1e-9)
    for lb, b in zip(log.log_b, prob.b):
        assert lb == pytest.approx(math.log(b), abs=1e-9)
    assert prob.log_a is None and prob.log_b is None


def test_log_domain_covariance_vs_closed_form(solved_pair):
    _, _, _, log = solved_pair
    _, _, cov = coupling_moments(GRID, log.pi)
    c_star = optimal_cross_covariance(VAR0, VAR1, EPS_K_MODERATE / 2.0)
    assert abs(cov - c_star) < 0.05


def test_log_domain_small_eps_converges(small_eps_log):
    # At eps_k = 0.02 the log path converges to a valid coupling: exact
    # marginals, total mass 1, nonnegative entries, and a covariance
    # matching the Gaussian closed form at eps = eps_k / 2.
    mu, nu, result = small_eps_log
    assert result.domain == "log"
    assert result.row_error < 1e-8
    assert result.col_error < 1e-8
    total = sum(sum(row) for row in result.pi)
    assert total == pytest.approx(1.0, abs=1e-9)
    assert all(v >= 0.0 for row in result.pi for v in row)
    for i in range(len(GRID)):
        assert sum(result.pi[i]) == pytest.approx(mu[i], abs=1e-8)
    for j in range(len(GRID)):
        assert sum(result.pi[i][j] for i in range(len(GRID))) == pytest.approx(
            nu[j], abs=1e-8
        )
    _, _, cov = coupling_moments(GRID, result.pi)
    c_star = optimal_cross_covariance(VAR0, VAR1, EPS_K_SMALL / 2.0)
    assert abs(cov - c_star) < 0.05


def test_small_eps_scalings_exceed_float_range(small_eps_log):
    # The reason the probability path cannot run here: the true
    # log-scalings span +884 / -1653, so exp(log a) overflows float
    # range (max ~709 in log) at the top and underflows to 0 at the
    # bottom. The log path never forms them.
    _, _, result = small_eps_log
    assert result.log_a is not None
    assert max(result.log_a) > 709.0
    assert min(result.log_a) < -745.0
    assert any(v == math.inf for v in result.a)


def test_small_eps_probability_path_outcome_documented(small_eps_log):
    # What the probability path does at eps_k = 0.02 (observed, not a
    # contract — the contract is the log-domain result asserted above):
    # its scalings overflow, the marginal errors go NaN, and it raises
    # RuntimeError without converging (300 iterations suffice to see
    # the NaNs; the default max_iter would churn 100_000 of them).
    mu, nu, log_result = small_eps_log
    try:
        sinkhorn_bridge(GRID, mu, nu, eps=EPS_K_SMALL, max_iter=300)
        outcome = "converged"
    except (RuntimeError, ValueError) as exc:
        outcome = f"{type(exc).__name__}: {exc}"
    # No assertion on `outcome`: if a future change makes the
    # probability path survive this eps, this test still passes — the
    # log-domain assertions above are the requirement. The observed
    # outcome at the time of writing is recorded in the comment above.
    assert log_result.row_error < 1e-8


def test_log_potentials_read_off_log_domain_result(small_eps_log):
    # log_potentials must use the stored log-potentials for a log-domain
    # result — taking log of the overflowed a / b would give inf — and
    # the coupling must reconstruct from them.
    _, _, result = small_eps_log
    log_a, log_b = log_potentials(result)
    assert log_a == result.log_a
    assert log_b == result.log_b
    rebuilt = reconstruct_coupling(GRID, EPS_K_SMALL, log_a, log_b)
    for i in range(len(GRID)):
        for j in range(len(GRID)):
            assert rebuilt[i][j] == pytest.approx(result.pi[i][j], abs=1e-12)


def test_log_domain_large_eps_is_near_product(demo_marginals):
    # eps -> inf: the kernel -> all ones, so the bridge -> the
    # independent product mu x nu, on either path.
    mu, nu = demo_marginals
    result = sinkhorn_bridge(GRID, mu, nu, eps=1e4, domain="log")
    for i in range(len(GRID)):
        for j in range(len(GRID)):
            assert result.pi[i][j] == pytest.approx(mu[i] * nu[j], abs=1e-3)


def test_log_domain_even_smaller_eps_converges(demo_marginals):
    # One decade smaller still: eps_k = 0.01 needs ~5k iterations at
    # tol 1e-9 and still lands on valid marginals.
    mu, nu = demo_marginals
    result = sinkhorn_bridge(GRID, mu, nu, eps=0.01, domain="log", tol=1e-9, max_iter=20_000)
    assert result.row_error < 1e-8
    assert result.col_error < 1e-8
    assert max(result.log_a) > 709.0


def test_rejects_bad_domain(demo_marginals):
    mu, nu = demo_marginals
    with pytest.raises(ValueError):
        sinkhorn_bridge(GRID, mu, nu, eps=1.0, domain="bogus")
    with pytest.raises(ValueError):
        sinkhorn_bridge(GRID, mu, nu, eps=0.0, domain="log")
