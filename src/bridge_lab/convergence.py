"""Grid-convergence check for the static bridge (1-D).

Solves the same Gaussian bridge on grids of decreasing spacing and
tabulates the coupling-covariance error against the closed-form
``c*``. The table answers "is the grid fine enough?" for the demo
configuration — and the answer has two parts, because two different
effects set the error:

* **Spacing (theta-function) error.** The discrete Sinkhorn problem
  with point-density Gaussian marginals and a Gaussian kernel is, up
  to Poisson-summation corrections, the *same* problem as the
  continuous one: a Gaussian sampled on a grid sums to a Gaussian,
  up to a periodic theta factor whose size is exponentially small in
  ``(kernel width / h)^2``. The kernel ``exp(-d^2 / eps_k)`` has
  width ``sqrt(eps_k / 2) = sqrt(eps)`` in bridge convention, so the
  corrections scale like ``exp(-2 pi^2 eps / h^2)`` (see
  :func:`theta_scale`). At ``h = 1.0`` with ``eps = 0.5`` they surface
  at ~4e-4; at ``h <= 0.5`` they are below 1e-12. The observed errors
  track that scale within about an order of magnitude across the
  tested ``(eps, h)`` combinations.
* **Range (tail-truncation) error.** The grid ``[lo, hi]`` drops the
  marginal tails outside it, and the normalization folds the missing
  mass back in. On the demo range ``[-6, 8]`` the ``N(2, 1.5)`` upper
  tail beyond 8 is ~5e-7 of mass, which floors the covariance error
  at ~1e-6 no matter how fine the grid gets — refining ``h`` from
  0.5 to 0.0625 does not reduce it. Widening the range to
  ``[-10, 12]`` removes the floor and the error falls to ~1e-15 at
  ``h = 0.5``.

So: on the demo range the error is spacing-limited only down to
``h = 0.5`` and range-limited below that. The 57-point demo grid
(``h = 0.25``) is already at the floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .gaussian import optimal_cross_covariance
from .sinkhorn import coupling_moments, discretize_gaussian, sinkhorn_bridge


@dataclass
class ConvergenceRow:
    """One row of the grid-convergence table.

    ``spacing`` is the grid step ``h``; ``n_points`` the grid size;
    ``cov`` the discrete coupling covariance; ``abs_error`` /
    ``rel_error`` its deviation from the closed-form ``c*``;
    ``iterations`` / ``row_error`` / ``col_error`` the Sinkhorn
    diagnostics for that grid.
    """

    spacing: float
    n_points: int
    cov: float
    abs_error: float
    rel_error: float
    iterations: int
    row_error: float
    col_error: float


def theta_scale(eps: float, h: float) -> float:
    """Order-of-magnitude scale of the spacing error, ``exp(-2 pi^2 eps / h^2)``.

    The Poisson-summation correction to a Gaussian sampled on a grid
    of spacing ``h`` is exponentially small in ``(kernel width /
    h)^2``; for the bridge kernel the width squared is ``eps``. This
    is a scale, not a bound — observed covariance errors track it
    within about an order of magnitude (see the tests) — and it is
    only the spacing part of the error: the range/truncation floor
    (see the module docstring) is separate and can dominate it.
    """
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if h <= 0.0:
        raise ValueError("h must be positive")
    return math.exp(-2.0 * math.pi**2 * eps / h**2)


def observed_order(err_coarse: float, err_fine: float) -> float:
    """Observed convergence order from two successive errors.

    ``log2(err_coarse / err_fine)``: 2.0 means the error quarters
    when ``h`` halves. Returns ``inf`` when the fine error is exactly
    zero (and the coarse error is not), ``0.0`` when both are zero,
    and ``-inf`` when the coarse error is zero but the fine one is
    not. A negative order means the error *grew* on refinement —
    the signature of an error floor (here: tail truncation) rather
    than a spacing effect.
    """
    if err_coarse < 0.0 or err_fine < 0.0:
        raise ValueError("errors must be nonnegative")
    if err_fine == 0.0:
        return math.inf if err_coarse > 0.0 else 0.0
    if err_coarse == 0.0:
        return -math.inf
    return math.log2(err_coarse / err_fine)


def grid_convergence_rows(
    m0: float,
    var0: float,
    m1: float,
    var1: float,
    eps: float,
    spacings: tuple[float, ...],
    lo: float,
    hi: float,
    tol: float = 1e-12,
) -> list[ConvergenceRow]:
    """Covariance error vs grid spacing for one Gaussian bridge.

    For each spacing ``h`` in ``spacings`` (in the given order),
    builds the grid ``lo, lo + h, ..., hi``, discretizes
    ``N(m0, var0)`` / ``N(m1, var1)`` on it, solves the static bridge
    with kernel parameter ``eps_k = 2 * eps`` (the demo convention),
    and reports the coupling covariance against the closed-form
    ``c*`` from :func:`bridge_lab.gaussian.optimal_cross_covariance`.

    ``(hi - lo)`` must be an exact multiple of every spacing (within
    1e-9 relative); otherwise the grid would not land on ``hi`` and
    the ranges would not be comparable across rows.
    """
    if not spacings:
        raise ValueError("spacings must be nonempty")
    if lo >= hi:
        raise ValueError("lo must be below hi")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    c_star = optimal_cross_covariance(var0, var1, eps)
    rows: list[ConvergenceRow] = []
    for h in spacings:
        if h <= 0.0:
            raise ValueError("spacings must be positive")
        n_float = (hi - lo) / h
        n = int(round(n_float)) + 1
        if abs(n_float - (n - 1)) > 1e-9 * max(1.0, n_float):
            raise ValueError(
                f"spacing {h} does not divide the range [{lo}, {hi}] exactly"
            )
        grid = [lo + h * i for i in range(n)]
        mu = discretize_gaussian(grid, m0, var0)
        nu = discretize_gaussian(grid, m1, var1)
        result = sinkhorn_bridge(grid, mu, nu, eps=2.0 * eps, tol=tol)
        _, _, cov = coupling_moments(grid, result.pi)
        abs_error = abs(cov - c_star)
        rows.append(
            ConvergenceRow(
                spacing=h,
                n_points=n,
                cov=cov,
                abs_error=abs_error,
                rel_error=abs_error / abs(c_star) if c_star != 0.0 else abs_error,
                iterations=result.iterations,
                row_error=result.row_error,
                col_error=result.col_error,
            )
        )
    return rows
