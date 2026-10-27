"""The Gaussian Schrödinger bridge in closed form (1-D).

For endpoints ``N(m0, var0)`` and ``N(m1, var1)`` and regularization
``eps``, the optimal coupling is jointly Gaussian. Its cross-covariance is

    c* = sqrt(var0 * var1 + (eps/2)^2) - eps/2,

the positive root of ``c^2 + eps * c = var0 * var1``. Sanity limits:
``eps -> 0`` gives ``c* -> sqrt(var0 * var1)`` (the comonotone optimal-
transport coupling), and ``eps -> inf`` gives ``c* -> 0`` (independent
endpoints: the reference measure itself).

Convention note: ``eps`` here is the SDE/Brownian convention — the pinned
bridges in the mixture representation have noise variance ``eps * t *
(1 - t)``. The grid kernel in :mod:`bridge_lab.sinkhorn` is written as
``exp(-(x_i - x_j)^2 / eps_k)``, which corresponds to bridge parameter
``eps = eps_k / 2``. See ``docs/DESIGN.md``.
"""

from __future__ import annotations

import math
import random


def optimal_cross_covariance(var0: float, var1: float, eps: float) -> float:
    """Optimal-coupling cross-covariance ``c* = sqrt(var0*var1 + (eps/2)^2) - eps/2``."""
    if var0 <= 0.0 or var1 <= 0.0:
        raise ValueError("variances must be positive")
    if eps < 0.0:
        raise ValueError("eps must be nonnegative")
    return math.sqrt(var0 * var1 + (eps / 2.0) ** 2) - eps / 2.0


def gaussian_bridge_marginal(
    m0: float, var0: float, m1: float, var1: float, eps: float, t: float
) -> tuple[float, float]:
    """Mean and variance of the bridge marginal at time ``t`` in [0, 1].

    From the bridge-mixture representation
    ``X_t = (1-t) X + t Y + sqrt(eps * t * (1-t)) Z`` with ``(X, Y)`` from
    the optimal coupling and ``Z ~ N(0, 1)`` independent:

        mean(t) = (1-t) m0 + t m1
        var(t)  = (1-t)^2 var0 + t^2 var1 + 2 t (1-t) c* + eps t (1-t)

    At ``t = 0`` this returns exactly ``(m0, var0)`` and at ``t = 1``
    exactly ``(m1, var1)``.
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError("t must lie in [0, 1]")
    c_star = optimal_cross_covariance(var0, var1, eps)
    mean = (1.0 - t) * m0 + t * m1
    var = (
        (1.0 - t) ** 2 * var0
        + t**2 * var1
        + 2.0 * t * (1.0 - t) * c_star
        + eps * t * (1.0 - t)
    )
    return mean, var


def sample_coupling(
    m0: float,
    var0: float,
    m1: float,
    var1: float,
    eps: float,
    n: int,
    rng: random.Random,
) -> list[tuple[float, float]]:
    """Draw ``n`` pairs ``(X, Y)`` from the Gaussian optimal coupling.

    Uses the conditional representation

        X ~ N(m0, var0),
        Y = m1 + (c*/var0) (X - m0) + sqrt(var1 - c*^2 / var0) * Z1,

    with ``Z1 ~ N(0, 1)`` independent. The residual variance
    ``var1 - c*^2 / var0 = eps * c* / var0`` is nonnegative because
    ``c*^2 = var0 * var1 - eps * c*``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    c_star = optimal_cross_covariance(var0, var1, eps)
    slope = c_star / var0
    resid_sd = math.sqrt(max(0.0, var1 - c_star * c_star / var0))
    sd0 = math.sqrt(var0)
    pairs = []
    for _ in range(n):
        x = m0 + sd0 * rng.gauss(0.0, 1.0)
        y = m1 + slope * (x - m0) + resid_sd * rng.gauss(0.0, 1.0)
        pairs.append((x, y))
    return pairs
