"""Particle simulation of bridge paths via the bridge-mixture representation.

The Schrödinger bridge path measure is a mixture of pinned Brownian
bridges: draw an endpoint pair ``(X, Y)`` from the optimal (static)
coupling, then run a Brownian bridge from ``X`` to ``Y``,

    X_t = (1-t) X + t Y + sqrt(eps) * B_t,

where ``B`` is a standard Brownian bridge (``B_0 = B_1 = 0``,
``Var(B_t) = t (1 - t)``). This mixture representation is exact for the
bridge problem — it is how the dynamic bridge is built from the static
one — not an SDE discretization.
"""

from __future__ import annotations

import math
import random

from .gaussian import sample_coupling


def sample_at_time(
    m0: float,
    var0: float,
    m1: float,
    var1: float,
    eps: float,
    t: float,
    n: int,
    rng: random.Random,
) -> list[float]:
    """Draw ``n`` samples of ``X_t`` from the single-time mixture formula.

    ``X_t = (1-t) X + t Y + sqrt(eps * t * (1-t)) Z`` with ``(X, Y)`` from
    the Gaussian optimal coupling and an independent ``Z ~ N(0, 1)`` per
    particle. At ``t = 1`` the noise term vanishes and the samples are
    exactly draws from the coupling's second marginal, ``N(m1, var1)``.
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError("t must lie in [0, 1]")
    noise_sd = math.sqrt(eps * t * (1.0 - t))
    return [
        (1.0 - t) * x + t * y + noise_sd * rng.gauss(0.0, 1.0)
        for x, y in sample_coupling(m0, var0, m1, var1, eps, n, rng)
    ]


def sample_paths(
    m0: float,
    var0: float,
    m1: float,
    var1: float,
    eps: float,
    times: list[float],
    n: int,
    rng: random.Random,
) -> list[list[float]]:
    """Draw ``n`` bridge paths evaluated on the time grid ``times``.

    ``times`` must be increasing, start at 0 and end at 1. Each particle
    shares one endpoint pair ``(X, Y)`` across all times (so the result is
    a genuine path, not independent marginals). The Brownian bridge is
    built as ``B_t = W_t - t * W_1`` from a Brownian motion ``W`` with
    variance rate ``eps`` sampled on the same grid, which gives the exact
    joint law of the bridge at the grid times.
    """
    if len(times) < 2 or times[0] != 0.0 or times[-1] != 1.0:
        raise ValueError("times must start at 0.0 and end at 1.0")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("times must be strictly increasing")
    if n <= 0:
        raise ValueError("n must be positive")

    paths = []
    for x, y in sample_coupling(m0, var0, m1, var1, eps, n, rng):
        # Brownian motion W (variance rate eps) on the time grid.
        w = [0.0]
        for a, b in zip(times, times[1:]):
            step = math.sqrt(eps * (b - a)) * rng.gauss(0.0, 1.0)
            w.append(w[-1] + step)
        w1 = w[-1]
        path = [
            (1.0 - t) * x + t * y + (w_t - t * w1)
            for t, w_t in zip(times, w)
        ]
        paths.append(path)
    return paths
