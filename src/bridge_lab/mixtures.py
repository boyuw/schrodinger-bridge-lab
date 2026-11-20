"""Non-Gaussian endpoints: bridges between Gaussian mixtures on the grid.

The static side of the lab already supports arbitrary marginals ``mu``,
``nu`` — only the Gaussian closed form in :mod:`bridge_lab.gaussian`
is tied to Gaussian endpoints. This module supplies the non-Gaussian
half of the story: discretize a Gaussian mixture onto the grid, solve
the bridge with the existing Sinkhorn solver, and report sanity
checks. There is no closed form for the coupling here, so every check
is labelled as such:

* the discretized moments match the analytic mixture moments
  (a discretization check, not a bridge check);
* the Sinkhorn marginal errors are small (the solver still works);
* the coupling covariance is positive and large — the bridge couples
  the two mixtures rather than mixing them up;
* the mode structure survives: with a split point between the modes
  of each marginal, the fraction of coupling mass on the same side of
  both splits (``coupling_same_side_mass``) is near 1 at small ``eps``
  and erodes toward the independent value as ``eps`` grows — the
  bridge maps left mode to left mode, right mode to right mode, and
  the regularization gradually forgets which mode is which;
* in the limit ``eps -> 0`` the coupling tends to the comonotone
  (identity) coupling, so for identical mixtures
  ``Cov(X, Y) -> Var(X)``; in the limit ``eps -> inf`` it tends to the
  independent product, so ``Cov(X, Y) -> 0``. Both limits are checked
  as inequalities, not equalities, because no closed form pins the
  rate.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .sinkhorn import discretize_gaussian


@dataclass(frozen=True)
class MixtureComponent:
    """One Gaussian component: ``weight`` of ``N(mean, var)``."""

    weight: float
    mean: float
    var: float


def discretize_mixture(
    grid: list[float], components: list[MixtureComponent]
) -> list[float]:
    """Probability vector on ``grid`` for a Gaussian mixture.

    Each component is discretized with
    :func:`bridge_lab.sinkhorn.discretize_gaussian` and the results are
    combined with the component weights. Weights must be nonnegative
    and sum to 1 (within 1e-6); every variance must be positive.
    """
    if not components:
        raise ValueError("components must be nonempty")
    total_weight = sum(c.weight for c in components)
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1 (got {total_weight})")
    if any(c.weight < 0.0 for c in components):
        raise ValueError("weights must be nonnegative")
    if any(c.var <= 0.0 for c in components):
        raise ValueError("variances must be positive")
    combined = [0.0] * len(grid)
    for c in components:
        discrete = discretize_gaussian(grid, c.mean, c.var)
        for i, w in enumerate(discrete):
            combined[i] += c.weight * w
    total = sum(combined)
    return [w / total for w in combined]


def mixture_moments(
    components: list[MixtureComponent],
) -> tuple[float, float]:
    """Analytic ``(mean, variance)`` of a Gaussian mixture.

    ``E[X] = sum_k w_k m_k`` and
    ``Var(X) = sum_k w_k (var_k + m_k^2) - E[X]^2``.
    """
    mean = sum(c.weight * c.mean for c in components)
    second = sum(c.weight * (c.var + c.mean**2) for c in components)
    return mean, second - mean**2


def coupling_same_side_mass(
    grid: list[float],
    pi: list[list[float]],
    split_x: float,
    split_y: float,
) -> float:
    """Fraction of coupling mass on the same side of both split points.

    Mass with ``(x < split_x and y < split_y)`` plus mass with
    ``(x >= split_x and y >= split_y)``. With the splits placed between
    the modes of the two marginals, this is the mass the bridge keeps
    mode-matched: 1 means every left-mode particle goes to the left
    mode, 0.5 is what the independent product gives for balanced
    modes.
    """
    mass = 0.0
    for i, x in enumerate(grid):
        for j, y in enumerate(grid):
            if (x < split_x and y < split_y) or (x >= split_x and y >= split_y):
                mass += pi[i][j]
    return mass


def sample_mixture_at_time(
    grid: list[float],
    pi: list[list[float]],
    eps: float,
    t: float,
    n: int,
    rng: random.Random,
) -> list[float]:
    """Draw ``n`` samples of ``X_t`` from a discrete-coupling bridge.

    Endpoint pairs ``(X, Y)`` are drawn from the coupling ``pi`` on the
    grid, then the pinned-bridge formula is applied:

        X_t = (1-t) X + t Y + sqrt(eps * t * (1-t)) Z,

    with ``Z ~ N(0, 1)`` independent — the same mixture representation
    as :func:`bridge_lab.simulate.sample_at_time`, but with the endpoint
    pair drawn from an arbitrary discrete coupling instead of the
    Gaussian closed-form coupling. ``eps`` is the bridge parameter
    (the grid kernel uses ``eps_k = 2 * eps``; see ``docs/DESIGN.md``).
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError("t must lie in [0, 1]")
    if eps < 0.0:
        raise ValueError("eps must be nonnegative")
    if n <= 0:
        raise ValueError("n must be positive")
    total = sum(sum(row) for row in pi)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"coupling must sum to 1 (got {total})")
    # Cumulative distribution over the flattened coupling, for
    # inverse-transform sampling of endpoint pairs.
    flat: list[float] = []
    cumulative = 0.0
    for row in pi:
        for p in row:
            cumulative += p
            flat.append(cumulative)
    noise_sd = math.sqrt(eps * t * (1.0 - t))
    samples = []
    size = len(grid)
    for _ in range(n):
        u = rng.random()
        # Binary search for the first cell whose cumulative mass exceeds u.
        lo, hi = 0, len(flat) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if flat[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        i, j = divmod(lo, size)
        x = grid[i]
        y = grid[j]
        samples.append(
            (1.0 - t) * x + t * y + noise_sd * rng.gauss(0.0, 1.0)
        )
    return samples
