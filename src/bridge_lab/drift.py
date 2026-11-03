"""The Doob h-transform drift — part 2 of the dynamic bridge.

The dynamic bridge is a Doob h-transform of the reference Brownian
motion. The reference motion has variance rate ``eps = eps_k / 2``
(the grid kernel ``exp(-(x - y)^2 / eps_k)`` is a transition density
with variance rate ``eps_k / 2``; see ``docs/DESIGN.md`` for the
factor-of-2 convention), so the h-transform drift is

    drift(t, x) = eps * d/dx log h(t, x) = (eps_k / 2) * grad log h(t, x),

with ``h`` from :mod:`bridge_lab.potentials`. The gradient is a finite
difference of ``log h`` on the grid: central differences at interior
points, one-sided differences at the two ends. On a uniform grid the
central difference is exact for quadratic ``log h`` — which is exactly
the case for a point-mass terminal potential (tested: the drift then
reduces to the Brownian-bridge pull ``-(x - x_j) / (1 - t)``).

This module stops at the drift. The Euler–Maruyama forward simulation
is the next part of the backlog item, deliberately not smuggled in
here: a wrong drift would otherwise be indistinguishable from a
discretization error in the simulation.
"""

from __future__ import annotations

import math

from .potentials import h_values


def log_h_values(
    grid: list[float], b: list[float], eps_k: float, t: float
) -> list[float]:
    """``log h(t, .)`` on ``grid`` (see :func:`bridge_lab.potentials.h_values`).

    Raises ``ValueError`` if any ``h`` value is nonpositive: ``log h``
    is undefined there, and a zero ``h`` means the terminal potential
    gives that grid point no mass under the reference kernel at all.
    """
    h = h_values(grid, b, eps_k, t)
    if any(v <= 0.0 for v in h):
        raise ValueError("h must be strictly positive to take log h")
    return [math.log(v) for v in h]


def gradient_log_h(grid: list[float], log_h: list[float]) -> list[float]:
    """Finite-difference gradient of ``log h`` on ``grid``.

    Interior points use central differences
    ``(log_h[i+1] - log_h[i-1]) / (x[i+1] - x[i-1])``; the ends use
    one-sided differences. Requires a strictly increasing grid of at
    least two points, matching ``log_h`` in length.
    """
    n = len(grid)
    if n < 2:
        raise ValueError("gradient needs a grid of at least two points")
    if len(log_h) != n:
        raise ValueError(f"log_h has length {len(log_h)}, expected {n}")
    if any(b <= a for a, b in zip(grid, grid[1:])):
        raise ValueError("grid must be strictly increasing")
    grad = [0.0] * n
    grad[0] = (log_h[1] - log_h[0]) / (grid[1] - grid[0])
    for i in range(1, n - 1):
        grad[i] = (log_h[i + 1] - log_h[i - 1]) / (grid[i + 1] - grid[i - 1])
    grad[n - 1] = (log_h[n - 1] - log_h[n - 2]) / (grid[n - 1] - grid[n - 2])
    return grad


def drift_values(
    grid: list[float], b: list[float], eps_k: float, t: float
) -> list[float]:
    """The h-transform drift ``(eps_k / 2) * grad log h(t, .)`` on ``grid``.

    ``b`` is the terminal Schrödinger scaling from a solved static
    bridge, ``eps_k`` the grid kernel parameter, and ``t`` in [0, 1].
    At ``t = 1`` this is ``(eps_k / 2) * grad log b`` directly.
    """
    log_h = log_h_values(grid, b, eps_k, t)
    grad = gradient_log_h(grid, log_h)
    return [(eps_k / 2.0) * g for g in grad]
