"""Forward SDE simulation of the dynamic bridge — part 3 of the dynamic bridge.

The dynamic bridge is the Doob h-transform of the reference Brownian
motion, whose variance rate is ``eps = eps_k / 2`` (see ``docs/DESIGN.md``
for the factor-of-2 convention). Its SDE is

    dX_t = drift(t, X_t) dt + sqrt(eps_k / 2) dW_t,

with ``drift`` from :mod:`bridge_lab.drift`. This module integrates it
with Euler–Maruyama on a time grid:

    X_{k+1} = X_k + drift(t_k, X_k) * dt + sqrt(eps_k / 2 * dt) * Z_k.

Two things to know about this discretization:

* The drift is only known on the spatial grid, so it is evaluated by
  linear interpolation between grid points (and clamped to the endpoint
  drift outside the grid). On the grid itself the interpolation is
  exact, so this adds no error at grid points.
* The drift is singular as ``t -> 1`` only in the point-mass limit; for
  a solved bridge with a smooth terminal potential it stays finite
  (tested in ``test_drift``), and the last Euler step lands exactly on
  ``t = 1``.

The starting points ``x0`` are an input, not drawn here: the SDE's
initial law is the start marginal ``mu``, and the caller samples it
(e.g. ``random.choices(grid, weights=mu)``). Keeping the sampler out
makes the module testable with deterministic starts.

This module stops at the simulation. The check of the simulated
time-marginals against the bridge-mixture representation is
:mod:`bridge_lab.marginals` (dynamic bridge, part 4): a simulation can
look plausible while its marginals are wrong, so the check lives in
its own module with Monte Carlo z-scores rather than being smuggled
in here.
"""

from __future__ import annotations

import math
import random

from .drift import drift_values


def _interp(grid: list[float], values: list[float], x: float) -> float:
    """Linear interpolation of ``values`` (given on ``grid``) at ``x``.

    Outside the grid the endpoint value is returned (clamped), so a
    particle that wanders past the grid edge still feels a finite
    drift instead of an extrapolation blow-up.
    """
    if x <= grid[0]:
        return values[0]
    if x >= grid[-1]:
        return values[-1]
    for i in range(len(grid) - 1):
        if grid[i] <= x <= grid[i + 1]:
            w = (x - grid[i]) / (grid[i + 1] - grid[i])
            return (1.0 - w) * values[i] + w * values[i + 1]
    return values[-1]  # unreachable; the clamps above cover the ends


def drift_at(grid: list[float], b: list[float], eps_k: float, t: float, x: float) -> float:
    """The h-transform drift at a single point ``x`` and time ``t``.

    Evaluates :func:`bridge_lab.drift.drift_values` on ``grid`` and
    linearly interpolates; exact at grid points, clamped outside the
    grid. ``b`` is the terminal Schrödinger scaling from a solved
    static bridge, ``eps_k`` the grid kernel parameter, ``t`` in [0, 1].
    """
    return _interp(grid, drift_values(grid, b, eps_k, t), x)


def simulate_sde(
    grid: list[float],
    b: list[float],
    eps_k: float,
    times: list[float],
    n: int,
    rng: random.Random,
    x0: list[float],
) -> list[list[float]]:
    """Euler–Maruyama paths of the h-transform SDE.

    ``times`` must be increasing, start at 0 and end at 1. ``x0`` holds
    the ``n`` starting positions (one per particle, drawn from the start
    marginal by the caller). Returns ``n`` paths, each a list of
    positions at ``times``. Deterministic under a fixed ``rng`` seed.
    """
    if len(times) < 2 or times[0] != 0.0 or times[-1] != 1.0:
        raise ValueError("times must start at 0.0 and end at 1.0")
    if any(d <= c for c, d in zip(times, times[1:])):
        raise ValueError("times must be strictly increasing")
    if n <= 0:
        raise ValueError("n must be positive")
    if len(x0) != n:
        raise ValueError(f"x0 has length {len(x0)}, expected {n}")
    if eps_k <= 0.0:
        raise ValueError("eps_k must be positive")

    diffusion = math.sqrt(eps_k / 2.0)
    paths = [[x] for x in x0]
    for t, t_next in zip(times, times[1:]):
        dt = t_next - t
        drift = drift_values(grid, b, eps_k, t)
        noise_sd = diffusion * math.sqrt(dt)
        for path in paths:
            path.append(
                path[-1] + _interp(grid, drift, path[-1]) * dt + noise_sd * rng.gauss(0.0, 1.0)
            )
    return paths
