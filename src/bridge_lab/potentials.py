"""Schrödinger potentials and the Doob h-function — part 1 of the dynamic bridge.

The static bridge from :mod:`bridge_lab.sinkhorn` is
``pi = diag(a) K diag(b)`` with ``K_ij = exp(-(x_i - x_j)^2 / eps_k)``.
The scalings ``a`` and ``b`` *are* the Schrödinger potentials (in
exponentiated form): the log-potentials are ``log a`` and ``log b``,
and the coupling reconstructs exactly as
``pi_ij = exp(log a_i + log b_j) * K_ij``.

The dynamic bridge is a Doob h-transform of the reference Brownian
motion. Its h-function at time ``t`` is the backward expectation of the
terminal potential under the reference transition kernel: with
``Delta = 1 - t`` remaining time, the unit-time transition density of
Brownian motion with variance rate ``eps_k / 2`` (see ``docs/DESIGN.md``
for the factor-of-2 convention) is proportional to
``exp(-(x - y)^2 / (eps_k * Delta))``, so for ``t < 1``

    h(t, x) = sum_j exp(-(x - x_j)^2 / (eps_k * (1 - t))) * b_j,

and at the terminal time ``h(1, x_j) = b_j`` exactly (the kernel
collapses to a point mass). Two exact checks anchor this definition:

* at ``t = 0``, ``h(0, x_i) = (K b)_i = mu_i / a_i``, the Sinkhorn
  update for ``a`` — so ``h(0, .)`` is determined by the start marginal
  and the ``a`` scaling, not by a fresh computation;
* at ``t = 1``, ``h`` is the terminal scaling ``b`` itself.

This module stops at the potentials and ``h``. The drift
``eps * grad log h(t, x)`` (finite differences of ``log h`` on the grid)
and the Euler–Maruyama forward simulation are the next parts of the
backlog item, deliberately not smuggled in here: ``h`` must be verified
on its own first, because every later drift error would otherwise be
indistinguishable from a potentials error.
"""

from __future__ import annotations

import math

from .sinkhorn import SinkhornResult, kernel_matrix


def log_potentials(result: SinkhornResult) -> tuple[list[float], list[float]]:
    """Return ``(log a, log b)`` for a solved static bridge.

    Raises ``ValueError`` if any scaling is nonpositive: a zero scaling
    belongs to a zero-marginal grid point, where the log-potential is
    ``-inf`` and no finite potential exists. The demo marginals are
    strictly positive on the grid, so this is a guard, not a limitation
    encountered in practice here.
    """
    if any(v <= 0.0 for v in result.a) or any(v <= 0.0 for v in result.b):
        raise ValueError("log-potentials require strictly positive scalings a and b")
    return [math.log(v) for v in result.a], [math.log(v) for v in result.b]


def reconstruct_coupling(
    grid: list[float], eps_k: float, log_a: list[float], log_b: list[float]
) -> list[list[float]]:
    """Rebuild ``pi`` from log-potentials: ``pi_ij = exp(log a_i + log b_j) K_ij``."""
    K = kernel_matrix(grid, eps_k)
    return [
        [math.exp(log_a[i] + log_b[j]) * K[i][j] for j in range(len(grid))]
        for i in range(len(grid))
    ]


def h_values(
    grid: list[float], b: list[float], eps_k: float, t: float
) -> list[float]:
    """The Doob h-function evaluated on ``grid`` at time ``t`` in [0, 1].

    For ``t < 1`` this is the reference-kernel expectation of ``b`` with
    remaining time ``1 - t`` (see the module docstring); for ``t = 1``
    it is ``b`` itself. The returned values are unnormalized kernel
    expectations — only ratios / ``log h`` differences matter for the
    drift, so no normalization constant is (or needs to be) applied.
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError("t must lie in [0, 1]")
    if eps_k <= 0.0:
        raise ValueError("eps_k must be positive")
    if len(grid) != len(b):
        raise ValueError(f"b has length {len(b)}, expected {len(grid)}")
    if t == 1.0:
        return list(b)
    remaining = eps_k * (1.0 - t)
    return [
        sum(math.exp(-((x - xj) ** 2) / remaining) * bj for xj, bj in zip(grid, b))
        for x in grid
    ]
