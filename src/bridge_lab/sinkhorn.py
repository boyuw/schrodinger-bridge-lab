"""The static Schrödinger bridge on a finite 1-D grid, via Sinkhorn iteration.

Given target marginals ``mu`` and ``nu`` (probability vectors over grid
points ``x_i``) and the reference kernel

    K_ij = exp(-(x_i - x_j)^2 / eps),

the static bridge is the coupling

    pi = diag(a) K diag(b)

whose row sums equal ``mu`` and whose column sums equal ``nu``. The positive
scalings ``a`` and ``b`` are found by Sinkhorn iteration (a.k.a. iterative
proportional fitting, IPF): alternately rescale rows to match ``mu`` and
columns to match ``nu`` until both marginal errors fall below ``tol``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SinkhornResult:
    """Result of :func:`sinkhorn_bridge`.

    ``pi[i][j]`` is the coupling mass between grid points ``x_i`` and
    ``x_j``; ``a`` and ``b`` are the scalings with
    ``pi[i][j] = a[i] * K_ij * b[j]``; ``row_error`` / ``col_error`` are the
    final max-absolute deviations of the marginals of ``pi`` from the
    targets.
    """

    pi: list[list[float]]
    a: list[float]
    b: list[float]
    iterations: int
    row_error: float
    col_error: float


def kernel_matrix(grid: list[float], eps: float) -> list[list[float]]:
    """Reference kernel ``K_ij = exp(-(x_i - x_j)^2 / eps)``."""
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    return [[math.exp(-((xi - xj) ** 2) / eps) for xj in grid] for xi in grid]


def _check_marginal(name: str, p: list[float], n: int) -> None:
    if len(p) != n:
        raise ValueError(f"{name} has length {len(p)}, expected {n}")
    if any(v < 0.0 for v in p):
        raise ValueError(f"{name} must be nonnegative")
    if abs(sum(p) - 1.0) > 1e-6:
        raise ValueError(f"{name} must sum to 1 (got {sum(p)})")


def sinkhorn_bridge(
    grid: list[float],
    mu: list[float],
    nu: list[float],
    eps: float,
    tol: float = 1e-12,
    max_iter: int = 100_000,
) -> SinkhornResult:
    """Solve the static bridge by Sinkhorn iteration.

    Iterates ``a = mu / (K b)`` and ``b = nu / (K^T a)`` (elementwise) until
    both marginal errors of ``pi = diag(a) K diag(b)`` are below ``tol``.
    Raises ``RuntimeError`` if ``max_iter`` is reached first.
    """
    n = len(grid)
    if n == 0:
        raise ValueError("grid must be nonempty")
    _check_marginal("mu", mu, n)
    _check_marginal("nu", nu, n)
    K = kernel_matrix(grid, eps)

    b = [1.0] * n
    a = [0.0] * n
    pi: list[list[float]] = []
    row_error = col_error = math.inf
    for iteration in range(1, max_iter + 1):
        # a = mu / (K b), then b = nu / (K^T a).
        for i in range(n):
            denom = sum(K[i][j] * b[j] for j in range(n))
            if denom <= 0.0:
                raise ValueError("row scaling underflowed; eps is too small for this grid")
            a[i] = mu[i] / denom
        for j in range(n):
            denom = sum(K[i][j] * a[i] for i in range(n))
            if denom <= 0.0:
                raise ValueError("column scaling underflowed; eps is too small for this grid")
            b[j] = nu[j] / denom

        pi = [[a[i] * K[i][j] * b[j] for j in range(n)] for i in range(n)]
        row_error = max(abs(sum(pi[i]) - mu[i]) for i in range(n))
        col_error = max(abs(sum(pi[i][j] for i in range(n)) - nu[j]) for j in range(n))
        if row_error < tol and col_error < tol:
            return SinkhornResult(pi, a, b, iteration, row_error, col_error)
    raise RuntimeError(
        f"Sinkhorn did not converge in {max_iter} iterations "
        f"(row_error={row_error:.3e}, col_error={col_error:.3e})"
    )


def coupling_moments(grid: list[float], pi: list[list[float]]) -> tuple[float, float, float]:
    """Return ``(E[X], E[Y], Cov(X, Y))`` under the coupling ``pi``.

    ``E[X] = sum_ij pi_ij x_i``, ``E[Y] = sum_ij pi_ij x_j``, and
    ``Cov(X, Y) = E[XY] - E[X] E[Y]`` with ``E[XY] = sum_ij pi_ij x_i x_j``.
    """
    n = len(grid)
    ex = sum(pi[i][j] * grid[i] for i in range(n) for j in range(n))
    ey = sum(pi[i][j] * grid[j] for i in range(n) for j in range(n))
    exy = sum(pi[i][j] * grid[i] * grid[j] for i in range(n) for j in range(n))
    return ex, ey, exy - ex * ey


def discretize_gaussian(grid: list[float], mean: float, var: float) -> list[float]:
    """Probability vector on ``grid`` proportional to the N(mean, var) density.

    Evaluates the Gaussian density at each grid point and normalizes; this
    is the standard midpoint-free discretization used throughout the demo
    and tests (tail mass outside the grid is dropped by the normalization).
    """
    if var <= 0.0:
        raise ValueError("var must be positive")
    weights = [math.exp(-((x - mean) ** 2) / (2.0 * var)) for x in grid]
    total = sum(weights)
    return [w / total for w in weights]
