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

Two arithmetic paths implement the same iteration:

* ``domain="probability"`` (the default): iterate on ``a`` and ``b``
  directly, ``a = mu / (K b)`` and ``b = nu / (K^T a)``. Simple and
  exact-looking at moderate ``eps``.
* ``domain="log"``: iterate on the log-potentials ``f = log a`` and
  ``g = log b`` with logsumexp updates,
  ``f_i = log mu_i - logsumexp_j(log K_ij + g_j)`` and symmetrically
  for ``g``. The scalings themselves grow like ``exp(1 / eps)`` and
  overflow ``float`` for small ``eps`` (on the demo grid the
  probability path already reaches ``a ~ 1e150`` at ``eps_k = 0.05``
  and produces NaNs at ``eps_k = 0.02``); the log-domain path never
  forms them, so it converges where the probability path cannot.

Both paths return the same :class:`SinkhornResult`; the log-domain
path additionally fills in ``log_a`` / ``log_b`` (finite log-potentials)
and records ``domain="log"``. For the log-domain path the exponentiated
``a`` / ``b`` may overflow to ``inf`` — the log-potentials are the
authoritative output there.
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
    targets. ``domain`` records which arithmetic path produced the
    result (``"probability"`` or ``"log"``). For the log-domain path,
    ``log_a`` / ``log_b`` hold the finite log-potentials ``log a`` /
    ``log b``; they are ``None`` for the probability-space path, whose
    ``a`` / ``b`` are exact. For the log-domain path ``a`` / ``b`` are
    the exponentiated log-potentials and may overflow to ``inf`` when
    the true scalings exceed float range — use ``log_a`` / ``log_b``
    there instead.
    """

    pi: list[list[float]]
    a: list[float]
    b: list[float]
    iterations: int
    row_error: float
    col_error: float
    domain: str = "probability"
    log_a: list[float] | None = None
    log_b: list[float] | None = None


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


def _logsumexp(terms: list[float]) -> float:
    """Stable ``log(sum(exp(t) for t in terms))``.

    Max-shifted so no intermediate overflows; a term of ``-inf``
    contributes nothing, and the result is ``-inf`` only when every
    term is ``-inf``.
    """
    m = max(terms)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(t - m) for t in terms))


def _safe_exp(v: float) -> float:
    """``exp(v)`` saturating to ``inf`` / ``0.0`` outside float range."""
    if v > 709.0:
        return math.inf
    if v < -745.0:
        return 0.0
    return math.exp(v)


def _sinkhorn_log_domain(
    grid: list[float],
    mu: list[float],
    nu: list[float],
    eps: float,
    tol: float,
    max_iter: int,
) -> SinkhornResult:
    """Sinkhorn iteration on the log-potentials ``f = log a``, ``g = log b``.

    The updates are the logs of the probability-space updates:

        f_i = log mu_i - logsumexp_j(log K_ij + g_j),
        g_j = log nu_j - logsumexp_i(log K_ij + f_i),

    with ``log K_ij = -(x_i - x_j)^2 / eps``. No intermediate ever
    exponentiates a scaling, so the iteration survives the small-``eps``
    regime where ``a`` and ``b`` overflow float range. The coupling is
    formed only at the end of each sweep, as ``pi_ij = exp(f_i +
    log K_ij + g_j)``; its entries are coupling masses (<= 1 in exact
    arithmetic), so that exponentiation cannot overflow. Convergence
    is judged on the marginal errors of ``pi`` exactly as in the
    probability-space path.
    """
    n = len(grid)
    log_k = [[-((xi - xj) ** 2) / eps for xj in grid] for xi in grid]
    log_mu = [math.log(v) if v > 0.0 else -math.inf for v in mu]
    log_nu = [math.log(v) if v > 0.0 else -math.inf for v in nu]

    f = [0.0] * n
    g = [0.0] * n
    pi: list[list[float]] = []
    row_error = col_error = math.inf
    for iteration in range(1, max_iter + 1):
        for i in range(n):
            denom = _logsumexp([log_k[i][j] + g[j] for j in range(n)])
            if denom == -math.inf:
                raise ValueError("log-domain row scaling underflowed; eps is too small for this grid")
            f[i] = log_mu[i] - denom
        for j in range(n):
            denom = _logsumexp([log_k[i][j] + f[i] for i in range(n)])
            if denom == -math.inf:
                raise ValueError("log-domain column scaling underflowed; eps is too small for this grid")
            g[j] = log_nu[j] - denom

        pi = [
            [math.exp(f[i] + log_k[i][j] + g[j]) for j in range(n)]
            for i in range(n)
        ]
        row_error = max(abs(sum(pi[i]) - mu[i]) for i in range(n))
        col_error = max(abs(sum(pi[i][j] for i in range(n)) - nu[j]) for j in range(n))
        if row_error < tol and col_error < tol:
            return SinkhornResult(
                pi,
                [_safe_exp(v) for v in f],
                [_safe_exp(v) for v in g],
                iteration,
                row_error,
                col_error,
                domain="log",
                log_a=list(f),
                log_b=list(g),
            )
    raise RuntimeError(
        f"Sinkhorn did not converge in {max_iter} iterations "
        f"(row_error={row_error:.3e}, col_error={col_error:.3e})"
    )


def sinkhorn_bridge(
    grid: list[float],
    mu: list[float],
    nu: list[float],
    eps: float,
    tol: float = 1e-12,
    max_iter: int = 100_000,
    domain: str = "probability",
) -> SinkhornResult:
    """Solve the static bridge by Sinkhorn iteration.

    Iterates ``a = mu / (K b)`` and ``b = nu / (K^T a)`` (elementwise) until
    both marginal errors of ``pi = diag(a) K diag(b)`` are below ``tol``.
    Raises ``RuntimeError`` if ``max_iter`` is reached first.

    ``domain`` selects the arithmetic path: ``"probability"`` (the
    default) iterates on the scalings directly; ``"log"`` iterates on
    the log-potentials with logsumexp updates and is the path to use
    when ``eps`` is small enough that the scalings overflow float
    range. Both paths solve the same problem and agree at moderate
    ``eps``; see the module docstring.
    """
    n = len(grid)
    if n == 0:
        raise ValueError("grid must be nonempty")
    if domain not in ("probability", "log"):
        raise ValueError(f"domain must be 'probability' or 'log', got {domain!r}")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    _check_marginal("mu", mu, n)
    _check_marginal("nu", nu, n)
    if domain == "log":
        return _sinkhorn_log_domain(grid, mu, nu, eps, tol, max_iter)
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
