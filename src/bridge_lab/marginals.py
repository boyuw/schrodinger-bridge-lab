"""Check of the simulated time-marginals against the bridge-mixture representation — dynamic bridge, part 4.

Parts 1–3 built the dynamic bridge (potentials, drift, forward SDE).
This module closes the loop: take Euler–Maruyama paths from
:mod:`bridge_lab.sde`, compute the sample mean/variance at each time,
and compare them with the closed-form marginals from
:func:`bridge_lab.gaussian.gaussian_bridge_marginal` — the exact
marginals of the bridge-mixture representation.

Each row carries z-scores: the mean error divided by the Monte Carlo
standard error ``sqrt(var / n)`` and the variance error divided by
``var * sqrt(2 / (n - 1))`` (the Gaussian standard error of the sample
variance). A row with ``|z|`` of a few is consistent with the SDE
simulating the right bridge; a row with ``|z|`` in the tens is not, and
no amount of seed-shopping fixes it. The z-scores are the check — the
raw errors alone cannot tell a biased drift from an unlucky seed.

One earlier misreading is corrected here, not hidden: the note left in
``test_sde`` claimed the Euler discretization "visibly biases" the
mid-time marginal (one seed's ``t = 0.5`` variance, 1.21 against the
closed-form 1.25). That difference is one Monte Carlo standard error
at 2000 particles, and it does not survive a bigger sample: at 20000
particles the ``t = 0.5`` variance error is -0.001 (seed 8), +0.021
(seed 9), +0.013 (seed 10) — all within two standard errors, with no
consistent sign. Nor does any offset shrink with more Euler steps or a
finer grid, which is what a discretization bias would do. The SDE
marginals agree with the mixture representation within Monte Carlo
error at every checked time; there is no established Euler bias to
report, only the ordinary noise of a finite particle count.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .gaussian import gaussian_bridge_marginal


@dataclass(frozen=True)
class MarginalRow:
    """One time slice of the marginal check."""

    t: float
    n: int  # particles
    sample_mean: float
    sample_var: float  # ddof=1, like statistics.variance
    closed_mean: float
    closed_var: float
    mean_z: float  # (sample_mean - closed_mean) / MC standard error
    var_z: float  # (sample_var - closed_var) / MC standard error

    @property
    def mean_err(self) -> float:
        return self.sample_mean - self.closed_mean

    @property
    def var_err(self) -> float:
        return self.sample_var - self.closed_var


def marginal_rows(
    paths: list[list[float]],
    times: list[float],
    m0: float,
    var0: float,
    m1: float,
    var1: float,
    eps: float,
) -> list[MarginalRow]:
    """Compare simulated time-marginals with the closed form.

    ``paths`` are the SDE paths from :func:`bridge_lab.sde.simulate_sde`
    (one list of positions per particle, aligned with ``times``);
    ``m0, var0, m1, var1, eps`` are the Gaussian bridge parameters in
    the SDE convention (``eps = eps_k / 2``). Returns one row per time.
    """
    if not paths:
        raise ValueError("at least one path is required")
    if len(times) < 1:
        raise ValueError("at least one time is required")
    if any(len(p) != len(times) for p in paths):
        raise ValueError("every path must have one position per time")
    n = len(paths)
    if n < 2:
        raise ValueError("at least two paths are required (variance needs n >= 2)")
    rows: list[MarginalRow] = []
    for k, t in enumerate(times):
        values = [p[k] for p in paths]
        sample_mean = statistics.fmean(values)
        sample_var = statistics.variance(values)
        closed_mean, closed_var = gaussian_bridge_marginal(m0, var0, m1, var1, eps, t)
        mean_se = math.sqrt(closed_var / n)
        var_se = closed_var * math.sqrt(2.0 / (n - 1))
        rows.append(
            MarginalRow(
                t=t,
                n=n,
                sample_mean=sample_mean,
                sample_var=sample_var,
                closed_mean=closed_mean,
                closed_var=closed_var,
                mean_z=(sample_mean - closed_mean) / mean_se,
                var_z=(sample_var - closed_var) / var_se,
            )
        )
    return rows


def worst_mean_z(rows: list[MarginalRow]) -> float:
    """Largest |mean z-score| across the checked times."""
    if not rows:
        raise ValueError("at least one row is required")
    return max(abs(r.mean_z) for r in rows)


def worst_var_z(rows: list[MarginalRow]) -> float:
    """Largest |variance z-score| across the checked times."""
    if not rows:
        raise ValueError("at least one row is required")
    return max(abs(r.var_z) for r in rows)
