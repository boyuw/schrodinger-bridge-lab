"""Demo: the three views of one Gaussian Schrödinger bridge, cross-checked.

Endpoints ``N(0, 1)`` -> ``N(2, 1.5)`` with bridge parameter ``eps = 0.5``.

1. Discretize both Gaussians on a grid and solve the static bridge with
   Sinkhorn; report the marginal errors and the empirical coupling
   covariance next to the closed-form ``c*``.
2. Print the closed-form bridge marginals at ``t = 0, .25, .5, .75, 1``.
3. Simulate particles from the mixture representation and compare the
   sample endpoint mean/variance with the target ``(m1, var1)``.
4. Simulate particles with the forward SDE (Euler–Maruyama on the
   h-transform drift) and compare the same endpoint statistics.
5. Check the SDE time-marginals against the closed form at every time
   (the dynamic bridge's part 4): sample mean/variance per time with
   Monte Carlo z-scores, reusing the section-4 paths.

Run with:  ``python -m bridge_lab.demo``

Convention note: the grid kernel is ``K_ij = exp(-(x_i - x_j)^2 / eps_k)``,
which corresponds to bridge parameter ``eps = eps_k / 2``, so the Sinkhorn
run below uses ``eps_k = 2 * eps`` for both halves to describe the same
bridge. See ``docs/DESIGN.md``.
"""

from __future__ import annotations

import random
import statistics

from .gaussian import gaussian_bridge_marginal, optimal_cross_covariance
from .marginals import marginal_rows, worst_mean_z, worst_var_z
from .sde import simulate_sde
from .simulate import sample_at_time
from .sinkhorn import coupling_moments, discretize_gaussian, sinkhorn_bridge

M0, VAR0 = 0.0, 1.0
M1, VAR1 = 2.0, 1.5
EPS = 0.5
SEED = 7
N_PARTICLES = 20_000


def main() -> None:
    print(f"Gaussian Schrödinger bridge: N({M0}, {VAR0}) -> N({M1}, {VAR1}), eps = {EPS}")
    print()

    # --- 1. Static bridge on a grid, solved by Sinkhorn -------------------
    grid = [-6.0 + 0.25 * i for i in range(57)]  # -6.00, -5.75, ..., 8.00
    mu = discretize_gaussian(grid, M0, VAR0)
    nu = discretize_gaussian(grid, M1, VAR1)
    result = sinkhorn_bridge(grid, mu, nu, eps=2.0 * EPS)
    _, _, cov_discrete = coupling_moments(grid, result.pi)
    c_star = optimal_cross_covariance(VAR0, VAR1, EPS)
    print("Static bridge (Sinkhorn on a 57-point grid, kernel eps_k = 2*eps = 1.0):")
    print(f"  iterations           : {result.iterations}")
    print(f"  row marginal error   : {result.row_error:.3e}")
    print(f"  col marginal error   : {result.col_error:.3e}")
    print(f"  coupling Cov(X, Y)   : {cov_discrete:.4f}  (grid approximation)")
    print(f"  closed-form c*       : {c_star:.4f}")
    print()

    # --- 2. Closed-form bridge marginals ----------------------------------
    print("Gaussian bridge marginals (closed form):")
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        mean, var = gaussian_bridge_marginal(M0, VAR0, M1, VAR1, EPS, t)
        print(f"  t = {t:4.2f}   mean = {mean:6.3f}   var = {var:6.3f}")
    print()

    # --- 3. Particle simulation -------------------------------------------
    rng = random.Random(SEED)
    endpoints = sample_at_time(M0, VAR0, M1, VAR1, EPS, t=1.0, n=N_PARTICLES, rng=rng)
    print(f"Particle simulation at t = 1 ({N_PARTICLES} particles, seed {SEED}):")
    print(f"  sample mean : {statistics.fmean(endpoints):.4f}   (target {M1})")
    print(f"  sample var  : {statistics.variance(endpoints):.4f}   (target {VAR1})")
    print()

    # --- 4. Forward SDE simulation ----------------------------------------
    rng_sde = random.Random(SEED + 1)
    x0 = rng_sde.choices(grid, weights=mu, k=N_PARTICLES)
    sde_times = [i / 40 for i in range(41)]
    sde_paths = simulate_sde(grid, result.b, 2.0 * EPS, sde_times, N_PARTICLES, rng_sde, x0)
    sde_endpoints = [p[-1] for p in sde_paths]
    print(f"SDE simulation at t = 1 ({N_PARTICLES} particles, 40 Euler steps, seed {SEED + 1}):")
    print(f"  sample mean : {statistics.fmean(sde_endpoints):.4f}   (target {M1})")
    print(f"  sample var  : {statistics.variance(sde_endpoints):.4f}   (target {VAR1})")
    print()

    # --- 5. SDE time-marginals vs the closed form (part 4) ----------------
    rows = marginal_rows(sde_paths, sde_times, M0, VAR0, M1, VAR1, EPS)
    print("SDE time-marginals vs closed form (same 20000 paths, 40 steps):")
    print(f"  {'t':>4}  {'sample mean':>11}  {'closed':>6}  "
          f"{'sample var':>10}  {'closed':>6}  {'mean z':>6}  {'var z':>6}")
    for r in rows:
        if r.t in (0.0, 0.25, 0.5, 0.75, 1.0):
            print(f"  {r.t:4.2f}  {r.sample_mean:11.4f}  {r.closed_mean:6.3f}  "
                  f"{r.sample_var:10.4f}  {r.closed_var:6.3f}  "
                  f"{r.mean_z:6.2f}  {r.var_z:6.2f}")
    print(f"  worst |z| over all 41 times: mean {worst_mean_z(rows):.2f}, "
          f"var {worst_var_z(rows):.2f} (Monte Carlo error, not drift bias)")


if __name__ == "__main__":
    main()
