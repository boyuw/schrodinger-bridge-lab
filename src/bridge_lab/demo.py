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
6. Solve the static bridge again at a small kernel parameter
   (``eps_k = 0.02``) with the log-domain Sinkhorn path, where the
   probability-space path overflows; report its marginal errors and
   coupling covariance next to the closed form.
7. Bridge between two bimodal Gaussian mixtures (no closed form):
   report the coupling covariance, the mode-matched mass, and the
   mid-time marginal from particles drawn through the discrete
   coupling — sanity checks only, labelled as such.
8. Grid-convergence check: the coupling-covariance error against
   the closed form at spacings 1.0 down to 0.0625 on the demo range,
   plus one wide-range row showing the ~1e-6 plateau is the tail-
   truncation floor, not spacing.

Run with:  ``python -m bridge_lab.demo``

Convention note: the grid kernel is ``K_ij = exp(-(x_i - x_j)^2 / eps_k)``,
which corresponds to bridge parameter ``eps = eps_k / 2``, so the Sinkhorn
run below uses ``eps_k = 2 * eps`` for both halves to describe the same
bridge. See ``docs/DESIGN.md``.
"""

from __future__ import annotations

import math
import random
import statistics

from .convergence import grid_convergence_rows, observed_order
from .gaussian import gaussian_bridge_marginal, optimal_cross_covariance
from .marginals import marginal_rows, worst_mean_z, worst_var_z
from .mixtures import (
    MixtureComponent,
    coupling_same_side_mass,
    discretize_mixture,
    mixture_moments,
    sample_mixture_at_time,
)
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
    # Rationalized c* stays accurate at huge eps, where the literal
    # sqrt-minus form cancels: at eps = 1e12 it returns exactly 0.0.
    huge_eps = 1e12
    c_huge = optimal_cross_covariance(VAR0, VAR1, huge_eps)
    literal_huge = math.sqrt(VAR0 * VAR1 + (huge_eps / 2.0) ** 2) - huge_eps / 2.0
    print(f"  c* at eps = 1e12     : {c_huge:.3e}  (rationalized; var0*var1/eps = "
          f"{VAR0 * VAR1 / huge_eps:.3e})")
    print(f"  literal form there   : {literal_huge:.3e}  (catastrophic cancellation)")
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
    print()

    # --- 6. Small eps: log-domain Sinkhorn --------------------------------
    # At eps_k = 0.02 the scalings a, b overflow float range (the true
    # log a spans +884 / -1653), so the probability-space path ends in
    # NaN marginal errors; the log-domain path never forms the scalings
    # and converges to a valid coupling with the same closed-form
    # covariance target (bridge eps = eps_k / 2 = 0.01).
    eps_k_small = 0.02
    try:
        sinkhorn_bridge(grid, mu, nu, eps=eps_k_small, max_iter=300)
        prob_outcome = "converged (unexpected at this eps)"
    except (RuntimeError, ValueError) as exc:
        prob_outcome = f"{type(exc).__name__}: {exc}"
    small = sinkhorn_bridge(grid, mu, nu, eps=eps_k_small, domain="log", tol=1e-9)
    _, _, cov_small = coupling_moments(grid, small.pi)
    c_star_small = optimal_cross_covariance(VAR0, VAR1, eps_k_small / 2.0)
    print(f"Small-eps static bridge (kernel eps_k = {eps_k_small}, bridge eps = 0.01):")
    print(f"  probability-space path : {prob_outcome}")
    print(f"  log-domain iterations  : {small.iterations}")
    print(f"  row marginal error     : {small.row_error:.3e}")
    print(f"  col marginal error     : {small.col_error:.3e}")
    print(f"  coupling Cov(X, Y)     : {cov_small:.4f}  (closed-form c* {c_star_small:.4f})")
    print(f"  log a range            : [{min(small.log_a):.1f}, {max(small.log_a):.1f}]"
          f"  (exp would overflow float range)")
    print()

    # --- 7. Non-Gaussian endpoints: bimodal mixtures ----------------------
    # No closed form for this coupling, so everything below is a sanity
    # check, labelled as such. mu is 0.5 N(-2, 0.5) + 0.5 N(2, 0.5)
    # (mean 0, var 4.5); nu is 0.5 N(-1, 0.5) + 0.5 N(3, 0.5)
    # (mean 1, var 4.5). The mode splits are at 0.0 and 1.0.
    mix_mu_components = [
        MixtureComponent(0.5, -2.0, 0.5),
        MixtureComponent(0.5, 2.0, 0.5),
    ]
    mix_nu_components = [
        MixtureComponent(0.5, -1.0, 0.5),
        MixtureComponent(0.5, 3.0, 0.5),
    ]
    mix_mu = discretize_mixture(grid, mix_mu_components)
    mix_nu = discretize_mixture(grid, mix_nu_components)
    mix_result = sinkhorn_bridge(grid, mix_mu, mix_nu, eps=2.0 * EPS)
    _, _, cov_mix = coupling_moments(grid, mix_result.pi)
    same_side = coupling_same_side_mass(grid, mix_result.pi, 0.0, 1.0)
    rng_mix = random.Random(SEED + 2)
    mid = sample_mixture_at_time(
        grid, mix_result.pi, EPS, t=0.5, n=N_PARTICLES, rng=rng_mix
    )
    mid_mean_formula = 0.5 * mixture_moments(mix_mu_components)[0] + 0.5 * mixture_moments(
        mix_nu_components
    )[0]
    mid_var_formula = (
        0.25 * mixture_moments(mix_mu_components)[1]
        + 0.25 * mixture_moments(mix_nu_components)[1]
        + 0.5 * cov_mix
        + EPS * 0.25
    )
    print("Non-Gaussian bridge: bimodal mixtures (sanity checks, no closed form):")
    print("  mu = 0.5 N(-2, 0.5) + 0.5 N(2, 0.5), "
          "nu = 0.5 N(-1, 0.5) + 0.5 N(3, 0.5), kernel eps_k = 1.0")
    print(f"  iterations           : {mix_result.iterations}")
    print(f"  row marginal error   : {mix_result.row_error:.3e}")
    print(f"  col marginal error   : {mix_result.col_error:.3e}")
    print(f"  coupling Cov(X, Y)   : {cov_mix:.4f}  (comonotone ceiling 4.5, "
          f"independent 0)")
    print(f"  mode-matched mass    : {same_side:.4f}  (1 = left->left and "
          f"right->right; independent product gives 0.5)")
    print(f"  mid-time t = 0.5 ({N_PARTICLES} particles, seed {SEED + 2}):")
    print(f"    sample mean : {statistics.fmean(mid):.4f}   (formula {mid_mean_formula:.4f})")
    print(f"    sample var  : {statistics.variance(mid):.4f}   (formula {mid_var_formula:.4f})")
    print()

    # --- 8. Grid-convergence check ----------------------------------------
    # Covariance error vs spacing on the demo range, against the
    # closed-form c* = 1.0. The error collapses ~170x from h = 1.0 to
    # h = 0.5 (theta-function corrections, exponentially small in
    # eps / h^2) and then plateaus at ~1e-6: that floor is the
    # N(2, 1.5) tail cut off at x = 8, not spacing — the wide-range
    # row below removes it and the error falls to ~1e-15.
    conv_rows = grid_convergence_rows(
        M0, VAR0, M1, VAR1, EPS, (1.0, 0.5, 0.25, 0.125, 0.0625), -6.0, 8.0
    )
    print("Grid convergence (demo range [-6, 8], closed-form c* = 1.0000):")
    print(f"  {'h':>6}  {'n':>4}  {'Cov':>8}  {'abs err':>9}  {'rel err':>9}  {'order':>5}")
    prev_err: float | None = None
    for r in conv_rows:
        order = (
            f"{observed_order(prev_err, r.abs_error):5.1f}"
            if prev_err is not None
            else "    -"
        )
        print(f"  {r.spacing:6.4f}  {r.n_points:4d}  {r.cov:8.6f}  "
              f"{r.abs_error:9.3e}  {r.rel_error:9.3e}  {order}")
        prev_err = r.abs_error
    wide_row = grid_convergence_rows(M0, VAR0, M1, VAR1, EPS, (0.5,), -10.0, 12.0)[0]
    print(f"  same h = 0.5 on [-10, 12]: abs err {wide_row.abs_error:.3e} "
          f"(the plateau above is the tail-truncation floor)")


if __name__ == "__main__":
    main()
