# Design notes — schrodinger-bridge-lab

A teaching implementation of the 1-D Schrödinger bridge. Correctness and
honest scoping over features: every formula below is implemented verbatim
and cross-checked by tests against an independent computation.

## The math, restated

### Static bridge (discrete)

Grid points `x_i`, target marginals `mu`, `nu` (probability vectors),
reference kernel

```
K_ij = exp(-(x_i - x_j)^2 / eps_k)
```

The static bridge is the coupling `pi = diag(a) K diag(b)` with row sums
`mu` and column sums `nu`. Sinkhorn iteration alternates

```
a <- mu / (K b),        b <- nu / (K^T a)
```

until both marginal errors are `< tol` (default `1e-12`).

### Gaussian closed form

Endpoints `N(m0, var0)`, `N(m1, var1)`, regularization `eps`. The optimal
coupling is jointly Gaussian with cross-covariance

```
c* = sqrt(var0 * var1 + (eps/2)^2) - eps/2,
```

the positive root of `c^2 + eps c = var0 var1`. Limits: `eps -> 0` gives
`sqrt(var0 var1)` (comonotone OT coupling); `eps -> inf` gives `0`
(independence).

### Bridge-mixture interpolation

At time `t in [0, 1]`,

```
X_t = (1-t) X + t Y + sqrt(eps * t * (1-t)) Z,
```

with `(X, Y)` from the optimal coupling and `Z ~ N(0, 1)` independent.
For a full path, the noise is one pinned Brownian bridge per particle:
`X_t = (1-t) X + t Y + sqrt(eps) B_t`. In the Gaussian case `X_t` is
Gaussian with

```
mean(t) = (1-t) m0 + t m1
var(t)  = (1-t)^2 var0 + t^2 var1 + 2 t (1-t) c* + eps t (1-t)
```

and the endpoints are exact: `t = 0` returns `(m0, var0)`, `t = 1`
returns `(m1, var1)`. The coupling is sampled via the conditional
`Y = m1 + (c*/var0)(X - m0) + sqrt(var1 - c*^2/var0) Z1`.

## The one convention subtlety (read this)

There are two epsilons in this repo, and they differ by a factor of 2:

- `eps` (Gaussian / SDE convention): the pinned bridges have noise
  variance `eps * t * (1-t)`, i.e. the reference is Brownian motion with
  variance rate `eps`, whose unit-time transition density is proportional
  to `exp(-(x-y)^2 / (2 eps))`. The closed form `c*` above belongs to
  this convention.
- `eps_k` (grid kernel, as written): `K_ij = exp(-(x_i-x_j)^2 / eps_k)`
  is a transition density with variance rate `eps_k / 2`.

So a Sinkhorn run at kernel parameter `eps_k` solves the bridge whose
closed-form parameter is `eps = eps_k / 2`, i.e. its coupling covariance
targets `optimal_cross_covariance(var0, var1, eps_k / 2)`. The demo and
tests use `eps_k = 2 * eps` so that all three views describe the *same*
bridge, and one test records the size of the gap (~0.16 for the demo
parameters) that results from ignoring the factor of 2. Both formulas are
implemented exactly as stated; the note exists so nobody "verifies" one
against the other with mismatched conventions.

## Design decisions

- **Standard library only** (`math`, `random`, `statistics`,
  `dataclasses`). Lists of floats instead of arrays: at 1-D teaching
  scale, readability beats vectorization, and there is nothing to install.
- **Randomness is injected.** Samplers take a `random.Random` instance;
  nothing touches global RNG state, so every result is deterministic
  under a fixed seed (tested).
- **Sinkhorn runs in probability space, not log space.** Simple and
  exact-looking at moderate `eps`; the trade-off is underflow for very
  small `eps` relative to grid spacing, which raises a clear error
  instead of silently producing zeros.
- **`c*` is implemented in the literal form** `sqrt(var0 var1 + (eps/2)^2)
  - eps/2`, not the algebraically equivalent rationalized form
  `var0 var1 / (sqrt(...) + eps/2)`. The literal form loses relative
  precision by cancellation when `eps >> var0 var1`; acceptable here
  (tests keep `eps <= 1e6` for the large-eps limit), and the rationalized
  form is listed under v2.
- **Discretization is density-evaluation + normalization.** No midpoint
  weights, no tail correction; the resulting error vs the continuum is
  quantified by the covariance test tolerance (0.05 on a 0.25 grid).
- **Paths are built as `W_t - t W_1`** from a Brownian motion sampled on
  the time grid, so each returned path is a genuine sample of the bridge
  at those times (shared endpoint pair, correlated noise), not a stack of
  independent marginal draws. `sample_at_time` is provided separately for
  single-time marginals.

## What is deliberately simplified

- 1-D, finite grids, Gaussian endpoints for the closed form / sampler.
- The dynamic bridge is obtained from the static one via the mixture
  representation (exact) and via the forward SDE (Euler–Maruyama on the
  h-transform drift, `bridge_lab.sde`); the simulated time-marginals
  are checked against the mixture representation in
  `bridge_lab.marginals` (part 4, landed), with Monte Carlo z-scores.
- No convergence acceleration, no log-domain stabilization, no
  multi-dimensional anything.

### Dynamic bridge, part 1: potentials and h (landed)

`bridge_lab.potentials` extracts the Schrödinger log-potentials
`log a`, `log b` from a solved static bridge and evaluates the Doob
h-function `h(t, x) = sum_j exp(-(x - x_j)^2 / (eps_k * (1-t))) * b_j`
for `t < 1`, with `h(1, .) = b`. It is anchored by two exact identities,
both tested: `h(0, x_i) = mu_i / a_i` (the Sinkhorn update for `a`)
and reconstruction `pi_ij = exp(log a_i + log b_j) K_ij`. The values
are unnormalized kernel expectations — only `log h` differences matter
for the drift — so no normalization constant is applied.

### Dynamic bridge, part 2: drift (landed)

`bridge_lab.drift` computes the h-transform drift
`drift(t, x) = (eps_k / 2) * grad log h(t, x)` — the `eps_k / 2` is the
reference Brownian motion's variance rate in the SDE convention (the
grid kernel `exp(-d^2 / eps_k)` has variance rate `eps_k / 2`). The
gradient is a finite difference of `log h` on the grid (central inside,
one-sided at the ends). It is anchored by an exact identity, tested:
for a point-mass terminal potential, `log h` is quadratic, the central
difference is exact on a uniform grid, and the drift reduces to the
Brownian-bridge pull `-(x - x_j) / (1 - t)`. The forward SDE simulation
(part 3) and its marginal check (part 4) are both landed below; the
drift was verified on its own first, before any simulation could hide
its errors.

### Dynamic bridge, part 3: forward SDE (landed)

`bridge_lab.sde` integrates `dX_t = drift(t, X_t) dt + sqrt(eps_k / 2) dW_t`
with Euler–Maruyama. The drift is known only on the grid, so it is
linearly interpolated (exact at grid points, clamped outside). The
starting points are an input — the SDE's initial law is the start
marginal, sampled by the caller — so the module stays testable with
deterministic starts. It is anchored by two checks, both tested: a
point-mass terminal potential makes the SDE a Brownian bridge, whose
endpoints concentrate on the target with spread `sqrt(eps * dt)` (the
last Euler step lands exactly on the target before the final noise
increment), and on the solved demo bridge the simulated endpoints
recover `N(2, 1.5)` (seed 11, 2000 particles, 50 steps: mean 2.019,
var 1.439). A note once left in the tests claimed the Euler
discretization "visibly biases" the mid-time marginal (one seed's
`t = 0.5` variance, 1.21 against closed-form 1.25); part 4 showed that
difference is one Monte Carlo standard error at 2000 particles and
vanishes at 20000, so the claim is corrected in `test_sde` and in
`bridge_lab.marginals` rather than left standing.

### Dynamic bridge, part 4: marginal check (landed)

`bridge_lab.marginals` closes the dynamic bridge: it takes the
Euler–Maruyama paths, computes the sample mean/variance at each time,
and compares them with `gaussian_bridge_marginal` — the exact
marginals of the mixture representation. Each row carries z-scores
against the Monte Carlo standard errors (`sqrt(var / n)` for the mean,
`var * sqrt(2 / (n - 1))` for the variance), because raw errors cannot
tell a biased drift from an unlucky seed. On the demo bridge the worst
|z| over all 41 times is 1.28 (mean) and 0.89 (variance) with 20000
particles, 40 steps; with seed 11, 2000 particles, 50 steps it is 0.95
and 1.35. The check has teeth, tested: doubling the diffusion
parameter blows the worst mean z-score to ~74, so a wrong SDE cannot
slip through. A second, independent cross-check compares the SDE
mid-time variance with the mixture-representation sampler
(`sample_at_time`) at `t = 0.5` — two different code paths, same
marginal, agreeing within ~1.3 combined Monte Carlo errors.

## What v2 could add

- Log-domain Sinkhorn for small `eps`, and the rationalized `c*` form.
- Non-Gaussian endpoints via discretized potentials (the static side
  already supports arbitrary `mu`, `nu`).
- Multi-dimensional Gaussian bridges (matrix square roots by hand in
  2-D), then general reference processes.
