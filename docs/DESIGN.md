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
  representation (exact), never via an SDE with a learned or computed
  drift.
- No convergence acceleration, no log-domain stabilization, no
  multi-dimensional anything.

## What v2 could add

- The **dynamic bridge via the Doob h-transform**: compute the drift
  `eps * grad log h(t, x)` from the Schrödinger system and simulate the
  forward SDE with Euler–Maruyama, then check it against the mixture
  representation — the natural "second half" of the theory.
- Log-domain Sinkhorn for small `eps`, and the rationalized `c*` form.
- Non-Gaussian endpoints via discretized potentials (the static side
  already supports arbitrary `mu`, `nu`).
- Multi-dimensional Gaussian bridges (matrix square roots by hand in
  2-D), then general reference processes.
