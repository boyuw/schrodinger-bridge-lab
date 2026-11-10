# schrodinger-bridge-lab

A minimal, dependency-free implementation of the **Schrödinger bridge**
in one dimension: a static bridge solved by Sinkhorn iteration, the
Gaussian bridge in closed form, and particle simulation of bridge paths.

Pure Python (standard library only); package and tests together are
~1,700 lines.

## The story

In 1931/32, Erwin Schrödinger posed a thought experiment from the early
quantum era: a cloud of independent Brownian particles is observed at two
times, and the observed endpoint distributions do not match what the
reference Brownian motion predicts. *What is the most likely evolution of
the cloud between the two observations?* The answer — the stochastic
control / large-deviation problem now called the Schrödinger bridge —
is the path measure closest to the reference Brownian motion (in KL
divergence) whose endpoint marginals are the observed ones.

Schrödinger's system of two coupled PDEs was rediscovered in the 1960s as
**iterative proportional fitting** (IPF), and the same alternating
rescaling is now known as **Sinkhorn iteration**, the workhorse of
entropic optimal transport. Almost a century later, the same mathematics
reappears in **diffusion-model theory**: a diffusion generative model is,
loosely, a learned bridge between a data distribution and noise, and
Schrödinger bridges are the natural generalization when both endpoints
are prescribed. This repo is the classical 1-D core of that story, small
enough to read in one sitting and verify against closed forms.

## What's here

| Module | What it implements |
| --- | --- |
| `bridge_lab.sinkhorn` | Static bridge on a finite grid: `pi = diag(a) K diag(b)` with `K_ij = exp(-(x_i - x_j)^2 / eps)`, scalings found by Sinkhorn iteration to marginal error < 1e-12. |
| `bridge_lab.gaussian` | Closed-form Gaussian bridge: optimal cross-covariance `c* = sqrt(var0*var1 + (eps/2)^2) - eps/2` and the time-`t` marginal mean/variance. |
| `bridge_lab.simulate` | Particle paths from the bridge-mixture representation `X_t = (1-t) X + t Y + sqrt(eps) B_t` over endpoint pairs drawn from the optimal coupling. |
| `bridge_lab.potentials` | Schrödinger log-potentials from a solved static bridge and the Doob h-function `h(t, x)` (dynamic bridge, part 1). |
| `bridge_lab.drift` | The Doob h-transform drift `(eps_k / 2) * grad log h` (dynamic bridge, part 2). |
| `bridge_lab.sde` | Euler–Maruyama forward simulation of the h-transform SDE (dynamic bridge, part 3). |
| `bridge_lab.marginals` | Check of the simulated time-marginals against the closed form, with Monte Carlo z-scores (dynamic bridge, part 4 — the dynamic bridge is now complete). |
| `bridge_lab.demo` | Cross-checks all three views of one bridge against each other. |

## Quick start

```bash
python -m bridge_lab.demo        # from an installed checkout, or with src/ on PYTHONPATH
python -m pytest                 # 50 tests, all against closed forms or exact properties
```

Demo output (endpoints `N(0, 1) -> N(2, 1.5)`, `eps = 0.5`):

```
Static bridge (Sinkhorn on a 57-point grid, kernel eps_k = 2*eps = 1.0):
  iterations           : 63
  row marginal error   : 9.749e-13
  col marginal error   : 3.469e-18
  coupling Cov(X, Y)   : 1.0000  (grid approximation)
  closed-form c*       : 1.0000

Gaussian bridge marginals (closed form):
  t = 0.00   mean =  0.000   var =  1.000
  t = 0.25   mean =  0.500   var =  1.125
  t = 0.50   mean =  1.000   var =  1.250
  t = 0.75   mean =  1.500   var =  1.375
  t = 1.00   mean =  2.000   var =  1.500

Particle simulation at t = 1 (20000 particles, seed 7):
  sample mean : 1.9928   (target 2.0)
  sample var  : 1.4897   (target 1.5)

SDE simulation at t = 1 (20000 particles, 40 Euler steps, seed 8):
  sample mean : 1.9889   (target 2.0)
  sample var  : 1.5026   (target 1.5)

SDE time-marginals vs closed form (same 20000 paths, 40 steps):
     t  sample mean  closed  sample var  closed  mean z   var z
  0.00      -0.0087   0.000      1.0040   1.000   -1.23    0.40
  0.25       0.4937   0.500      1.1285   1.125   -0.85    0.31
  0.50       0.9983   1.000      1.2491   1.250   -0.21   -0.07
  0.75       1.4942   1.500      1.3729   1.375   -0.71   -0.15
  1.00       1.9889   2.000      1.5026   1.500   -1.28    0.17
  worst |z| over all 41 times: mean 1.28, var 0.89 (Monte Carlo error, not drift bias)
```

## Honesty box

- **1-D only**, and the dynamics are built from the *static* bridge:
  the bridge-mixture representation (exact for this problem) mixes
  pinned Brownian bridges over the static optimal coupling, and the
  dynamic bridge is simulated by discretizing the Doob h-transform
  SDE (Euler–Maruyama) and checked against the closed-form marginals
  with Monte Carlo z-scores — worst |z| 1.28 (mean) and 0.89 (variance)
  over all 41 times on the demo bridge.
- **No neural anything.** No learned scores, no function approximation;
  this is the classical problem, not a diffusion-model implementation.
- **One convention subtlety**, documented in
  [docs/DESIGN.md](docs/DESIGN.md): the grid kernel
  `exp(-(x_i - x_j)^2 / eps_k)` corresponds to bridge parameter
  `eps = eps_k / 2`. Read that note before comparing numbers across
  modules.

## Reading

- E. Schrödinger, *Über die Umkehrung der Naturgesetze* (1931) and the
  1932 follow-up — the original thought experiment.
- C. Léonard, *A survey of the Schrödinger problem and some of its
  connections with optimal transport* (2014).
- Y. Chen, T. Georgiou, M. Pavon, *Stochastic control liaisons:
  Richard Sinkhorn meets Gaspard Monge on a Schrödinger bridge* (2021).

## License

MIT, © Boyu Wang.
