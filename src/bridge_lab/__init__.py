"""bridge_lab: a minimal 1-D Schrödinger bridge.

Three pieces, each implementing one classical formula:

- ``sinkhorn``: the static bridge on a finite grid, solved by Sinkhorn
  iteration (iterative proportional fitting).
- ``gaussian``: the closed-form Gaussian bridge (optimal cross-covariance
  and time-t marginals).
- ``simulate``: particle paths from the bridge-mixture representation.
- ``potentials``: Schrödinger log-potentials from a solved static bridge
  and the Doob h-function ``h(t, x)`` (dynamic bridge, part 1).
- ``drift``: the Doob h-transform drift ``(eps_k / 2) * grad log h``
  (dynamic bridge, part 2).
- ``sde``: Euler–Maruyama forward simulation of the h-transform SDE
  (dynamic bridge, part 3).
- ``marginals``: check of the simulated time-marginals against the
  bridge-mixture closed form, with Monte Carlo z-scores (dynamic
  bridge, part 4 — the dynamic bridge is now complete).

Standard library only.
"""

from .drift import drift_values, gradient_log_h, log_h_values
from .gaussian import gaussian_bridge_marginal, optimal_cross_covariance, sample_coupling
from .marginals import MarginalRow, marginal_rows, worst_mean_z, worst_var_z
from .potentials import h_values, log_potentials, reconstruct_coupling
from .sde import drift_at, simulate_sde
from .simulate import sample_at_time, sample_paths
from .sinkhorn import SinkhornResult, discretize_gaussian, sinkhorn_bridge

__all__ = [
    "MarginalRow",
    "SinkhornResult",
    "discretize_gaussian",
    "drift_at",
    "drift_values",
    "gaussian_bridge_marginal",
    "gradient_log_h",
    "h_values",
    "log_h_values",
    "log_potentials",
    "marginal_rows",
    "optimal_cross_covariance",
    "reconstruct_coupling",
    "sample_at_time",
    "sample_coupling",
    "sample_paths",
    "simulate_sde",
    "sinkhorn_bridge",
    "worst_mean_z",
    "worst_var_z",
]

__version__ = "0.1.0"
