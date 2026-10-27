"""bridge_lab: a minimal 1-D Schrödinger bridge.

Three pieces, each implementing one classical formula:

- ``sinkhorn``: the static bridge on a finite grid, solved by Sinkhorn
  iteration (iterative proportional fitting).
- ``gaussian``: the closed-form Gaussian bridge (optimal cross-covariance
  and time-t marginals).
- ``simulate``: particle paths from the bridge-mixture representation.

Standard library only.
"""

from .gaussian import gaussian_bridge_marginal, optimal_cross_covariance, sample_coupling
from .simulate import sample_at_time, sample_paths
from .sinkhorn import SinkhornResult, discretize_gaussian, sinkhorn_bridge

__all__ = [
    "SinkhornResult",
    "discretize_gaussian",
    "gaussian_bridge_marginal",
    "optimal_cross_covariance",
    "sample_at_time",
    "sample_coupling",
    "sample_paths",
    "sinkhorn_bridge",
]

__version__ = "0.1.0"
