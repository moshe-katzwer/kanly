"""Commented refactor of the experimental synthetic-control estimators.

See README.md for the equations, migration notes, and comparison with Powell's
papers. The original generalized_synthetic_control sandbox remains available.
"""

from .gsc import FitOptions, GSCResult, gsc
from .inference import PlaceboResult, permutation_placebos, time_placebos
from .isc import ISCResult, isc
from .model import JointModel, ModelOptions, build_model
from .panel import PanelData

__all__ = [
    "gsc", "isc", "build_model", "time_placebos", "permutation_placebos",
    "GSCResult", "ISCResult", "PlaceboResult", "PanelData", "JointModel",
    "ModelOptions", "FitOptions",
]
