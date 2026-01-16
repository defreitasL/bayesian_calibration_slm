"""slmcal: NSGA-II + Bayesian calibration workflow for shoreline models."""

from .data import TimeSeriesDataset
from .core.workflow import CalibrationWorkflow
from .models.base import ShorelineModel, ParameterSpec
from .models.blackbox import BlackBoxModel
from .preprocess import PreprocessResult, preprocess_legacy_yates

__all__ = [
    "TimeSeriesDataset",
    "CalibrationWorkflow",
    "ShorelineModel",
    "ParameterSpec",
    "BlackBoxModel",
    "PreprocessResult",
    "preprocess_legacy_yates",
]
