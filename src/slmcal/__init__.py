"""slmcal: NSGA-II + Bayesian calibration workflow for shoreline models.

The package keeps heavy optional dependencies (PyMC, ArviZ, plotting stacks,
pySPADS/IHSetUtils) out of top-level import so that lightweight imports such as
``from slmcal.data import TimeSeriesDataset`` stay fast and robust.
"""
from __future__ import annotations

from .data import TimeSeriesDataset, MultiTransectDataset
from .models.base import ShorelineModel, ParameterSpec
from .models.blackbox import BlackBoxModel

__all__ = [
    "TimeSeriesDataset",
    "MultiTransectDataset",
    "CalibrationWorkflow",
    "MixtureOfExpertsWorkflow",
    "GateConfig",
    "MoEEnsembleWorkflow",
    "GateSoftmaxConfig",
    "BayesMoEEnsembleWorkflow",
    "BayesGateConfig",
    "ShorelineModel",
    "ParameterSpec",
    "BlackBoxModel",
    "PreprocessResult",
    "preprocess_legacy_yates",
    "preprocess_legacy_no_index",
]


def __getattr__(name: str):
    if name == "CalibrationWorkflow":
        from .core.workflow import CalibrationWorkflow
        return CalibrationWorkflow
    if name in {"MixtureOfExpertsWorkflow", "GateConfig"}:
        from .core.moe_workflow import MixtureOfExpertsWorkflow, GateConfig
        return {"MixtureOfExpertsWorkflow": MixtureOfExpertsWorkflow, "GateConfig": GateConfig}[name]
    if name in {"MoEEnsembleWorkflow", "GateSoftmaxConfig"}:
        from .core.moe_ensemble import MoEEnsembleWorkflow, GateSoftmaxConfig
        return {"MoEEnsembleWorkflow": MoEEnsembleWorkflow, "GateSoftmaxConfig": GateSoftmaxConfig}[name]
    if name in {"BayesMoEEnsembleWorkflow", "BayesGateConfig"}:
        from .core.moe_ensemble_bayes import BayesMoEEnsembleWorkflow, BayesGateConfig
        return {"BayesMoEEnsembleWorkflow": BayesMoEEnsembleWorkflow, "BayesGateConfig": BayesGateConfig}[name]
    if name in {"PreprocessResult", "preprocess_legacy_yates", "preprocess_legacy_no_index"}:
        from .preprocess import PreprocessResult, preprocess_legacy_yates, preprocess_legacy_no_index
        return {
            "PreprocessResult": PreprocessResult,
            "preprocess_legacy_yates": preprocess_legacy_yates,
            "preprocess_legacy_no_index": preprocess_legacy_no_index,
        }[name]
    raise AttributeError(f"module 'slmcal' has no attribute {name!r}")
