from __future__ import annotations

from .base import ParameterSpec, ShorelineModel, transform_raw_to_physical

# -------------------------
# Helper: safe optional imports
# -------------------------
def _missing_class(name: str, err: Exception):
    class _Missing:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"{name} failed to import. This is usually due to missing optional "
                f"dependencies (e.g. numba/pymc/arviz) or an internal import error."
            ) from err

    _Missing.__name__ = name
    return _Missing


# -------------------------
# Core models (optional deps like numba)
# -------------------------
try:  # pragma: no cover
    from .yates09 import Yates09Model
except Exception as _e:  # pragma: no cover
    Yates09Model = _missing_class("Yates09Model", _e)  # type: ignore

try:  # pragma: no cover
    from .vitousek21 import Vitousek21YatesModel
except Exception as _e:  # pragma: no cover
    Vitousek21YatesModel = _missing_class("Vitousek21YatesModel", _e)  # type: ignore

try:  # pragma: no cover
    from .miller_dean import MD04Model
except Exception as _e:  # pragma: no cover
    MD04Model = _missing_class("MD04Model", _e)  # type: ignore

try:  # pragma: no cover
    from .blackbox import BlackBoxModel
except Exception as _e:  # pragma: no cover
    BlackBoxModel = _missing_class("BlackBoxModel", _e)  # type: ignore

try:  # pragma: no cover
    from .shorefor import ShoreForModel, YoreForModel
except Exception as _e:  # pragma: no cover
    ShoreForModel = _missing_class("ShoreForModel", _e)  # type: ignore
    YoreForModel = _missing_class("YoreForModel", _e)  # type: ignore


# -------------------------
# Rotation + IH-MOOSE (numba strongly recommended)
# -------------------------
try:  # pragma: no cover
    from .jaramillo21 import Jaramillo21aModel
except Exception as _e:  # pragma: no cover
    Jaramillo21aModel = _missing_class("Jaramillo21aModel", _e)  # type: ignore

try:  # pragma: no cover
    from .ih_moose import IHMooseModel, IHMooseConfig, IHMoosePar2Params
except Exception as _e:  # pragma: no cover
    IHMooseModel = _missing_class("IHMooseModel", _e)  # type: ignore
    IHMooseConfig = _missing_class("IHMooseConfig", _e)  # type: ignore
    IHMoosePar2Params = _missing_class("IHMoosePar2Params", _e)  # type: ignore


# -------------------------
# Expert interface + optional expert handlers
# -------------------------
from .expert import Expert, ExpertPrediction

try:  # pragma: no cover
    from .expert_yates_bayes import PhysicsBayesExpert, PhysicsBayesConfig
except Exception as _e:  # pragma: no cover
    PhysicsBayesExpert = _missing_class("PhysicsBayesExpert", _e)  # type: ignore
    PhysicsBayesConfig = _missing_class("PhysicsBayesConfig", _e)  # type: ignore

try:  # pragma: no cover
    from .expert_spads_bayes import SPADSBayesExpert, SPADSBayesConfig
except Exception as _e:  # pragma: no cover
    SPADSBayesExpert = _missing_class("SPADSBayesExpert", _e)  # type: ignore
    SPADSBayesConfig = _missing_class("SPADSBayesConfig", _e)  # type: ignore


try:  # pragma: no cover
    from .expert_spads_regression_bayes import SPADSRegressionBayesExpert, SPADSRegressionBayesConfig
except Exception as _e:  # pragma: no cover
    SPADSRegressionBayesExpert = _missing_class("SPADSRegressionBayesExpert", _e)  # type: ignore
    SPADSRegressionBayesConfig = _missing_class("SPADSRegressionBayesConfig", _e)  # type: ignore

try:  # pragma: no cover
    from .expert_bart_direct import BARTDirectExpert, BARTDirectConfig
except Exception as _e:  # pragma: no cover
    BARTDirectExpert = _missing_class("BARTDirectExpert", _e)  # type: ignore
    BARTDirectConfig = _missing_class("BARTDirectConfig", _e)  # type: ignore

try:  # pragma: no cover
    from .expert_sarimax import SARIMAXExpert, SARIMAXConfig
except Exception as _e:  # pragma: no cover
    SARIMAXExpert = _missing_class("SARIMAXExpert", _e)  # type: ignore
    SARIMAXConfig = _missing_class("SARIMAXConfig", _e)  # type: ignore


__all__ = [
    "ParameterSpec",
    "ShorelineModel",
    "transform_raw_to_physical",
    "Yates09Model",
    "Vitousek21YatesModel",
    "MD04Model",
    "BlackBoxModel",
    "ShoreForModel",
    "YoreForModel",
    "Jaramillo21aModel",
    "IHMooseModel",
    "IHMooseConfig",
    "IHMoosePar2Params",
    "Expert",
    "ExpertPrediction",
    "PhysicsBayesExpert",
    "PhysicsBayesConfig",
    "SPADSBayesExpert",
    "SPADSBayesConfig",
    "SPADSRegressionBayesExpert",
    "SPADSRegressionBayesConfig",
    "BARTDirectExpert",
    "BARTDirectConfig",
    "SARIMAXExpert",
    "SARIMAXConfig",
]