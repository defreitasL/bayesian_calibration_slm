from __future__ import annotations

"""Optional BART helpers (PyMC-BART).

This module supports the *two-stage* MoE workflow used in slmcal:

1) Calibrate a physics-based shoreline model (e.g., Yates09) using the existing
   black-box PyMC calibration.
2) Fit a BART model on the *residuals* (obs - physics_prediction) to learn a
   flexible discrepancy / correction term.

`pymc_bart` is intentionally an optional dependency, so all imports are local.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class BartConfig:
    """Configuration for a residual BART regression."""

    m: int = 50
    alpha: float | None = None
    beta: float | None = None

    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    random_seed: int = 42
    cores: int | None = None

    # Likelihood on residuals
    sigma_prior_scale: float | None = None
    target_accept: float = 0.9


def _require_pymc_bart():
    try:
        import pymc_bart as pmb  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "pymc_bart is required for the BART residual expert. "
            "Install it (e.g., `pip install pymc-bart`) and try again."
        ) from e


def fit_bart_residuals(
    X_obs: np.ndarray,
    residuals: np.ndarray,
    cfg: BartConfig = BartConfig(),
    feature_names: Sequence[str] | None = None,
    progressbar: bool = True,
):
    """Fit a BART model to residuals.

    Parameters
    ----------
    X_obs
        (n_obs, n_features) design matrix.
    residuals
        (n_obs,) residual series.
    cfg
        BART configuration.

    Returns
    -------
    idata
        ArviZ InferenceData from `pm.sample`.
    bart_model
        The PyMC model instance, useful for out-of-sample predictions via
        `pm.set_data` + `pm.sample_posterior_predictive`.
    """

    _require_pymc_bart()

    import pymc as pm
    import pymc_bart as pmb

    X_obs = np.asarray(X_obs, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    if X_obs.ndim != 2:
        raise ValueError("X_obs must be 2D (n_obs, n_features)")
    if residuals.ndim != 1:
        raise ValueError("residuals must be 1D")
    if X_obs.shape[0] != residuals.shape[0]:
        raise ValueError("X_obs and residuals must have the same number of rows")

    n_obs, n_feat = X_obs.shape
    feat = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(n_feat)]

    coords = {"point": np.arange(n_obs), "feature": feat}

    sigma0 = float(np.std(residuals)) if residuals.size else 1.0
    sigma_scale = float(cfg.sigma_prior_scale) if cfg.sigma_prior_scale is not None else max(sigma0, 1e-6)

    with pm.Model(coords=coords) as model:
        X = pm.Data("X", X_obs, dims=("point", "feature"))

        bart_kwargs = {"m": int(cfg.m)}
        if cfg.alpha is not None:
            bart_kwargs["alpha"] = float(cfg.alpha)
        if cfg.beta is not None:
            bart_kwargs["beta"] = float(cfg.beta)

        mu = pmb.BART("mu", X=X, Y=residuals, dims=("point",), **bart_kwargs)
        sigma = pm.HalfNormal("sigma", sigma=sigma_scale)
        pm.Normal("residuals", mu=mu, sigma=sigma, observed=residuals, dims=("point",))

        idata = pm.sample(
            draws=int(cfg.draws),
            tune=int(cfg.tune),
            chains=int(cfg.chains),
            random_seed=int(cfg.random_seed),
            target_accept=float(cfg.target_accept),
            cores=cfg.cores,
            progressbar=progressbar,
        )


    return idata, model


def predict_bart_mean(
    idata,
    model,
    X_new: np.ndarray,
    *,
    random_seed: int = 42,
    var_name: str = "mu",
    predictions: bool = True,
):
    """Predict the BART mean function at new X.

    Notes
    -----
    We use `pm.set_data` + `pm.sample_posterior_predictive` to ensure compatibility
    with recent PyMC-BART versions.

    Returns
    -------
    mu_draws
        Array with shape (n_samples, n_new).
    """

    _require_pymc_bart()

    import pymc as pm

    X_new = np.asarray(X_new, dtype=float)
    if X_new.ndim != 2:
        raise ValueError("X_new must be 2D (n_new, n_features)")

    n_new = X_new.shape[0]

    with model:
        pm.set_data({"X": X_new}, coords={"point": np.arange(n_new)})
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=[var_name],
            predictions=predictions,
            random_seed=int(random_seed),
        )

    group = "predictions" if predictions else "posterior_predictive"
    arr = getattr(ppc, group)[var_name].values  # (chain, draw, point)
    flat = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    return flat
