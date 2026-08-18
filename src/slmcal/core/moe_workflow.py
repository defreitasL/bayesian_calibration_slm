from __future__ import annotations

"""Mixture-of-Experts workflow (physics + data-driven discrepancy).

This file implements an *upscaled* workflow (Option C) built on top of the
existing slmcal framework:

- Expert 1: a physics-based shoreline model (e.g., Yates09), calibrated with the
  existing black-box PyMC calibration.
- Expert 2: a BART discrepancy model fitted to the residuals of Expert 1.
- Gate: a simple logistic gate (time/covariate driven) that blends the two
  experts through time.

Design choices
--------------
We implement a **two-stage** BART residual expert. In PyMC-BART, the BART random
variable needs the training targets `Y` when it is defined. Because the
physics-model prediction depends on unknown parameters, a fully joint
(Yates-params + BART) model is awkward in practice. Two-stage calibration is a
robust, pragmatic compromise:

1) Calibrate the physics model.
2) Compute residuals using a summary physics prediction (posterior median by
   default) and fit BART on those residuals.
3) Fit a gate that prefers the physics model a priori, but can turn on the BART
   correction when needed.

You still get uncertainty from:
- physics posterior (parameter uncertainty)
- BART posterior (discrepancy uncertainty)
- gate posterior (blending uncertainty)

The final predictive ensemble is obtained by Monte Carlo combining draws from
these three posteriors.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from slmcal.data import TimeSeriesDataset
from slmcal.models.base import ShorelineModel, transform_raw_to_physical
from slmcal.bayes.pymc import Prior, bayesian_calibrate
from slmcal.bayes.bart import BartConfig, fit_bart_residuals, predict_bart_mean
from slmcal.features import ewma_irregular


@dataclass(frozen=True)
class GateConfig:
    """Configuration for the logistic gate."""

    # Which covariates drive the gate
    mode: str = "time"  # "time" | "covariates"

    # Prior preference for the physics model
    alpha_prior_mean: float = 2.0
    alpha_prior_sd: float = 1.0
    beta_prior_sd: float = 1.0

    draws: int = 2000
    tune: int = 2000
    chains: int = 4
    random_seed: int = 42
    cores: int | None = None

    # observation noise in gate fit
    sigma_prior_scale: float | None = None
    target_accept: float = 0.9


@dataclass
class MoEResult:
    """Holds fitted posteriors and key intermediate arrays."""

    trace_physics: object
    idata_bart: object
    bart_model: object
    idata_gate: object

    yhat_physics_obs: np.ndarray
    residuals_obs: np.ndarray

    X_bart_obs: np.ndarray
    X_bart_full: np.ndarray

    X_gate_obs: np.ndarray
    X_gate_full: np.ndarray


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(train, axis=0)
    sd = np.std(train, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (train - mu) / sd, (test - mu) / sd, mu, sd


def build_design_matrix(
    dataset: TimeSeriesDataset,
    forcing_names: Sequence[str] = ("E",),
    *,
    include_time: bool = True,
    include_doy: bool = False,
    at: str = "obs",
) -> tuple[np.ndarray, list[str]]:
    """Create a simple feature matrix from dataset forcings (+ optional time features)."""

    at_l = at.lower().strip()
    if at_l not in {"obs", "full"}:
        raise ValueError("at must be 'obs' or 'full'")

    if at_l == "obs":
        idx = np.asarray(dataset.idx_obs, dtype=int)
    else:
        idx = np.arange(dataset.time.shape[0], dtype=int)

    cols: list[np.ndarray] = []
    names: list[str] = []

    if include_time:
        t = np.asarray(idx, dtype=float)
        # scale to roughly [-1, 1]
        if t.size > 1:
            t = (t - t.mean()) / (t.std() if t.std() > 0 else 1.0)
        cols.append(t)
        names.append("t")

    if include_doy:
        if not np.issubdtype(dataset.time.dtype, np.datetime64):
            raise ValueError("include_doy requires datetime64 time axis")
        import pandas as pd

        dt = pd.DatetimeIndex(dataset.time[idx])
        doy = dt.dayofyear.to_numpy(dtype=float)
        ang = 2 * np.pi * (doy - 1.0) / 365.25
        cols.append(np.sin(ang))
        cols.append(np.cos(ang))
        names.extend(["sin_doy", "cos_doy"])

    for fn in forcing_names:
        if fn not in dataset.forcings:
            raise KeyError(f"forcing '{fn}' not found in dataset.forcings")
        v = np.asarray(dataset.forcings[fn], dtype=float)[idx]
        cols.append(v)
        names.append(str(fn))

    X = np.column_stack(cols).astype(float)
    return X, names


class MixtureOfExpertsWorkflow:
    """Mixture-of-Experts workflow: physics model + BART residual expert."""

    def __init__(
        self,
        model: ShorelineModel,
        dataset_cal: TimeSeriesDataset,
        prior: Prior,
        dataset_full: TimeSeriesDataset | None = None,
    ) -> None:
        self.model = model
        self.dataset_cal = dataset_cal
        self.dataset_full = dataset_full or dataset_cal
        self.prior = prior

    def fit(
        self,
        *,
        # --- physics calibration ---
        physics_draws: int = 2000,
        physics_tune: int = 1000,
        physics_chains: int = 4,
        physics_random_seed: int = 42,
        physics_likelihood: str = "normal",
        physics_sigma: float | None = None,
        physics_estimate_sigma: bool = False,
        physics_include_bias: bool = True,
        physics_estimate_initial_position: bool = False,
        physics_nu_fixed: float | None = None,
        physics_nu_bounds: tuple[float, float] = (2.0, 50.0),
        physics_cores: int | None = None,
        # --- BART residual expert ---
        bart_cfg: BartConfig = BartConfig(),
        bart_forcings: Sequence[str] = ("E",),
        bart_include_time: bool = True,
        bart_include_doy: bool = False,
        bart_residual_tau_days: float | None = None,
        # --- gate ---)
        gate_cfg: GateConfig = GateConfig(),
        gate_forcings: Sequence[str] = ("E",),
        gate_include_time: bool = True,
        gate_include_doy: bool = False,
        progress: bool = True,
    ) -> MoEResult:
        """Fit the three components: physics posterior, BART residual, and gate."""

        # -------------------------
        # 1) Physics calibration
        # -------------------------
        trace_physics, _ppc = bayesian_calibrate(
            model=self.model,
            dataset=self.dataset_cal,
            prior=self.prior,
            draws=int(physics_draws),
            tune=int(physics_tune),
            chains=int(physics_chains),
            random_seed=int(physics_random_seed),
            likelihood=str(physics_likelihood),
            sigma=physics_sigma,
            estimate_sigma=bool(physics_estimate_sigma),
            include_bias=bool(physics_include_bias),
            estimate_initial_position=bool(physics_estimate_initial_position),
            nu_fixed=physics_nu_fixed,
            nu_bounds=physics_nu_bounds,
            cores=physics_cores,
        )

        raw = trace_physics.posterior["raw_par"].values  # (chain, draw, n_par)
        raw_flat = raw.reshape(raw.shape[0] * raw.shape[1], raw.shape[2])
        raw_med = np.median(raw_flat, axis=0)

        # optional bias and y0
        bias = 0.0
        if physics_include_bias and "bias" in trace_physics.posterior:
            bias = float(np.median(trace_physics.posterior["bias"].values))

        y0_val = float(self.dataset_cal.y0)
        if physics_estimate_initial_position and "y0" in trace_physics.posterior:
            y0_val = float(np.median(trace_physics.posterior["y0"].values))

        # deterministic physics prediction (posterior median)
        phys_med = transform_raw_to_physical(raw_med, self.model.parameters)
        ds_med = TimeSeriesDataset(
            time=self.dataset_cal.time,
            forcings=self.dataset_cal.forcings,
            obs_time=self.dataset_cal.obs_time,
            obs=self.dataset_cal.obs,
            y0=y0_val,
            idx_obs=self.dataset_cal.idx_obs,
            dt=self.dataset_cal.dt,
        )
        yhat_full = self.model.simulate(phys_med, ds_med)
        yhat_obs = np.asarray(yhat_full[ds_med.idx_obs], dtype=float) + bias

        residuals_raw = np.asarray(ds_med.obs, dtype=float) - yhat_obs

        # Optional: smooth the residual target to focus BART on low-frequency discrepancy.
        # This is often important when forcings are high-frequency but shoreline observations
        # are irregular and residuals contain substantial noise.
        if bart_residual_tau_days is not None:
            residuals = ewma_irregular(ds_med.obs_time, residuals_raw, tau_days=float(bart_residual_tau_days))
        else:
            residuals = residuals_raw

        # -------------------------
        # 2) BART residual expert
        # -------------------------
        X_bart_obs_raw, bart_names = build_design_matrix(
            self.dataset_cal,
            forcing_names=bart_forcings,
            include_time=bart_include_time,
            include_doy=bart_include_doy,
            at="obs",
        )
        X_bart_full_raw, _ = build_design_matrix(
            self.dataset_full,
            forcing_names=bart_forcings,
            include_time=bart_include_time,
            include_doy=bart_include_doy,
            at="full",
        )

        X_bart_obs, X_bart_full, _, _ = _standardize(X_bart_obs_raw, X_bart_full_raw)

        idata_bart, bart_model = fit_bart_residuals(
            X_obs=X_bart_obs,
            residuals=residuals,
            cfg=bart_cfg,
            feature_names=bart_names,
            progressbar=progress,
        )

        # in-sample BART mean (posterior mean over draws)
        mu_bart_obs_draws = predict_bart_mean(idata_bart, bart_model, X_bart_obs, random_seed=bart_cfg.random_seed)
        delta_obs_mean = np.mean(mu_bart_obs_draws, axis=0)

        # -------------------------
        # 3) Gate fit (logistic)
        # -------------------------
        X_gate_obs_raw, gate_names = build_design_matrix(
            self.dataset_cal,
            forcing_names=gate_forcings,
            include_time=gate_include_time,
            include_doy=gate_include_doy,
            at="obs",
        )
        X_gate_full_raw, _ = build_design_matrix(
            self.dataset_full,
            forcing_names=gate_forcings,
            include_time=gate_include_time,
            include_doy=gate_include_doy,
            at="full",
        )
        X_gate_obs, X_gate_full, _, _ = _standardize(X_gate_obs_raw, X_gate_full_raw)

        idata_gate = self._fit_gate(
            y_obs=self.dataset_cal.obs,
            yhat_physics_obs=yhat_obs,
            delta_obs=delta_obs_mean,
            X_gate_obs=X_gate_obs,
            cfg=gate_cfg,
            feature_names=gate_names,
            progressbar=progress,
        )

        return MoEResult(
            trace_physics=trace_physics,
            idata_bart=idata_bart,
            bart_model=bart_model,
            idata_gate=idata_gate,
            yhat_physics_obs=yhat_obs,
            residuals_obs=residuals_raw,
            X_bart_obs=X_bart_obs,
            X_bart_full=X_bart_full,
            X_gate_obs=X_gate_obs,
            X_gate_full=X_gate_full,
        )

    @staticmethod
    def _fit_gate(
        *,
        y_obs: np.ndarray,
        yhat_physics_obs: np.ndarray,
        delta_obs: np.ndarray,
        X_gate_obs: np.ndarray,
        cfg: GateConfig,
        feature_names: Sequence[str],
        progressbar: bool,
    ):
        import pymc as pm

        y_obs = np.asarray(y_obs, dtype=float)
        yhat_physics_obs = np.asarray(yhat_physics_obs, dtype=float)
        delta_obs = np.asarray(delta_obs, dtype=float)
        X_gate_obs = np.asarray(X_gate_obs, dtype=float)

        if y_obs.ndim != 1:
            raise ValueError("y_obs must be 1D")
        if X_gate_obs.ndim != 2:
            raise ValueError("X_gate_obs must be 2D")
        if y_obs.shape[0] != X_gate_obs.shape[0]:
            raise ValueError("X_gate_obs rows must match y_obs")

        n_obs, n_feat = X_gate_obs.shape
        coords = {"point": np.arange(n_obs), "feature": list(feature_names)}

        sigma0 = float(np.std(y_obs - yhat_physics_obs))
        sigma_scale = float(cfg.sigma_prior_scale) if cfg.sigma_prior_scale is not None else max(sigma0, 1e-6)

        with pm.Model(coords=coords) as model:
            Xg = pm.Data("Xg", X_gate_obs, dims=("point", "feature"))

            alpha = pm.Normal("alpha", mu=float(cfg.alpha_prior_mean), sigma=float(cfg.alpha_prior_sd))
            beta = pm.Normal("beta", mu=0.0, sigma=float(cfg.beta_prior_sd), dims=("feature",))

            lin = alpha + pm.math.dot(Xg, beta)
            w = pm.Deterministic("w", pm.math.sigmoid(lin), dims=("point",))

            mu = yhat_physics_obs + (1.0 - w) * delta_obs
            sigma = pm.HalfNormal("sigma", sigma=sigma_scale)
            pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs, dims=("point",))

            idata = pm.sample(
                draws=int(cfg.draws),
                tune=int(cfg.tune),
                chains=int(cfg.chains),
                random_seed=int(cfg.random_seed),
                target_accept=float(cfg.target_accept),
                cores=cfg.cores,
                progressbar=progressbar,
            )

        return idata

    @staticmethod
    def _extract_flat(idata, var_name: str) -> np.ndarray:
        arr = idata.posterior[var_name].values
        return arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])

    def predict_full(
        self,
        result: MoEResult,
        *,
        n_joint_draws: int = 2000,
        random_seed: int = 42,
    ) -> dict[str, np.ndarray]:
        """Monte Carlo MoE prediction on the *full* dataset time axis."""

        return self.predict_full_with_models(
            result,
            bart_model=result.bart_model,
            n_joint_draws=n_joint_draws,
            random_seed=random_seed,
        )

    def predict_full_with_models(
        self,
        result: MoEResult,
        *,
        bart_model,
        n_joint_draws: int = 2000,
        random_seed: int = 42,
    ) -> dict[str, np.ndarray]:
        """Same as predict_full, but requires the BART model object.

        The BART model is required to perform out-of-sample predictions in recent
        PyMC-BART versions via `pm.set_data` + `pm.sample_posterior_predictive`.
        """

        rng = np.random.default_rng(int(random_seed))

        # --- physics posterior samples
        raw = result.trace_physics.posterior["raw_par"].values
        raw_flat = raw.reshape(raw.shape[0] * raw.shape[1], raw.shape[2])

        idx_p = rng.choice(raw_flat.shape[0], size=int(n_joint_draws), replace=raw_flat.shape[0] < n_joint_draws)
        raw_sel = raw_flat[idx_p]

        # bias samples (optional)
        if "bias" in result.trace_physics.posterior:
            bias_flat = self._extract_flat(result.trace_physics, "bias")
            bias_sel = bias_flat[idx_p]
        else:
            bias_sel = np.zeros(int(n_joint_draws), dtype=float)

        # y0 samples (optional)
        if "y0" in result.trace_physics.posterior:
            y0_flat = self._extract_flat(result.trace_physics, "y0")
            y0_sel = y0_flat[idx_p]
        else:
            y0_sel = np.full(int(n_joint_draws), float(self.dataset_full.y0), dtype=float)

        # run physics model per selected draw
        n_t = self.dataset_full.time.shape[0]
        y_phys = np.empty((n_t, int(n_joint_draws)), dtype=float)
        for j, (rp, b, y0j) in enumerate(zip(raw_sel, bias_sel, y0_sel)):
            phys = transform_raw_to_physical(rp, self.model.parameters)
            dsj = TimeSeriesDataset(
                time=self.dataset_full.time,
                forcings=self.dataset_full.forcings,
                obs_time=self.dataset_full.obs_time,
                obs=self.dataset_full.obs,
                y0=float(y0j),
                idx_obs=self.dataset_full.idx_obs,
                dt=self.dataset_full.dt,
            )
            yy = self.model.simulate(phys, dsj)
            y_phys[:, j] = np.asarray(yy, dtype=float) + float(b)

        # --- BART discrepancy mean draws at full X
        mu_bart_full = predict_bart_mean(
            result.idata_bart,
            bart_model,
            result.X_bart_full,
            random_seed=random_seed,
        )  # (n_bart_samples, n_t)

        idx_d = rng.choice(mu_bart_full.shape[0], size=int(n_joint_draws), replace=mu_bart_full.shape[0] < n_joint_draws)
        delta_full = mu_bart_full[idx_d].T  # (n_t, n_joint_draws)

        # --- gate weights at full X (manual evaluation)
        alpha = self._extract_flat(result.idata_gate, "alpha")  # (n_gate_samples,)
        beta = self._extract_flat(result.idata_gate, "beta")    # (n_gate_samples, n_feat)

        idx_g = rng.choice(alpha.shape[0], size=int(n_joint_draws), replace=alpha.shape[0] < n_joint_draws)
        alpha_sel = alpha[idx_g]
        beta_sel = beta[idx_g]

        # compute w_full per joint draw
        Xg = np.asarray(result.X_gate_full, dtype=float)  # (n_t, n_feat)
        w_full = np.empty((n_t, int(n_joint_draws)), dtype=float)
        for j in range(int(n_joint_draws)):
            lin = float(alpha_sel[j]) + Xg @ beta_sel[j]
            lin = np.clip(lin, -60.0, 60.0)  # numerical stability
            w_full[:, j] = 1.0 / (1.0 + np.exp(-lin))

        # combine
        y_moe = y_phys + (1.0 - w_full) * delta_full

        # summary statistics
        per5 = np.percentile(y_moe, 5, axis=1)
        per50 = np.percentile(y_moe, 50, axis=1)
        per95 = np.percentile(y_moe, 95, axis=1)

        return {
            "time": np.asarray(self.dataset_full.time),
            "ensemble": y_moe,
            "p05": per5,
            "p50": per50,
            "p95": per95,
            "w_mean": np.mean(w_full, axis=1),
        }
