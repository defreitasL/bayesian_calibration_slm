from __future__ import annotations

r"""Bayesian Mixture-of-Experts ensemble (multi-expert).

This module complements :mod:`slmcal.core.moe_ensemble` (frequentist gate)
with a **Bayesian** gate that learns time-/covariate-dependent mixture weights
from observations.

Why a Bayesian gate?
-------------------
The frequentist gate in :class:`~slmcal.core.moe_ensemble.MoEEnsembleWorkflow`
is trained on a *winner-takes-all* label (which expert has the smallest error
at each observation). This is fast, but it discards information about
*how much* better an expert is and can overfit.

Here we fit a gate by maximizing the **predictive density** of a mixture model:

.. math::

    p(y_i | x_i) = \sum_{k=1}^K w_k(x_i)\; \mathcal{N}(y_i; \mu_{ik}, \sigma_{ik}^2 + \sigma_\text{extra}^2)

where (mu_ik, sigma_ik) are estimated from each expert's predictive draws at
observation times (so expert uncertainty is propagated).

This is a pragmatic, scalable MoE for shoreline-model ensembles:
1) Calibrate each expert (e.g., Yates09 and MD04) and obtain predictive draws.
2) Fit a Bayesian softmax gate on observations using the mixture likelihood.
3) Combine expert draws using gate-weight draws for full-period predictions.
"""

from dataclasses import dataclass
from typing import Sequence, Literal
import math
import re

import numpy as np
import arviz as az
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

from slmcal.data import TimeSeriesDataset
from slmcal.features import build_feature_matrix, ewma_irregular
from slmcal.models.expert import Expert, ExpertPrediction


@dataclass
class BayesGateConfig:
    """Configuration for the Bayesian softmax gate.

    Notes
    -----
    The legacy behaviour is preserved by default:
    - ``feature_selection='manual'`` uses ``forcing_names`` exactly as provided.
    - ``dynamic_ewma_tau_days=None`` disables temporal smoothing of the gate.

    New optional features:
    - ``feature_selection='differential_screen'`` ranks candidate forcings by how
      strongly they explain *relative expert skill* at observation times.
    - ``dynamic_ewma_tau_days`` may be a numeric tau in days or ``"auto"`` to
      select a smoothing scale by blocked time-series cross-validation.
    """

    forcing_names: Sequence[str] = ("E", "hs", "tp")
    include_doy: bool = True
    include_time: bool = False
    standardize: bool = True

    # Gate structure
    gate_mode: Literal["linear", "categorical_bmu"] = "linear"

    # Optional automatic feature selection for a more parsimonious linear gate.
    feature_selection: Literal["manual", "differential_screen"] = "manual"
    candidate_forcing_names: Sequence[str] | None = None
    max_selected_features: int = 4
    min_abs_correlation: float = 0.0
    selection_score: Literal["max_abs_corr", "mean_abs_corr"] = "max_abs_corr"
    screening_one_per_family: bool = False

    # Optional temporal smoothing of the linear gate inputs/logits via EWMA on the
    # design matrix (in days). None / <=0 keeps the legacy unsmoothed gate.
    # ``"auto"`` triggers a lightweight empirical-Bayes hyperparameter search.
    dynamic_ewma_tau_days: float | Literal["auto"] | None = None
    tau_auto_metric: Literal["logscore"] = "logscore"
    tau_auto_grid: Sequence[float] = (0.0, 3.0, 7.0, 14.0, 30.0, 60.0, 90.0)
    tau_auto_n_splits: int = 3
    tau_auto_one_se_rule: bool = True
    tau_auto_use_acf_center: bool = True
    tau_auto_min_train_obs: int = 48

    # Priors for softmax coefficients
    alpha_sd: float = 2.0
    beta_sd: float = 1.0

    # Categorical BMU gate options
    categorical_forcing_name: str = "bmus"
    bmu_partial_pooling: bool = True
    bmu_sigma_sd: float = 1.0
    use_bmu_dwell_time: bool = False
    dwell_time_transform: Literal["log1p", "none"] = "log1p"
    dwell_time_sd: float = 0.5

    # Robustness controls for the gate weights
    # - weight_floor shrinks weights toward the uniform mixture so no expert
    #   is ever completely ignored.
    # - entropy_reg_strength softly rewards higher-entropy weights in the
    #   posterior (0 disables the regularizer).
    weight_floor: float = 0.0
    entropy_reg_strength: float = 0.0

    # Optional extra noise term in the mixture likelihood (added in quadrature)
    estimate_sigma_extra: bool = True
    sigma_extra_prior_scale: float | None = None

    # Gate likelihood
    likelihood: Literal["normal", "studentt"] = "normal"

    # Student-t nu inference (used only if likelihood="studentt")
    # We sample nu = nu_min + Exponential(scale=nu_scale)
    nu_min: float = 2.0
    nu_scale: float = 10.0

    # MCMC
    draws: int = 2000
    tune: int = 2000
    chains: int = 4
    random_seed: int = 42
    cores: int | None = None
    target_accept: float = 0.9

@dataclass
class BayesMoEEnsembleResult:
    time: np.ndarray
    weights_mean: np.ndarray  # (n_time, K)
    expert_predictions: list[ExpertPrediction]
    ensemble: ExpertPrediction
    idata_gate: object
    gate_feature_names: list[str] | None = None
    selected_forcing_names: list[str] | None = None
    screening_scores: dict[str, float] | None = None
    selected_tau_days: float | None = None
    tau_search_summary: list[dict[str, float]] | None = None
    gate_diagnostics: dict[str, object] | None = None


def _expert_moments_at_obs(pred: ExpertPrediction, idx_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu, sd) at observation indices from expert draws."""

    d = np.asarray(pred.draws, dtype=float)  # (n_draws, n_time)
    y = d[:, idx_obs]  # (n_draws, n_obs)
    mu = np.mean(y, axis=0)
    sd = np.std(y, axis=0, ddof=1)
    sd = np.where(sd <= 0.0, 1e-6, sd)
    return mu, sd

def _safe_abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Absolute Pearson correlation, robust to NaNs / constants."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 3:
        return 0.0
    xx = x[mask]
    yy = y[mask]
    if np.nanstd(xx) <= 0.0 or np.nanstd(yy) <= 0.0:
        return 0.0
    return float(abs(np.corrcoef(xx, yy)[0, 1]))


def _normal_logpdf(y: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    sd = np.where(sd <= 0.0, 1e-6, sd)
    return -0.5 * ((y[:, None] - mu) / sd) ** 2 - np.log(sd) - 0.5 * np.log(2.0 * np.pi)


def _studentt_logpdf(y: np.ndarray, mu: np.ndarray, sd: np.ndarray, nu: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    sd = np.where(sd <= 0.0, 1e-6, sd)
    nu = float(max(nu, 2.001))
    z = (y[:, None] - mu) / sd
    return (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
        - np.log(sd)
        - ((nu + 1.0) / 2.0) * np.log1p((z**2) / nu)
    )


def _corr_with_time(time_like: np.ndarray, values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    t = np.arange(x.shape[0], dtype=float) if time_like is None else np.arange(np.asarray(time_like).shape[0], dtype=float)
    return _safe_abs_corr(t, x)


def _median_dt_days(time: np.ndarray) -> float:
    t = np.asarray(time)
    if t.size < 2:
        return 1.0
    if np.issubdtype(t.dtype, np.datetime64):
        dt = (t[1:] - t[:-1]) / np.timedelta64(1, "D")
        dt = np.asarray(dt, dtype=float)
    else:
        dt = np.diff(np.asarray(t, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if dt.size == 0:
        return 1.0
    return float(np.median(dt))


def _reference_studentt_nu(cfg: BayesGateConfig) -> float:
    return float(max(cfg.nu_min + cfg.nu_scale, cfg.nu_min + 2.0))


def _resolve_numeric_tau(tau: float | str | None) -> float | None:
    if tau is None:
        return None
    if isinstance(tau, str):
        val = tau.strip().lower()
        if val == "auto":
            return None
        return float(val)
    tau_f = float(tau)
    if not np.isfinite(tau_f) or tau_f <= 0.0:
        return None
    return tau_f



def _normalize_category_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
    except TypeError:
        pass
    return value


def _extract_category_levels(values: np.ndarray) -> list[object]:
    levels: list[object] = []
    seen: set[object] = set()
    for raw in np.asarray(values).ravel():
        val = _normalize_category_value(raw)
        if val is None or val in seen:
            continue
        seen.add(val)
        levels.append(val)
    try:
        levels = sorted(levels)
    except TypeError:
        pass
    return levels


def _map_categories_to_indices(values: np.ndarray, levels: Sequence[object]) -> np.ndarray:
    level_to_idx = {lvl: i for i, lvl in enumerate(list(levels))}
    out = np.full(np.asarray(values).shape[0], -1, dtype=int)
    for i, raw in enumerate(np.asarray(values).ravel()):
        val = _normalize_category_value(raw)
        if val is None:
            continue
        idx = level_to_idx.get(val)
        if idx is not None:
            out[i] = int(idx)
    return out


def _compute_category_dwell_time_days(time: np.ndarray, categories: np.ndarray) -> np.ndarray:
    time = np.asarray(time)
    cats = np.asarray(categories)
    n = cats.shape[0]
    dwell = np.zeros(n, dtype=float)
    if n == 0:
        return dwell

    if np.issubdtype(time.dtype, np.datetime64):
        t_num = time.astype("datetime64[ns]").astype(np.int64) / (24.0 * 3600.0 * 1e9)
    else:
        t_num = np.asarray(time, dtype=float)

    run_start = 0
    prev = _normalize_category_value(cats[0])
    for i in range(1, n):
        cur = _normalize_category_value(cats[i])
        same = (prev is not None) and (cur is not None) and (cur == prev)
        if same:
            delta = float(t_num[i] - t_num[run_start])
            dwell[i] = max(delta, 0.0) if np.isfinite(delta) else max(dwell[i - 1], 0.0)
        else:
            run_start = i
            dwell[i] = 0.0
        prev = cur
    return dwell


def _prepare_bmu_gate_inputs(
    dataset: TimeSeriesDataset,
    *,
    forcing_name: str,
    use_dwell_time: bool,
    dwell_time_transform: str,
    standardize: bool,
    ref_levels: Sequence[object] | None = None,
    ref_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[object], np.ndarray | None, np.ndarray | None, list[str], tuple[np.ndarray, np.ndarray]]:
    if dataset.idx_obs is None:
        raise ValueError("dataset.idx_obs is required to build BMU gate inputs")
    if forcing_name not in dataset.forcings:
        raise KeyError(f"Forcing '{forcing_name}' not found in dataset.forcings")

    idx_obs = np.asarray(dataset.idx_obs, dtype=int)
    raw = np.asarray(dataset.forcings[forcing_name])
    levels = list(ref_levels) if ref_levels is not None else _extract_category_levels(raw[idx_obs])
    if len(levels) == 0:
        raise ValueError(f"No valid categories were found in forcing '{forcing_name}' at observation times")

    bmu_full_idx = _map_categories_to_indices(raw, levels)
    bmu_obs_idx = bmu_full_idx[idx_obs]

    feat_names: list[str] = []
    mu_x = np.zeros(0, dtype=float)
    sig_x = np.ones(0, dtype=float)
    dwell_obs = None
    dwell_full = None

    if use_dwell_time:
        dwell_full = _compute_category_dwell_time_days(np.asarray(dataset.time), raw)
        transform = str(dwell_time_transform).lower().strip()
        if transform == "log1p":
            dwell_full = np.log1p(np.clip(dwell_full, 0.0, None))
            feat_names = ["log1p_dwell_time_days"]
        elif transform in ("none", "identity"):
            feat_names = ["dwell_time_days"]
        else:
            raise ValueError(
                "dwell_time_transform must be 'log1p' or 'none', "
                f"got {dwell_time_transform!r}"
            )
        dwell_obs = np.asarray(dwell_full[idx_obs], dtype=float)

        if standardize:
            if ref_stats is None:
                mu_x = np.asarray([np.nanmean(dwell_obs)], dtype=float)
                sig_x = np.asarray([np.nanstd(dwell_obs)], dtype=float)
                sig_x = np.where(np.isfinite(sig_x) & (sig_x > 0.0), sig_x, 1.0)
            else:
                mu_x = np.asarray(ref_stats[0], dtype=float)
                sig_x = np.asarray(ref_stats[1], dtype=float)
                sig_x = np.where(np.isfinite(sig_x) & (sig_x > 0.0), sig_x, 1.0)
            dwell_obs = (dwell_obs - mu_x[0]) / sig_x[0]
            dwell_full = (np.asarray(dwell_full, dtype=float) - mu_x[0]) / sig_x[0]
        else:
            mu_x = np.zeros(1, dtype=float)
            sig_x = np.ones(1, dtype=float)

    return bmu_obs_idx, bmu_full_idx, levels, dwell_obs, dwell_full, feat_names, (mu_x, sig_x)


def _categorical_softmax_from_params(
    mu_class: np.ndarray,
    gamma: np.ndarray,
    bmu_idx: np.ndarray,
    *,
    dwell: np.ndarray | None = None,
    rho_dwell: np.ndarray | None = None,
    weight_floor: float = 0.0,
) -> np.ndarray:
    bmu_idx = np.asarray(bmu_idx, dtype=int)
    mu_class = np.asarray(mu_class, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    K1 = mu_class.shape[0]
    K = K1 + 1
    n = bmu_idx.shape[0]

    logits_k1 = np.broadcast_to(mu_class[None, :], (n, K1)).astype(float).copy()
    known = bmu_idx >= 0
    if np.any(known):
        logits_k1[known, :] = gamma[:, bmu_idx[known]].T

    if dwell is not None and rho_dwell is not None:
        dwell = np.asarray(dwell, dtype=float)
        rho_dwell = np.asarray(rho_dwell, dtype=float)
        logits_k1 += dwell[:, None] * rho_dwell[None, :]

    logits = np.zeros((n, K), dtype=float)
    logits[:, :K1] = logits_k1
    logits = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(logits)
    w = ez / np.sum(ez, axis=1, keepdims=True)

    eps_floor = float(weight_floor or 0.0)
    if eps_floor < 0.0 or eps_floor >= 1.0:
        raise ValueError(f"weight_floor must be in [0, 1), got {eps_floor!r}")
    if eps_floor > 0.0:
        w = (1.0 - eps_floor) * w + (eps_floor / float(K))
    return w


def _estimate_skill_persistence_tau_days(
    *,
    dataset: TimeSeriesDataset,
    y_obs: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    cfg: BayesGateConfig,
) -> float | None:
    if y_obs.size < 8:
        return None
    like = str(cfg.likelihood).lower().strip()
    if like == "studentt":
        logp = _studentt_logpdf(y_obs, mu, sd, _reference_studentt_nu(cfg))
    else:
        logp = _normal_logpdf(y_obs, mu, sd)
    if logp.shape[1] < 2:
        return None
    delta = np.asarray(logp[:, 0] - logp[:, -1], dtype=float)
    delta = delta - np.nanmean(delta)
    mask = np.isfinite(delta)
    delta = delta[mask]
    if delta.size < 8:
        return None
    var = float(np.nanvar(delta))
    if not np.isfinite(var) or var <= 0.0:
        return None
    max_lag = min(delta.size - 1, max(2, delta.size // 3))
    if max_lag < 2:
        return None
    target = math.exp(-1.0)
    lag_cross = None
    for lag in range(1, max_lag + 1):
        a = delta[:-lag]
        b = delta[lag:]
        if a.size < 3 or np.nanstd(a) <= 0.0 or np.nanstd(b) <= 0.0:
            continue
        ac = float(np.corrcoef(a, b)[0, 1])
        if not np.isfinite(ac):
            continue
        if ac <= target:
            lag_cross = lag
            break
    if lag_cross is None:
        lag_cross = max_lag
    dt_days = _median_dt_days(np.asarray(dataset.obs_time))
    tau_days = float(lag_cross) * float(dt_days)
    if not np.isfinite(tau_days) or tau_days <= 0.0:
        return None
    return tau_days


def _build_tau_candidate_grid(cfg: BayesGateConfig, tau_hint_days: float | None) -> list[float]:
    vals: list[float] = []
    for tau in cfg.tau_auto_grid:
        tau_f = float(tau)
        if np.isfinite(tau_f) and tau_f >= 0.0:
            vals.append(tau_f)

    if bool(getattr(cfg, "tau_auto_use_acf_center", True)) and tau_hint_days is not None and np.isfinite(tau_hint_days):
        for fac in (0.5, 1.0, 2.0):
            vals.append(float(max(0.0, fac * tau_hint_days)))

    vals = sorted({round(v, 6) for v in vals if np.isfinite(v) and v >= 0.0})
    if not vals:
        vals = [0.0, 3.0, 7.0, 14.0, 30.0, 60.0]
    return [float(v) for v in vals]


def _build_forward_block_splits(n_obs: int, n_splits: int, min_train_obs: int) -> list[tuple[np.ndarray, np.ndarray]]:
    n_obs = int(n_obs)
    if n_obs < 4:
        return []
    min_train_obs = max(3, min(int(min_train_obs), n_obs - 1))
    remaining = n_obs - min_train_obs
    if remaining <= 0:
        train = np.arange(max(1, n_obs - 1), dtype=int)
        val = np.arange(max(1, n_obs - 1), n_obs, dtype=int)
        return [(train, val)] if val.size > 0 else []

    n_splits = max(1, int(n_splits))
    val_block = max(1, remaining // n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    train_end = min_train_obs
    while train_end < n_obs:
        val_end = min(n_obs, train_end + val_block)
        train_idx = np.arange(train_end, dtype=int)
        val_idx = np.arange(train_end, val_end, dtype=int)
        if train_idx.size >= 3 and val_idx.size >= 1:
            splits.append((train_idx, val_idx))
        train_end = val_end
    if not splits:
        train = np.arange(max(1, n_obs - 1), dtype=int)
        val = np.arange(max(1, n_obs - 1), n_obs, dtype=int)
        if val.size > 0:
            splits.append((train, val))
    return splits


def _map_theta_unpack(theta: np.ndarray, K: int, p: int, estimate_sigma_extra: bool) -> tuple[np.ndarray, np.ndarray, float]:
    K1 = K - 1
    alpha = np.asarray(theta[:K1], dtype=float)
    beta = np.asarray(theta[K1:K1 + (K1 * p)], dtype=float).reshape(K1, p)
    sigma_extra = 0.0
    if estimate_sigma_extra:
        sigma_extra = float(np.exp(theta[-1]))
    return alpha, beta, sigma_extra


def _mixture_logscore_numpy(
    *,
    y_obs: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    weights: np.ndarray,
    cfg: BayesGateConfig,
    sigma_extra: float = 0.0,
) -> np.ndarray:
    sig = np.sqrt(np.asarray(sd, dtype=float) ** 2 + float(max(sigma_extra, 0.0)) ** 2)
    like = str(cfg.likelihood).lower().strip()
    if like == "studentt":
        logp = _studentt_logpdf(y_obs, mu, sig, _reference_studentt_nu(cfg))
    else:
        logp = _normal_logpdf(y_obs, mu, sig)
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 1e-12, 1.0)
    return logsumexp(np.log(w) + logp, axis=1)


def _fit_gate_map_for_tau(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    mu_train: np.ndarray,
    sd_train: np.ndarray,
    cfg: BayesGateConfig,
) -> dict[str, np.ndarray | float | bool]:
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    mu_train = np.asarray(mu_train, dtype=float)
    sd_train = np.asarray(sd_train, dtype=float)

    n_obs, p = X_train.shape
    K = int(mu_train.shape[1])
    K1 = K - 1
    use_sigma_extra = bool(cfg.estimate_sigma_extra)
    sigma0 = float(max(np.std(y_train - np.mean(mu_train, axis=1)), 1e-3))
    theta0 = np.zeros(K1 + (K1 * p) + (1 if use_sigma_extra else 0), dtype=float)
    if use_sigma_extra:
        theta0[-1] = math.log(max(0.1 * sigma0, 1e-3))

    alpha_sd = float(max(cfg.alpha_sd, 1e-6))
    beta_sd = float(max(cfg.beta_sd, 1e-6))
    sigma_scale = float(cfg.sigma_extra_prior_scale) if cfg.sigma_extra_prior_scale is not None else max(sigma0, 1e-3)

    def objective(theta: np.ndarray) -> float:
        alpha, beta, sigma_extra = _map_theta_unpack(theta, K=K, p=p, estimate_sigma_extra=use_sigma_extra)
        w = _softmax_from_params(alpha, beta, X_train, weight_floor=float(getattr(cfg, "weight_floor", 0.0) or 0.0))
        logscore = _mixture_logscore_numpy(
            y_obs=y_train,
            mu=mu_train,
            sd=sd_train,
            weights=w,
            cfg=cfg,
            sigma_extra=sigma_extra,
        )
        obj = -float(np.sum(logscore))
        obj += 0.5 * float(np.sum((alpha / alpha_sd) ** 2))
        if beta.size > 0:
            obj += 0.5 * float(np.sum((beta / beta_sd) ** 2))
        if use_sigma_extra:
            sigma_extra = float(max(sigma_extra, 1e-9))
            obj += 0.5 * (sigma_extra / sigma_scale) ** 2 - math.log(sigma_extra)
        ent_strength = float(getattr(cfg, "entropy_reg_strength", 0.0) or 0.0)
        if ent_strength > 0.0:
            w_safe = np.clip(w, 1e-12, 1.0)
            ent = -np.sum(w_safe * np.log(w_safe), axis=1) / math.log(float(K))
            obj -= ent_strength * float(np.sum(ent))
        return float(obj)

    res = minimize(objective, theta0, method="L-BFGS-B")
    alpha, beta, sigma_extra = _map_theta_unpack(np.asarray(res.x, dtype=float), K=K, p=p, estimate_sigma_extra=use_sigma_extra)
    return {
        "alpha": alpha,
        "beta": beta,
        "sigma_extra": float(sigma_extra),
        "objective": float(res.fun),
        "success": bool(res.success),
    }


def _auto_select_gate_tau_days(
    *,
    dataset: TimeSeriesDataset,
    y_obs: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    selected_forcing_names: Sequence[str],
    cfg: BayesGateConfig,
) -> tuple[float | None, list[dict[str, float]]]:
    tau_hint = _estimate_skill_persistence_tau_days(dataset=dataset, y_obs=y_obs, mu=mu, sd=sd, cfg=cfg)
    tau_grid = _build_tau_candidate_grid(cfg, tau_hint)
    splits = _build_forward_block_splits(
        n_obs=int(y_obs.shape[0]),
        n_splits=int(getattr(cfg, "tau_auto_n_splits", 3)),
        min_train_obs=int(getattr(cfg, "tau_auto_min_train_obs", 48)),
    )
    if not splits:
        return None, []

    summary: list[dict[str, float]] = []
    for tau in tau_grid:
        X_obs_tau, _X_full_tau, _feat_names, _stats = _prepare_gate_matrices(
            dataset,
            forcing_names=selected_forcing_names,
            include_doy=cfg.include_doy,
            include_time=cfg.include_time,
            standardize=cfg.standardize,
            dynamic_ewma_tau_days=float(tau),
            ref_stats=None,
        )
        split_scores: list[float] = []
        for train_idx, val_idx in splits:
            fit_res = _fit_gate_map_for_tau(
                X_train=X_obs_tau[train_idx],
                y_train=y_obs[train_idx],
                mu_train=mu[train_idx],
                sd_train=sd[train_idx],
                cfg=cfg,
            )
            w_val = _softmax_from_params(
                np.asarray(fit_res["alpha"], dtype=float),
                np.asarray(fit_res["beta"], dtype=float),
                X_obs_tau[val_idx],
                weight_floor=float(getattr(cfg, "weight_floor", 0.0) or 0.0),
            )
            logscore_val = _mixture_logscore_numpy(
                y_obs=y_obs[val_idx],
                mu=mu[val_idx],
                sd=sd[val_idx],
                weights=w_val,
                cfg=cfg,
                sigma_extra=float(fit_res["sigma_extra"]),
            )
            split_scores.append(float(np.mean(logscore_val)))
        if split_scores:
            mean_score = float(np.mean(split_scores))
            std_score = float(np.std(split_scores, ddof=1)) if len(split_scores) > 1 else 0.0
            sem_score = std_score / math.sqrt(len(split_scores)) if len(split_scores) > 0 else 0.0
            summary.append({
                "tau_days": float(tau),
                "mean_logscore": mean_score,
                "std_logscore": std_score,
                "sem_logscore": sem_score,
                "n_splits": float(len(split_scores)),
                "tau_hint_days": float(tau_hint) if tau_hint is not None else np.nan,
            })

    if not summary:
        return None, []

    best_idx = int(np.argmax([row["mean_logscore"] for row in summary]))
    best_mean = float(summary[best_idx]["mean_logscore"])
    best_sem = float(summary[best_idx].get("sem_logscore", 0.0))
    selected = float(summary[best_idx]["tau_days"])

    if bool(getattr(cfg, "tau_auto_one_se_rule", True)):
        threshold = best_mean - best_sem
        eligible = [row for row in summary if float(row["mean_logscore"]) >= threshold]
        if eligible:
            selected = float(max(eligible, key=lambda row: float(row["tau_days"]))["tau_days"])

    return selected, summary


def _posterior_mean_gate_diagnostics(
    *,
    idata,
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    dataset: TimeSeriesDataset,
    feature_names: Sequence[str],
    selected_forcing_names: Sequence[str],
    expert_names: Sequence[str],
    cfg: BayesGateConfig,
) -> dict[str, object]:
    K = int(mu.shape[1])
    alpha_mean = np.asarray(idata.posterior["alpha"].values, dtype=float).mean(axis=(0, 1))
    beta_mean = np.asarray(idata.posterior["beta"].values, dtype=float).mean(axis=(0, 1))
    sigma_extra_mean = 0.0
    if "sigma_extra" in idata.posterior:
        sigma_extra_mean = float(np.asarray(idata.posterior["sigma_extra"].values, dtype=float).mean())

    obs_weights = _softmax_from_params(
        alpha_mean,
        beta_mean,
        np.asarray(X_obs, dtype=float),
        weight_floor=float(getattr(cfg, "weight_floor", 0.0) or 0.0),
    )
    intercept_only_weights = _softmax_from_params(
        alpha_mean,
        beta_mean,
        np.zeros((1, np.asarray(X_obs).shape[1]), dtype=float),
        weight_floor=float(getattr(cfg, "weight_floor", 0.0) or 0.0),
    )[0]

    like = str(cfg.likelihood).lower().strip()
    sig = np.sqrt(np.asarray(sd, dtype=float) ** 2 + sigma_extra_mean ** 2)
    if like == "studentt":
        logp = _studentt_logpdf(y_obs, mu, sig, _reference_studentt_nu(cfg))
    else:
        logp = _normal_logpdf(y_obs, mu, sig)

    delta_logp = None
    if logp.shape[1] == 2:
        delta_logp = np.asarray(logp[:, 0] - logp[:, 1], dtype=float)

    feature_time_corr: dict[str, float] = {}
    feature_delta_corr: dict[str, float] = {}
    idx_obs = np.asarray(dataset.idx_obs, dtype=int)
    for name in selected_forcing_names:
        vals = np.asarray(dataset.forcings[name], dtype=float)[idx_obs]
        feature_time_corr[str(name)] = _corr_with_time(np.asarray(dataset.obs_time), vals)
        if delta_logp is None:
            feature_delta_corr[str(name)] = np.nan
        else:
            feature_delta_corr[str(name)] = _safe_abs_corr(vals, delta_logp)

    return {
        "expert_names": list(expert_names),
        "baseline_expert": str(expert_names[-1]) if expert_names else None,
        "intercept_only_weights": intercept_only_weights,
        "obs_weights_mean": obs_weights,
        "delta_logp_obs": delta_logp,
        "feature_time_corr": feature_time_corr,
        "feature_delta_logp_corr": feature_delta_corr,
        "selected_forcing_names": list(selected_forcing_names),
        "feature_names": list(feature_names),
        "obs_time": np.asarray(dataset.obs_time),
    }


def _posterior_mean_categorical_gate_diagnostics(
    *,
    idata,
    bmu_obs_idx: np.ndarray,
    dwell_obs: np.ndarray | None,
    bmu_levels: Sequence[object],
    expert_names: Sequence[str],
    cfg: BayesGateConfig,
) -> dict[str, object]:
    mu_class_mean = np.asarray(idata.posterior["mu_class"].values, dtype=float).mean(axis=(0, 1))
    gamma_mean = np.asarray(idata.posterior["gamma"].values, dtype=float).mean(axis=(0, 1))
    rho_mean = None
    if "rho_dwell" in idata.posterior:
        rho_mean = np.asarray(idata.posterior["rho_dwell"].values, dtype=float).mean(axis=(0, 1))

    obs_weights = _categorical_softmax_from_params(
        mu_class_mean,
        gamma_mean,
        np.asarray(bmu_obs_idx, dtype=int),
        dwell=dwell_obs,
        rho_dwell=rho_mean,
        weight_floor=float(getattr(cfg, "weight_floor", 0.0) or 0.0),
    )
    intercept_only_weights = _categorical_softmax_from_params(
        mu_class_mean,
        gamma_mean,
        np.full(1, -1, dtype=int),
        dwell=(np.zeros(1, dtype=float) if dwell_obs is not None else None),
        rho_dwell=rho_mean,
        weight_floor=float(getattr(cfg, "weight_floor", 0.0) or 0.0),
    )[0]

    category_weights = np.empty((len(bmu_levels), len(expert_names)), dtype=float)
    for i in range(len(bmu_levels)):
        category_weights[i, :] = _categorical_softmax_from_params(
            mu_class_mean,
            gamma_mean,
            np.asarray([i], dtype=int),
            dwell=(np.zeros(1, dtype=float) if dwell_obs is not None else None),
            rho_dwell=rho_mean,
            weight_floor=float(getattr(cfg, "weight_floor", 0.0) or 0.0),
        )[0]

    counts = np.bincount(np.clip(np.asarray(bmu_obs_idx, dtype=int), 0, max(len(bmu_levels) - 1, 0)), minlength=len(bmu_levels))
    if np.any(np.asarray(bmu_obs_idx, dtype=int) < 0):
        counts = counts.astype(int)
    return {
        "expert_names": list(expert_names),
        "baseline_expert": str(expert_names[-1]) if expert_names else None,
        "gate_mode": "categorical_bmu",
        "obs_weights_mean": obs_weights,
        "intercept_only_weights": intercept_only_weights,
        "bmu_levels": list(bmu_levels),
        "bmu_counts_obs": counts,
        "weights_by_bmu_at_entry": category_weights,
        "uses_dwell_time": bool(dwell_obs is not None),
    }


def _forcing_family(name: str) -> str:
    """Return a coarse forcing family name for screening diversity."""
    name = str(name)
    m = re.match(r"^(.*)_(?:ewma|movavg)\d+$", name)
    return m.group(1) if m else name


def _apply_screening_diversity(
    ranked: Sequence[str],
    cfg: BayesGateConfig,
) -> list[str]:
    """Prune redundant screened features while preserving ranking order."""
    ranked_list = list(ranked)
    max_k = int(cfg.max_selected_features) if cfg.max_selected_features is not None else len(ranked_list)
    max_k = max(1, min(max_k, len(ranked_list)))
    if not bool(cfg.screening_one_per_family):
        return ranked_list[:max_k]

    selected: list[str] = []
    used_families: set[str] = set()
    for name in ranked_list:
        fam = _forcing_family(name)
        if fam in used_families:
            continue
        selected.append(name)
        used_families.add(fam)
        if len(selected) >= max_k:
            break

    if not selected:
        selected = ranked_list[:max_k]
    return selected


def _screen_forcing_names_by_differential_skill(
    *,
    dataset: TimeSeriesDataset,
    candidate_names: Sequence[str],
    y_obs: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    cfg: BayesGateConfig,
) -> tuple[list[str], dict[str, float]]:
    """Rank forcings by association with relative expert skill.

    The screening target is the differential pointwise expert predictive skill,
    approximated from Gaussian expert predictive moments at observation times.
    For K experts, we compare each non-baseline expert against the baseline
    expert used by the softmax parameterization (the last expert).
    """
    if dataset.idx_obs is None:
        raise ValueError("dataset.idx_obs is required for differential screening")

    idx_obs = np.asarray(dataset.idx_obs, dtype=int)
    like = str(cfg.likelihood).lower().strip()
    if like == "studentt":
        logp = _studentt_logpdf(y_obs, mu, sd, _reference_studentt_nu(cfg))
    else:
        logp = _normal_logpdf(y_obs, mu, sd)
    baseline = logp.shape[1] - 1
    diffs = [logp[:, k] - logp[:, baseline] for k in range(logp.shape[1] - 1)]
    if not diffs:
        raise ValueError("Differential screening requires at least 2 experts")

    scores: dict[str, float] = {}
    for name in candidate_names:
        if name not in dataset.forcings:
            raise KeyError(f"Forcing '{name}' not found in dataset.forcings")
        x = np.asarray(dataset.forcings[name], dtype=float)[idx_obs]
        corr_vals = np.asarray([_safe_abs_corr(x, d) for d in diffs], dtype=float)
        if cfg.selection_score == "mean_abs_corr":
            score = float(np.nanmean(corr_vals))
        else:
            score = float(np.nanmax(corr_vals))
        scores[name] = score

    ranked = sorted(candidate_names, key=lambda n: scores.get(n, 0.0), reverse=True)
    ranked = [n for n in ranked if scores.get(n, 0.0) >= float(cfg.min_abs_correlation)]
    if not ranked:
        ranked = sorted(candidate_names, key=lambda n: scores.get(n, 0.0), reverse=True)

    selected = _apply_screening_diversity(ranked, cfg)
    return selected, scores


def _resolve_gate_forcing_names(
    *,
    dataset: TimeSeriesDataset,
    y_obs: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    cfg: BayesGateConfig,
) -> tuple[list[str], dict[str, float] | None]:
    """Return the forcing names to use in the gate.

    Defaults to the legacy manual list. When ``feature_selection`` is enabled,
    it ranks ``candidate_forcing_names`` (or ``forcing_names`` if omitted)
    using differential expert skill at observation times.
    """
    mode = str(cfg.feature_selection).lower().strip()
    if mode == "manual":
        return list(cfg.forcing_names), None
    if mode != "differential_screen":
        raise ValueError(
            "feature_selection must be 'manual' or 'differential_screen', "
            f"got {cfg.feature_selection!r}"
        )

    candidates = list(cfg.candidate_forcing_names) if cfg.candidate_forcing_names is not None else list(cfg.forcing_names)
    if len(candidates) == 0:
        raise ValueError("Differential screening requested but no candidate forcings were provided")

    selected, scores = _screen_forcing_names_by_differential_skill(
        dataset=dataset,
        candidate_names=candidates,
        y_obs=y_obs,
        mu=mu,
        sd=sd,
        cfg=cfg,
    )
    print("[MoE gate] differential screening selected forcings:", selected)
    return selected, scores


def _prepare_gate_matrices(
    dataset: TimeSeriesDataset,
    *,
    forcing_names: Sequence[str],
    include_doy: bool,
    include_time: bool,
    standardize: bool,
    dynamic_ewma_tau_days: float | None,
    ref_stats: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, list[str], tuple[np.ndarray, np.ndarray]]:
    """Build gate design matrices for obs and full grid.

    The matrix is first built on the full forcing grid, optionally smoothed in
    time using EWMA, and then sampled at observation times. Standardization is
    always learned from the observation rows and then reused on the full grid,
    which keeps fit and predict consistent.
    """
    X_full_raw, feat_names, _ = build_feature_matrix(
        dataset,
        forcing_names,
        include_doy=include_doy,
        include_time=include_time,
        at="full",
        standardize=False,
    )

    tau = None if dynamic_ewma_tau_days is None else float(dynamic_ewma_tau_days)
    if tau is not None and tau > 0.0 and X_full_raw.shape[1] > 0:
        X_full_proc = np.empty_like(X_full_raw, dtype=float)
        for j in range(X_full_raw.shape[1]):
            X_full_proc[:, j] = ewma_irregular(dataset.time, X_full_raw[:, j], tau_days=tau)
    else:
        X_full_proc = np.asarray(X_full_raw, dtype=float)

    if dataset.idx_obs is None:
        raise ValueError("dataset.idx_obs is required to build gate matrices")
    idx_obs = np.asarray(dataset.idx_obs, dtype=int)
    X_obs_raw = np.asarray(X_full_proc[idx_obs], dtype=float)

    if not standardize or X_full_proc.shape[1] == 0:
        if ref_stats is None:
            mu_x = np.zeros(X_full_proc.shape[1], dtype=float)
            sig_x = np.ones(X_full_proc.shape[1], dtype=float)
        else:
            mu_x, sig_x = ref_stats
        return X_obs_raw, X_full_proc, feat_names, (np.asarray(mu_x, dtype=float), np.asarray(sig_x, dtype=float))

    if ref_stats is None:
        mu_x = np.nanmean(X_obs_raw, axis=0)
        sig_x = np.nanstd(X_obs_raw, axis=0)
        sig_x = np.where(np.isfinite(sig_x) & (sig_x > 0.0), sig_x, 1.0)
    else:
        mu_x, sig_x = ref_stats
        mu_x = np.asarray(mu_x, dtype=float)
        sig_x = np.asarray(sig_x, dtype=float)
        sig_x = np.where(np.isfinite(sig_x) & (sig_x > 0.0), sig_x, 1.0)

    X_obs = (X_obs_raw - mu_x) / sig_x
    X_full = (X_full_proc - mu_x) / sig_x
    return X_obs, X_full, feat_names, (mu_x, sig_x)


class BayesMoEEnsembleWorkflow:
    """Fit experts and a Bayesian softmax gate using a mixture likelihood."""

    def __init__(
        self,
        experts: Sequence[Expert],
        gate_cfg: BayesGateConfig = BayesGateConfig(),
        *,
        name: str = "moe_bayes",
    ) -> None:
        self.experts = list(experts)
        self.gate_cfg = gate_cfg
        self.name = name

        self._idata_gate = None
        self._ref_stats: tuple[np.ndarray, np.ndarray] | None = None
        self._last_result: BayesMoEEnsembleResult | None = None
        self._gate_mode: str = str(gate_cfg.gate_mode).lower().strip()
        self._selected_forcing_names: list[str] = list(gate_cfg.forcing_names)
        self._selected_tau_days: float | None = _resolve_numeric_tau(gate_cfg.dynamic_ewma_tau_days)
        self._tau_search_summary: list[dict[str, float]] | None = None
        self._gate_feature_names: list[str] | None = None
        self._screening_scores: dict[str, float] | None = None
        self._gate_diagnostics: dict[str, object] | None = None
        self._bmu_levels: list[object] | None = None

    @property
    def idata_gate(self):
        return self._idata_gate

    def fit(
        self,
        dataset_cal: TimeSeriesDataset,
        *,
        n_gate_draws: int = 400,
        random_seed: int = 42,
        progress: bool = True,
    ) -> "BayesMoEEnsembleWorkflow":
        """Fit experts then infer gate posterior on observation times."""

        # 1) Fit experts
        for ex in self.experts:
            ex.fit(dataset_cal)

        if dataset_cal.idx_obs is None:
            raise ValueError("dataset_cal.idx_obs is required")
        idx_obs = np.asarray(dataset_cal.idx_obs, dtype=int)

        # 2) Expert predictive moments at obs
        preds_obs: list[ExpertPrediction] = []
        mu_list = []
        sd_list = []
        for ex in self.experts:
            pred = ex.predict_draws(dataset_cal, n_draws=int(n_gate_draws), random_seed=int(random_seed))
            preds_obs.append(pred)
            mu_k, sd_k = _expert_moments_at_obs(pred, idx_obs)
            mu_list.append(mu_k)
            sd_list.append(sd_k)

        mu = np.column_stack(mu_list)  # (n_obs, K)
        sd = np.column_stack(sd_list)  # (n_obs, K)

        y_obs = np.asarray(dataset_cal.obs, dtype=float)
        if y_obs.shape[0] != mu.shape[0]:
            raise ValueError("Mismatch: dataset_cal.obs length and gate obs features")

        gate_mode = str(self.gate_cfg.gate_mode).lower().strip()
        self._gate_mode = gate_mode

        if gate_mode == "categorical_bmu":
            self._selected_forcing_names = [str(self.gate_cfg.categorical_forcing_name)]
            self._screening_scores = None
            self._selected_tau_days = None
            self._tau_search_summary = None

            bmu_obs_idx, _bmu_full_cal_idx, bmu_levels, dwell_obs, _dwell_full_cal, feat_names, ref_stats = _prepare_bmu_gate_inputs(
                dataset_cal,
                forcing_name=str(self.gate_cfg.categorical_forcing_name),
                use_dwell_time=bool(self.gate_cfg.use_bmu_dwell_time),
                dwell_time_transform=str(self.gate_cfg.dwell_time_transform),
                standardize=bool(self.gate_cfg.standardize),
                ref_levels=None,
                ref_stats=None,
            )
            self._bmu_levels = list(bmu_levels)
            self._ref_stats = ref_stats
            self._gate_feature_names = list(feat_names)

            self._idata_gate = _fit_bayes_categorical_bmu_gate(
                bmu_obs_idx=bmu_obs_idx,
                dwell_obs=dwell_obs,
                bmu_levels=bmu_levels,
                y_obs=y_obs,
                mu=mu,
                sd=sd,
                cfg=self.gate_cfg,
                progress=progress,
            )

            self._gate_diagnostics = _posterior_mean_categorical_gate_diagnostics(
                idata=self._idata_gate,
                bmu_obs_idx=bmu_obs_idx,
                dwell_obs=dwell_obs,
                bmu_levels=bmu_levels,
                expert_names=[ex.name for ex in self.experts],
                cfg=self.gate_cfg,
            )
        elif gate_mode == "linear":
            # 3) Optional differential-skill screening + parsimonious gate +
            # optional temporal smoothing of the gate inputs.
            selected_forcing_names, screening_scores = _resolve_gate_forcing_names(
                dataset=dataset_cal,
                y_obs=y_obs,
                mu=mu,
                sd=sd,
                cfg=self.gate_cfg,
            )
            self._selected_forcing_names = list(selected_forcing_names)
            self._screening_scores = screening_scores

            tau_cfg = self.gate_cfg.dynamic_ewma_tau_days
            tau_search_summary: list[dict[str, float]] | None = None
            if isinstance(tau_cfg, str) and tau_cfg.strip().lower() == "auto":
                selected_tau, tau_search_summary = _auto_select_gate_tau_days(
                    dataset=dataset_cal,
                    y_obs=y_obs,
                    mu=mu,
                    sd=sd,
                    selected_forcing_names=self._selected_forcing_names,
                    cfg=self.gate_cfg,
                )
                self._selected_tau_days = selected_tau
                self._tau_search_summary = tau_search_summary
                print(f"[MoE gate] selected dynamic_ewma_tau_days={self._selected_tau_days}")
            else:
                self._selected_tau_days = _resolve_numeric_tau(tau_cfg)
                self._tau_search_summary = None

            X_obs, _X_full_cal, feat_names, ref_stats = _prepare_gate_matrices(
                dataset_cal,
                forcing_names=self._selected_forcing_names,
                include_doy=self.gate_cfg.include_doy,
                include_time=self.gate_cfg.include_time,
                standardize=self.gate_cfg.standardize,
                dynamic_ewma_tau_days=self._selected_tau_days,
                ref_stats=None,
            )
            self._ref_stats = ref_stats
            self._gate_feature_names = list(feat_names)

            # 4) Fit Bayesian softmax gate
            self._idata_gate = _fit_bayes_softmax_gate(
                X_obs=X_obs,
                y_obs=y_obs,
                mu=mu,
                sd=sd,
                cfg=self.gate_cfg,
                feature_names=feat_names,
                progress=progress,
            )

            self._gate_diagnostics = _posterior_mean_gate_diagnostics(
                idata=self._idata_gate,
                X_obs=X_obs,
                y_obs=y_obs,
                mu=mu,
                sd=sd,
                dataset=dataset_cal,
                feature_names=feat_names,
                selected_forcing_names=self._selected_forcing_names,
                expert_names=[ex.name for ex in self.experts],
                cfg=self.gate_cfg,
            )
        else:
            raise ValueError(f"Unsupported gate_mode {self.gate_cfg.gate_mode!r}. Use 'linear' or 'categorical_bmu'.")

        return self

    def predict(
        self,
        dataset_full: TimeSeriesDataset,
        *,
        n_draws: int = 2000,
        random_seed: int = 42,
    ) -> BayesMoEEnsembleResult:
        if self._idata_gate is None:
            raise RuntimeError("Call fit() before predict()")

        rng = np.random.default_rng(int(random_seed))

        gate_mode = str(self._gate_mode).lower().strip()
        if gate_mode == "categorical_bmu":
            _bmu_obs_unused, bmu_full_idx, _bmu_levels, _dwell_obs_unused, dwell_full, _names, _ = _prepare_bmu_gate_inputs(
                dataset_full,
                forcing_name=str(self.gate_cfg.categorical_forcing_name),
                use_dwell_time=bool(self.gate_cfg.use_bmu_dwell_time),
                dwell_time_transform=str(self.gate_cfg.dwell_time_transform),
                standardize=bool(self.gate_cfg.standardize),
                ref_levels=(self._bmu_levels or []),
                ref_stats=self._ref_stats,
            )
        else:
            # Gate features on full grid (with the same selected forcings /
            # standardization / temporal smoothing used during fit).
            _X_obs_unused, X_full, _names, _ = _prepare_gate_matrices(
                dataset_full,
                forcing_names=self._selected_forcing_names,
                include_doy=self.gate_cfg.include_doy,
                include_time=self.gate_cfg.include_time,
                standardize=self.gate_cfg.standardize,
                dynamic_ewma_tau_days=self._selected_tau_days,
                ref_stats=self._ref_stats,
            )

        # Expert draws
        preds: list[ExpertPrediction] = []
        for ex in self.experts:
            preds.append(ex.predict_draws(dataset_full, n_draws=int(n_draws), random_seed=int(random_seed)))

        K = len(preds)
        n_t = dataset_full.time.shape[0]

        # Sample gate coefficient draws
        gate_params = _sample_gate_params(self._idata_gate, K=K, n=int(n_draws), rng=rng, gate_mode=gate_mode)
        sigma_extra_s = gate_params.get("sigma_extra")

        # Compute weights per joint draw
        w = np.empty((n_t, int(n_draws), K), dtype=float)
        for j in range(int(n_draws)):
            if gate_mode == "categorical_bmu":
                w[:, j, :] = _categorical_softmax_from_params(
                    gate_params["mu_class"][j],
                    gate_params["gamma"][j],
                    bmu_full_idx,
                    dwell=dwell_full,
                    rho_dwell=gate_params.get("rho_dwell", None)[j] if gate_params.get("rho_dwell", None) is not None else None,
                    weight_floor=float(getattr(self.gate_cfg, "weight_floor", 0.0) or 0.0),
                )
            else:
                w[:, j, :] = _softmax_from_params(
                    gate_params["alpha"][j],
                    gate_params["beta"][j],
                    X_full,
                    weight_floor=float(getattr(self.gate_cfg, "weight_floor", 0.0) or 0.0),
                )

        # Blend draws
        ens_draws = np.zeros((int(n_draws), n_t), dtype=float)
        for k in range(K):
            d_k = np.asarray(preds[k].draws, dtype=float)
            if d_k.shape != (int(n_draws), n_t):
                raise ValueError(
                    f"Expert '{preds[k].name}' returned draws with shape {d_k.shape}, "
                    f"but MoE predict expected {(int(n_draws), n_t)}."
                )
            ens_draws += (w[:, :, k].T) * d_k

        # Optional: add sigma_extra as additional observation noise
        # (this is *not* epistemic; treat as aleatory for predictive envelopes)
        if sigma_extra_s is not None and np.any(sigma_extra_s > 0):
            eps = rng.normal(0.0, sigma_extra_s[:, None], size=ens_draws.shape)
            ens_draws = ens_draws + eps

        ens = ExpertPrediction(name=self.name, time=np.asarray(dataset_full.time), draws=ens_draws)

        weights_mean = np.mean(w, axis=1)  # (n_t, K)
        result = BayesMoEEnsembleResult(
            time=np.asarray(dataset_full.time),
            weights_mean=weights_mean,
            expert_predictions=preds,
            ensemble=ens,
            idata_gate=self._idata_gate,
            gate_feature_names=list(self._gate_feature_names) if self._gate_feature_names is not None else None,
            selected_forcing_names=list(self._selected_forcing_names),
            screening_scores=dict(self._screening_scores) if self._screening_scores is not None else None,
            selected_tau_days=(float(self._selected_tau_days) if self._selected_tau_days is not None else None),
            tau_search_summary=(list(self._tau_search_summary) if self._tau_search_summary is not None else None),
            gate_diagnostics=self._gate_diagnostics,
        )
        self._last_result = result
        return result


    def save_default_plots(
        self,
        *,
        out_dir,
        dataset_full: TimeSeriesDataset,
        result: BayesMoEEnsembleResult | None = None,
        n_draws_plot: int | None = None,
        random_seed: int = 42,
        split_date: np.datetime64 | str | None = None,
        make_expert_plots: bool = True,
        expert_draws_plot: int = 1500,
    ) -> None:
        """Save default plots for the full MoE workflow (experts + ensemble + gate).

        It creates two folders inside ``out_dir``:

        - ``experts/<expert_name>/``: default plot set per expert (if supported)
        - ``moe/``: default plot set for the MoE ensemble, including:
          * posterior predictive envelope
          * model comparison (experts vs MoE)
          * gate-weights heatmap
          * gate sampler diagnostics table (r_hat / ESS)

        Notes
        -----
        This uses :func:`slmcal.plotting.make_default_calibration_plots`.
        """
        from pathlib import Path
        from slmcal.plotting import make_default_calibration_plots
        from slmcal.plotting.plots import (
            plot_categorical_bmu_effect_strength,
            plot_categorical_bmu_gate_heatmap,
            plot_categorical_bmu_gate_timeline,
            save_categorical_bmu_gate_summary,
        )

        out_dir = Path(out_dir)

        gate_mode = str(self._gate_mode).lower().strip()

        # Ensure we have a result
        if result is None:
            result = self._last_result
        if result is None:
            if n_draws_plot is None:
                n_draws_plot = 2000
            result = self.predict(dataset_full, n_draws=int(n_draws_plot), random_seed=int(random_seed))

        # Cal/val masks from split_date (if provided)
        obs_mask_cal = None
        obs_mask_val = None
        split_dt = None
        if split_date is not None and np.issubdtype(dataset_full.obs_time.dtype, np.datetime64):
            split_dt = np.datetime64(split_date)
            obs_mask_cal = np.asarray(dataset_full.obs_time) < split_dt
            obs_mask_val = np.asarray(dataset_full.obs_time) >= split_dt

        # --- Expert plots ---
        if make_expert_plots:
            exp_root = out_dir / "experts"
            exp_root.mkdir(parents=True, exist_ok=True)
            for ex in self.experts:
                sub = exp_root / getattr(ex, "name", "expert")
                try:
                    if hasattr(ex, "save_default_plots"):
                        ex.save_default_plots(
                            out_dir=sub,
                            dataset_full=dataset_full,
                            n_draws=int(expert_draws_plot),
                            random_seed=int(random_seed),
                            split_date=split_dt,
                        )
                    else:
                        # Minimal expert plots: envelope only
                        pred = ex.predict_draws(dataset_full, n_draws=int(expert_draws_plot), random_seed=int(random_seed))
                        dd = np.asarray(pred.draws, dtype=float)
                        per1 = np.nanpercentile(dd, 1, axis=0)
                        per5 = np.nanpercentile(dd, 5, axis=0)
                        per10 = np.nanpercentile(dd, 10, axis=0)
                        per50 = np.nanpercentile(dd, 50, axis=0)
                        per90 = np.nanpercentile(dd, 90, axis=0)
                        per95 = np.nanpercentile(dd, 95, axis=0)
                        per99 = np.nanpercentile(dd, 99, axis=0)
                        mini = np.nanmin(dd, axis=0)
                        maxi = np.nanmax(dd, axis=0)
                        make_default_calibration_plots(
                            out_dir=sub,
                            time=np.asarray(dataset_full.time),
                            per5=per5,
                            per10=per10,
                            per50=per50,
                            per90=per90,
                            per95=per95,
                            per1=per1,
                            per99=per99,
                            mini=mini,
                            maxi=maxi,
                            draws=dd,
                            obs_time=np.asarray(dataset_full.obs_time),
                            obs=np.asarray(dataset_full.obs, dtype=float),
                            label=getattr(ex, "name", "expert"),
                            obs_mask_cal=obs_mask_cal,
                            obs_mask_val=obs_mask_val,
                            split_date=split_dt,
                        )
                except Exception as e:
                    print(f"[WARN] Could not save plots for expert {getattr(ex,'name','expert')}: {e}")

        # --- MoE ensemble plots ---
        moe_dir = out_dir / "moe"
        moe_dir.mkdir(parents=True, exist_ok=True)

        dd = np.asarray(result.ensemble.draws, dtype=float)
        per1 = np.nanpercentile(dd, 1, axis=0)
        per5 = np.nanpercentile(dd, 5, axis=0)
        per10 = np.nanpercentile(dd, 10, axis=0)
        per50 = np.nanpercentile(dd, 50, axis=0)
        per90 = np.nanpercentile(dd, 90, axis=0)
        per95 = np.nanpercentile(dd, 95, axis=0)
        per99 = np.nanpercentile(dd, 99, axis=0)
        mini = np.nanmin(dd, axis=0)
        maxi = np.nanmax(dd, axis=0)

        compare_models: dict[str, dict[str, np.ndarray]] = {}
        for pred in result.expert_predictions:
            ddd = np.asarray(pred.draws, dtype=float)
            compare_models[pred.name] = {
                "per1": np.nanpercentile(ddd, 1, axis=0),
                "per5": np.nanpercentile(ddd, 5, axis=0),
                "per10": np.nanpercentile(ddd, 10, axis=0),
                "per50": np.nanpercentile(ddd, 50, axis=0),
                "per90": np.nanpercentile(ddd, 90, axis=0),
                "per95": np.nanpercentile(ddd, 95, axis=0),
                "per99": np.nanpercentile(ddd, 99, axis=0),
                "mini": np.nanmin(ddd, axis=0),
                "maxi": np.nanmax(ddd, axis=0),
            }
        compare_models[result.ensemble.name] = {
            "per1": per1,
            "per5": per5,
            "per10": per10,
            "per50": per50,
            "per90": per90,
            "per95": per95,
            "per99": per99,
            "mini": mini,
            "maxi": maxi,
        }

        make_default_calibration_plots(
            out_dir=moe_dir,
            time=np.asarray(dataset_full.time),
            per5=per5,
            per10=per10,
            per50=per50,
            per90=per90,
            per95=per95,
            per1=per1,
            per99=per99,
            mini=mini,
            maxi=maxi,
            draws=dd,
            obs_time=np.asarray(dataset_full.obs_time),
            obs=np.asarray(dataset_full.obs, dtype=float),
            trace=result.idata_gate,
            ppc=None,
            label=self.name,
            obs_mask_cal=obs_mask_cal,
            obs_mask_val=obs_mask_val,
            split_date=split_dt,
            compare_models=compare_models,
            gate_weights=np.asarray(result.weights_mean, dtype=float),
            gate_model_names=[p.name for p in result.expert_predictions],
        )

        if gate_mode == "categorical_bmu":
            try:
                moe_diag_dir = moe_dir / "diagnostics"
                moe_diag_dir.mkdir(parents=True, exist_ok=True)

                bmu_obs_idx, bmu_full_idx, bmu_levels, _dwell_obs, _dwell_full, _feat_names, _ = _prepare_bmu_gate_inputs(
                    dataset_full,
                    forcing_name=str(self.gate_cfg.categorical_forcing_name),
                    use_dwell_time=bool(self.gate_cfg.use_bmu_dwell_time),
                    dwell_time_transform=str(self.gate_cfg.dwell_time_transform),
                    standardize=bool(self.gate_cfg.standardize),
                    ref_levels=(self._bmu_levels or []),
                    ref_stats=self._ref_stats,
                )

                plot_categorical_bmu_gate_heatmap(
                    weights=np.asarray(result.weights_mean, dtype=float),
                    bmu_full_idx=np.asarray(bmu_full_idx, dtype=int),
                    bmu_obs_idx=np.asarray(bmu_obs_idx, dtype=int),
                    bmu_levels=list(bmu_levels),
                    model_names=[p.name for p in result.expert_predictions],
                    out_dir=moe_dir,
                    label=self.name,
                    idata=result.idata_gate,
                    weight_floor=float(getattr(self.gate_cfg, "weight_floor", 0.0) or 0.0),
                )

                plot_categorical_bmu_gate_timeline(
                    time=np.asarray(dataset_full.time),
                    weights=np.asarray(result.weights_mean, dtype=float),
                    bmu_full_idx=np.asarray(bmu_full_idx, dtype=int),
                    bmu_levels=list(bmu_levels),
                    model_names=[p.name for p in result.expert_predictions],
                    out_dir=moe_dir,
                    label=self.name,
                    split_date=split_dt,
                )

                save_categorical_bmu_gate_summary(
                    weights=np.asarray(result.weights_mean, dtype=float),
                    bmu_full_idx=np.asarray(bmu_full_idx, dtype=int),
                    bmu_obs_idx=np.asarray(bmu_obs_idx, dtype=int),
                    bmu_levels=list(bmu_levels),
                    model_names=[p.name for p in result.expert_predictions],
                    out_dir=moe_diag_dir,
                    label=self.name,
                    idata=result.idata_gate,
                    weight_floor=float(getattr(self.gate_cfg, "weight_floor", 0.0) or 0.0),
                )

                plot_categorical_bmu_effect_strength(
                    idata=result.idata_gate,
                    model_names=[p.name for p in result.expert_predictions],
                    out_dir=moe_diag_dir,
                    label=self.name,
                )
            except Exception as e:
                print(f"[WARN] Could not save categorical BMU gate diagnostics: {e}")



def _softmax_from_params(
    alpha: np.ndarray,
    beta: np.ndarray,
    X: np.ndarray,
    *,
    weight_floor: float = 0.0,
) -> np.ndarray:
    """Compute softmax weights for one coefficient draw.

    Parameters
    ----------
    alpha
        (K-1,) intercepts for classes 0..K-2. Class K-1 is baseline zeros.
    beta
        (K-1, p) coefficients.
    X
        (n, p) feature matrix.
    weight_floor
        Shrinkage toward the uniform mixture. With ``weight_floor=0.1`` and
        ``K=2``, each expert receives at least 5% weight.
    """

    X = np.asarray(X, dtype=float)
    K1, p = beta.shape
    n = X.shape[0]
    logits = np.zeros((n, K1 + 1), dtype=float)
    logits[:, :K1] = alpha[None, :] + X @ beta.T
    logits = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(logits)
    w = ez / np.sum(ez, axis=1, keepdims=True)

    eps_floor = float(weight_floor or 0.0)
    if eps_floor < 0.0 or eps_floor >= 1.0:
        raise ValueError(f"weight_floor must be in [0, 1), got {eps_floor!r}")
    if eps_floor > 0.0:
        w = (1.0 - eps_floor) * w + (eps_floor / float(K1 + 1))
    return w


def _fit_bayes_softmax_gate(
    *,
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    cfg: BayesGateConfig,
    feature_names: Sequence[str],
    progress: bool,
    gate_likelihood: str | None = None,   # <-- NEW
):
    """Fit a Bayesian softmax gate using mixture likelihood (Normal or Student-t)."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    X_obs = np.asarray(X_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)

    n_obs, p = X_obs.shape
    K = mu.shape[1]
    if K < 2:
        raise ValueError("Need at least 2 experts")
    if mu.shape != sd.shape:
        raise ValueError("mu and sd must have the same shape")
    if mu.shape[0] != n_obs:
        raise ValueError("mu rows must match X_obs rows")

    like = (gate_likelihood or cfg.likelihood).lower().strip()
    if like not in ("normal", "studentt"):
        raise ValueError(f"gate_likelihood must be 'normal' or 'studentt', got {like!r}")

    coords = {
        "point": np.arange(n_obs),
        "feature": list(feature_names),
        "class": np.arange(K - 1),
        "expert": np.arange(K),
    }

    resid0 = y_obs - np.mean(mu, axis=1)
    sigma0 = float(np.std(resid0))
    sigma_scale = float(cfg.sigma_extra_prior_scale) if cfg.sigma_extra_prior_scale is not None else max(sigma0, 1e-6)

    with pm.Model(coords=coords) as model:
        # ---- standardize X (so beta magnitudes are comparable across features) ----
        if bool(cfg.standardize):
            X_mean = X_obs.mean(axis=0)
            X_std = X_obs.std(axis=0)
            X_std[X_std == 0] = 1.0
            X_use = (X_obs - X_mean) / X_std
        else:
            X_use = X_obs

        Xd = pm.Data("X", X_use, dims=("point", "feature"))
        y = pm.Data("y_obs", y_obs, dims=("point",))
        mu_d = pm.Data("mu", mu, dims=("point", "expert"))
        sd_d = pm.Data("sd", sd, dims=("point", "expert"))

        alpha = pm.Normal("alpha", mu=0.0, sigma=float(cfg.alpha_sd), dims=("class",))
        beta = pm.Normal("beta", mu=0.0, sigma=float(cfg.beta_sd), dims=("class", "feature"))

        logits_k1 = alpha + pt.dot(Xd, beta.T)  # (n_obs, K-1)
        logits = pt.concatenate([logits_k1, pt.zeros((n_obs, 1))], axis=1)  # baseline = last expert

        # stable log-softmax for mixture weights
        logw_raw = logits - pm.math.logsumexp(logits, axis=1, keepdims=True)
        w_raw = pt.exp(logw_raw)

        eps_floor = float(getattr(cfg, "weight_floor", 0.0) or 0.0)
        if eps_floor < 0.0 or eps_floor >= 1.0:
            raise ValueError(f"weight_floor must be in [0, 1), got {eps_floor!r}")

        if eps_floor > 0.0:
            w = (1.0 - eps_floor) * w_raw + (eps_floor / float(K))
            logw = pt.log(w)
        else:
            w = w_raw
            logw = logw_raw

        pm.Deterministic("w", w, dims=("point", "expert"))

        ent_strength = float(getattr(cfg, "entropy_reg_strength", 0.0) or 0.0)
        if ent_strength > 0.0:
            w_safe = pt.clip(w, 1e-12, 1.0)
            ent = -pt.sum(w_safe * pt.log(w_safe), axis=1) / np.log(float(K))
            pm.Deterministic("gate_entropy", ent, dims=("point",))
            pm.Potential("entropy_regularizer", ent_strength * pt.sum(ent))

        if cfg.estimate_sigma_extra:
            sigma_extra = pm.HalfNormal("sigma_extra", sigma=sigma_scale)
        else:
            sigma_extra = pt.as_tensor_variable(0.0)

        sig = pt.sqrt(sd_d**2 + sigma_extra**2)  # (n_obs, K)

        # log-likelihood per expert
        if like == "normal":
            logp = (
                -0.5 * ((y[:, None] - mu_d) / sig) ** 2
                - pt.log(sig)
                - 0.5 * np.log(2.0 * np.pi)
            )
        else:
            # infer nu (not fixed)
            # nu = nu_min + Exponential(scale=nu_scale)
            nu_minus = pm.Exponential("nu_minus", lam=1.0 / float(cfg.nu_scale))
            nu = pm.Deterministic("nu", float(cfg.nu_min) + nu_minus)

            z = (y[:, None] - mu_d) / sig
            logp = (
                pt.gammaln((nu + 1.0) / 2.0)
                - pt.gammaln(nu / 2.0)
                - 0.5 * pt.log(nu * np.pi)
                - pt.log(sig)
                - ((nu + 1.0) / 2.0) * pt.log1p((z**2) / nu)
            )

        # mixture loglik per obs
        loglike = pt.logsumexp(logw + logp, axis=1)
        pm.Potential("likelihood", pt.sum(loglike))

        idata = pm.sample(
            draws=int(cfg.draws),
            tune=int(cfg.tune),
            chains=int(cfg.chains),
            random_seed=int(cfg.random_seed),
            target_accept=float(cfg.target_accept),
            cores=cfg.cores,
            progressbar=bool(progress),
        )

        print(az.summary(idata))

    return idata


def _fit_bayes_categorical_bmu_gate(
    *,
    bmu_obs_idx: np.ndarray,
    dwell_obs: np.ndarray | None,
    bmu_levels: Sequence[object],
    y_obs: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    cfg: BayesGateConfig,
    progress: bool,
    gate_likelihood: str | None = None,
):
    """Fit a Bayesian softmax gate indexed by categorical BMU states."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    bmu_obs_idx = np.asarray(bmu_obs_idx, dtype=int)
    y_obs = np.asarray(y_obs, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    dwell_obs = None if dwell_obs is None else np.asarray(dwell_obs, dtype=float)

    n_obs = y_obs.shape[0]
    K = mu.shape[1]
    C = len(bmu_levels)
    if K < 2:
        raise ValueError("Need at least 2 experts")
    if C < 1:
        raise ValueError("Categorical BMU gate requires at least one observed category")
    if mu.shape != sd.shape:
        raise ValueError("mu and sd must have the same shape")
    if mu.shape[0] != n_obs or bmu_obs_idx.shape[0] != n_obs:
        raise ValueError("BMU gate inputs must match the number of observations")
    if dwell_obs is not None and dwell_obs.shape[0] != n_obs:
        raise ValueError("dwell_obs length must match y_obs length")

    like = (gate_likelihood or cfg.likelihood).lower().strip()
    if like not in ("normal", "studentt"):
        raise ValueError(f"gate_likelihood must be 'normal' or 'studentt', got {like!r}")

    coords = {
        "point": np.arange(n_obs),
        "class": np.arange(K - 1),
        "expert": np.arange(K),
        "bmu_category": [str(v) for v in bmu_levels],
    }

    resid0 = y_obs - np.mean(mu, axis=1)
    sigma0 = float(np.std(resid0))
    sigma_scale = float(cfg.sigma_extra_prior_scale) if cfg.sigma_extra_prior_scale is not None else max(sigma0, 1e-6)

    with pm.Model(coords=coords) as model:
        bmu_idx_d = pm.Data("bmu_idx", bmu_obs_idx.astype("int32"), dims=("point",))
        y = pm.Data("y_obs", y_obs, dims=("point",))
        mu_d = pm.Data("mu", mu, dims=("point", "expert"))
        sd_d = pm.Data("sd", sd, dims=("point", "expert"))

        mu_class = pm.Normal("mu_class", mu=0.0, sigma=float(cfg.alpha_sd), dims=("class",))
        if bool(getattr(cfg, "bmu_partial_pooling", True)):
            sigma_class = pm.HalfNormal("sigma_class", sigma=float(getattr(cfg, "bmu_sigma_sd", 1.0)), dims=("class",))
            z_bmu = pm.Normal("z_bmu", mu=0.0, sigma=1.0, dims=("class", "bmu_category"))
            gamma = pm.Deterministic(
                "gamma",
                mu_class[:, None] + sigma_class[:, None] * z_bmu,
                dims=("class", "bmu_category"),
            )
        else:
            gamma = pm.Normal(
                "gamma",
                mu=mu_class[:, None],
                sigma=float(max(cfg.beta_sd, 1e-6)),
                dims=("class", "bmu_category"),
            )

        known = pt.ge(bmu_idx_d, 0)
        safe_idx = pt.cast(pt.switch(known, bmu_idx_d, 0), "int32")
        gamma_lookup = gamma.T[safe_idx]
        global_logits = pt.tile(mu_class[None, :], (n_obs, 1))
        logits_k1 = pt.switch(known[:, None], gamma_lookup, global_logits)

        if dwell_obs is not None:
            dwell_d = pm.Data("dwell_time", dwell_obs, dims=("point",))
            rho_dwell = pm.Normal("rho_dwell", mu=0.0, sigma=float(getattr(cfg, "dwell_time_sd", 0.5)), dims=("class",))
            logits_k1 = logits_k1 + dwell_d[:, None] * rho_dwell[None, :]

        logits = pt.concatenate([logits_k1, pt.zeros((n_obs, 1))], axis=1)
        logw_raw = logits - pm.math.logsumexp(logits, axis=1, keepdims=True)
        w_raw = pt.exp(logw_raw)

        eps_floor = float(getattr(cfg, "weight_floor", 0.0) or 0.0)
        if eps_floor < 0.0 or eps_floor >= 1.0:
            raise ValueError(f"weight_floor must be in [0, 1), got {eps_floor!r}")

        if eps_floor > 0.0:
            w = (1.0 - eps_floor) * w_raw + (eps_floor / float(K))
            logw = pt.log(w)
        else:
            w = w_raw
            logw = logw_raw

        pm.Deterministic("w", w, dims=("point", "expert"))

        ent_strength = float(getattr(cfg, "entropy_reg_strength", 0.0) or 0.0)
        if ent_strength > 0.0:
            w_safe = pt.clip(w, 1e-12, 1.0)
            ent = -pt.sum(w_safe * pt.log(w_safe), axis=1) / np.log(float(K))
            pm.Deterministic("gate_entropy", ent, dims=("point",))
            pm.Potential("entropy_regularizer", ent_strength * pt.sum(ent))

        if cfg.estimate_sigma_extra:
            sigma_extra = pm.HalfNormal("sigma_extra", sigma=sigma_scale)
        else:
            sigma_extra = pt.as_tensor_variable(0.0)

        sig = pt.sqrt(sd_d**2 + sigma_extra**2)

        if like == "normal":
            logp = (
                -0.5 * ((y[:, None] - mu_d) / sig) ** 2
                - pt.log(sig)
                - 0.5 * np.log(2.0 * np.pi)
            )
        else:
            nu_minus = pm.Exponential("nu_minus", lam=1.0 / float(cfg.nu_scale))
            nu = pm.Deterministic("nu", float(cfg.nu_min) + nu_minus)
            z = (y[:, None] - mu_d) / sig
            logp = (
                pt.gammaln((nu + 1.0) / 2.0)
                - pt.gammaln(nu / 2.0)
                - 0.5 * pt.log(nu * np.pi)
                - pt.log(sig)
                - ((nu + 1.0) / 2.0) * pt.log1p((z**2) / nu)
            )

        loglike = pt.logsumexp(logw + logp, axis=1)
        pm.Potential("likelihood", pt.sum(loglike))

        idata = pm.sample(
            draws=int(cfg.draws),
            tune=int(cfg.tune),
            chains=int(cfg.chains),
            random_seed=int(cfg.random_seed),
            target_accept=float(cfg.target_accept),
            cores=cfg.cores,
            progressbar=bool(progress),
        )

        print(az.summary(idata))

    return idata


def _sample_gate_params(idata, *, K: int, n: int, rng: np.random.Generator, gate_mode: str = "linear"):
    """Sample gate parameters from idata for either linear or categorical gates."""
    mode = str(gate_mode).lower().strip()
    posterior = idata.posterior

    if mode == "linear":
        alpha = posterior["alpha"].values
        beta = posterior["beta"].values
        alpha_f = alpha.reshape(alpha.shape[0] * alpha.shape[1], alpha.shape[2])
        beta_f = beta.reshape(beta.shape[0] * beta.shape[1], beta.shape[2], beta.shape[3])

        m = alpha_f.shape[0]
        idx = rng.choice(m, size=int(n), replace=(m < n))
        alpha_s = alpha_f[idx]
        beta_s = beta_f[idx]

        if alpha_s.shape[1] != (K - 1):
            raise ValueError("Gate alpha dimension does not match number of experts")

        out = {"alpha": alpha_s, "beta": beta_s}
    elif mode == "categorical_bmu":
        mu_class = posterior["mu_class"].values
        gamma = posterior["gamma"].values
        mu_class_f = mu_class.reshape(mu_class.shape[0] * mu_class.shape[1], mu_class.shape[2])
        gamma_f = gamma.reshape(gamma.shape[0] * gamma.shape[1], gamma.shape[2], gamma.shape[3])

        m = mu_class_f.shape[0]
        idx = rng.choice(m, size=int(n), replace=(m < n))
        mu_class_s = mu_class_f[idx]
        gamma_s = gamma_f[idx]
        if mu_class_s.shape[1] != (K - 1):
            raise ValueError("Gate mu_class dimension does not match number of experts")

        out = {"mu_class": mu_class_s, "gamma": gamma_s}
        if "rho_dwell" in posterior:
            rho = posterior["rho_dwell"].values
            rho_f = rho.reshape(rho.shape[0] * rho.shape[1], rho.shape[2])
            out["rho_dwell"] = rho_f[idx]
    else:
        raise ValueError(f"Unsupported gate_mode {gate_mode!r}")

    sigma_extra_s = None
    if "sigma_extra" in posterior:
        se = posterior["sigma_extra"].values.reshape(-1)
        sigma_extra_s = se[rng.choice(se.shape[0], size=int(n), replace=(se.shape[0] < n))]
    out["sigma_extra"] = sigma_extra_s
    return out
