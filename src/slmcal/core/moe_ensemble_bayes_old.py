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

import numpy as np
import arviz as az

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
    - ``dynamic_ewma_tau_days`` applies an EWMA smoother to the gate design
      matrix before the softmax, yielding temporally smoother weights.
    """

    forcing_names: Sequence[str] = ("E", "hs", "tp")
    include_doy: bool = True
    include_time: bool = False
    standardize: bool = True

    # Optional automatic feature selection for a more parsimonious gate.
    feature_selection: Literal["manual", "differential_screen"] = "manual"
    candidate_forcing_names: Sequence[str] | None = None
    max_selected_features: int = 4
    min_abs_correlation: float = 0.0
    selection_score: Literal["max_abs_corr", "mean_abs_corr"] = "max_abs_corr"

    # Optional temporal smoothing of the gate inputs/logits via EWMA on the
    # design matrix (in days). None / <=0 keeps the legacy unsmoothed gate.
    dynamic_ewma_tau_days: float | None = None

    # Priors for softmax coefficients
    alpha_sd: float = 2.0
    beta_sd: float = 1.0

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

    max_k = int(cfg.max_selected_features) if cfg.max_selected_features is not None else len(ranked)
    max_k = max(1, min(max_k, len(ranked)))
    selected = list(ranked[:max_k])
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
        self._selected_forcing_names: list[str] = list(gate_cfg.forcing_names)
        self._gate_feature_names: list[str] | None = None
        self._screening_scores: dict[str, float] | None = None

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

        X_obs, _X_full_cal, feat_names, ref_stats = _prepare_gate_matrices(
            dataset_cal,
            forcing_names=self._selected_forcing_names,
            include_doy=self.gate_cfg.include_doy,
            include_time=self.gate_cfg.include_time,
            standardize=self.gate_cfg.standardize,
            dynamic_ewma_tau_days=self.gate_cfg.dynamic_ewma_tau_days,
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

        # Gate features on full grid (with the same selected forcings /
        # standardization / temporal smoothing used during fit).
        _X_obs_unused, X_full, _names, _ = _prepare_gate_matrices(
            dataset_full,
            forcing_names=self._selected_forcing_names,
            include_doy=self.gate_cfg.include_doy,
            include_time=self.gate_cfg.include_time,
            standardize=self.gate_cfg.standardize,
            dynamic_ewma_tau_days=self.gate_cfg.dynamic_ewma_tau_days,
            ref_stats=self._ref_stats,
        )

        # Expert draws
        preds: list[ExpertPrediction] = []
        for ex in self.experts:
            preds.append(ex.predict_draws(dataset_full, n_draws=int(n_draws), random_seed=int(random_seed)))

        K = len(preds)
        n_t = dataset_full.time.shape[0]

        # Sample gate coefficient draws
        alpha_s, beta_s, sigma_extra_s = _sample_gate_params(self._idata_gate, K=K, n=int(n_draws), rng=rng)

        # Compute weights per joint draw
        w = np.empty((n_t, int(n_draws), K), dtype=float)
        for j in range(int(n_draws)):
            w[:, j, :] = _softmax_from_params(alpha_s[j], beta_s[j], X_full)

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

        out_dir = Path(out_dir)

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



def _softmax_from_params(alpha: np.ndarray, beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute softmax weights for one coefficient draw.

    Parameters
    ----------
    alpha
        (K-1,) intercepts for classes 0..K-2. Class K-1 is baseline zeros.
    beta
        (K-1, p) coefficients.
    X
        (n, p) feature matrix.
    """

    X = np.asarray(X, dtype=float)
    K1, p = beta.shape
    n = X.shape[0]
    logits = np.zeros((n, K1 + 1), dtype=float)
    logits[:, :K1] = alpha[None, :] + X @ beta.T
    # baseline (last class) stays at 0
    logits = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(logits)
    return ez / np.sum(ez, axis=1, keepdims=True)


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
        logw = logits - pm.math.logsumexp(logits, axis=1, keepdims=True)
        pm.Deterministic("w", pm.math.softmax(logits, axis=1), dims=("point", "expert"))

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


def _sample_gate_params(idata, *, K: int, n: int, rng: np.random.Generator):
    """Sample (alpha, beta, sigma_extra) from idata, returning arrays for n draws."""

    # alpha: (chain, draw, K-1)
    alpha = idata.posterior["alpha"].values
    beta = idata.posterior["beta"].values
    alpha_f = alpha.reshape(alpha.shape[0] * alpha.shape[1], alpha.shape[2])
    beta_f = beta.reshape(beta.shape[0] * beta.shape[1], beta.shape[2], beta.shape[3])

    m = alpha_f.shape[0]
    idx = rng.choice(m, size=int(n), replace=(m < n))
    alpha_s = alpha_f[idx]
    beta_s = beta_f[idx]

    sigma_extra_s = None
    if "sigma_extra" in idata.posterior:
        se = idata.posterior["sigma_extra"].values.reshape(-1)
        sigma_extra_s = se[rng.choice(se.shape[0], size=int(n), replace=(se.shape[0] < n))]

    # sanity check
    if alpha_s.shape[1] != (K - 1):
        raise ValueError("Gate alpha dimension does not match number of experts")

    return alpha_s, beta_s, sigma_extra_s
