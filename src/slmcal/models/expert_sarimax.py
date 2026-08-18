from __future__ import annotations

"""SARIMAX shoreline expert.

This expert predicts shoreline position y(t) using a SARIMAX state-space model
from `statsmodels`.

Statsmodels is treated as an optional dependency.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from slmcal.data import TimeSeriesDataset
from slmcal.features import build_feature_matrix, interp_obs_to_forcing

from .expert import ExpertPrediction


def _require_statsmodels():
    try:
        import statsmodels.api as sm  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "statsmodels is required for SARIMAXExpert. Install it with `pip install statsmodels`."
        ) from e


@dataclass
class SARIMAXConfig:
    forcing_names: Sequence[str] = ("E", "hs", "tp")
    include_doy: bool = True
    include_time: bool = False
    standardize: bool = True

    calibration_mode: str = "obs_only"  # or "interp_to_forcing"
    interp_kind: str = "linear"

    order: tuple[int, int, int] = (1, 0, 0)
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)
    trend: str | None = "c"  # constant

    enforce_stationarity: bool = True
    enforce_invertibility: bool = True

    # Draw sampling
    noise_scale: float | None = None  # if None, use predicted variance


class SARIMAXExpert:
    def __init__(self, cfg: SARIMAXConfig = SARIMAXConfig(), name: str = "sarimax"):
        self.cfg = cfg
        self.name = name
        self._res = None
        self._ref_stats: tuple[np.ndarray, np.ndarray] | None = None

    def fit(self, dataset_cal: TimeSeriesDataset) -> "SARIMAXExpert":
        _require_statsmodels()
        import statsmodels.api as sm

        cfg = self.cfg
        mode = cfg.calibration_mode.lower().strip()
        if mode not in ("obs_only", "interp_to_forcing"):
            raise ValueError("calibration_mode must be 'obs_only' or 'interp_to_forcing'")

        # Build exogenous matrix on full forcing time grid
        X_full, _, ref_stats = build_feature_matrix(
            dataset_cal,
            cfg.forcing_names,
            include_doy=cfg.include_doy,
            include_time=cfg.include_time,
            at="full",
            standardize=cfg.standardize,
        )

        # Endog aligned to forcing time grid
        if mode == "obs_only":
            y = np.full(dataset_cal.time.shape[0], np.nan, dtype=float)
            idx = np.asarray(dataset_cal.idx_obs, dtype=int)
            y[idx] = np.asarray(dataset_cal.obs, dtype=float)
        else:
            y = interp_obs_to_forcing(dataset_cal.time, dataset_cal.obs_time, dataset_cal.obs, kind=cfg.interp_kind)

        mod = sm.tsa.SARIMAX(
            endog=y,
            exog=X_full if X_full.shape[1] > 0 else None,
            order=cfg.order,
            seasonal_order=cfg.seasonal_order,
            trend=cfg.trend,
            enforce_stationarity=cfg.enforce_stationarity,
            enforce_invertibility=cfg.enforce_invertibility,
        )

        self._res = mod.fit(disp=False)
        self._ref_stats = ref_stats
        return self

    def predict_draws(
        self,
        dataset_full: TimeSeriesDataset,
        *,
        n_draws: int,
        random_seed: int = 42,
    ) -> ExpertPrediction:
        if self._res is None:
            raise RuntimeError("Call fit() before predict_draws()")

        cfg = self.cfg
        X_full, _, _ = build_feature_matrix(
            dataset_full,
            cfg.forcing_names,
            include_doy=cfg.include_doy,
            include_time=cfg.include_time,
            at="full",
            standardize=cfg.standardize,
            ref_stats=self._ref_stats,
        )


        n_full = int(dataset_full.time.shape[0])
        # number of time steps used during fit (forcing-grid length)
        try:
            n_fit = int(getattr(self._res, "nobs"))
        except Exception:
            n_fit = int(getattr(getattr(self._res, "model", None), "nobs", n_full))

        # Statsmodels only expects *out-of-sample* exog values when extending the sample.
        # So we compute in-sample prediction separately and then append a forecast if needed.
        if n_full <= n_fit:
            pred_in = self._res.get_prediction(start=0, end=n_full - 1)
            mean = np.asarray(pred_in.predicted_mean, dtype=float)

            try:
                var = np.asarray(pred_in.var_pred_mean, dtype=float)
            except Exception:
                resid = np.asarray(getattr(self._res, "resid", np.array([])), dtype=float)
                var0 = float(np.nanvar(resid)) if resid.size else 1.0
                var = np.full(mean.shape, var0, dtype=float)
        else:
            # In-sample part
            pred_in = self._res.get_prediction(start=0, end=n_fit - 1)
            mean_in = np.asarray(pred_in.predicted_mean, dtype=float)
            try:
                var_in = np.asarray(pred_in.var_pred_mean, dtype=float)
            except Exception:
                resid = np.asarray(getattr(self._res, "resid", np.array([])), dtype=float)
                var0 = float(np.nanvar(resid)) if resid.size else 1.0
                var_in = np.full(mean_in.shape, var0, dtype=float)

            # Out-of-sample forecast
            steps = int(n_full - n_fit)
            if X_full.shape[1] > 0:
                exog_out = X_full[n_fit:, :]
            else:
                exog_out = None

            fc = self._res.get_forecast(steps=steps, exog=exog_out)
            mean_out = np.asarray(fc.predicted_mean, dtype=float)
            try:
                var_out = np.asarray(fc.var_pred_mean, dtype=float)
            except Exception:
                resid = np.asarray(getattr(self._res, "resid", np.array([])), dtype=float)
                var0 = float(np.nanvar(resid)) if resid.size else 1.0
                var_out = np.full(mean_out.shape, var0, dtype=float)

            mean = np.concatenate([mean_in, mean_out], axis=0)
            var = np.concatenate([var_in, var_out], axis=0)


        rng = np.random.default_rng(int(random_seed))
        if cfg.noise_scale is not None:
            sig = float(cfg.noise_scale)
            eps = rng.normal(0.0, sig, size=(int(n_draws), mean.shape[0]))
        else:
            sig = np.sqrt(np.clip(var, 1e-12, None))
            eps = rng.normal(0.0, 1.0, size=(int(n_draws), mean.shape[0])) * sig[None, :]

        draws = mean[None, :] + eps
        return ExpertPrediction(name=self.name, time=np.asarray(dataset_full.time), draws=draws)
