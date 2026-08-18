from __future__ import annotations

"""Direct BART shoreline expert.

This expert predicts shoreline position y(t) directly as a function of
forcings/features, **not** as a residual correction.

Requires optional dependency: `pymc-bart`.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from slmcal.data import TimeSeriesDataset
from slmcal.features import build_feature_matrix, interp_obs_to_forcing
from slmcal.bayes.bart import BartConfig, fit_bart_residuals, predict_bart_mean

from .expert import ExpertPrediction


@dataclass
class BARTDirectConfig:
    forcing_names: Sequence[str] = ("E", "hs", "tp")
    include_doy: bool = True
    include_time: bool = False
    standardize: bool = True

    # "obs_only" (default): train on observation timestamps only
    # "interp_to_forcing": interpolate obs to forcing time grid and train on dense targets
    calibration_mode: str = "obs_only"
    interp_kind: str = "linear"

    # underlying PyMC-BART settings
    bart: BartConfig = BartConfig()

    # add Gaussian noise to the mean-function draws using posterior sigma
    add_noise: bool = True


class BARTDirectExpert:
    """BART regression expert for shoreline position."""

    def __init__(self, cfg: BARTDirectConfig = BARTDirectConfig(), name: str = "bart"):
        self.cfg = cfg
        self.name = name
        self._idata = None
        self._model = None
        self._feat_names: list[str] | None = None
        self._ref_stats: tuple[np.ndarray, np.ndarray] | None = None

    def fit(self, dataset_cal: TimeSeriesDataset) -> "BARTDirectExpert":
        cfg = self.cfg

        mode = cfg.calibration_mode.lower().strip()
        if mode not in ("obs_only", "interp_to_forcing"):
            raise ValueError("calibration_mode must be 'obs_only' or 'interp_to_forcing'")

        if mode == "obs_only":
            X, feat_names, ref_stats = build_feature_matrix(
                dataset_cal,
                cfg.forcing_names,
                include_doy=cfg.include_doy,
                include_time=cfg.include_time,
                at="obs",
                standardize=cfg.standardize,
            )
            y = np.asarray(dataset_cal.obs, dtype=float)
        else:
            y_interp = interp_obs_to_forcing(
                dataset_cal.time,
                dataset_cal.obs_time,
                dataset_cal.obs,
                kind=cfg.interp_kind,
            )
            ok = np.isfinite(y_interp)
            ds_tmp = dataset_cal
            X_full, feat_names, ref_stats = build_feature_matrix(
                ds_tmp,
                cfg.forcing_names,
                include_doy=cfg.include_doy,
                include_time=cfg.include_time,
                at="full",
                standardize=cfg.standardize,
            )
            X = X_full[ok]
            y = y_interp[ok]

        idata, model = fit_bart_residuals(
            X_obs=X,
            residuals=y,
            cfg=cfg.bart,
            feature_names=feat_names,
        )

        self._idata = idata
        self._model = model
        self._feat_names = list(feat_names)
        self._ref_stats = ref_stats
        return self

    def predict_draws(
        self,
        dataset_full: TimeSeriesDataset,
        *,
        n_draws: int,
        random_seed: int = 42,
    ) -> ExpertPrediction:
        if self._idata is None or self._model is None:
            raise RuntimeError("Call fit() before predict_draws()")

        cfg = self.cfg
        X_full, feat_names, _ = build_feature_matrix(
            dataset_full,
            cfg.forcing_names,
            include_doy=cfg.include_doy,
            include_time=cfg.include_time,
            at="full",
            standardize=cfg.standardize,
            ref_stats=self._ref_stats,
        )

        # mu_draws shape: (n_samples, n_time)
        mu_draws = predict_bart_mean(
            self._idata,
            self._model,
            X_full,
            random_seed=int(random_seed),
            var_name="mu",
            predictions=True,
        )

        # subsample to requested n_draws
        rng = np.random.default_rng(int(random_seed))
        n_avail = mu_draws.shape[0]
        if n_avail == 0:
            raise RuntimeError("No BART draws available")
        idx = rng.choice(n_avail, size=int(n_draws), replace=(n_draws > n_avail))
        draws = mu_draws[idx]

        if cfg.add_noise and "sigma" in getattr(self._idata, "posterior", {}):
            sig = self._idata.posterior["sigma"].values.reshape(-1)
            sig_idx = rng.choice(sig.shape[0], size=int(n_draws), replace=(n_draws > sig.shape[0]))
            eps = rng.normal(0.0, sig[sig_idx][:, None], size=draws.shape)
            draws = draws + eps

        return ExpertPrediction(name=self.name, time=np.asarray(dataset_full.time), draws=np.asarray(draws, dtype=float))
