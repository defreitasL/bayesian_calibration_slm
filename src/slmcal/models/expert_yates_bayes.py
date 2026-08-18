from __future__ import annotations

"""Bayesian physics expert wrapper.

This expert wraps the existing `CalibrationWorkflow` and returns predictive
draws on the full forcing grid by forward-simulating the shoreline model for
posterior parameter samples.

Notes
-----
This is a convenience wrapper. It is intentionally lightweight and keeps the
existing workflow unchanged.
"""

from dataclasses import dataclass

import numpy as np
import arviz as az
from slmcal.data import TimeSeriesDataset
from slmcal.models.base import ShorelineModel, transform_raw_to_physical

from .expert import ExpertPrediction


@dataclass
class PhysicsBayesConfig:
    # NSGA-II pre-calibration
    metrics: tuple[str, ...] = ("kge", "pbias", "spearman")
    nsga2_backend: str = "fast_optimization"

    # Prior
    prior_kind: str = "mvn"  # "mvn" | "kde" | "copula_kde"
    cov_scale: float = 1.0
    bw_method: str | float | None = "scott"
    jitter: float = 1e-9
    copula_method: str = "factor"
    copula_grid_size: int = 1024
    copula_rank: int | None = None
    copula_explained_var: float = 0.99
    copula_shrinkage: float = 0.05
    use_valid_only: str = "valid_only"  # "all" | "valid_only"

    # Bayesian calibration
    draws: int = 2000
    tune: int = 1000
    chains: int = 4
    random_seed: int = 42
    likelihood: str = "normal"
    sigma: float | None = None
    estimate_sigma: bool = True
    include_bias: bool = True
    cores: int | None = None

    # Predictive noise (if sigma exists in the trace)
    add_noise: bool = True


class PhysicsBayesExpert:
    """Expert that uses Bayesian calibrated physics model."""

    def __init__(self, model: ShorelineModel, cfg: PhysicsBayesConfig = PhysicsBayesConfig(), name: str = "physics"):
        self.model = model
        self.cfg = cfg
        self.name = name
        self.trace = None
        self.ppc = None
        # Keep this loose to avoid import-time dependency issues.
        # We set it in fit().
        self.workflow = None

    def fit(self, dataset_cal: TimeSeriesDataset) -> "PhysicsBayesExpert":
        # Lazy import to avoid failing at `import slmcal.models` time when optional
        # dependencies (e.g., pymc, fast_optimization) are not available.
        from slmcal.core.workflow import CalibrationWorkflow

        cfg = self.cfg
        wf = CalibrationWorkflow(model=self.model, dataset=dataset_cal)
        wf.precalibrate_nsga2(metrics=cfg.metrics, backend=cfg.nsga2_backend)
        wf.build_prior_from_nsga2(
            kind=cfg.prior_kind,
            cov_scale=cfg.cov_scale,
            bw_method=cfg.bw_method,
            jitter=cfg.jitter,
            copula_method=cfg.copula_method,
            copula_grid_size=cfg.copula_grid_size,
            copula_rank=cfg.copula_rank,
            copula_explained_var=cfg.copula_explained_var,
            copula_shrinkage=cfg.copula_shrinkage,
            use_valid_only=cfg.use_valid_only,
        )
        trace, ppc = wf.bayesian_calibrate(
            draws=cfg.draws,
            tune=cfg.tune,
            chains=cfg.chains,
            random_seed=cfg.random_seed,
            likelihood=cfg.likelihood,
            sigma=cfg.sigma,
            estimate_sigma=cfg.estimate_sigma,
            include_bias=cfg.include_bias,
            cores=cfg.cores,
        )
        self.trace = trace
        print(az.summary(trace))
        self.ppc = ppc
        self.workflow = wf

        return self

    def predict_draws(
        self,
        dataset_full: TimeSeriesDataset,
        *,
        n_draws: int,
        random_seed: int = 42,
    ) -> ExpertPrediction:
        if self.trace is None:
            raise RuntimeError("Call fit() before predict_draws()")

        rng = np.random.default_rng(int(random_seed))

        raw = self.trace.posterior["raw_par"].values  # (chain, draw, d)
        raw_flat = raw.reshape(raw.shape[0] * raw.shape[1], raw.shape[2])
        n_avail = raw_flat.shape[0]
        idx = rng.choice(n_avail, size=int(n_draws), replace=(n_draws > n_avail))
        raw_s = raw_flat[idx]

        # Optional bias and sigma
        bias = None
        if "bias" in self.trace.posterior:
            b = self.trace.posterior["bias"].values.reshape(-1)
            bias = b[rng.choice(b.shape[0], size=int(n_draws), replace=(n_draws > b.shape[0]))]

        sigma = None
        if "sigma" in self.trace.posterior:
            s = self.trace.posterior["sigma"].values.reshape(-1)
            sigma = s[rng.choice(s.shape[0], size=int(n_draws), replace=(n_draws > s.shape[0]))]

        draws = np.empty((int(n_draws), dataset_full.time.shape[0]), dtype=float)
        for i in range(int(n_draws)):
            phys = transform_raw_to_physical(raw_s[i], self.model.parameters)
            y = np.asarray(self.model.simulate(phys, dataset_full), dtype=float)
            if bias is not None:
                y = y + float(bias[i])
            draws[i, :] = y

        if self.cfg.add_noise and sigma is not None:
            eps = rng.normal(0.0, sigma[:, None], size=draws.shape)
            draws = draws + eps

        return ExpertPrediction(name=self.name, time=np.asarray(dataset_full.time), draws=draws)


    def save_default_plots(
        self,
        *,
        out_dir,
        dataset_full: TimeSeriesDataset,
        n_draws: int = 2000,
        random_seed: int = 42,
        label: str | None = None,
        split_date: np.datetime64 | str | None = None,
    ) -> None:
        """Save the package's default plot set for this expert.

        This is a convenience wrapper around :func:`slmcal.plotting.make_default_calibration_plots`.

        Parameters
        ----------
        out_dir:
            Output directory for plots.
        dataset_full:
            Full-period dataset (forcing grid + all observations).
        n_draws:
            Number of predictive draws to build envelopes.
        split_date:
            Optional datetime split used to tag cal vs val observation points.
        """
        from pathlib import Path
        from slmcal.plotting import make_default_calibration_plots

        if self.workflow is None or self.trace is None:
            raise RuntimeError("Call fit() before save_default_plots()")

        # Predictive draws on the full grid
        pred = self.predict_draws(dataset_full, n_draws=int(n_draws), random_seed=int(random_seed))
        d = np.asarray(pred.draws, dtype=float)

        # Envelope stats
        per1 = np.nanpercentile(d, 1, axis=0)
        per5 = np.nanpercentile(d, 5, axis=0)
        per10 = np.nanpercentile(d, 10, axis=0)
        per50 = np.nanpercentile(d, 50, axis=0)
        per90 = np.nanpercentile(d, 90, axis=0)
        per95 = np.nanpercentile(d, 95, axis=0)
        per99 = np.nanpercentile(d, 99, axis=0)
        mini = np.nanmin(d, axis=0)
        maxi = np.nanmax(d, axis=0)

        # Prior / posterior raw parameters for the prior-vs-posterior panel
        prior_raw = None
        try:
            if self.workflow.nsga2_result is not None and hasattr(self.workflow.nsga2_result, "individuals_raw"):
                prior_raw = np.asarray(self.workflow.nsga2_result.individuals_raw, dtype=float)
        except Exception:
            prior_raw = None

        raw = self.trace.posterior["raw_par"].values
        posterior_raw = raw.reshape(raw.shape[0] * raw.shape[1], raw.shape[2])

        # Cal/val masks from split_date (if provided)
        obs_mask_cal = None
        obs_mask_val = None
        split_dt = None
        if split_date is not None and np.issubdtype(dataset_full.obs_time.dtype, np.datetime64):
            split_dt = np.datetime64(split_date)
            obs_mask_cal = np.asarray(dataset_full.obs_time) < split_dt
            obs_mask_val = np.asarray(dataset_full.obs_time) >= split_dt

        make_default_calibration_plots(
            out_dir=Path(out_dir),
            prior_raw=prior_raw,
            posterior_raw=posterior_raw,
            time=np.asarray(dataset_full.time),
            per5=per5,
            per50=per50,
            per95=per95,
            per10=per10,
            per90=per90,    
            per1=per1,
            per99=per99,
            mini=mini,
            maxi=maxi,
            draws=d,
            obs_time=np.asarray(dataset_full.obs_time),
            obs=np.asarray(dataset_full.obs, dtype=float),
            trace=self.trace,
            ppc=self.ppc,
            param_names=[p.name for p in self.model.parameters],
            label=label or self.name,
            obs_mask_cal=obs_mask_cal,
            obs_mask_val=obs_mask_val,
            split_date=split_dt,
        )

