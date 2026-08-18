from __future__ import annotations

"""Bayesian expert wrapper for SPADS (CEEMDAN + frequency-matched regression).

This integrates the *deterministic* SPADS reconstruction (as implemented in the
official `pySPADS` package) into the Bayesian MoE framework by adding a simple
Bayesian observation model around the SPADS mean prediction.

Design goals
------------
- Keep the SPADS *physics/statistical* core identical to pySPADS (same steps).
- Keep imports lazy so users without SPADS deps can still import slmcal.
- Provide posterior predictive draws on the model grid for MoE aggregation.

Notes
-----
SPADS is fundamentally a hindcast/nowcast method (CEEMDAN is data-adaptive).
For best results you should build IMFs on the *full* period and only fit
regressions on the calibration subset.

In practice, this class supports both:
- fit(dataset_cal) + predict_draws(dataset_full): will (re)build the SPADS state
  on `dataset_full` and fit on the calibration window implied by `dataset_cal`.

"""

from dataclasses import dataclass
from typing import Iterable, Sequence, Literal

import numpy as np
from pathlib import Path
import hashlib
try:
    import pymc as pm  # type: ignore
    import arviz as az
    import pytensor.tensor as pt
except Exception as e:  # pragma: no cover
    raise ImportError(
        "SPADSBayesExpert requires PyMC + ArviZ. Install your bayes extras/environment."
    ) from e

from slmcal.data import TimeSeriesDataset
from slmcal.models.expert import ExpertPrediction


@dataclass(frozen=True)
class SPADSBayesConfig:
    # --- SPADS core ---
    forcing_names: Sequence[str] = ("E", "Hs", "Tp")
    noise_levels: Sequence[float] = (0.2,)
    num_trials: int = 100
    freq_threshold: float = 0.5
    exclude_trend: bool = False
    reject_noise: bool = False
    reject_noise_alpha: float = 0.95
    reg_model: str = "mreg2"
    fit_intercept: bool = False
    normalize: bool = False
    parallel: bool = True
    processes: int | 6 = 6
    parallel_kind: str = "thread"

    # --- Optional caching (speed) ---
    # If cache_dir is set, the deterministic SPADS prediction on a given dataset
    # grid + configuration is cached on disk. This avoids re-running CEEMDAN when
    # iterating on IH-MOOSE coupling or Bayesian wrappers.
    cache_dir: str | None = None
    cache_predictions: bool = True
    # Cache CEEMDAN decompositions (IMFs) for signal and each driver.
    # This is especially useful when repeatedly re-calibrating SPADS with
    # different calibration windows but identical forcing series.
    cache_imfs: bool = True
    cache_tag: str = "spads_yhat_v1"
    cache_verbose: bool = False

    # --- Reference restoration (absolute shoreline) ---
    # Some SPADS/CEEMDAN pipelines effectively output a de-meaned or detrended
    # reconstruction. These options restore the reference level/trend using the
    # interpolated calibration observations (the same series used to build the
    # CEEMDAN state).
    #
    # - 'none': no correction
    # - 'offset_first': add a constant so yhat matches y_ref at the first cal sample
    # - 'offset_mean': add a constant so mean(yhat) matches mean(y_ref) on cal window
    # - 'lintrend_cal': add a polynomial (default: linear) fit to (y_ref - yhat) on cal window
    # - 'lintrend_full': same as above, but fitted on the full period
    restore_reference: Literal['none','offset_first','offset_mean','lintrend_cal','lintrend_full'] = 'lintrend_cal'
    restore_trend_degree: int = 1

    # --- Bayesian observation model around SPADS mean ---
    likelihood: str = "studentt"  # 'normal' or 'studentt'
    include_bias: bool = True
    sigma: float = 10.0
    estimate_sigma: bool = True
    sigma_prior_scale: float = 20.0

    # --- Residual structure ---
    residual_model: Literal["iid", "ar1", "ou"] = "iid"
    ar1_rho_lower: float = -0.99
    ar1_rho_upper: float = 0.99

    # --- OU/CAR(1) settings (time-aware AR1 for irregular sampling) ---
    # tau is correlation timescale (days): rho(dt)=exp(-dt/tau)
    ou_tau_loc_days: float = 30.0     # median-ish timescale
    ou_tau_log_sd: float = 1.0        # log-space spread (bigger = weaker prior)

    # Student-t nu inference
    nu_min: float = 2.0
    nu_scale: float = 10.0

    # sampling
    draws: int = 1500
    tune: int = 2000
    chains: int = 4
    target_accept: float = 0.9
    cores: int | None = None


class SPADSBayesExpert:
    """SPADS expert with a Bayesian error model."""

    def __init__(self, cfg: SPADSBayesConfig | None = None, name: str = "SPADS"):
        self.cfg = cfg or SPADSBayesConfig()
        self.name = str(name)

        # Fitted bayes objects
        self.trace = None
        self.ppc = None

        # Calibration dataset reference (used to infer cal window)
        self._dataset_cal: TimeSeriesDataset | None = None

        # Optional context/state dataset used to build the SPADS decomposition once
        # on a larger/full grid, while still fitting the regression and Bayesian
        # error model using calibration observations only.
        self._dataset_context: TimeSeriesDataset | None = None

        # Cached SPADS deterministic mean on last state grid
        self._cache_time_key: tuple[int, int] | None = None
        self._cache_yhat_full: np.ndarray | None = None

    # -------------------------
    # Expert API
    # -------------------------
    def set_full_context(self, dataset_full: TimeSeriesDataset | None) -> "SPADSBayesExpert":
        """Set an optional full/context dataset for the deterministic SPADS state.

        When this is provided, CEEMDAN + frequency matching are computed once on
        ``dataset_full`` and reused for any prediction window that is an exact
        subset of that time grid. Calibration regressions and the Bayesian error
        model still use calibration observations only.
        """
        self._dataset_context = dataset_full
        self._cache_time_key = None
        self._cache_yhat_full = None
        return self

    def fit(self, dataset_cal: TimeSeriesDataset) -> "SPADSBayesExpert":
        # We store dataset_cal; the heavy SPADS state will be built on-demand
        self._dataset_cal = dataset_cal

        # Build SPADS mean for the calibration window. If a full/context dataset
        # has been provided via ``set_full_context()``, the CEEMDAN state is built
        # only once on that larger grid and then sliced back to calibration here.
        yhat_cal = self._spads_mean(dataset_cal)

        # Fit Bayesian observation model at observation points
        idx_obs = np.asarray(dataset_cal.idx_obs, dtype=int)
        y_obs = np.asarray(dataset_cal.obs, dtype=float)
        mu_obs = np.asarray(yhat_cal[idx_obs], dtype=float)

        # Lazy imports (so slmcal can be imported without PyMC installed)
        try:
            import pymc as pm  # type: ignore
            import arviz as az  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "SPADSBayesExpert requires PyMC + ArviZ. Install your bayes extras/environment."\
            ) from e

        cfg = self.cfg

        # OU/CAR(1) model settings
        t_obs = np.asarray(dataset_cal.obs_time)
        # sort to ensure increasing time (recommended)
        order = np.argsort(t_obs)
        t_obs = t_obs[order]
        y_obs = y_obs[order]
        mu_obs = mu_obs[order]

        dt_days = (t_obs[1:] - t_obs[:-1]) / np.timedelta64(1, "D")
        dt_days = np.asarray(dt_days, dtype=float)
        dt_days = np.maximum(dt_days, 1e-6)  # avoid zeros

        with pm.Model() as model:
            mu = pm.Data("mu_obs", mu_obs)

            # if cfg.include_bias:
            #     bias = pm.Normal("bias", mu=0.0, sigma=50.0)
            # else:
            #     bias = 0.0

            # if cfg.estimate_sigma:
            #     sigma = pm.HalfNormal("sigma", sigma=float(cfg.sigma_prior_scale))
            # else:
            #     sigma = float(cfg.sigma)

            # like = str(cfg.likelihood).lower().strip()
            # if like == "normal":
            #     pm.Normal("y", mu=mu + bias, sigma=sigma, observed=y_obs)
            # elif like in ("studentt", "student_t", "t"):
            #     nu_minus2 = pm.Exponential("nu_minus2", lam=1 / 10.0)
            #     nu = pm.Deterministic("nu", nu_minus2 + 2.0)
            #     pm.StudentT("y", mu=mu + bias, sigma=sigma, nu=nu, observed=y_obs)
            # else:
            #     raise ValueError(f"Unsupported likelihood: {cfg.likelihood}")

            like = str(cfg.likelihood).lower().strip()
            resmod = str(cfg.residual_model).lower().strip()

            # Make sigma/bias always tensor-safe
            if cfg.include_bias:
                bias = pm.Normal("bias", mu=0.0, sigma=50.0)
            else:
                bias = pt.as_tensor_variable(0.0)

            if cfg.estimate_sigma:
                sigma = pm.HalfNormal("sigma", sigma=float(cfg.sigma_prior_scale))
            else:
                sigma = pt.as_tensor_variable(float(cfg.sigma))

            # Student-t df (inferred)
            if like in ("studentt", "student_t", "t"):
                nu_minus = pm.Exponential("nu_minus", lam=1.0 / float(cfg.nu_scale))
                nu = pm.Deterministic("nu", float(cfg.nu_min) + nu_minus)
            else:
                nu = None

            # Convenience: stable logpdfs (Normal / Student-t) for potentials
            def _logpdf_normal_loc(x, loc, sig):
                return (
                    -0.5 * ((x - loc) / sig) ** 2
                    - pt.log(sig)
                    - 0.5 * np.log(2.0 * np.pi)
                )

            def _logpdf_studentt_loc(x, loc, sig, nu_):
                z2 = ((x - loc) / sig) ** 2
                return (
                    pt.gammaln((nu_ + 1.0) / 2.0)
                    - pt.gammaln(nu_ / 2.0)
                    - 0.5 * pt.log(nu_ * np.pi)
                    - pt.log(sig)
                    - ((nu_ + 1.0) / 2.0) * pt.log1p(z2 / nu_)
                )

            # ------------------------------------------------------------------
            # Likelihood
            # ------------------------------------------------------------------
            if resmod == "iid":
                # Keep your original RVs (so PPC works)
                if like == "normal":
                    pm.Normal("y", mu=mu + bias, sigma=sigma, observed=y_obs)
                elif like in ("studentt", "student_t", "t"):
                    pm.StudentT("y", mu=mu + bias, sigma=sigma, nu=nu, observed=y_obs)
                else:
                    raise ValueError(f"Unsupported likelihood: {cfg.likelihood}")

            elif resmod == "ar1":
                # AR1 over the observation sequence (in obs order, assumes equal spacing in index order)
                rho = pm.Uniform("rho", lower=float(cfg.ar1_rho_lower), upper=float(cfg.ar1_rho_upper))

                y_data = pt.as_tensor_variable(np.asarray(y_obs, dtype=float))
                r = y_data - (mu + bias)

                eps = 1e-6
                sig0 = sigma / pt.sqrt(pt.maximum(eps, 1.0 - rho * rho))

                r0 = r[0]
                e = r[1:] - rho * r[:-1]

                if like == "normal":
                    logp0 = _logpdf_normal_loc(r0, 0.0, sig0)
                    logpe = _logpdf_normal_loc(e, 0.0, sigma)
                elif like in ("studentt", "student_t", "t"):
                    logp0 = _logpdf_studentt_loc(r0, 0.0, sig0, nu)
                    logpe = _logpdf_studentt_loc(e, 0.0, sigma, nu)
                else:
                    raise ValueError(f"Unsupported likelihood: {cfg.likelihood}")

                pm.Potential("likelihood", logp0 + pt.sum(logpe))

            elif resmod == "ou":
                # Time-aware AR(1) / OU / CAR(1) on irregular observation times:
                # rho_i = exp(-dt_i / tau)
                # r0 ~ dist(0, sigma)
                # r_i | r_{i-1} ~ dist(loc=rho_i*r_{i-1}, scale=sigma*sqrt(1-rho_i^2))
                tau = pm.LogNormal(
                    "tau",
                    mu=float(np.log(cfg.ou_tau_loc_days)),
                    sigma=float(cfg.ou_tau_log_sd),
                )

                dt = pt.as_tensor_variable(np.asarray(dt_days, dtype=float))  # (N-1,)
                rho = pt.exp(-dt / tau)  # (N-1,)

                y_data = pt.as_tensor_variable(np.asarray(y_obs, dtype=float))
                r = y_data - (mu + bias)  # (N,)

                eps = 1e-6
                sig0 = sigma
                sig_i = sigma * pt.sqrt(pt.maximum(eps, 1.0 - rho * rho))  # (N-1,)

                r0 = r[0]
                ri = r[1:]
                loc_i = rho * r[:-1]

                if like == "normal":
                    lp0 = _logpdf_normal_loc(r0, 0.0, sig0)
                    lpi = _logpdf_normal_loc(ri, loc_i, sig_i)
                elif like in ("studentt", "student_t", "t"):
                    lp0 = _logpdf_studentt_loc(r0, 0.0, sig0, nu)
                    lpi = _logpdf_studentt_loc(ri, loc_i, sig_i, nu)
                else:
                    raise ValueError(f"Unsupported likelihood: {cfg.likelihood}")

                pm.Potential("likelihood", lp0 + pt.sum(lpi))

            else:
                raise ValueError(f"Unsupported residual_model: {cfg.residual_model}")

            import inspect

            sample_kwargs = dict(
                draws=int(cfg.draws),
                tune=int(cfg.tune),
                chains=int(cfg.chains),
                target_accept=float(cfg.target_accept),
                random_seed=42,
                compute_convergence_checks=True,
            )
            sig = inspect.signature(pm.sample)
            if cfg.cores is not None and 'cores' in sig.parameters:
                sample_kwargs['cores'] = int(cfg.cores)
            # PyMC versions use either progressbar or progress
            if 'progressbar' in sig.parameters:
                sample_kwargs['progressbar'] = True
            if 'progress' in sig.parameters:
                sample_kwargs['progress'] = True

            self.trace = pm.sample(**sample_kwargs)

            print(az.summary(self.trace))

            # Optional PPC at obs points (useful for plots)
            try:
                self.ppc = pm.sample_posterior_predictive(self.trace, var_names=["y"], random_seed=123)
            except Exception:
                self.ppc = None

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

        yhat_full = self._spads_mean(dataset_full)

        # Pull posterior samples
        post = self.trace.posterior
        # Flatten chain/draw dims (chain, draw, ...) -> (n_samples, ...)
        def _flat(name: str) -> np.ndarray:
            v = np.asarray(post[name].values)
            flat = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            # Common case: scalar RV -> (n_samples,)
            if flat.ndim == 1:
                return flat
            # Scalars sometimes come out as (n_samples, 1)
            if flat.ndim == 2 and flat.shape[1] == 1:
                return flat[:, 0]
            return flat

        rng = np.random.default_rng(int(random_seed))

        n_post = None
        if 'bias' in post:
            bias_s = _flat('bias').astype(float)
            n_post = bias_s.shape[0]
        else:
            bias_s = None

        if 'sigma' in post:
            sigma_s = _flat('sigma').astype(float)
            n_post = sigma_s.shape[0] if n_post is None else n_post
        else:
            # fixed sigma
            if n_post is None:
                n_post = int(max(1, n_draws))
            sigma_s = np.full((n_post,), float(self.cfg.sigma), dtype=float)

        if bias_s is None:
            bias_s = np.zeros((n_post,), dtype=float)

        like = str(self.cfg.likelihood).lower().strip()
        if like in ('studentt', 'student_t', 't') and 'nu' in post:
            nu_s = _flat('nu').astype(float)
        else:
            nu_s = None

        # Subsample / resample posterior to the requested draw count.
        # Important: this method must ALWAYS return exactly ``n_draws`` rows so
        # MoE ensembles can combine experts draw-by-draw. When more predictive
        # draws are requested than posterior samples available, we resample with
        # replacement instead of silently returning only ``n_post`` samples.
        n_post = int(bias_s.shape[0])
        if n_draws <= 0:
            raise ValueError("n_draws must be positive")
        sel = rng.choice(n_post, size=int(n_draws), replace=(int(n_draws) > n_post))

        bias_sel = bias_s[sel]
        sigma_sel = sigma_s[sel]
        if nu_s is not None:
            nu_sel = nu_s[sel]
        else:
            nu_sel = None

        mu = yhat_full[None, :] + bias_sel[:, None]

        use_ar1 = (str(self.cfg.residual_model).lower().strip() == "ar1") and ("rho" in post)
        if use_ar1:
            rho_s = _flat("rho").astype(float)
        else:
            rho_s = None

        use_ou = (str(self.cfg.residual_model).lower().strip() == "ou") and ("tau" in post)
        if use_ou:
            tau_s = _flat("tau").astype(float)
        else:
            tau_s = None


        def _innov(shape, sig, nu_val):
            if nu_val is None:
                return rng.normal(loc=0.0, scale=sig, size=shape)
            # numpy standard_t supports array df if broadcasted
            return rng.standard_t(df=nu_val, size=shape) * sig

        T = mu.shape[1]

        def _innov(shape, sig, nu_val):
            if nu_val is None:
                return rng.normal(loc=0.0, scale=sig, size=shape)
            return rng.standard_t(df=float(nu_val), size=shape) * sig

        # Build dt (days) for full grid for OU
        if use_ou:
            t_full = np.asarray(dataset_full.time)
            dt_full = (t_full[1:] - t_full[:-1]) / np.timedelta64(1, "D")
            dt_full = np.asarray(dt_full, dtype=float)
            dt_full = np.maximum(dt_full, 1e-6)  # avoid zeros
        else:
            dt_full = None

        if (not use_ar1) and (not use_ou):
            # iid residuals
            if nu_sel is None:
                eps = rng.normal(loc=0.0, scale=1.0, size=mu.shape) * sigma_sel[:, None]
            else:
                eps = rng.standard_t(df=nu_sel[:, None], size=mu.shape) * sigma_sel[:, None]

        elif use_ar1:
            # AR1 residuals on the full grid (assumes equal spacing in index order)
            rho_sel = rho_s[sel]
            eps = np.empty(mu.shape, dtype=float)

            for i in range(mu.shape[0]):
                rho = float(rho_sel[i])
                sig = float(sigma_sel[i])
                nuv = None if nu_sel is None else float(nu_sel[i])

                # Stationary-ish initial variance
                sig0 = sig / np.sqrt(max(1e-6, 1.0 - rho * rho))
                eps[i, 0] = _innov(1, sig0, nuv)[0]

                innov = _innov(T - 1, sig, nuv)
                for t in range(1, T):
                    eps[i, t] = rho * eps[i, t - 1] + innov[t - 1]

        elif use_ou:
            # OU / CAR(1) residuals on the full grid with time-aware rho(dt)=exp(-dt/tau)
            tau_sel = tau_s[sel]
            eps = np.empty(mu.shape, dtype=float)

            for i in range(mu.shape[0]):
                tau = float(tau_sel[i])
                sig = float(sigma_sel[i])
                nuv = None if nu_sel is None else float(nu_sel[i])

                # initial residual from stationary-ish distribution
                eps[i, 0] = _innov(1, sig, nuv)[0]

                # time-aware recursion
                for t in range(1, T):
                    rho_t = float(np.exp(-dt_full[t - 1] / max(1e-6, tau)))
                    sig_t = sig * np.sqrt(max(1e-6, 1.0 - rho_t * rho_t))
                    eps[i, t] = rho_t * eps[i, t - 1] + _innov(1, sig_t, nuv)[0]

        else:
            raise RuntimeError("Unexpected residual model state")

        # # Add observation noise to produce predictive draws
        # if nu_sel is None:
        #     eps = rng.normal(loc=0.0, scale=1.0, size=mu.shape) * sigma_sel[:, None]
        # else:
        #     # numpy's standard_t uses df
        #     eps = rng.standard_t(df=nu_sel[:, None], size=mu.shape) * sigma_sel[:, None]

        draws = mu + eps

        return ExpertPrediction(name=self.name, time=np.asarray(dataset_full.time), draws=np.asarray(draws, dtype=float))

    # -------------------------
    # Convenience: default plots
    # -------------------------
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
        from pathlib import Path
        from slmcal.plotting import make_default_calibration_plots

        pred = self.predict_draws(dataset_full, n_draws=int(n_draws), random_seed=int(random_seed))
        d = np.asarray(pred.draws, dtype=float)

        per1 = np.nanpercentile(d, 1, axis=0)
        per5 = np.nanpercentile(d, 5, axis=0)
        per10 = np.nanpercentile(d, 10, axis=0)
        per50 = np.nanpercentile(d, 50, axis=0)
        pero90 = np.nanpercentile(d, 90, axis=0)
        per95 = np.nanpercentile(d, 95, axis=0)
        per99 = np.nanpercentile(d, 99, axis=0)
        mini = np.nanmin(d, axis=0)
        maxi = np.nanmax(d, axis=0)

        obs_mask_cal = None
        obs_mask_val = None
        split_dt = None
        if split_date is not None and np.issubdtype(dataset_full.obs_time.dtype, np.datetime64):
            split_dt = np.datetime64(split_date)
            obs_mask_cal = np.asarray(dataset_full.obs_time) < split_dt
            obs_mask_val = np.asarray(dataset_full.obs_time) >= split_dt

        make_default_calibration_plots(
            out_dir=Path(out_dir),
            prior_raw=None,
            posterior_raw=None,
            time=np.asarray(dataset_full.time),
            per5=per5,
            per10=per10,
            per50=per50,
            per90=pero90,
            per95=per95,
            per1=per1,
            per99=per99,
            mini=mini,
            maxi=maxi,
            draws=d,
            obs_time=np.asarray(dataset_full.obs_time),
            obs=np.asarray(dataset_full.obs, dtype=float),
            trace=self.trace,
            ppc=self.ppc,
            param_names=(),
            label=label or self.name,
            obs_mask_cal=obs_mask_cal,
            obs_mask_val=obs_mask_val,
            split_date=split_dt,
        )

    # -------------------------
    # SPADS deterministic core
    # -------------------------
    def _extract_from_state_grid(
        self,
        *,
        yhat_state: np.ndarray,
        dataset_state: TimeSeriesDataset,
        dataset_request: TimeSeriesDataset,
    ) -> np.ndarray:
        """Extract a requested time grid from a cached SPADS state grid.

        The requested grid must be an exact subset of the state grid. This is the
        common case in the examples, where ``dataset_cal`` is a contiguous slice of
        ``dataset_full`` built from the same parent record.
        """
        t_state = np.asarray(dataset_state.time)
        t_req = np.asarray(dataset_request.time)

        if t_req.shape == t_state.shape and np.array_equal(t_req, t_state):
            return np.asarray(yhat_state, dtype=float)

        if t_req.size == 0:
            return np.asarray([], dtype=float)

        # searchsorted works for both datetime64 and numeric monotonic grids
        idx = np.searchsorted(t_state, t_req)
        if np.any(idx < 0) or np.any(idx >= t_state.size):
            raise ValueError(
                'Requested SPADS grid is not contained in the stored full/context grid.'
            )

        if not np.array_equal(t_state[idx], t_req):
            raise ValueError(
                'Requested SPADS grid must be an exact subset of the stored full/context grid.'
            )

        return np.asarray(yhat_state[idx], dtype=float)

    def _spads_mean(self, dataset: TimeSeriesDataset) -> np.ndarray:
        """Compute deterministic SPADS mean on the requested dataset time grid.

        If ``set_full_context()`` has been used, the CEEMDAN state is built once on
        that larger grid and any request on an exact subset is served by slicing the
        cached state-grid prediction, avoiding a second decomposition on a truncated
        series.
        """
        dataset_state = self._dataset_context if self._dataset_context is not None else dataset

        t_state = np.asarray(dataset_state.time)
        key = (int(t_state.__array_interface__["data"][0]), int(t_state.size))
        if self._cache_time_key != key or self._cache_yhat_full is None:
            yhat_state = self._run_spads_pipeline(dataset_state)
            self._cache_time_key = key
            self._cache_yhat_full = np.asarray(yhat_state, dtype=float)

        return self._extract_from_state_grid(
            yhat_state=np.asarray(self._cache_yhat_full, dtype=float),
            dataset_state=dataset_state,
            dataset_request=dataset,
        )

    def _run_spads_pipeline(self, dataset: TimeSeriesDataset) -> np.ndarray:
        """Run pySPADS steps end-to-end on a single dataset.

        This builds a daily shoreline series by interpolating observations to the
        forcing grid, then runs CEEMDAN + frequency-matching regression for each
        noise level and averages reconstructions.
        """
        cfg = self.cfg

        # ------------------------------------------------------------------
        # Optional on-disk cache (massive speed-up for repeated runs)
        # ------------------------------------------------------------------
        cache_path = None
        if cfg.cache_dir and cfg.cache_predictions:
            try:
                cache_dir = Path(str(cfg.cache_dir))
                cache_dir.mkdir(parents=True, exist_ok=True)

                if self._dataset_cal is None:
                    raise RuntimeError("SPADSBayesExpert.fit must be called before SPADS caching can be used")

                # Build a stable digest from: time grid + selected forcings + cal obs.
                h = hashlib.sha1()
                t64 = np.asarray(dataset.time).astype("datetime64[ns]").view("int64")
                h.update(t64.tobytes())
                for fn in cfg.forcing_names:
                    arr = np.asarray(dataset.forcings[fn], dtype=np.float64)
                    h.update(arr.tobytes())

                cal = self._dataset_cal
                ot64 = np.asarray(cal.obs_time).astype("datetime64[ns]").view("int64")
                h.update(ot64.tobytes())
                if cal.obs is not None:
                    h.update(np.asarray(cal.obs, dtype=np.float64).tobytes())

                # And the configuration knobs that change the output.
                cfg_bytes = (
                    f"{cfg.cache_tag}|{list(cfg.forcing_names)}|{list(cfg.noise_levels)}|"
                    f"{cfg.num_trials}|{cfg.freq_threshold}|{cfg.exclude_trend}|{cfg.reject_noise}|"
                    f"{cfg.reject_noise_alpha}|{cfg.reg_model}|{cfg.fit_intercept}|{cfg.normalize}|"
                    f"{cfg.restore_reference}|{cfg.restore_trend_degree}"
                ).encode("utf-8")
                h.update(cfg_bytes)

                key = h.hexdigest()[:16]
                cache_path = cache_dir / f"spads_yhat_{key}.npz"
                if cache_path.exists():
                    z = np.load(cache_path)
                    yhat = np.asarray(z["yhat"], dtype=float)
                    if yhat.shape[0] == np.asarray(dataset.time).shape[0]:
                        if cfg.cache_verbose:
                            print(f"[SPADS cache] loaded {cache_path}")
                        return yhat
            except Exception:
                # Cache is best-effort; any issue should not crash the run.
                cache_path = None

        # Lazy import pySPADS + pandas
        try:
            import pandas as pd
            from pySPADS.pipeline import steps
            from pySPADS.processing.dataclasses import TrendModel
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "SPADSBayesExpert requires pySPADS + its dependencies (pandas, sklearn, PyEMD)."\
            ) from e
        
        def _sanitize_imfs(df, *, name: str):
            """Ensure no NaN/inf in IMF dataframe (CEEMDAN can occasionally produce them)."""
            # Replace inf -> nan
            df = df.replace([np.inf, -np.inf], np.nan)

            if df.isna().any().any():
                # 1) interpolate along time (datetime index)
                try:
                    df = df.interpolate(method="time", limit_direction="both")
                except Exception:
                    # fallback: plain interpolation
                    df = df.interpolate(limit_direction="both")

                # 2) still NaNs? forward/back fill
                df = df.ffill().bfill()

                # 3) if a whole column is NaN (rare), drop it or set to 0
                all_nan_cols = df.columns[df.isna().all(axis=0)]
                if len(all_nan_cols) > 0:
                    # safer to drop “broken” components than to crash
                    df = df.drop(columns=list(all_nan_cols))

                # 4) any remaining NaNs: set to 0 (removes that component’s effect)
                df = df.fillna(0.0)

                # Optional: quick debug
                # print(f"[SPADS] Sanitized NaNs in {name}")
            return df

        # -------------------------
        # Optional IMF cache helpers
        # -------------------------
        def _save_df_npz(path: Path, df) -> None:
            """Best-effort: save a pandas.DataFrame (datetime index) to NPZ."""
            try:
                idx = pd.to_datetime(df.index).astype("datetime64[ns]").view("int64")
                cols = np.asarray([str(c) for c in df.columns], dtype=object)
                arr = np.asarray(df.to_numpy(), dtype=np.float64)
                np.savez_compressed(path, idx=idx, cols=cols, arr=arr)
            except Exception:
                return

        def _load_df_npz(path: Path):
            """Best-effort: load a pandas.DataFrame (datetime index) from NPZ."""
            z = np.load(path, allow_pickle=True)
            idx = pd.to_datetime(np.asarray(z["idx"], dtype=np.int64))
            cols = [str(c) for c in list(z["cols"])]
            arr = np.asarray(z["arr"], dtype=np.float64)
            return pd.DataFrame(arr, index=idx, columns=cols)

        def _imf_cache_path(kind: str, name: str, key: str) -> Path | None:
            if not (cfg.cache_dir and cfg.cache_imfs):
                return None
            try:
                d = Path(str(cfg.cache_dir))
                d.mkdir(parents=True, exist_ok=True)
                return d / f"spads_imf_{kind}_{name}_{key}.npz"
            except Exception:
                return None

        def _digest_series(ser: "pd.Series", *, noise: float) -> str:
            """Digest for CEEMDAN inputs (time grid + values + key params)."""
            h = hashlib.sha1()
            t64 = pd.to_datetime(ser.index).astype("datetime64[ns]").view("int64")
            h.update(np.asarray(t64, dtype=np.int64).tobytes())
            h.update(np.asarray(ser.to_numpy(), dtype=np.float64).tobytes())
            h.update(str(noise).encode("utf-8"))
            h.update(str(int(cfg.num_trials)).encode("utf-8"))
            # Include parallel knobs; some implementations seed per-worker.
            h.update(str(bool(cfg.parallel)).encode("utf-8"))
            h.update(str(int(getattr(cfg, "processes", 0) or 0)).encode("utf-8"))
            h.update(str(getattr(cfg, "parallel_kind", "")).encode("utf-8"))
            return h.hexdigest()[:16]

        def _decompose_cached(ser: "pd.Series", *, name: str, noise: float, kind: str):
            """Run steps.decompose with an optional on-disk IMF cache."""
            key = _digest_series(ser, noise=noise)
            p = _imf_cache_path(kind, name, key)

            if p is not None and p.exists():
                try:
                    df = _load_df_npz(p)
                    if cfg.cache_verbose:
                        print(f"[SPADS cache] loaded IMFs: {p}")
                    return df
                except Exception:
                    pass

            df = steps.decompose(
                ser,
                float(noise),
                num_trials=int(cfg.num_trials),
                progress=False,
                parallel=bool(cfg.parallel),
                processes=cfg.processes,
                parallel_kind=str(cfg.parallel_kind),
            )
            df = _sanitize_imfs(df, name=f"{kind}:{name}")

            if p is not None:
                _save_df_npz(p, df)
                if cfg.cache_verbose:
                    print(f"[SPADS cache] saved IMFs: {p}")
            return df

        if not np.issubdtype(np.asarray(dataset.time).dtype, np.datetime64):
            raise ValueError("SPADS requires datetime64 time axis")

        t_full = pd.to_datetime(np.asarray(dataset.time))

        # Build full-grid signal by time interpolation of **calibration** observations
        # (avoids leaking validation obs into CEEMDAN + regression).
        if self._dataset_cal is None:
            raise RuntimeError('SPADSBayesExpert.fit must be called before running SPADS on full dataset')

        cal = self._dataset_cal
        cal_start = pd.to_datetime(np.asarray(cal.time)[0])
        cal_end = pd.to_datetime(np.asarray(cal.time)[-1])

        y_obs_ser = pd.Series(
            data=np.asarray(cal.obs, dtype=float),
            index=pd.to_datetime(np.asarray(cal.obs_time)),
        ).sort_index()

        # collapse duplicates if any
        if y_obs_ser.index.duplicated().any():
            y_obs_ser = y_obs_ser.groupby(level=0).mean()

        # ensure tz-naive
        y_obs_ser.index = pd.to_datetime(y_obs_ser.index).tz_localize(None)
        t_full = pd.to_datetime(t_full).tz_localize(None)

        # union index so interpolation has anchors
        idx_all = y_obs_ser.index.union(t_full)

        y_full_ser = (
            y_obs_ser.reindex(idx_all)
            .sort_index()
            .interpolate(method="time")
            .reindex(t_full)
            .ffill()
            .bfill()
        )

        n_nan = int(y_full_ser.isna().sum())
        print("NaNs in y_full_ser:", n_nan)
        if n_nan > 0:
            raise ValueError(
                f"SPADS: y_full_ser still has {n_nan} NaNs after union interpolation. "
                f"obs range [{y_obs_ser.index.min()}..{y_obs_ser.index.max()}], "
                f"forcing range [{t_full.min()}..{t_full.max()}]."
            )

        # Drivers (full grid)
        drivers = {}
        for fn in cfg.forcing_names:
            if fn not in dataset.forcings:
                raise KeyError(f"Missing forcing '{fn}' required by SPADS")
            drivers[fn] = pd.Series(np.asarray(dataset.forcings[fn], dtype=float), index=t_full)

        signal_name = "shoreline"

        # Iterate noise levels
        pred_by_noise = {}

        for noise in cfg.noise_levels:
            imfs = {}
            # Decompose signal
            imf_sig = _decompose_cached(y_full_ser, name="shoreline", noise=float(noise), kind="signal")

            if cfg.reject_noise:
                imf_sig = steps.reject_noise(imf_sig, noise_threshold=float(cfg.reject_noise_alpha))

            print("NaNs in imf_sig:", imf_sig.isna().sum().sum())

            imfs[signal_name] = imf_sig

            # Decompose drivers
            for k, ser in drivers.items():
                imf_k = _decompose_cached(ser, name=str(k), noise=float(noise), kind="driver")
                if cfg.reject_noise:
                    imf_k = steps.reject_noise(imf_k, noise_threshold=float(cfg.reject_noise_alpha))
                imfs[k] = imf_k

            nearest = steps.match_frequencies(
                imfs,
                signal=signal_name,
                threshold=float(cfg.freq_threshold),
                exclude_trend=bool(cfg.exclude_trend),
            )

            # Fit regression only on the calibration window (no leakage).
            imfs_cal = {
                k: v.loc[cal_start:cal_end] if hasattr(v, 'loc') else v
                for k, v in imfs.items()
            }
            imfs_cal = {
                k: _sanitize_imfs(v, name=f"cal:{k}") if hasattr(v, "isna") else v
                for k, v in imfs_cal.items()
            }

            coefs = steps.fit(
                imfs_cal,
                nearest,
                signal=signal_name,
                model=str(cfg.reg_model),
                fit_intercept=bool(cfg.fit_intercept),
                normalize=bool(cfg.normalize),
            )

            start_date = str(t_full[0].date())
            end_date = str(t_full[-1].date())

            comp_pred = steps.predict(
                imfs,
                nearest,
                signal=signal_name,
                coefficients=coefs,
                start_date=start_date,
                end_date=end_date,
                exclude_trend=bool(cfg.exclude_trend),
            )

            pred_by_noise[float(noise)] = comp_pred

        total = steps.combine_predictions(pred_by_noise, trend=TrendModel(coeff=0.0, intercept=0.0))

        # Align to model grid (should already match daily)
        total = total.reindex(t_full)
        yhat = np.asarray(total.to_numpy(), dtype=float)
        if yhat.shape[0] != np.asarray(dataset.time).shape[0]:
            raise RuntimeError("SPADS prediction length mismatch")

        # ------------------------------------------------------------------
        # Restore reference level/trend if SPADS output is de-meaned/detrended
        # ------------------------------------------------------------------
        try:
            mode = str(cfg.restore_reference).lower().strip()
        except Exception:
            mode = 'lintrend_cal'

        if mode not in ('none', 'offset_first', 'offset_mean', 'lintrend_cal', 'lintrend_full'):
            mode = 'lintrend_cal'

        if mode != 'none':
            # Reference series used to build CEEMDAN state (interpolated cal obs)
            y_ref = np.asarray(y_full_ser.to_numpy(), dtype=float)

            # Numeric time in days from start
            t64 = np.asarray(t_full.to_numpy(), dtype='datetime64[ns]')
            tt = (t64 - t64[0]) / np.timedelta64(1, 'D')
            tt = np.asarray(tt, dtype=float)

            # Calibration window mask (or full)
            if mode == 'lintrend_full':
                m = np.isfinite(y_ref) & np.isfinite(yhat) & np.isfinite(tt)
            else:
                m_win = (t_full >= cal_start) & (t_full <= cal_end)
                m = np.asarray(m_win, dtype=bool) & np.isfinite(y_ref) & np.isfinite(yhat) & np.isfinite(tt)

            if m.sum() >= 1:
                if mode == 'offset_first':
                    i0 = int(np.flatnonzero(m)[0])
                    c0 = float(y_ref[i0] - yhat[i0])
                    yhat = yhat + c0
                elif mode == 'offset_mean':
                    c0 = float(np.mean((y_ref - yhat)[m]))
                    yhat = yhat + c0
                else:
                    deg = int(getattr(cfg, 'restore_trend_degree', 1) or 1)
                    deg = max(0, min(3, deg))
                    # Need at least deg+1 samples for polyfit
                    if m.sum() < (deg + 1):
                        c0 = float(np.mean((y_ref - yhat)[m]))
                        yhat = yhat + c0
                    else:
                        # Fit residual polynomial on selected window
                        rr = (y_ref - yhat)[m]
                        pp = np.polyfit(tt[m], rr, deg=deg)
                        corr = np.polyval(pp, tt)
                        yhat = yhat + corr

        # Save cache (best-effort)
        if cache_path is not None:
            try:
                np.savez_compressed(cache_path, yhat=np.asarray(yhat, dtype=np.float64))
                if cfg.cache_verbose:
                    print(f"[SPADS cache] saved {cache_path}")
            except Exception:
                pass

        return yhat
