from __future__ import annotations

"""Bayesian IMF-regression expert for SPADS.

This expert keeps the CEEMDAN decomposition + frequency matching from pySPADS
fixed, but treats the per-IMF regression coefficients as Bayesian parameters.

Compared with :class:`SPADSBayesExpert`, this class moves part of the SPADS
internal calibration into the Bayesian layer instead of placing a Bayesian
error model only on the reconstructed shoreline mean.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Sequence
import hashlib

import numpy as np

try:
    import pymc as pm  # type: ignore
    import arviz as az  # type: ignore
    import pytensor.tensor as pt  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "SPADSRegressionBayesExpert requires PyMC + ArviZ."
    ) from e

from slmcal.data import TimeSeriesDataset
from slmcal.models.expert import ExpertPrediction


@dataclass(frozen=True)
class SPADSRegressionBayesConfig:
    # --- SPADS core ---
    forcing_names: Sequence[str] = ("E", "Hs", "Tp")
    noise_levels: Sequence[float] = (0.2,)
    num_trials: int = 100
    freq_threshold: float = 0.5
    exclude_trend: bool = False
    reject_noise: bool = False
    reject_noise_alpha: float = 0.95
    normalize: bool = False
    parallel: bool = True
    processes: int = 6
    parallel_kind: str = "thread"

    # --- Optional caching ---
    cache_dir: str | None = None
    cache_imfs: bool = True
    cache_verbose: bool = False

    # --- Reference restoration ---
    restore_reference: Literal[
        "none", "offset_first", "offset_mean", "lintrend_cal", "lintrend_full"
    ] = "lintrend_cal"
    restore_trend_degree: int = 1

    # --- Bayesian regression ---
    fit_intercept: bool = True
    coef_prior: Literal["normal", "hierarchical", "empirical_mvn"] = "hierarchical"
    coef_prior_scale: float = 1.0
    intercept_prior_scale: float = 20.0
    empirical_prior_scale: float = 4.0
    empirical_prior_ridge: float = 1e-6
    empirical_prior_jitter: float = 1e-6
    likelihood: Literal["normal", "studentt"] = "normal"
    estimate_component_sigma: bool = True
    component_sigma: float = 1.0
    component_sigma_prior_scale: float = 2.0
    nu_min: float = 2.0
    nu_scale: float = 10.0

    # --- prediction ---
    predictive_mode_default: Literal["latent", "component_ppc"] = "latent"

    # --- sampling ---
    sampler: Literal["nuts", "demetropolisz"] = "nuts"
    draws: int = 1500
    tune: int = 2000
    chains: int = 4
    target_accept: float = 0.9
    cores: int | None = None
    random_seed: int = 42
    init_from_deterministic: bool = True
    init_jitter_rel: float = 0.10
    init_jitter_abs: float = 1e-3


@dataclass
class _SPADSRegComponent:
    uid: str
    noise: float
    component: int
    feature_names: list[str]
    X_cal: np.ndarray
    y_cal: np.ndarray
    X_full: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    beta_init: np.ndarray
    alpha_init: float
    sigma_init: float
    beta_prior_mean: np.ndarray
    beta_prior_cov: np.ndarray
    alpha_prior_mean: float
    alpha_prior_sd: float


class SPADSRegressionBayesExpert:
    """SPADS expert with Bayesian inference over IMF-regression coefficients."""

    def __init__(self, cfg: SPADSRegressionBayesConfig | None = None, name: str = "SPADSRegBayes"):
        self.cfg = cfg or SPADSRegressionBayesConfig()
        self.name = str(name)

        self.trace = None
        self.ppc = None
        self._dataset_cal: TimeSeriesDataset | None = None
        self._fit_state: dict | None = None
        self._pred_state_cache: dict[tuple[int, int], dict] = {}

    # ------------------------------------------------------------------
    # Public expert API
    # ------------------------------------------------------------------
    def fit(self, dataset_cal: TimeSeriesDataset) -> "SPADSRegressionBayesExpert":
        self._dataset_cal = dataset_cal
        state = self._build_spads_state(dataset_cal)
        self._fit_state = state

        cfg = self.cfg
        components: list[_SPADSRegComponent] = state["components"]
        if not components:
            raise RuntimeError("SPADSRegressionBayesExpert: no valid IMF regression components were assembled.")

        with pm.Model() as model:
            for j, comp in enumerate(components):
                X = np.asarray(comp.X_cal, dtype=float)
                y = np.asarray(comp.y_cal, dtype=float)
                n_feat = int(X.shape[1])

                if n_feat <= 0:
                    continue

                prior_kind = str(cfg.coef_prior).lower().strip()
                if prior_kind == "hierarchical":
                    tau = pm.HalfNormal(f"tau_{j}", sigma=float(cfg.coef_prior_scale))
                    beta = pm.Normal(f"beta_{j}", mu=0.0, sigma=tau, shape=n_feat)
                elif prior_kind == "empirical_mvn":
                    beta = pm.MvNormal(
                        f"beta_{j}",
                        mu=np.asarray(comp.beta_prior_mean, dtype=float),
                        cov=np.asarray(comp.beta_prior_cov, dtype=float),
                        shape=n_feat,
                    )
                else:
                    beta = pm.Normal(
                        f"beta_{j}",
                        mu=0.0,
                        sigma=float(cfg.coef_prior_scale),
                        shape=n_feat,
                    )

                if cfg.fit_intercept:
                    if prior_kind == "empirical_mvn":
                        alpha = pm.Normal(
                            f"alpha_{j}",
                            mu=float(comp.alpha_prior_mean),
                            sigma=float(comp.alpha_prior_sd),
                        )
                    else:
                        alpha = pm.Normal(
                            f"alpha_{j}",
                            mu=0.0,
                            sigma=float(cfg.intercept_prior_scale),
                        )
                else:
                    alpha = pt.as_tensor_variable(0.0)

                mu = alpha + pt.dot(X, beta)

                if cfg.estimate_component_sigma:
                    sigma = pm.HalfNormal(
                        f"sigma_{j}",
                        sigma=float(cfg.component_sigma_prior_scale),
                    )
                else:
                    sigma = pt.as_tensor_variable(float(cfg.component_sigma))

                like = str(cfg.likelihood).lower().strip()
                if like == "normal":
                    pm.Normal(f"y_{j}", mu=mu, sigma=sigma, observed=y)
                elif like in ("studentt", "student_t", "t"):
                    nu_minus = pm.Exponential(f"nu_minus_{j}", lam=1.0 / float(cfg.nu_scale))
                    nu = pm.Deterministic(f"nu_{j}", float(cfg.nu_min) + nu_minus)
                    pm.StudentT(f"y_{j}", mu=mu, sigma=sigma, nu=nu, observed=y)
                else:
                    raise ValueError(f"Unsupported likelihood: {cfg.likelihood}")

            import inspect

            sample_kwargs = dict(
                draws=int(cfg.draws),
                tune=int(cfg.tune),
                chains=int(cfg.chains),
                random_seed=int(cfg.random_seed),
                compute_convergence_checks=True,
            )
            sig = inspect.signature(pm.sample)
            if cfg.cores is not None and "cores" in sig.parameters:
                sample_kwargs["cores"] = int(cfg.cores)
            if "progressbar" in sig.parameters:
                sample_kwargs["progressbar"] = True
            if "progress" in sig.parameters:
                sample_kwargs["progress"] = True
            if "idata_kwargs" in sig.parameters:
                sample_kwargs["idata_kwargs"] = {"log_likelihood": True}

            initvals = self._build_initvals(components)
            if initvals is not None:
                if "initvals" in sig.parameters:
                    sample_kwargs["initvals"] = initvals
                elif "initval" in sig.parameters:
                    sample_kwargs["initval"] = initvals

            sampler_kind = str(cfg.sampler).lower().strip()
            if sampler_kind in {"demetropolisz", "demetz", "de"}:
                sample_kwargs["step"] = pm.DEMetropolisZ()
            elif sampler_kind == "nuts":
                if "target_accept" in sig.parameters:
                    sample_kwargs["target_accept"] = float(cfg.target_accept)
            else:
                raise ValueError(f"Unsupported sampler: {cfg.sampler}")

            self.trace = pm.sample(**sample_kwargs)
            try:
                print(az.summary(self.trace))
            except Exception:
                pass

        return self

    def predict_draws(
        self,
        dataset_full: TimeSeriesDataset,
        *,
        n_draws: int,
        random_seed: int = 42,
        mode: Literal["latent", "component_ppc"] | None = None,
    ) -> ExpertPrediction:
        if self.trace is None:
            raise RuntimeError("Call fit() before predict_draws()")
        if self._dataset_cal is None:
            raise RuntimeError("Missing calibration dataset state")

        mode_eff = str(mode or self.cfg.predictive_mode_default).lower().strip()
        if mode_eff not in ("latent", "component_ppc"):
            raise ValueError("mode must be 'latent' or 'component_ppc'")

        state = self._build_spads_state(dataset_full)
        components: list[_SPADSRegComponent] = state["components"]
        if not components:
            raise RuntimeError("No SPADS regression components available for prediction")

        post = self.trace.posterior
        rng = np.random.default_rng(int(random_seed))

        def _flat(name: str) -> np.ndarray:
            arr = np.asarray(post[name].values)
            return arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])

        beta0 = _flat("beta_0")
        n_post = int(beta0.shape[0])
        if n_draws <= 0:
            raise ValueError("n_draws must be positive")
        sel = rng.choice(n_post, size=int(n_draws), replace=(int(n_draws) > n_post))

        beta_blocks: list[np.ndarray] = []
        alpha_blocks: list[np.ndarray] = []
        sigma_blocks: list[np.ndarray] = []
        nu_blocks: list[np.ndarray | None] = []

        for j, comp in enumerate(components):
            beta_j = _flat(f"beta_{j}")[sel]
            beta_blocks.append(np.asarray(beta_j, dtype=float))

            if self.cfg.fit_intercept and f"alpha_{j}" in post:
                alpha_j = _flat(f"alpha_{j}")[sel]
                alpha_blocks.append(np.asarray(alpha_j, dtype=float).reshape(-1))
            else:
                alpha_blocks.append(np.zeros((len(sel),), dtype=float))

            if self.cfg.estimate_component_sigma and f"sigma_{j}" in post:
                sigma_j = _flat(f"sigma_{j}")[sel]
                sigma_blocks.append(np.asarray(sigma_j, dtype=float).reshape(-1))
            else:
                sigma_blocks.append(np.full((len(sel),), float(self.cfg.component_sigma), dtype=float))

            nu_name = f"nu_{j}"
            if nu_name in post:
                nu_j = _flat(nu_name)[sel]
                nu_blocks.append(np.asarray(nu_j, dtype=float).reshape(-1))
            else:
                nu_blocks.append(None)

        T = int(np.asarray(dataset_full.time).size)
        draws = np.empty((int(n_draws), T), dtype=float)
        noise_levels = np.asarray(state["noise_levels"], dtype=float)

        groups: dict[float, list[int]] = {}
        for j, comp in enumerate(components):
            groups.setdefault(float(comp.noise), []).append(j)

        for i in range(int(n_draws)):
            by_noise = []
            for noise in noise_levels:
                idxs = groups.get(float(noise), [])
                if not idxs:
                    continue
                y_sum = np.zeros((T,), dtype=float)
                for j in idxs:
                    comp = components[j]
                    beta = beta_blocks[j][i]
                    alpha = float(alpha_blocks[j][i])
                    mu = alpha + np.asarray(comp.X_full, dtype=float) @ np.asarray(beta, dtype=float)

                    if mode_eff == "component_ppc":
                        sig = float(sigma_blocks[j][i])
                        nu_j = nu_blocks[j]
                        if nu_j is None:
                            mu = mu + rng.normal(0.0, sig, size=mu.shape)
                        else:
                            mu = mu + rng.standard_t(df=float(nu_j[i]), size=mu.shape) * sig

                    y_sum = y_sum + np.asarray(mu, dtype=float)
                by_noise.append(y_sum)

            if not by_noise:
                raise RuntimeError("Prediction failed: no reconstructed noise-level predictions were assembled")

            y_draw = np.mean(np.vstack(by_noise), axis=0)
            y_draw = self._restore_reference_draw(y_draw, state)
            draws[i] = np.asarray(y_draw, dtype=float)

        return ExpertPrediction(
            name=self.name,
            time=np.asarray(dataset_full.time),
            draws=np.asarray(draws, dtype=float),
        )

    def save_default_plots(
        self,
        *,
        out_dir,
        dataset_full: TimeSeriesDataset,
        n_draws: int = 2000,
        random_seed: int = 42,
        label: str | None = None,
        split_date: np.datetime64 | str | None = None,
        mode: Literal["latent", "component_ppc"] | None = None,
    ) -> None:
        from pathlib import Path
        from slmcal.plotting import make_default_calibration_plots

        pred = self.predict_draws(
            dataset_full,
            n_draws=int(n_draws),
            random_seed=int(random_seed),
            mode=mode,
        )
        d = np.asarray(pred.draws, dtype=float)

        per1 = np.nanpercentile(d, 1, axis=0)
        per5 = np.nanpercentile(d, 5, axis=0)
        per10 = np.nanpercentile(d, 10, axis=0)
        per50 = np.nanpercentile(d, 50, axis=0)
        per90 = np.nanpercentile(d, 90, axis=0)
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
            per90=per90,
            per95=per95,
            per1=per1,
            per99=per99,
            mini=mini,
            maxi=maxi,
            draws=d,
            obs_time=np.asarray(dataset_full.obs_time),
            obs=np.asarray(dataset_full.obs, dtype=float),
            trace=self.trace,
            ppc=None,
            param_names=(),
            label=label or self.name,
            obs_mask_cal=obs_mask_cal,
            obs_mask_val=obs_mask_val,
            split_date=split_dt,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_initvals(self, components: list[_SPADSRegComponent]) -> list[dict] | None:
        cfg = self.cfg
        if int(cfg.chains) <= 0:
            return None

        rng = np.random.default_rng(int(cfg.random_seed))
        initvals: list[dict] = []
        prior_kind = str(cfg.coef_prior).lower().strip()
        like = str(cfg.likelihood).lower().strip()

        for _ in range(int(cfg.chains)):
            d: dict[str, np.ndarray | float] = {}
            for j, comp in enumerate(components):
                beta0 = np.asarray(comp.beta_init, dtype=float) if cfg.init_from_deterministic else np.zeros_like(comp.beta_init)

                if prior_kind == "empirical_mvn":
                    try:
                        beta_cov = self._ensure_spd(
                            np.asarray(comp.beta_prior_cov, dtype=float) * max(float(cfg.init_jitter_rel), 1e-3),
                            float(cfg.empirical_prior_jitter),
                        )
                        beta_draw = rng.multivariate_normal(np.asarray(comp.beta_prior_mean, dtype=float), beta_cov)
                    except Exception:
                        scale = float(cfg.init_jitter_abs) + float(cfg.init_jitter_rel) * np.maximum(np.abs(beta0), 1.0)
                        beta_draw = beta0 + rng.normal(0.0, scale, size=beta0.shape)
                else:
                    scale = float(cfg.init_jitter_abs) + float(cfg.init_jitter_rel) * np.maximum(np.abs(beta0), 1.0)
                    beta_draw = beta0 + rng.normal(0.0, scale, size=beta0.shape)

                d[f"beta_{j}"] = np.asarray(beta_draw, dtype="float64")

                if prior_kind == "hierarchical":
                    tau0 = max(float(np.nanstd(beta0)), float(cfg.init_jitter_abs), 0.1 * float(cfg.coef_prior_scale), 1e-6)
                    d[f"tau_{j}"] = float(abs(tau0 * np.exp(rng.normal(0.0, 0.15))))

                if cfg.fit_intercept:
                    alpha0 = float(comp.alpha_init if cfg.init_from_deterministic else 0.0)
                    alpha_scale = float(cfg.init_jitter_abs) + float(cfg.init_jitter_rel) * max(abs(alpha0), 1.0)
                    d[f"alpha_{j}"] = float(alpha0 + rng.normal(0.0, alpha_scale))

                if cfg.estimate_component_sigma:
                    sigma0 = max(float(comp.sigma_init), float(cfg.init_jitter_abs), 1e-6)
                    d[f"sigma_{j}"] = float(abs(sigma0 * np.exp(rng.normal(0.0, 0.10))))

                if like in {"studentt", "student_t", "t"}:
                    d[f"nu_minus_{j}"] = float(max(1e-6, float(cfg.nu_scale) * np.exp(rng.normal(0.0, 0.10))))

            initvals.append(d)

        return initvals

    def _compute_empirical_regression_stats(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray, float, float]:
        cfg = self.cfg
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, p = X.shape
        ridge = max(float(cfg.empirical_prior_ridge), 0.0)
        jitter = max(float(cfg.empirical_prior_jitter), 1e-12)

        if cfg.fit_intercept:
            Xa = np.column_stack([np.ones((n,), dtype=float), X])
            penalty = np.eye(p + 1, dtype=float)
            penalty[0, 0] = 0.0
        else:
            Xa = X
            penalty = np.eye(p, dtype=float)

        A = Xa.T @ Xa
        A_reg = A + ridge * penalty
        b = Xa.T @ y

        try:
            theta_hat = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            theta_hat = np.linalg.pinv(A_reg) @ b

        resid = y - Xa @ theta_hat
        rss = float(np.dot(resid, resid))
        dof = max(1, int(n - Xa.shape[1]))
        sigma2 = max(rss / float(dof), 1e-12)

        cov_theta = float(cfg.empirical_prior_scale) * sigma2 * np.linalg.pinv(A_reg)
        cov_theta = self._ensure_spd(cov_theta, jitter)

        if cfg.fit_intercept:
            alpha_mean = float(theta_hat[0])
            beta_mean = np.asarray(theta_hat[1:], dtype=float)
            alpha_sd = float(np.sqrt(max(cov_theta[0, 0], jitter)))
            beta_cov = self._ensure_spd(np.asarray(cov_theta[1:, 1:], dtype=float), jitter)
        else:
            alpha_mean = 0.0
            beta_mean = np.asarray(theta_hat, dtype=float)
            alpha_sd = float(max(cfg.intercept_prior_scale, jitter))
            beta_cov = self._ensure_spd(np.asarray(cov_theta, dtype=float), jitter)

        sigma_init = float(np.sqrt(max(sigma2, 1e-12)))
        return beta_mean, alpha_mean, sigma_init, beta_mean.copy(), beta_cov, alpha_sd

    def _ensure_spd(self, cov: np.ndarray, jitter: float) -> np.ndarray:
        cov = np.asarray(cov, dtype=float)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
        cov = np.atleast_2d(cov)
        cov = 0.5 * (cov + cov.T)

        scale = max(1.0, float(np.mean(np.abs(np.diag(cov)))) if cov.shape[0] else 1.0)
        eye = np.eye(cov.shape[0], dtype=float)
        cov = cov + float(jitter) * scale * eye

        try:
            eigmin = float(np.min(np.linalg.eigvalsh(cov)))
        except np.linalg.LinAlgError:
            eigmin = -1.0
        if not np.isfinite(eigmin) or eigmin <= 0.0:
            cov = cov + (abs(eigmin) + float(jitter) * scale) * eye

        return cov

    # ------------------------------------------------------------------
    # Internal SPADS state construction
    # ------------------------------------------------------------------
    def _build_spads_state(self, dataset: TimeSeriesDataset) -> dict:
        if self._fit_state is not None and self._dataset_cal is not None:
            try:
                t_cur = np.asarray(dataset.time)
                t_cal = np.asarray(self._dataset_cal.time)
                if t_cur.size == t_cal.size and np.array_equal(t_cur, t_cal):
                    return self._fit_state
            except Exception:
                pass

        t = np.asarray(dataset.time)
        key = (int(t.__array_interface__["data"][0]), int(t.size))
        if key in self._pred_state_cache:
            return self._pred_state_cache[key]

        state = self._run_spads_state_build(dataset)
        self._pred_state_cache[key] = state
        return state

    def _run_spads_state_build(self, dataset: TimeSeriesDataset) -> dict:
        cfg = self.cfg
        if self._dataset_cal is None:
            raise RuntimeError("fit() must be called before SPADS state building")

        try:
            import pandas as pd
            from pySPADS.pipeline import steps
            from pySPADS.processing.reconstruct import get_X, get_y
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "SPADSRegressionBayesExpert requires pySPADS + pandas + PyEMD/pyeemd."
            ) from e

        def _sanitize_imfs(df, *, name: str):
            df = df.replace([np.inf, -np.inf], np.nan)
            if df.isna().any().any():
                try:
                    df = df.interpolate(method="time", limit_direction="both")
                except Exception:
                    df = df.interpolate(limit_direction="both")
                df = df.ffill().bfill()
                all_nan_cols = df.columns[df.isna().all(axis=0)]
                if len(all_nan_cols) > 0:
                    df = df.drop(columns=list(all_nan_cols))
                df = df.fillna(0.0)
            return df

        def _save_df_npz(path: Path, df) -> None:
            try:
                idx = pd.to_datetime(df.index).astype("datetime64[ns]").view("int64")
                cols = np.asarray([str(c) for c in df.columns], dtype=object)
                arr = np.asarray(df.to_numpy(), dtype=np.float64)
                np.savez_compressed(path, idx=idx, cols=cols, arr=arr)
            except Exception:
                return

        def _load_df_npz(path: Path):
            z = np.load(path, allow_pickle=True)
            idx = pd.to_datetime(np.asarray(z["idx"], dtype=np.int64))
            cols = [int(c) if str(c).lstrip("-").isdigit() else str(c) for c in list(z["cols"])]
            arr = np.asarray(z["arr"], dtype=np.float64)
            return pd.DataFrame(arr, index=idx, columns=cols)

        def _imf_cache_path(kind: str, name: str, key: str) -> Path | None:
            if not (cfg.cache_dir and cfg.cache_imfs):
                return None
            try:
                d = Path(str(cfg.cache_dir))
                d.mkdir(parents=True, exist_ok=True)
                return d / f"spads_reg_imf_{kind}_{name}_{key}.npz"
            except Exception:
                return None

        def _digest_series(ser: "pd.Series", *, noise: float) -> str:
            h = hashlib.sha1()
            t64 = pd.to_datetime(ser.index).astype("datetime64[ns]").view("int64")
            h.update(np.asarray(t64, dtype=np.int64).tobytes())
            h.update(np.asarray(ser.to_numpy(), dtype=np.float64).tobytes())
            h.update(str(noise).encode("utf-8"))
            h.update(str(int(cfg.num_trials)).encode("utf-8"))
            h.update(str(bool(cfg.parallel)).encode("utf-8"))
            h.update(str(int(cfg.processes)).encode("utf-8"))
            h.update(str(cfg.parallel_kind).encode("utf-8"))
            return h.hexdigest()[:16]

        def _decompose_cached(ser: "pd.Series", *, name: str, noise: float, kind: str):
            key = _digest_series(ser, noise=float(noise))
            p = _imf_cache_path(kind, name, key)
            if p is not None and p.exists():
                try:
                    df = _load_df_npz(p)
                    if cfg.cache_verbose:
                        print(f"[SPADSReg cache] loaded IMFs: {p}")
                    return df
                except Exception:
                    pass

            df = steps.decompose(
                ser,
                float(noise),
                num_trials=int(cfg.num_trials),
                progress=False,
                parallel=bool(cfg.parallel),
                processes=int(cfg.processes),
                parallel_kind=str(cfg.parallel_kind),
            )
            df = _sanitize_imfs(df, name=f"{kind}:{name}")
            if p is not None:
                _save_df_npz(p, df)
                if cfg.cache_verbose:
                    print(f"[SPADSReg cache] saved IMFs: {p}")
            return df

        if not np.issubdtype(np.asarray(dataset.time).dtype, np.datetime64):
            raise ValueError("SPADS requires datetime64 time axis")

        t_full = pd.to_datetime(np.asarray(dataset.time)).tz_localize(None)
        cal = self._dataset_cal
        cal_start = pd.to_datetime(np.asarray(cal.time)[0]).tz_localize(None)
        cal_end = pd.to_datetime(np.asarray(cal.time)[-1]).tz_localize(None)

        y_obs_ser = pd.Series(
            data=np.asarray(cal.obs, dtype=float),
            index=pd.to_datetime(np.asarray(cal.obs_time)),
        ).sort_index()
        if y_obs_ser.index.duplicated().any():
            y_obs_ser = y_obs_ser.groupby(level=0).mean()
        y_obs_ser.index = pd.to_datetime(y_obs_ser.index).tz_localize(None)

        idx_all = y_obs_ser.index.union(t_full)
        y_full_ser = (
            y_obs_ser.reindex(idx_all)
            .sort_index()
            .interpolate(method="time")
            .reindex(t_full)
            .ffill()
            .bfill()
        )
        if int(y_full_ser.isna().sum()) > 0:
            raise ValueError("SPADSRegressionBayesExpert: interpolated shoreline series still contains NaNs")

        drivers: dict[str, "pd.Series"] = {}
        for fn in cfg.forcing_names:
            if fn not in dataset.forcings:
                raise KeyError(f"Missing forcing '{fn}' required by SPADS")
            drivers[str(fn)] = pd.Series(np.asarray(dataset.forcings[fn], dtype=float), index=t_full)

        signal_name = "shoreline"
        components: list[_SPADSRegComponent] = []
        nearest_by_noise: dict[float, object] = {}

        is_frozen_prediction = self._fit_state is not None and self._dataset_cal is not None
        try:
            t_cal = np.asarray(self._dataset_cal.time)
            t_cur = np.asarray(dataset.time)
            if t_cur.size == t_cal.size and np.array_equal(t_cur, t_cal):
                is_frozen_prediction = False
        except Exception:
            pass

        fit_components: list[_SPADSRegComponent] = []
        if is_frozen_prediction and self._fit_state is not None:
            fit_components = list(self._fit_state.get("components", []))
            nearest_by_noise = dict(self._fit_state.get("nearest_by_noise", {}))

        if is_frozen_prediction:
            imfs_by_noise: dict[float, dict[str, object]] = {}
            for noise in cfg.noise_levels:
                noise_f = float(noise)
                imfs_noise: dict[str, object] = {}
                for k, ser in drivers.items():
                    imf_k = _decompose_cached(ser, name=str(k), noise=noise_f, kind="driver")
                    if cfg.reject_noise:
                        imf_k = steps.reject_noise(imf_k, noise_threshold=float(cfg.reject_noise_alpha))
                    imfs_noise[k] = _sanitize_imfs(imf_k, name=f"driver:{k}@{noise_f}")
                imfs_by_noise[noise_f] = imfs_noise

            for comp_counter, ref_comp in enumerate(fit_components):
                noise_f = float(ref_comp.noise)
                nearest = nearest_by_noise.get(noise_f)
                if nearest is None:
                    continue
                imfs_noise = imfs_by_noise.get(noise_f, {})
                X_full_df = get_X(imfs_noise, nearest, signal_name, ref_comp.component, t_full)
                X_full_df.columns = [str(c) for c in X_full_df.columns]
                X_full_df = X_full_df.reindex(columns=list(ref_comp.feature_names), fill_value=0.0)
                X_full = np.asarray(X_full_df, dtype=float)
                if cfg.normalize:
                    X_full = (X_full - ref_comp.x_mean[None, :]) / ref_comp.x_scale[None, :]
                if not np.all(np.isfinite(X_full)):
                    X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
                components.append(replace(ref_comp, uid=f"n{int(comp_counter):03d}", X_full=X_full))
        else:
            comp_counter = 0
            for noise in cfg.noise_levels:
                noise_f = float(noise)
                imfs: dict[str, object] = {}
                imf_sig = _decompose_cached(y_full_ser, name="shoreline", noise=noise_f, kind="signal")
                if cfg.reject_noise:
                    imf_sig = steps.reject_noise(imf_sig, noise_threshold=float(cfg.reject_noise_alpha))
                imfs[signal_name] = _sanitize_imfs(imf_sig, name=f"signal@{noise_f}")

                for k, ser in drivers.items():
                    imf_k = _decompose_cached(ser, name=str(k), noise=noise_f, kind="driver")
                    if cfg.reject_noise:
                        imf_k = steps.reject_noise(imf_k, noise_threshold=float(cfg.reject_noise_alpha))
                    imfs[k] = _sanitize_imfs(imf_k, name=f"driver:{k}@{noise_f}")

                nearest = steps.match_frequencies(
                    imfs,
                    signal=signal_name,
                    threshold=float(cfg.freq_threshold),
                    exclude_trend=bool(cfg.exclude_trend),
                )
                nearest_by_noise[noise_f] = nearest.copy()

                imfs_cal = {
                    k: v.loc[cal_start:cal_end] if hasattr(v, "loc") else v
                    for k, v in imfs.items()
                }
                imfs_cal = {
                    k: _sanitize_imfs(v, name=f"cal:{k}@{noise_f}") if hasattr(v, "isna") else v
                    for k, v in imfs_cal.items()
                }

                index_cal = imfs_cal[signal_name].index
                index_full = t_full

                output_columns = list(imfs[signal_name].columns)
                if cfg.exclude_trend and len(output_columns) > 0:
                    output_columns = output_columns[:-1]

                for component in output_columns:
                    if component not in nearest.index:
                        continue
                    X_cal_df = get_X(imfs_cal, nearest, signal_name, component, index_cal)
                    if X_cal_df.shape[1] == 0:
                        continue
                    y_cal_ser = get_y(imfs_cal, signal_name, component, index_cal)
                    X_full_df = get_X(imfs, nearest, signal_name, component, index_full)

                    X_cal_df.columns = [str(c) for c in X_cal_df.columns]
                    X_full_df.columns = [str(c) for c in X_full_df.columns]

                    keep_cols = [c for c in X_cal_df.columns if c in X_full_df.columns]
                    if len(keep_cols) == 0:
                        continue
                    X_cal_df = X_cal_df.loc[:, keep_cols]
                    X_full_df = X_full_df.loc[:, keep_cols]

                    good_cols: list[str] = []
                    for col in X_cal_df.columns:
                        x = np.asarray(X_cal_df[col], dtype=float)
                        if np.all(np.isfinite(x)) and (np.nanstd(x) > 0):
                            good_cols.append(str(col))
                    if len(good_cols) == 0:
                        continue
                    X_cal_df = X_cal_df.loc[:, good_cols]
                    X_full_df = X_full_df.loc[:, good_cols]

                    X_cal = np.asarray(X_cal_df, dtype=float)
                    X_full = np.asarray(X_full_df, dtype=float)
                    y_cal = np.asarray(y_cal_ser, dtype=float)

                    x_mean = np.zeros((X_cal.shape[1],), dtype=float)
                    x_scale = np.ones((X_cal.shape[1],), dtype=float)
                    if cfg.normalize:
                        x_mean = np.nanmean(X_cal, axis=0)
                        x_scale = np.nanstd(X_cal, axis=0)
                        x_scale = np.where(x_scale > 0, x_scale, 1.0)
                        X_cal = (X_cal - x_mean[None, :]) / x_scale[None, :]
                        X_full = (X_full - x_mean[None, :]) / x_scale[None, :]

                    if (not np.all(np.isfinite(X_cal))) or (not np.all(np.isfinite(y_cal))) or (not np.all(np.isfinite(X_full))):
                        continue
                    if X_cal.shape[0] != y_cal.shape[0] or X_cal.shape[0] == 0:
                        continue

                    beta_init, alpha_init, sigma_init, beta_prior_mean, beta_prior_cov, alpha_prior_sd = self._compute_empirical_regression_stats(X_cal, y_cal)

                    uid = f"n{int(comp_counter):03d}"
                    components.append(
                        _SPADSRegComponent(
                            uid=uid,
                            noise=noise_f,
                            component=int(component),
                            feature_names=[str(c) for c in X_cal_df.columns],
                            X_cal=X_cal,
                            y_cal=y_cal,
                            X_full=X_full,
                            x_mean=x_mean,
                            x_scale=x_scale,
                            beta_init=beta_init,
                            alpha_init=float(alpha_init),
                            sigma_init=float(sigma_init),
                            beta_prior_mean=np.asarray(beta_prior_mean, dtype=float),
                            beta_prior_cov=np.asarray(beta_prior_cov, dtype=float),
                            alpha_prior_mean=float(alpha_init),
                            alpha_prior_sd=float(alpha_prior_sd),
                        )
                    )
                    comp_counter += 1

        state = {
            "components": components,
            "nearest_by_noise": nearest_by_noise,
            "noise_levels": [float(n) for n in cfg.noise_levels],
            "t_full": np.asarray(t_full.to_numpy(), dtype="datetime64[ns]"),
            "y_ref": np.asarray(y_full_ser.to_numpy(), dtype=float),
            "cal_start": np.datetime64(cal_start.to_datetime64()),
            "cal_end": np.datetime64(cal_end.to_datetime64()),
        }
        return state

    def _restore_reference_draw(self, yhat: np.ndarray, state: dict) -> np.ndarray:
        mode = str(self.cfg.restore_reference).lower().strip()
        if mode == "none":
            return np.asarray(yhat, dtype=float)

        yhat = np.asarray(yhat, dtype=float).copy()
        y_ref = np.asarray(state["y_ref"], dtype=float)
        t64 = np.asarray(state["t_full"], dtype="datetime64[ns]")
        cal_start = np.datetime64(state["cal_start"])
        cal_end = np.datetime64(state["cal_end"])

        tt = (t64 - t64[0]) / np.timedelta64(1, "D")
        tt = np.asarray(tt, dtype=float)

        if mode == "lintrend_full":
            m = np.isfinite(y_ref) & np.isfinite(yhat) & np.isfinite(tt)
        else:
            m_win = (t64 >= cal_start) & (t64 <= cal_end)
            m = np.asarray(m_win, dtype=bool) & np.isfinite(y_ref) & np.isfinite(yhat) & np.isfinite(tt)

        if int(np.sum(m)) < 1:
            return yhat

        if mode == "offset_first":
            i0 = int(np.flatnonzero(m)[0])
            yhat = yhat + float(y_ref[i0] - yhat[i0])
            return yhat

        if mode == "offset_mean":
            yhat = yhat + float(np.mean((y_ref - yhat)[m]))
            return yhat

        deg = int(getattr(self.cfg, "restore_trend_degree", 1) or 1)
        deg = max(0, min(3, deg))
        if int(np.sum(m)) < (deg + 1):
            yhat = yhat + float(np.mean((y_ref - yhat)[m]))
            return yhat

        rr = (y_ref - yhat)[m]
        pp = np.polyfit(tt[m], rr, deg=deg)
        corr = np.polyval(pp, tt)
        yhat = yhat + corr
        return yhat
