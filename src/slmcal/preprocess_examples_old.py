from __future__ import annotations

"""Example helpers shared across multiple scripts.

These utilities keep example scripts short and consistent.

They intentionally live in the library (under ``slmcal``) so that:
  - Example scripts don't copy/paste boilerplate.
  - Small API changes (dataset layout, posterior variable names, etc.) can be
    handled centrally.

This module is *not* required by the core library workflows.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import xarray as xr

from .models.base import transform_raw_to_physical
from .data import MultiTransectDataset, TimeSeriesDataset
from .models.ih_moose import IHMooseModel
from .preprocess import PreprocessResultMoose2D, preprocess_legacy_moose2d, preprocess_legacy_millerDean
from .features import add_ewma_forcings


@dataclass(frozen=True)
class Moose2DExampleDatasets:
    """Bundle of datasets commonly used by IH-MOOSE examples."""

    prep: PreprocessResultMoose2D
    ds_cal_2d: MultiTransectDataset
    ds_full_2d: MultiTransectDataset
    ds_cs_cal: TimeSeriesDataset | None
    ds_cs_full: TimeSeriesDataset | None
    ds_ref_cal: TimeSeriesDataset | None
    ds_ref_full: TimeSeriesDataset | None
    ds_rot_cal: TimeSeriesDataset | None


def _default_forcings(prep: PreprocessResultMoose2D, *, split: bool) -> dict[str, np.ndarray]:
    """Build a forcing dict with the standard IH-MOOSE variables."""

    suf = "_splited" if split else ""

    def get(name: str) -> np.ndarray:
        return np.asarray(getattr(prep, f"{name}{suf}"), dtype=float)

    # Keep a superset of forcings. Models will ignore unused keys.
    return {
        "P": get("P"),
        "dir": get("dir"),
        "E": get("E"),
        "Hs": get("hs"),
        "Tp": get("tp"),
        "hb": get("hb"),
        "depthb": get("depthb"),
        "sl": get("sl"),
        "wast": get("wast"),
        "Omega": get("Omega"),
        "cos_dir": get("cos_dir"),
        "sin_dir": get("sin_dir"),
    }


def build_moose2d_example_datasets(
    ds: xr.Dataset,
    *,
    start_date: str,
    split_date: str,
    end_date: str | None,
    params: Mapping[str, Any],
    include_cs_cal: bool = True,
    include_ref_timeseries: bool = False,
    include_rot_cal: bool = True,
) -> Moose2DExampleDatasets:
    """Build standard datasets for IH-MOOSE examples from an xarray Dataset."""

    prep = preprocess_legacy_moose2d(
        ds,
        start_date=start_date,
        split_date=split_date,
        end_date=end_date,
        params=dict(params),
    )

    forc_cal = _default_forcings(prep, split=True)
    forc_full = _default_forcings(prep, split=False)

    ref_rot = float(prep.ref_rot) if prep.ref_rot is not None and np.isfinite(float(prep.ref_rot)) else None

    ds_cal_2d = MultiTransectDataset(
        time=prep.time_splited,
        forcings=forc_cal,
        obs_time=prep.time_obs_splited,
        obs=prep.Obs_splited,
        alpha0=prep.alpha0,
        mask_nan_obs=prep.mask_nan_obs_splited,
        dt=prep.dt_splited,
        xi=prep.xi,
        yi=prep.yi,
        xf=prep.xf,
        yf=prep.yf,
        phi=prep.phi,
        x_pivotal=float(prep.x_pivotal),
        y_pivotal=float(prep.y_pivotal),
        phi_pivotal=float(prep.phi_pivotal),
        ref_transect=int(prep.ref_transect),
        hberm=float(prep.hberm),
        ref_rot=ref_rot,
    )

    ds_full_2d = MultiTransectDataset(
        time=prep.time,
        forcings=forc_full,
        obs_time=prep.time_obs,
        obs=prep.Obs,
        alpha0=prep.alpha0,
        mask_nan_obs=prep.mask_nan_obs,
        dt=prep.dt,
        xi=prep.xi,
        yi=prep.yi,
        xf=prep.xf,
        yf=prep.yf,
        phi=prep.phi,
        x_pivotal=float(prep.x_pivotal),
        y_pivotal=float(prep.y_pivotal),
        phi_pivotal=float(prep.phi_pivotal),
        ref_transect=int(prep.ref_transect),
        hberm=float(prep.hberm),
        ref_rot=ref_rot,
    )

    ds_cs_cal = ds_cal_2d.to_timeseries(int(prep.ref_transect)) if include_cs_cal else None
    ds_cs_full = ds_full_2d.to_timeseries(int(prep.ref_transect)) if include_cs_cal else None
    ds_ref_cal = ds_cal_2d.to_timeseries_ref() if include_ref_timeseries else None
    ds_ref_full = ds_full_2d.to_timeseries_ref() if include_ref_timeseries else None

    ds_rot_cal: TimeSeriesDataset | None = None
    if include_rot_cal:
        if np.asarray(prep.rot_splited).size == 0:
            raise ValueError("Dataset has no 'rot' in the calibration window; cannot build rotation prior.")

        good = ~np.asarray(prep.mask_nan_rot_splited, dtype=bool)
        ds_rot_cal = TimeSeriesDataset(
            time=prep.time_splited,
            forcings={"P": prep.P_splited, "dir": prep.dir_splited},
            obs_time=np.asarray(prep.time_obs_splited)[good],
            rot=np.asarray(prep.rot_splited, dtype=float)[good],
            idx_obs=np.asarray(prep.idx_obs_splited, dtype=int)[good],
            dt=prep.dt_splited,
            ref_rot=ref_rot,
        )

    return Moose2DExampleDatasets(
        prep=prep,
        ds_cal_2d=ds_cal_2d,
        ds_full_2d=ds_full_2d,
        ds_cs_cal=ds_cs_cal,
        ds_cs_full=ds_cs_full,
        ds_ref_cal=ds_ref_cal,
        ds_ref_full=ds_ref_full,
        ds_rot_cal=ds_rot_cal,
    )


def sample_posterior_draws(
    trace,
    model,
    dataset_full,
    n_draws: int,
    *,
    seed: int = 123,
    dtype=np.float32,
    max_bytes: int = 300_000_000,
    verbose: bool = True,
    progress_every: int = 25,
    mode: str = "latent",
    sigma_fixed: float | None = None,
    include_bias: bool = True,
    noise_at_obs_only: bool = False,
):
    """Draw posterior predictive simulations with bounded memory.

    This helper supports both 1D time-series models (e.g. Yates/Miller-Dean)
    and 2D IH-MOOSE-style models.

    Returns
    -------
    y_draws : ndarray
        Shape ``(n_draws, n_time)`` for 1D models or ``(n_draws, n_time, n_tr)``
        for 2D models.
    alpha_draws : ndarray | None
        Rotation/auxiliary series when the model provides it, otherwise ``None``.
    """

    rng = np.random.default_rng(int(seed))
    post = trace.posterior

    if "raw_par" in post:
        raw = post["raw_par"].values
        raw_flat = raw.reshape(raw.shape[0] * raw.shape[1], raw.shape[2])
    else:
        blocks: list[np.ndarray] = []
        bi = 0
        while f"raw_par_{bi}" in post:
            b = post[f"raw_par_{bi}"].values
            blocks.append(b.reshape(-1, b.shape[-1]))
            bi += 1
        if not blocks:
            raise KeyError("Posterior does not contain 'raw_par' or any 'raw_par_i' blocks.")
        raw_flat = np.concatenate(blocks, axis=1)

    n_avail = int(raw_flat.shape[0])

    bias_flat = None
    if include_bias and ("bias" in post):
        try:
            bias_flat = post["bias"].values.reshape(-1).astype(float)
        except Exception:
            bias_flat = None

    y0_flat = None
    if "y0" in post:
        try:
            y0_flat = post["y0"].values.reshape(-1).astype(float)
        except Exception:
            y0_flat = None

    sigma_flat = None
    if "sigma" in post:
        try:
            sigma_flat = post["sigma"].values.reshape(-1).astype(float)
        except Exception:
            sigma_flat = None
    elif "log_sigma" in post:
        try:
            sigma_flat = np.exp(post["log_sigma"].values.reshape(-1).astype(float))
        except Exception:
            sigma_flat = None

    if sigma_flat is None and sigma_fixed is not None:
        sigma_flat = np.full((n_avail,), float(sigma_fixed), dtype=float)

    n_t = int(np.asarray(dataset_full.time).size)
    obs_arr = getattr(dataset_full, "obs", None)
    obs_arr = None if obs_arr is None else np.asarray(obs_arr)
    is_2d = bool(obs_arr is not None and obs_arr.ndim == 2)
    n_tr = int(obs_arr.shape[1]) if is_2d else 1

    has_alpha = hasattr(model, "simulate_with_alpha") or hasattr(model, "predict_alpha")
    item = np.dtype(dtype).itemsize
    alpha_mult = 1 if has_alpha else 0
    est_bytes = int(n_draws) * int(n_t) * int(n_tr + alpha_mult) * item
    if est_bytes > int(max_bytes):
        cap = max(1, int(max_bytes // (int(n_t) * int(n_tr + alpha_mult) * item)))
        if verbose:
            mb = est_bytes / 1e6
            mb_cap = (cap * n_t * (n_tr + alpha_mult) * item) / 1e6
            print(f"[ppc] Requested {n_draws} draws (~{mb:.1f} MB) exceeds max_bytes={max_bytes} bytes.")
            print(f"[ppc] Capping predictive draws to {cap} (~{mb_cap:.1f} MB).")
        n_draws = cap

    idx = rng.choice(n_avail, size=int(n_draws), replace=(n_draws > n_avail))

    if is_2d:
        y_draws = np.empty((int(n_draws), n_t, n_tr), dtype=dtype)
    else:
        y_draws = np.empty((int(n_draws), n_t), dtype=dtype)
    alpha_draws = np.empty((int(n_draws), n_t), dtype=dtype) if has_alpha else None

    mode_l = str(mode).lower().strip()
    if mode_l not in ("latent", "mean", "ppc"):
        raise ValueError(f"Invalid mode={mode!r}. Use 'latent', 'mean', or 'ppc'.")

    obs_idx = None
    if noise_at_obs_only and mode_l == "ppc":
        ii = getattr(dataset_full, "idx_obs", None)
        if ii is not None:
            try:
                obs_idx = np.asarray(ii, dtype=np.int64)
            except Exception:
                obs_idx = None

    use_fast = hasattr(model, "simulate_with_alpha")
    use_predict_alpha = hasattr(model, "predict_alpha")

    for i, j in enumerate(idx):
        phys = transform_raw_to_physical(raw_flat[j], model.parameters)
        y0_j = None
        if y0_flat is not None and y0_flat.size == n_avail:
            y0_j = float(y0_flat[j])

        if use_fast:
            y, alpha = model.simulate_with_alpha(phys, dataset_full)
        else:
            if y0_j is not None:
                try:
                    y = model.simulate(phys, dataset_full, y0=y0_j)
                except TypeError:
                    y = model.simulate(phys, dataset_full)
            else:
                y = model.simulate(phys, dataset_full)

            if use_predict_alpha:
                alpha = model.predict_alpha(phys, dataset_full)
            else:
                alpha = None

        mu = np.asarray(y, dtype=float)
        if mode_l in ("mean", "ppc") and bias_flat is not None and bias_flat.size == n_avail:
            mu = mu + float(bias_flat[j])

        if mode_l == "ppc":
            if sigma_flat is None or sigma_flat.size != n_avail:
                raise ValueError(
                    "mode='ppc' requires sigma in trace (sigma/log_sigma) or sigma_fixed."
                )
            sig = float(sigma_flat[j])
            if obs_idx is None:
                mu = mu + rng.normal(0.0, sig, size=mu.shape)
            else:
                try:
                    eps = np.zeros_like(mu, dtype=float)
                    if mu.ndim == 1:
                        m = (obs_idx >= 0) & (obs_idx < mu.shape[0])
                        ii = obs_idx[m]
                        eps[ii] = rng.normal(0.0, sig, size=ii.size)
                    else:
                        n_t_mu, n_tr_mu = mu.shape
                        tt = obs_idx // n_tr_mu
                        tr = obs_idx % n_tr_mu
                        m = (tt >= 0) & (tt < n_t_mu) & (tr >= 0) & (tr < n_tr_mu)
                        tt = tt[m]
                        tr = tr[m]
                        eps[tt, tr] = rng.normal(0.0, sig, size=tt.size)
                    mu = mu + eps
                except Exception:
                    mu = mu + rng.normal(0.0, sig, size=mu.shape)

        y_draws[i] = np.asarray(mu, dtype=dtype)
        if alpha_draws is not None:
            if alpha is None:
                alpha_draws[i] = np.full((n_t,), np.nan, dtype=dtype)
            else:
                alpha_draws[i] = np.asarray(alpha, dtype=dtype)

        if verbose and progress_every and ((i + 1) % int(progress_every) == 0 or (i + 1) == int(n_draws)):
            print(f"[ppc] {i+1}/{int(n_draws)} posterior draws simulated")

    return y_draws, alpha_draws

def build_datasets_MoE(
        ds: xr.Dataset, 
        md_params: dict[str, float], 
        split_date: str, 
        start_date: str, end_date: str | None, 
        use_ewma: bool | False, 
        ewma_keys: tuple[str, ...] | None, 
        ewma_taus: tuple[float, ...] | None
        ) -> tuple[TimeSeriesDataset, TimeSeriesDataset]:

    prep = preprocess_legacy_millerDean(
        ds,
        start_date=start_date,
        split_date=split_date,
        end_date=end_date,
        params=md_params,
        cop_vars=True
    )


    # Calibration dataset (forcing grid within [start, split))
    ds_cal = TimeSeriesDataset(
        time=prep.time_splited,
        forcings={
            "E": prep.E_splited,
            "P": prep.P_splited,
            "Hs": prep.hs_splited,
            "Tp": prep.tp_splited,
            "dir": prep.dir_splited,
            "hb": prep.hb_splited,
            "depthb": prep.depthb_splited,
            "sl": prep.sl_splited,
            "wast": prep.wast_splited,
            "Omega": prep.Omega_splited,
            "cos_dir": prep.cos_dir_splited,
            "sin_dir": prep.sin_dir_splited,
            "tm": prep.tm_splitted,
            "mslp": prep.mslp_splitted,
            "sp": prep.sp_splitted,
            "sst": prep.sst_splitted,
        },
        obs_time=prep.time_obs_splited,
        obs=prep.Obs_splited,
        y0=float(prep.Obs_splited[0]),
        idx_obs=prep.idx_obs_splited,
        dt=prep.dt_splited,
        hberm=np.float64(prep.hberm),
    )

    # Full dataset
    ds_full = TimeSeriesDataset(
        time=prep.time,
        forcings={
            "E": prep.E,
            "P": prep.P,
            "Hs": prep.hs,
            "Tp": prep.tp,
            "dir": prep.dir,
            "hb": prep.hb,
            "depthb": prep.depthb,
            "sl": prep.sl,
            "wast": prep.wast,
            "Omega": prep.Omega,
            "cos_dir": prep.cos_dir,
            "sin_dir": prep.sin_dir,
            "tm": prep.tm,
            "mslp": prep.mslp,
            "sp": prep.sp,
            "sst": prep.sst,
        },
        obs_time=prep.time_obs,
        obs=prep.Obs,
        y0=float(prep.Obs[0]),
        idx_obs=None,
        dt=prep.dt,
        hberm=np.float64(prep.hberm),
    )

    if use_ewma:
        add_ewma_forcings(ds_cal, keys=ewma_keys, taus_days=ewma_taus, in_place=True)
        add_ewma_forcings(ds_full, keys=ewma_keys, taus_days=ewma_taus, in_place=True)

    return ds_cal, ds_full