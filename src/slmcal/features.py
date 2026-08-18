from __future__ import annotations

"""Feature engineering helpers.

This module contains lightweight utilities to build *slow-varying* covariates
from irregularly sampled time series (e.g., wave climate proxies) for use in
the MoE (BART + gate) workflow.

Key utilities:
- `ewma_irregular`: exponential moving average with continuous-time decay,
  robust to irregular time axes (datetime64 or float).
- `moving_average_irregular`: trailing simple moving average on irregular time axes.

Typical use:
- build EWMA features on the **forcing time grid** (dense / regular),
  then sample them at observation times using the dataset's `idx_obs`.
"""

from dataclasses import replace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .data import TimeSeriesDataset


def ewma_irregular(time, x, tau_days: float = 30.0) -> np.ndarray:
    """Exponential moving average with continuous-time decay.

    Parameters
    ----------
    time
        1D array of timestamps (datetime64 or float), monotonic.
    x
        1D array of values aligned with `time`.
    tau_days
        e-folding timescale in **days**.

    Returns
    -------
    y
        1D array of EWMA values, same shape as `x`.
    """
    t = np.asarray(time)
    x = np.asarray(x, dtype=float)

    if t.ndim != 1 or x.ndim != 1:
        raise ValueError("time and x must be 1D")
    if t.shape[0] != x.shape[0]:
        raise ValueError("time and x must have the same length")
    if t.shape[0] == 0:
        return x.copy()

    y = np.empty_like(x, dtype=float)
    y[0] = x[0]

    # dt in days
    if np.issubdtype(t.dtype, np.datetime64):
        dt = (t[1:] - t[:-1]) / np.timedelta64(1, "D")
        dt = dt.astype(float)
    else:
        dt = np.diff(t.astype(float))

    dt = np.clip(dt, 1e-6, None)
    alpha = 1.0 - np.exp(-dt / float(tau_days))  # (n-1,)

    for i in range(1, len(x)):
        y[i] = y[i - 1] + alpha[i - 1] * (x[i] - y[i - 1])

    return y


def moving_average_irregular(time, x, window_days: float = 30.0) -> np.ndarray:
    """Trailing simple moving average on an irregular time axis.

    Parameters
    ----------
    time
        1D array of timestamps (datetime64 or float), monotonic.
    x
        1D array of values aligned with `time`. NaNs are ignored within each
        moving window.
    window_days
        Width of the trailing averaging window in **days**. At each time step
        `i`, the result is the arithmetic mean of samples whose timestamps fall
        in ``[time[i] - window_days, time[i]]``.

    Returns
    -------
    y
        1D array of moving-average values, same shape as `x`.
    """
    t = np.asarray(time)
    x = np.asarray(x, dtype=float)

    if t.ndim != 1 or x.ndim != 1:
        raise ValueError("time and x must be 1D")
    if t.shape[0] != x.shape[0]:
        raise ValueError("time and x must have the same length")
    if t.shape[0] == 0:
        return x.copy()

    window_days = float(window_days)
    if window_days <= 0.0:
        raise ValueError("window_days must be > 0")

    if np.issubdtype(t.dtype, np.datetime64):
        t_num = (t.astype("datetime64[ns]") - t[0].astype("datetime64[ns]")) / np.timedelta64(1, "D")
        t_num = np.asarray(t_num, dtype=float)
    else:
        t_num = np.asarray(t, dtype=float)

    finite = np.isfinite(x)
    x_valid = np.where(finite, x, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(x_valid, dtype=float)))
    ccnt = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))

    y = np.empty_like(x, dtype=float)
    left = 0
    for i in range(len(x)):
        tmin = t_num[i] - window_days
        while left < i and t_num[left] < tmin:
            left += 1
        total = csum[i + 1] - csum[left]
        count = int(ccnt[i + 1] - ccnt[left])
        y[i] = total / count if count > 0 else np.nan

    return y


def add_ewma_forcings(
    dataset: TimeSeriesDataset,
    *,
    keys: Sequence[str] = ("E",),
    taus_days: Sequence[float] = (30.0, 180.0),
    name_fmt: str = "{key}_ewma{tau}",
    in_place: bool = True,
) -> TimeSeriesDataset:
    """Add EWMA-smoothed forcings to a dataset.

    EWMA is computed on the dataset's *forcing time grid* (`dataset.time`) for
    each `key` in `dataset.forcings`.

    Parameters
    ----------
    dataset
        The input dataset.
    keys
        Forcing names to smooth (must exist in dataset.forcings).
    taus_days
        EWMA decay timescales in days.
    name_fmt
        Name template for new features. Receives `{key}` and `{tau}` (int days).
    in_place
        If True (default), mutate `dataset.forcings` and return `dataset`.
        If False, returns a new dataset with an updated forcings dict.

    Returns
    -------
    TimeSeriesDataset
        The updated dataset.
    """
    forc = dict(dataset.forcings)  # ensure mutable copy

    for key in keys:
        if key not in forc:
            raise KeyError(f"Forcing '{key}' not found in dataset.forcings")

        x = np.asarray(forc[key], dtype=float)
        for tau in taus_days:
            tau_i = int(round(float(tau)))
            name = name_fmt.format(key=key, tau=tau_i)
            forc[name] = ewma_irregular(dataset.time, x, tau_days=float(tau))

    if in_place:
        dataset.forcings = forc  # type: ignore[assignment]
        return dataset

    return TimeSeriesDataset(
        time=dataset.time,
        forcings=forc,
        obs_time=dataset.obs_time,
        obs=dataset.obs,
        y0=dataset.y0,
        idx_obs=dataset.idx_obs,
        dt=dataset.dt,
    )


def add_movavg_forcings(
    dataset: TimeSeriesDataset,
    *,
    keys: Sequence[str] = ("E",),
    windows_days: Sequence[float] = (30.0, 180.0),
    name_fmt: str = "{key}_movavg{window}",
    in_place: bool = True,
) -> TimeSeriesDataset:
    """Add trailing moving-average forcings to a dataset.

    The moving average is computed on the dataset's *forcing time grid*
    (``dataset.time``) for each requested forcing key.
    """
    forc = dict(dataset.forcings)

    for key in keys:
        if key not in forc:
            raise KeyError(f"Forcing '{key}' not found in dataset.forcings")

        x = np.asarray(forc[key], dtype=float)
        for window in windows_days:
            window_i = int(round(float(window)))
            name = name_fmt.format(key=key, window=window_i)
            forc[name] = moving_average_irregular(dataset.time, x, window_days=float(window))

    if in_place:
        dataset.forcings = forc  # type: ignore[assignment]
        return dataset

    return TimeSeriesDataset(
        time=dataset.time,
        forcings=forc,
        obs_time=dataset.obs_time,
        obs=dataset.obs,
        y0=dataset.y0,
        idx_obs=dataset.idx_obs,
        dt=dataset.dt,
    )


def interp_obs_to_forcing(
    time: np.ndarray,
    obs_time: np.ndarray,
    obs: np.ndarray,
    *,
    kind: str = "linear",
) -> np.ndarray:
    """Interpolate observations to the forcing time grid.

    Values outside the observation time range are returned as NaN (no extrapolation).
    """
    t = np.asarray(time)
    to = np.asarray(obs_time)
    y = np.asarray(obs, dtype=float)
    if t.size == 0:
        return np.asarray([], dtype=float)
    if to.size == 0:
        return np.full(t.shape[0], np.nan, dtype=float)

    # convert time to numeric days for interpolation
    if np.issubdtype(t.dtype, np.datetime64):
        t_num = (t.astype("datetime64[ns]") - t[0].astype("datetime64[ns]")) / np.timedelta64(1, "D")
    else:
        t_num = t.astype(float)

    if np.issubdtype(to.dtype, np.datetime64):
        to_num = (to.astype("datetime64[ns]") - t[0].astype("datetime64[ns]")) / np.timedelta64(1, "D")
    else:
        to_num = to.astype(float)

    # ensure sorted obs
    order = np.argsort(to_num)
    to_num = to_num[order]
    y = y[order]

    # linear interpolation (no extrapolation)
    yi = np.interp(t_num, to_num, y, left=np.nan, right=np.nan)
    # np.interp doesn't support nan left/right for older numpy; force masks
    yi = np.asarray(yi, dtype=float)
    yi[t_num < to_num[0]] = np.nan
    yi[t_num > to_num[-1]] = np.nan
    return yi


def build_feature_matrix(
    dataset: TimeSeriesDataset,
    forcing_names: Sequence[str],
    *,
    include_doy: bool = False,
    include_time: bool = False,
    at: str = "full",  # "full" or "obs"
    standardize: bool = True,
    ref_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[str], tuple[np.ndarray, np.ndarray]]:
    """Build a design matrix from dataset forcings and optional time features.

    Parameters
    ----------
    dataset
        Input dataset.
    forcing_names
        Forcing keys from dataset.forcings.
    include_doy
        Add sin/cos of day-of-year.
    include_time
        Add a normalized time feature (0..1).
    at
        "full" uses the full forcing grid; "obs" samples at observation indices.
    standardize
        If True, z-score features using (mean,std). If ref_stats provided,
        use those stats.
    ref_stats
        (mean, std) arrays used for standardization.

    Returns
    -------
    X, names, (mean,std)
    """
    if at not in ("full", "obs"):
        raise ValueError("at must be 'full' or 'obs'")

    if at == "full":
        idx = slice(None)
        t = dataset.time
    else:
        if dataset.idx_obs is None:
            raise ValueError("dataset.idx_obs is required for at='obs'")
        idx = np.asarray(dataset.idx_obs, dtype=int)
        t = dataset.time[idx]

    cols = []
    names: list[str] = []

    for k in forcing_names:
        if k not in dataset.forcings:
            raise KeyError(f"Forcing '{k}' not found in dataset.forcings")
        v = np.asarray(dataset.forcings[k], dtype=float)[idx]
        cols.append(v)
        names.append(k)

    if include_doy:
        if not np.issubdtype(np.asarray(t).dtype, np.datetime64):
            raise ValueError("include_doy requires datetime64 time axis")
        tt = pd.to_datetime(t)
        doy = tt.dayofyear.to_numpy(dtype=float)
        ang = 2.0 * np.pi * (doy / 365.25)
        cols.append(np.sin(ang))
        names.append("sin_doy")
        cols.append(np.cos(ang))
        names.append("cos_doy")

    if include_time:
        if np.issubdtype(np.asarray(t).dtype, np.datetime64):
            tt = (t.astype("datetime64[ns]") - t[0].astype("datetime64[ns]")) / np.timedelta64(1, "D")
            tt = tt.astype(float)
        else:
            tt = np.asarray(t, dtype=float)
            tt = tt - tt[0]
        if tt.size > 0 and np.nanmax(tt) > 0:
            tt = tt / np.nanmax(tt)
        cols.append(tt)
        names.append("t_norm")

    X = np.column_stack(cols).astype(float) if cols else np.zeros((len(t), 0), dtype=float)

    if standardize and X.shape[1] > 0:
        if ref_stats is None:
            mu = np.nanmean(X, axis=0)
            sig = np.nanstd(X, axis=0)
            sig = np.where(sig <= 0, 1.0, sig)
        else:
            mu, sig = ref_stats
            sig = np.where(sig <= 0, 1.0, sig)
        X = (X - mu) / sig
        return X, names, (mu, sig)

    mu = np.zeros((X.shape[1],), dtype=float)
    sig = np.ones((X.shape[1],), dtype=float)
    return X, names, (mu, sig)
