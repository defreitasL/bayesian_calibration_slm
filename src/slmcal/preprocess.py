from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class PreprocessResult:
    # Full series (cropped so forcing starts at start_date)
    time: np.ndarray
    E: np.ndarray
    dt: np.ndarray

    time_obs: np.ndarray
    Obs: np.ndarray

    # Calibration window split
    start_date: np.datetime64
    end_date: np.datetime64

    time_splited: np.ndarray
    E_splited: np.ndarray
    dt_splited: np.ndarray

    time_obs_splited: np.ndarray
    Obs_splited: np.ndarray

    # Index helpers (legacy-compatible)
    idx_obs: np.ndarray          # indices into Obs/time_obs within [start_date, end_date)
    idx_obs_splited: np.ndarray  # indices mapping time_obs_splited -> time_splited (nearest)


def _mk_dt(time: np.ndarray) -> np.ndarray:
    # hours between consecutive timesteps
    t = pd.to_datetime(time)
    dt_hours = np.asarray([(t[i + 1] - t[i]).total_seconds() / 3600.0 for i in range(len(t) - 1)], dtype=float)
    return dt_hours


def _mk_nearest_idx(time: np.ndarray, query_time: np.ndarray) -> np.ndarray:
    # vectorized nearest neighbour indices
    t = pd.to_datetime(time).values.astype("datetime64[ns]")
    q = pd.to_datetime(query_time).values.astype("datetime64[ns]")
    # compute via argmin of abs difference
    # Using broadcasting is fine here; lengths are manageable.
    # For huge arrays, replace with searchsorted-based nearest.
    idx = np.argmin(np.abs(t[:, None] - q[None, :]), axis=0)
    return idx.astype(int)


def preprocess_legacy_yates(
    ds: xr.Dataset,
    start_date: str | np.datetime64 | pd.Timestamp,
    end_date: str | np.datetime64 | pd.Timestamp,
    hs_var: str = "hs",
    obs_var: str = "obs",
    time_var: str = "time",
    time_obs_var: str = "time_obs",
) -> PreprocessResult:
    """
    - build E = hs**2
    - crop forcing time series so `time >= start_date`
    - split calibration window [start_date, end_date)
    - compute `idx_obs` and `idx_obs_splited` using nearest-neighbour argmin.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    time = pd.to_datetime(ds[time_var].values)
    hs = np.asarray(ds[hs_var].values, dtype=float)
    E = hs ** 2

    # crop forcing to start_date (legacy)
    idx0 = np.where(time >= start)[0]
    time = time[idx0]
    E = E[idx0]

    Obs = np.asarray(ds[obs_var].values, dtype=float)
    time_obs = pd.to_datetime(ds[time_obs_var].values)

    dt = _mk_dt(time.values)

    # split calibration window
    idx = np.where((time >= start) & (time < end))[0]
    time_spl = time[idx]
    E_spl = E[idx]
    dt_spl = _mk_dt(time_spl.values)

    idx_obs = np.where((time_obs >= start) & (time_obs < end))[0]
    Obs_spl = Obs[idx_obs]
    time_obs_spl = time_obs[idx_obs]

    idx_obs_spl = _mk_nearest_idx(time_spl.values, time_obs_spl.values)

    return PreprocessResult(
        time=time.values.astype("datetime64[ns]"),
        E=E,
        dt=dt,
        time_obs=time_obs.values.astype("datetime64[ns]"),
        Obs=Obs,
        start_date=np.datetime64(start.to_datetime64()),
        end_date=np.datetime64(end.to_datetime64()),
        time_splited=time_spl.values.astype("datetime64[ns]"),
        E_splited=E_spl,
        dt_splited=dt_spl,
        time_obs_splited=time_obs_spl.values.astype("datetime64[ns]"),
        Obs_splited=Obs_spl,
        idx_obs=idx_obs.astype(int),
        idx_obs_splited=idx_obs_spl.astype(int),
    )
