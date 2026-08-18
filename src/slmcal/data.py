from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass
class TimeSeriesDataset:
    """Container for time series forcings and observations.

    Attributes
    ----------
    time
        1D array of model timesteps (monotonic). Can be datetime64 or float.
    forcings
        Mapping of forcing name -> 1D array aligned with `time`.
    obs_time
        1D array of observation timestamps.
    obs
        1D array of shoreline observations aligned with `obs_time`.
    y0
        Initial shoreline position (optional; used by many 1-line models).
    idx_obs
        Integer indices mapping `obs_time` onto `time` (nearest neighbour).
    dt
        Time-step size array for the model (same length as `time` - 1).
    """

    time: np.ndarray
    forcings: Mapping[str, np.ndarray]
    obs_time: np.ndarray
    obs: np.ndarray | None = None
    rot: np.ndarray | None = None
    ref_rot: float | None = None
    y0: float | None = None
    alpha0: float | None = None
    idx_obs: np.ndarray | None = None
    dt: np.ndarray | None = None
    hberm: np.float64 | None = None
    DY0: np.float64 | None = None

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time)
        self.obs_time = np.asarray(self.obs_time)
        self.obs = np.asarray(self.obs, dtype=float) if self.obs is not None else None
        self.rot = np.asarray(self.rot, dtype=float) if self.rot is not None else None

        if self.obs is not None and self.obs.ndim != 1:
            raise ValueError("obs must be 1D")
        if self.rot is not None and self.rot.ndim != 1:
            raise ValueError("rot must be 1D")

        # Ensure forcings are 1D and aligned with time
        for k, v in self.forcings.items():
            vv = np.asarray(v)
            if vv.shape[0] != self.time.shape[0]:
                raise ValueError(f"forcing '{k}' length does not match time")

        if self.idx_obs is None:
            self.idx_obs = self._nearest_indices(self.time, self.obs_time)

        if self.dt is None:
            self.dt = self._dt_from_time(self.time)

        if self.y0 is None:
            # default: use the first observation if it exists
            if self.obs is not None and self.obs.size > 0:
                self.y0 = float(self.obs[0])
        
        if self.alpha0 is None:
            # default: use the first rotation observation if it exists
            if self.rot is not None and self.rot.size > 0:
                self.alpha0 = float(self.rot[0])

    @staticmethod
    def _nearest_indices(time: np.ndarray, obs_time: np.ndarray) -> np.ndarray:
        """Map obs_time to nearest indices in time."""
        # simple, robust nearest neighbour mapping
        idx = np.empty(obs_time.shape[0], dtype=int)
        for i, t in enumerate(obs_time):
            idx[i] = int(np.argmin(np.abs(time - t)))
        return idx

    @staticmethod
    def _dt_from_time(time: np.ndarray) -> np.ndarray:
        """Compute dt in hours from datetime64, robust to pandas-like inputs."""
        time = np.asarray(time)
        if np.issubdtype(time.dtype, np.datetime64):
            t = time.astype("datetime64[ns]")
            dt_ns = np.diff(t).astype("timedelta64[ns]").astype(np.int64)
            dt = (dt_ns / (3600.0 * 1e9)).astype(np.float64)
            if np.any(~np.isfinite(dt)) or np.any(dt <= 0):
                good = np.isfinite(dt) & (dt > 0)
                fallback = float(np.median(dt[good])) if np.any(good) else 24.0
                dt = np.where(good, dt, fallback).astype(np.float64)
            return dt
        return np.diff(time).astype(float)

    def subset(self, start: int, end: int) -> "TimeSeriesDataset":
        """Return a sliced dataset (by index) with obs remapped."""
        time = self.time[start:end]
        forcings = {k: np.asarray(v)[start:end] for k, v in self.forcings.items()}

        # keep obs that fall within the new time window
        idx_in = np.where((self.idx_obs >= start) & (self.idx_obs < end))[0]
        obs_time = self.obs_time[idx_in]
        obs = self.obs[idx_in] if self.obs is not None else None
        rot = self.rot[idx_in] if self.rot is not None else None

        # remap indices to the new 0-based time
        idx_obs = (self.idx_obs[idx_in] - start).astype(int)

        y0 = self.y0
        if obs is not None and obs.size > 0:
            y0 = float(obs[0])

        alpha0 = self.alpha0
        if rot is not None and rot.size > 0:
            alpha0 = float(rot[0])

        return TimeSeriesDataset(
            time=time,
            forcings=forcings,
            obs_time=obs_time,
            obs=obs,
            y0=y0,
            alpha0=alpha0,
            idx_obs=idx_obs,
            rot=rot,
            dt=self._dt_from_time(time),
            hberm=self.hberm,
        )



@dataclass
class MultiTransectDataset:
    """Container for (time, transect) shoreline observations with 1D forcings.

    This dataset is used by IH-MOOSE Bayesian calibration where the likelihood is built on
    observations with shape (time_obs, n_transects). Internally we flatten valid observations
    into a 1D vector (obs_flat) and provide indices (idx_obs) to extract corresponding model
    predictions from y_model.reshape(-1), where y_model has shape (time, n_transects).

    Required arrays
    --------------
    - obs: (time_obs, n_transects)
    - mask_nan_obs: bool mask of same shape (True where obs is NaN / invalid)
    - xi, yi, xf, yf: transect endpoints arrays of length n_transects
    - x_pivotal, y_pivotal: pivot point (scalar)
    - ref_transect: integer index of the reference transect (closest to pivot)

    Forcings are 1D arrays aligned with `time`.
    """

    time: np.ndarray
    forcings: Mapping[str, np.ndarray]
    obs_time: np.ndarray
    obs: np.ndarray  # (time_obs, n_transects)
    mask_nan_obs: np.ndarray | None = None
    y0: float | None = None
    alpha0: float | None = None
    dt: np.ndarray | None = None

    # geometry
    xi: np.ndarray | None = None
    yi: np.ndarray | None = None
    xf: np.ndarray | None = None
    yf: np.ndarray | None = None
    phi: np.ndarray | None = None
    x_pivotal: float | None = None
    y_pivotal: float | None = None
    phi_pivotal: float | None = None
    ref_transect: int | None = None
    ref_i1: int | None = None
    ref_i2: int | None = None
    ref_w1: float | None = None
    ref_w2: float | None = None
    xi_ref: float | None = None
    yi_ref: float | None = None
    xf_ref: float | None = None
    yf_ref: float | None = None

    # flattened view (computed)
    obs_flat: np.ndarray | None = None
    idx_obs: np.ndarray | None = None  # indices into y.reshape(-1)
    idx_t_obs: np.ndarray | None = None  # nearest forcing index for each time_obs row
    rot: np.ndarray | None = None  # rotation signal (if available)
    ref_rot: float | None = None  # reference rotation angle (if available)

    idx_t_flat: np.ndarray | None = None
    idx_obs_tr_flat: np.ndarray | None = None
    unique_t: np.ndarray | None = None
    obs_unique_row: np.ndarray | None = None

    hberm: float | None = None

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time)
        self.obs_time = np.asarray(self.obs_time)
        self.obs = np.asarray(self.obs, dtype=float)
        if self.rot is not None:
            self.rot = np.asarray(self.rot, dtype=float)
            self.alpha0 = float(self.rot[0]) if self.rot.size > 0 else None

        if self.obs.ndim != 2:
            raise ValueError("MultiTransectDataset.obs must be 2D (time_obs, transect)")

        n_tr = self.obs.shape[1]

        # forcings check
        for k, v in self.forcings.items():
            vv = np.asarray(v)
            if vv.shape[0] != self.time.shape[0]:
                raise ValueError(f"forcing '{k}' length does not match time")

        # dt
        if self.dt is None:
            self.dt = TimeSeriesDataset._dt_from_time(self.time)

        # y0 default: first valid observation on ref_transect if possible, else global first valid
        if self.y0 is None:
            if self.ref_transect is not None:
                tr = int(self.ref_transect)
                col = self.obs[:, tr]
                m = np.isfinite(col)
                self.y0 = float(col[m][0]) if np.any(m) else float(np.nanmean(self.obs))
            else:
                m = np.isfinite(self.obs)
                self.y0 = float(self.obs[m][0]) if np.any(m) else 0.0

        # mask
        if self.mask_nan_obs is None:
            self.mask_nan_obs = ~np.isfinite(self.obs)
        else:
            self.mask_nan_obs = np.asarray(self.mask_nan_obs, dtype=bool)
            if self.mask_nan_obs.shape != self.obs.shape:
                raise ValueError("mask_nan_obs must have same shape as obs")

        # map obs_time -> time indices (nearest)
        self.idx_t_obs = TimeSeriesDataset._nearest_indices(self.time, self.obs_time)

        # build flattened observation vector and indices
        valid = ~self.mask_nan_obs
        obs_flat = self.obs[valid].astype(float)

        # indices into y.reshape(-1) plus companion indices
        idx_list = np.empty(obs_flat.size, dtype=int)
        idx_t_flat = np.empty(obs_flat.size, dtype=int)
        idx_tr_flat = np.empty(obs_flat.size, dtype=int)

        k = 0
        for i_obs in range(self.obs.shape[0]):
            ti = int(self.idx_t_obs[i_obs])
            base = ti * n_tr
            for j in range(n_tr):
                if valid[i_obs, j]:
                    idx_list[k] = base + j
                    idx_t_flat[k] = ti
                    idx_tr_flat[k] = j
                    k += 1

        self.obs_flat = obs_flat
        self.idx_obs = idx_list

        # extra helpers for fast likelihood evaluation (simulate_obs)
        self.idx_t_flat = idx_t_flat
        self.idx_obs_tr_flat = idx_tr_flat
        self.unique_t, inv = np.unique(idx_t_flat, return_inverse=True)
        self.obs_unique_row = inv.astype(np.int64)

    def to_timeseries(self, transect: int) -> TimeSeriesDataset:
        """Extract a 1D TimeSeriesDataset for a single transect (dropping NaNs)."""
        tr = int(transect)
        col = self.obs[:, tr]
        good = np.isfinite(col) & (~self.mask_nan_obs[:, tr])
        obs_time = self.obs_time[good]
        obs = col[good]
        idx_obs = self.idx_t_obs[good].astype(int)
        return TimeSeriesDataset(
            time=self.time,
            forcings=self.forcings,
            obs_time=obs_time,
            obs=obs,
            y0=float(obs[0]) if obs.size else (self.y0 if self.y0 is not None else 0.0),
            idx_obs=idx_obs,
            dt=self.dt,
            hberm=getattr(self, "hberm", None),
        )
    
    def to_timeseries_ref(self) -> TimeSeriesDataset:
        """Interpolated reference transect time series (between two closest transects to pivotal)."""
        if self.ref_i1 is None or self.ref_i2 is None or self.ref_w1 is None or self.ref_w2 is None:
            # fallback to closest discrete transect
            if self.ref_transect is None:
                raise ValueError("No reference transect info available")
            return self.to_timeseries(int(self.ref_transect))

        i1 = int(self.ref_i1)
        i2 = int(self.ref_i2)
        w1 = float(self.ref_w1)
        w2 = float(self.ref_w2)

        col1 = self.obs[:, i1]
        col2 = self.obs[:, i2]
        m1 = np.isfinite(col1) & (~self.mask_nan_obs[:, i1])
        m2 = np.isfinite(col2) & (~self.mask_nan_obs[:, i2])

        out = np.full(self.obs.shape[0], np.nan, dtype=float)
        good = m1 & m2
        out[good] = w1 * col1[good] + w2 * col2[good]
        out[m1 & ~m2] = col1[m1 & ~m2]
        out[~m1 & m2] = col2[~m1 & m2]

        keep = np.isfinite(out)
        obs_time = self.obs_time[keep]
        obs = out[keep]
        idx_obs = self.idx_t_obs[keep].astype(int)

        return TimeSeriesDataset(
            time=self.time,
            forcings=self.forcings,
            obs_time=obs_time,
            obs=obs,
            y0=float(obs[0]) if obs.size else (self.y0 if self.y0 is not None else 0.0),
            idx_obs=idx_obs,
            dt=self.dt,
            hberm=getattr(self, "hberm", None),
            alpha0=getattr(self, "alpha0", None),
        )


