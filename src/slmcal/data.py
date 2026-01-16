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
    obs: np.ndarray
    y0: float | None = None
    idx_obs: np.ndarray | None = None
    dt: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time)
        self.obs_time = np.asarray(self.obs_time)
        self.obs = np.asarray(self.obs, dtype=float)

        if self.obs.ndim != 1:
            raise ValueError("obs must be 1D")

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
            self.y0 = float(self.obs[0])

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
        """Compute dt in **hours** from a datetime64 time axis, or unitless diffs for float."""
        if np.issubdtype(time.dtype, np.datetime64):
            dt = (time[1:] - time[:-1]).astype('timedelta64[s]').astype(float) / 3600.0
            return dt
        return np.diff(time).astype(float)

    def subset(self, start: int, end: int) -> "TimeSeriesDataset":
        """Return a sliced dataset (by index) with obs remapped."""
        time = self.time[start:end]
        forcings = {k: np.asarray(v)[start:end] for k, v in self.forcings.items()}

        # keep obs that fall within the new time window
        idx_in = np.where((self.idx_obs >= start) & (self.idx_obs < end))[0]
        obs_time = self.obs_time[idx_in]
        obs = self.obs[idx_in]

        # remap indices to the new 0-based time
        idx_obs = (self.idx_obs[idx_in] - start).astype(int)

        return TimeSeriesDataset(
            time=time,
            forcings=forcings,
            obs_time=obs_time,
            obs=obs,
            y0=self.y0,
            idx_obs=idx_obs,
            dt=self._dt_from_time(time),
        )
