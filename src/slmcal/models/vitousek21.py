from __future__ import annotations

"""Vitousek et al. (2021) reformulation of the Yates et al. (2009) model.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from slmcal.data import TimeSeriesDataset
from slmcal.models.base import ParameterSpec

try:  # pragma: no cover - optional speed-up
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


_EPS = 1.0e-12


def _as_positive(x: np.ndarray) -> np.ndarray:
    return np.exp(np.asarray(x, dtype=float))


def _identity(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _vitousek21_core_py(
    hs: np.ndarray,
    dt_hours: np.ndarray,
    delta_y: float,
    delta_t_acc_days: float,
    delta_t_ero_days: float,
    hhat: float,
    y0: float,
    exact: bool,
) -> np.ndarray:
    n = int(hs.size)
    y = np.empty(n, dtype=np.float64)
    y[0] = float(y0)

    # The ODE is written for a shoreline anomaly relative to the initial
    # shoreline/reference.  The returned series is converted back to absolute
    # shoreline coordinates by adding y0.
    yr = 0.0
    hhat_safe = max(float(hhat), _EPS)
    hhat2 = hhat_safe * hhat_safe

    # robust fallback for malformed dt vectors
    if dt_hours.size > 0:
        good = np.isfinite(dt_hours) & (dt_hours > 0.0)
        dt_fallback = float(np.median(dt_hours[good])) if np.any(good) else 24.0
    else:
        dt_fallback = 24.0

    for i in range(n - 1):
        h = float(hs[i])
        if (not np.isfinite(h)) or h <= 0.0:
            h = hhat_safe

        y_eq = -float(delta_y) * ((h * h - hhat2) / hhat2)
        tendency = y_eq - yr

        if tendency >= 0.0:
            delta_t_days = float(delta_t_acc_days)
        else:
            delta_t_days = float(delta_t_ero_days)
        if (not np.isfinite(delta_t_days)) or delta_t_days <= 0.0:
            delta_t_days = _EPS

        # tau is in hours because TimeSeriesDataset.dt is in hours for datetime
        # time vectors.
        tau_hours = 24.0 * delta_t_days * hhat_safe / max(h, _EPS)
        tau_hours = max(tau_hours, _EPS)

        dti = float(dt_hours[i]) if i < dt_hours.size else dt_fallback
        if (not np.isfinite(dti)) or dti <= 0.0:
            dti = dt_fallback

        if exact:
            # Exact update for dY/dt=(Yeq-Y)/tau over one step with constant
            # forcing.  This is numerically stable even for coarse forcing.
            yr = y_eq + (yr - y_eq) * np.exp(-dti / tau_hours)
        else:
            yr = yr + dti * tendency / tau_hours

        if not np.isfinite(yr):
            yr = 0.0
        y[i + 1] = float(y0) + yr

    return y


if njit is not None:  # pragma: no cover - compiled at runtime
    _vitousek21_core = njit(cache=True)(_vitousek21_core_py)
else:  # pragma: no cover
    _vitousek21_core = _vitousek21_core_py


@dataclass
class Vitousek21YatesModel:
    """Adapter for the Vitousek et al. (2021) Yates09 reformulation.

    Parameters
    ----------
    bounds_raw:
        Optional bounds in raw parameter space.  By default parameters are
        sampled in log-space, so these are log-bounds.
    split_timescales:
        If False, calibrate the parsimonious parameter vector
        ``[DeltaY, DeltaT, Hhat]``.  If True, calibrate
        ``[DeltaY, DeltaT_acc, DeltaT_ero, Hhat]``.
    positive_transform:
        ``"log"`` samples positive parameters in log-space. ``"linear"`` keeps
        the physical positive values directly in raw space.
    integration:
        ``"exact"`` uses the stable exact relaxation update; ``"euler"`` uses a
        forward Euler step.

    Notes
    -----
    ``DeltaT`` is expressed in days for readability; internally it is converted
    to hours because ``TimeSeriesDataset.dt`` is in hours for datetime inputs.
    """

    bounds_raw: np.ndarray | None = None
    split_timescales: bool = False
    positive_transform: Literal["log", "linear"] = "log"
    integration: Literal["exact", "euler"] = "exact"

    def __post_init__(self) -> None:
        mode = str(self.positive_transform).lower().strip()
        if mode not in {"log", "linear"}:
            raise ValueError("positive_transform must be 'log' or 'linear'")
        integ = str(self.integration).lower().strip()
        if integ not in {"exact", "euler"}:
            raise ValueError("integration must be 'exact' or 'euler'")

        if self.bounds_raw is None:
            # Physical defaults: DeltaY [1, 150] m, DeltaT [0.25, 365] days,
            # Hhat [0.1, 6] m.  These are intentionally broad but not absurd.
            physical_bounds = (
                np.array([[1.0, 150.0], [0.25, 365.0], [0.1, 6.0]], dtype=float)
                if not self.split_timescales
                else np.array([[1.0, 150.0], [0.25, 365.0], [0.25, 365.0], [0.1, 6.0]], dtype=float)
            )
            bnd = np.log(physical_bounds) if mode == "log" else physical_bounds
        else:
            bnd = np.asarray(self.bounds_raw, dtype=float)

        expected = 4 if self.split_timescales else 3
        if bnd.shape != (expected, 2):
            raise ValueError(f"bounds_raw must have shape ({expected}, 2)")
        self._bounds_raw = bnd

        transform = _as_positive if mode == "log" else _identity
        if self.split_timescales:
            names = ["log_delta_y", "log_delta_t_acc_days", "log_delta_t_ero_days", "log_hhat"] if mode == "log" else ["delta_y", "delta_t_acc_days", "delta_t_ero_days", "hhat"]
        else:
            names = ["log_delta_y", "log_delta_t_days", "log_hhat"] if mode == "log" else ["delta_y", "delta_t_days", "hhat"]

        self._parameters = [
            ParameterSpec(name, tuple(self._bounds_raw[i]), transform=transform)
            for i, name in enumerate(names)
        ]

    @property
    def parameters(self) -> list[ParameterSpec]:
        return list(self._parameters)

    def _wave_height(self, dataset: TimeSeriesDataset) -> np.ndarray:
        f = dataset.forcings
        for key in ("Hs", "hs", "H", "h"):
            if key in f:
                return np.asarray(f[key], dtype=np.float64)
        if "E" in f:
            e = np.asarray(f["E"], dtype=np.float64)
            return np.sqrt(np.maximum(e, 0.0))
        raise KeyError("Vitousek21YatesModel requires forcing 'E' or one of 'Hs'/'hs'.")

    def simulate(
        self,
        physical_params: np.ndarray,
        dataset: TimeSeriesDataset,
        y0: float | None = None,
    ) -> np.ndarray:
        p = np.asarray(physical_params, dtype=float).reshape(-1)
        expected = 4 if self.split_timescales else 3
        if p.size != expected:
            raise ValueError(f"Expected {expected} physical parameters, got {p.size}.")

        if self.split_timescales:
            delta_y, delta_t_acc_days, delta_t_ero_days, hhat = map(float, p)
        else:
            delta_y, delta_t_days, hhat = map(float, p)
            delta_t_acc_days = delta_t_ero_days = delta_t_days

        if y0 is None:
            y0 = float(dataset.y0) if dataset.y0 is not None else float(np.asarray(dataset.obs, dtype=float)[0])
        else:
            y0 = float(y0)

        hs = self._wave_height(dataset)
        dt = np.asarray(dataset.dt, dtype=np.float64)
        exact = str(self.integration).lower().strip() == "exact"

        return np.asarray(
            _vitousek21_core(
                hs.astype(np.float64),
                dt.astype(np.float64),
                float(delta_y),
                float(delta_t_acc_days),
                float(delta_t_ero_days),
                float(hhat),
                float(y0),
                bool(exact),
            ),
            dtype=float,
        )

    @staticmethod
    def to_legacy_yates_params(
        delta_y: float,
        delta_t_days: float,
        hhat: float,
    ) -> tuple[float, float, float]:
        """Return one equivalent ``(a, b, C)`` triplet for the original Y09 form.

        This is mainly a diagnostic helper.  The reformulated model does not need
        ``a``, ``b`` or ``C`` during simulation.
        """
        b = float(hhat) ** 2
        a = -b / float(delta_y)
        # Convert DeltaT from days to hours for consistency with legacy C units.
        c = 1.0 / (float(delta_t_days) * 24.0 * a * np.sqrt(b))
        return float(a), float(b), float(c)
