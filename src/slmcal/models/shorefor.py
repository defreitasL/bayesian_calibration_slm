from __future__ import annotations

"""Fast JIT ShoreFor/YoreFor shoreline model adapters.

The classes in this module follow the same small adapter interface used by
``Yates09Model`` and ``MD04Model``: raw calibration parameters are declared via
``ParameterSpec`` and converted to physical parameters before a fast Numba
forward model is called from ``simulate``.

Implemented models
------------------
ShoreForModel
    Davidson/Splinter ShoreFor formulation based on the disequilibrium of the
    dimensionless fall velocity, using either:

    - ``mode='auto_r'``: parameters ``ks_plus, phi_days, b`` and an automatic
      erosion/accretion ratio ``r`` computed from the forcing balance.
    - ``mode='independent'``: parameters ``ks_plus, ks_minus, phi_days, b``.

YoreForModel
    Lightweight convolution/Yates-equivalent expert inspired by Vitousek et al.
    (2025). It removes explicit shoreline-position dependence and represents
    shoreline response as an exponential-memory convolution of a wave-forcing
    anomaly. It is useful as a fast, structurally distinct expert in MoE tests.

Conventions
-----------
- ``dataset.dt`` is expected in hours, as in ``TimeSeriesDataset``; internally it
  is converted to days.
- ShoreFor ``b`` is in m/day.
- ``P`` and ``Omega`` are taken from ``dataset.forcings`` if present. Otherwise
  they are computed from configurable forcing names.
- The antecedent ``Omega_eq`` is computed as an O(n) recursive exponential-memory
  approximation to the published 10^(-age/phi) weighting, which is much faster
  for repeated Bayesian/NSGA evaluations.
"""

import numpy as np
from numba import njit

from slmcal.models.base import ParameterSpec
from slmcal.data import TimeSeriesDataset

_EPS = 1.0e-12
_LN10 = 2.302585092994046


@njit(fastmath=True, cache=True)
def _safe_mean_nb(x: np.ndarray) -> float:
    s = 0.0
    n = 0
    for i in range(x.size):
        v = x[i]
        if np.isfinite(v):
            s += v
            n += 1
    if n == 0:
        return 0.0
    return s / n


@njit(fastmath=True, cache=True)
def _safe_std_nb(x: np.ndarray) -> float:
    mu = _safe_mean_nb(x)
    s2 = 0.0
    n = 0
    for i in range(x.size):
        v = x[i]
        if np.isfinite(v):
            d = v - mu
            s2 += d * d
            n += 1
    if n <= 1:
        return 1.0
    out = np.sqrt(s2 / (n - 1))
    if not np.isfinite(out) or out < _EPS:
        return 1.0
    return out


@njit(fastmath=True, cache=True)
def _linear_detrend_nb(y: np.ndarray) -> np.ndarray:
    """Return ``y`` minus its least-squares linear trend against integer index."""
    n = y.size
    out = np.empty(n, dtype=np.float64)
    if n <= 1:
        for i in range(n):
            out[i] = y[i]
        return out

    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0
    m = 0.0
    for i in range(n):
        yi = y[i]
        if np.isfinite(yi):
            xi = float(i)
            sx += xi
            sy += yi
            sxx += xi * xi
            sxy += xi * yi
            m += 1.0

    if m <= 1.0:
        for i in range(n):
            out[i] = y[i]
        return out

    den = m * sxx - sx * sx
    if np.abs(den) < _EPS:
        slope = 0.0
        intercept = sy / m
    else:
        slope = (m * sxy - sx * sy) / den
        intercept = (sy - slope * sx) / m

    for i in range(n):
        out[i] = y[i] - (intercept + slope * float(i))
    return out


@njit(fastmath=True, cache=True)
def _omega_eq_recursive_nb(omega: np.ndarray, dt_days: np.ndarray, phi_days: float) -> np.ndarray:
    """Fast antecedent equilibrium Omega using recursive 10^(-age/phi) memory.

    The current sample is not used for its own equilibrium state. For timestep i,
    ``Omega_eq[i]`` uses information up to ``omega[i-1]``. This avoids look-ahead
    in calibration and matches the antecedent-memory interpretation of ShoreFor.
    """
    n = omega.size
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    if not np.isfinite(phi_days) or phi_days <= _EPS:
        phi_days = 1.0

    state = omega[0] if np.isfinite(omega[0]) else 0.0
    out[0] = state
    for i in range(1, n):
        dtd = dt_days[i - 1]
        if not np.isfinite(dtd) or dtd <= 0.0:
            dtd = 1.0
        # Recursive equivalent of weights proportional to 10^(-age/phi).
        a = np.exp(-_LN10 * dtd / phi_days)
        prev = omega[i - 1]
        if not np.isfinite(prev):
            prev = state
        state = a * state + (1.0 - a) * prev
        out[i] = state
    return out


@njit(fastmath=True, cache=True)
def _shorefor_auto_r_nb(P: np.ndarray, domega_norm: np.ndarray, detrend_for_r: bool) -> float:
    n = P.size
    F = np.empty(n, dtype=np.float64)
    for i in range(n):
        p = P[i]
        if not np.isfinite(p) or p < 0.0:
            p = 0.0
        F[i] = np.sqrt(p) * domega_norm[i]

    if detrend_for_r:
        F_use = _linear_detrend_nb(F)
    else:
        F_use = F

    sp = 0.0
    sm = 0.0
    for i in range(n):
        v = F_use[i]
        if v > 0.0:
            sp += v
        elif v < 0.0:
            sm += -v
    if sm <= _EPS:
        return 1.0
    r = np.abs(sp / sm)
    if not np.isfinite(r) or r <= 0.0:
        return 1.0
    return r


@njit(fastmath=True, cache=True)
def shorefor_auto_r_nb(
    params: np.ndarray,
    P: np.ndarray,
    Omega: np.ndarray,
    dt_days: np.ndarray,
    y0: float,
    detrend_for_r: bool = True,
) -> np.ndarray:
    """Classic ShoreFor with parameters [ks_plus, phi_days, b_m_per_day]."""
    ks_plus = params[0]
    phi_days = params[1]
    b = params[2]

    n = P.size
    y = np.empty(n, dtype=np.float64)
    if n == 0:
        return y

    omega_eq = _omega_eq_recursive_nb(Omega, dt_days, phi_days)
    domega = np.empty(n, dtype=np.float64)
    for i in range(n):
        domega[i] = omega_eq[i] - Omega[i]
    sig = _safe_std_nb(domega)

    domega_norm = np.empty(n, dtype=np.float64)
    for i in range(n):
        domega_norm[i] = domega[i] / sig

    r = _shorefor_auto_r_nb(P, domega_norm, detrend_for_r)
    ks_minus = r * ks_plus

    y[0] = y0
    for i in range(1, n):
        p = P[i - 1]
        if not np.isfinite(p) or p < 0.0:
            p = 0.0
        f = np.sqrt(p) * domega_norm[i - 1]
        if domega_norm[i - 1] >= 0.0:
            rate = ks_plus * f + b
        else:
            rate = ks_minus * f + b
        dtd = dt_days[i - 1]
        if not np.isfinite(dtd) or dtd <= 0.0:
            dtd = 1.0
        y[i] = y[i - 1] + dtd * rate
    return y


@njit(fastmath=True, cache=True)
def shorefor_independent_nb(
    params: np.ndarray,
    P: np.ndarray,
    Omega: np.ndarray,
    dt_days: np.ndarray,
    y0: float,
) -> np.ndarray:
    """Classic ShoreFor with [ks_plus, ks_minus, phi_days, b_m_per_day]."""
    ks_plus = params[0]
    ks_minus = params[1]
    phi_days = params[2]
    b = params[3]

    n = P.size
    y = np.empty(n, dtype=np.float64)
    if n == 0:
        return y

    omega_eq = _omega_eq_recursive_nb(Omega, dt_days, phi_days)
    domega = np.empty(n, dtype=np.float64)
    for i in range(n):
        domega[i] = omega_eq[i] - Omega[i]
    sig = _safe_std_nb(domega)

    y[0] = y0
    for i in range(1, n):
        domn = domega[i - 1] / sig
        p = P[i - 1]
        if not np.isfinite(p) or p < 0.0:
            p = 0.0
        f = np.sqrt(p) * domn
        if domn >= 0.0:
            rate = ks_plus * f + b
        else:
            rate = ks_minus * f + b
        dtd = dt_days[i - 1]
        if not np.isfinite(dtd) or dtd <= 0.0:
            dtd = 1.0
        y[i] = y[i - 1] + dtd * rate
    return y


@njit(fastmath=True, cache=True)
def yorefor_nb(params: np.ndarray, F: np.ndarray, dt_days: np.ndarray, y0: float) -> np.ndarray:
    """Convolution/Yates-like expert with [delta_y, tau_days, b_m_per_day].

    ``F`` is a wave-forcing proxy, usually wave energy ``E`` or power ``P``.
    The model smooths the zero-mean normalized forcing anomaly with an
    exponential memory kernel.
    """
    delta_y = params[0]
    tau_days = params[1]
    b = params[2]

    n = F.size
    y = np.empty(n, dtype=np.float64)
    if n == 0:
        return y
    if not np.isfinite(tau_days) or tau_days <= _EPS:
        tau_days = 1.0

    fbar = _safe_mean_nb(F)
    if not np.isfinite(fbar) or np.abs(fbar) < _EPS:
        fbar = 1.0

    y[0] = y0
    for i in range(1, n):
        dtd = dt_days[i - 1]
        if not np.isfinite(dtd) or dtd <= 0.0:
            dtd = 1.0
        lam = np.exp(-dtd / tau_days)
        yeq = -delta_y * (F[i - 1] - fbar) / fbar
        y[i] = lam * y[i - 1] + (1.0 - lam) * yeq + b * dtd
    return y


def _as_float64_1d(x) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())


def _positive_exp(x: np.ndarray) -> np.ndarray:
    return np.exp(np.asarray(x, dtype=float))


def _identity(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)


class ShoreForModel:
    """Adapter for the ShoreFor equilibrium shoreline model.

    Parameters
    ----------
    mode:
        ``'auto_r'`` calibrates ``ks_plus``, ``phi_days`` and ``b`` and computes
        ``ks_minus = r * ks_plus`` from the forcing balance. ``'independent'``
        calibrates ``ks_plus`` and ``ks_minus`` independently.
    bounds_raw:
        Optional raw-space bounds. By default rate/memory parameters are sampled
        in log-space and the residual trend ``b`` is sampled directly in m/day.
    p_name, omega_name:
        Names of precomputed forcing arrays. Defaults match ``preprocess_examples``.
    hs_name, tp_name, ws:
        Fallback calculation of ``Omega = Hs / (Tp * ws)`` and ``P = Hs**2 * Tp``
        when precomputed ``P``/``Omega`` are not available.
    detrend_for_r:
        Whether to remove a linear trend from the forcing before computing the
        automatic erosion/accretion ratio ``r``.
    """

    def __init__(
        self,
        bounds_raw: np.ndarray | None = None,
        *,
        mode: str = "auto_r",
        p_name: str = "P",
        omega_name: str = "Omega",
        hs_name: str = "Hs",
        tp_name: str = "Tp",
        ws: float = 0.03,
        detrend_for_r: bool = True,
    ):
        mode_l = str(mode).lower().strip()
        if mode_l not in {"auto_r", "independent"}:
            raise ValueError("mode must be 'auto_r' or 'independent'")
        self.mode = mode_l
        self.p_name = str(p_name)
        self.omega_name = str(omega_name)
        self.hs_name = str(hs_name)
        self.tp_name = str(tp_name)
        self.ws = float(ws)
        self.detrend_for_r = bool(detrend_for_r)

        if bounds_raw is None:
            if self.mode == "auto_r":
                bounds_raw = np.array(
                    [
                        [np.log(1e-4), np.log(1.0)],      # log(ks_plus)
                        [np.log(5.0), np.log(1000.0)],    # log(phi_days)
                        [-0.10, 0.10],                    # b [m/day]
                    ],
                    dtype=float,
                )
            else:
                bounds_raw = np.array(
                    [
                        [np.log(1e-4), np.log(1.0)],      # log(ks_plus)
                        [np.log(1e-4), np.log(2.0)],      # log(ks_minus)
                        [np.log(5.0), np.log(1000.0)],    # log(phi_days)
                        [-0.10, 0.10],                    # b [m/day]
                    ],
                    dtype=float,
                )

        self._bounds_raw = np.asarray(bounds_raw, dtype=float)
        if self.mode == "auto_r":
            self._parameters = [
                ParameterSpec("log_ks_plus", tuple(self._bounds_raw[0]), transform=_positive_exp),
                ParameterSpec("log_phi_days", tuple(self._bounds_raw[1]), transform=_positive_exp),
                ParameterSpec("b_m_per_day", tuple(self._bounds_raw[2]), transform=_identity),
            ]
        else:
            self._parameters = [
                ParameterSpec("log_ks_plus", tuple(self._bounds_raw[0]), transform=_positive_exp),
                ParameterSpec("log_ks_minus", tuple(self._bounds_raw[1]), transform=_positive_exp),
                ParameterSpec("log_phi_days", tuple(self._bounds_raw[2]), transform=_positive_exp),
                ParameterSpec("b_m_per_day", tuple(self._bounds_raw[3]), transform=_identity),
            ]

    @property
    def parameters(self) -> list[ParameterSpec]:
        return list(self._parameters)

    def _forcing_arrays(self, dataset: TimeSeriesDataset) -> tuple[np.ndarray, np.ndarray]:
        forc = dataset.forcings
        if self.p_name in forc:
            P = _as_float64_1d(forc[self.p_name])
        else:
            hs = _as_float64_1d(forc[self.hs_name])
            tp = _as_float64_1d(forc[self.tp_name])
            P = hs * hs * tp

        if self.omega_name in forc:
            Omega = _as_float64_1d(forc[self.omega_name])
        else:
            hs = _as_float64_1d(forc[self.hs_name])
            tp = _as_float64_1d(forc[self.tp_name])
            ws = self.ws if np.isfinite(self.ws) and self.ws > 0.0 else 0.03
            Omega = hs / (tp * ws + _EPS)

        if P.shape[0] != dataset.time.shape[0] or Omega.shape[0] != dataset.time.shape[0]:
            raise ValueError("ShoreFor forcing arrays must be aligned with dataset.time")
        return P, Omega

    def simulate(self, physical_params: np.ndarray, dataset: TimeSeriesDataset, y0: float | None = None) -> np.ndarray:
        P, Omega = self._forcing_arrays(dataset)
        dt_days = _as_float64_1d(np.asarray(dataset.dt, dtype=float) / 24.0)
        if y0 is None:
            if dataset.y0 is not None:
                y0 = float(dataset.y0)
            elif dataset.obs is not None and dataset.obs.size > 0:
                y0 = float(dataset.obs[0])
            else:
                y0 = 0.0
        params = _as_float64_1d(physical_params)
        if self.mode == "auto_r":
            y = shorefor_auto_r_nb(params, P, Omega, dt_days, float(y0), self.detrend_for_r)
        else:
            y = shorefor_independent_nb(params, P, Omega, dt_days, float(y0))
        return np.asarray(y, dtype=float)


class YoreForModel:
    """Fast convolution/Yates-equivalent expert for MoE and Bayesian calibration.

    Raw parameters
    --------------
    - ``raw[0] = log(delta_y)`` -> response amplitude in m.
    - ``raw[1] = log(tau_days)`` -> exponential memory time scale in days.
    - ``raw[2] = b`` -> residual trend in m/day.

    The forcing defaults to ``E`` if available, otherwise ``P`` or ``Hs**2``.
    """

    def __init__(
        self,
        bounds_raw: np.ndarray | None = None,
        *,
        forcing_name: str = "E",
        fallback_p_name: str = "P",
        hs_name: str = "Hs",
    ):
        self.forcing_name = str(forcing_name)
        self.fallback_p_name = str(fallback_p_name)
        self.hs_name = str(hs_name)
        if bounds_raw is None:
            bounds_raw = np.array(
                [
                    [np.log(1.0), np.log(200.0)],       # log(delta_y) [m]
                    [np.log(5.0), np.log(1000.0)],      # log(tau_days)
                    [-0.10, 0.10],                      # b [m/day]
                ],
                dtype=float,
            )
        self._bounds_raw = np.asarray(bounds_raw, dtype=float)
        self._parameters = [
            ParameterSpec("log_delta_y", tuple(self._bounds_raw[0]), transform=_positive_exp),
            ParameterSpec("log_tau_days", tuple(self._bounds_raw[1]), transform=_positive_exp),
            ParameterSpec("b_m_per_day", tuple(self._bounds_raw[2]), transform=_identity),
        ]

    @property
    def parameters(self) -> list[ParameterSpec]:
        return list(self._parameters)

    def _forcing_array(self, dataset: TimeSeriesDataset) -> np.ndarray:
        forc = dataset.forcings
        if self.forcing_name in forc:
            F = _as_float64_1d(forc[self.forcing_name])
        elif self.fallback_p_name in forc:
            F = _as_float64_1d(forc[self.fallback_p_name])
        else:
            hs = _as_float64_1d(forc[self.hs_name])
            F = hs * hs
        if F.shape[0] != dataset.time.shape[0]:
            raise ValueError("YoreFor forcing array must be aligned with dataset.time")
        return F

    def simulate(self, physical_params: np.ndarray, dataset: TimeSeriesDataset, y0: float | None = None) -> np.ndarray:
        F = self._forcing_array(dataset)
        dt_days = _as_float64_1d(np.asarray(dataset.dt, dtype=float) / 24.0)
        if y0 is None:
            if dataset.y0 is not None:
                y0 = float(dataset.y0)
            elif dataset.obs is not None and dataset.obs.size > 0:
                y0 = float(dataset.obs[0])
            else:
                y0 = 0.0
        params = _as_float64_1d(physical_params)
        return np.asarray(yorefor_nb(params, F, dt_days, float(y0)), dtype=float)
