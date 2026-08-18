from __future__ import annotations

"""Miller and Dean model and adapter for the bayesian calibration.

The goal is to make adding new experts trivial: put a handler in
`slmcal.models` implementing the `Expert` protocol.
"""

import numpy as np
from numba import njit
from math import exp

from slmcal.models.base import ParameterSpec
from slmcal.data import TimeSeriesDataset


@njit(fastmath=True, cache=True)
def millerDean(hb, depthb, sl, wast, dt, Hberm, Y0, kacr, kero, Yini):
    n = hb.shape[0]

    wl =  sl #0.106 * hb +
    denom = Hberm + depthb
    yeq = np.empty(n)
    # kacr_ = np.empty(n)
    # kero_ = np.empty(n)
    for i in range(n):
        yeq[i] = Y0 - wast[i] * wl[i] / denom[i]
        # kacr_[i] = kacr * hb[i] ** 2
        # kero_[i] = kero * hb[i] ** 2

    Y = np.empty(n)
    Y[0] = Yini
    for i in range(1, n):
        prev = Y[i-1]
        cur_eq = yeq[i]
        # delta = cur_eq - prev
        delta = prev - cur_eq
        # cond = 1 si acreción, 0 si erosión
        k = kacr if prev < cur_eq else kero
        # k = kacr_[i] if prev < cur_eq else kero_[i]
        # Y[i] = prev + k * dt[i-1] * delta
        Y[i] = cur_eq + exp(-k * dt[i-1]) * delta
       
       
        # prev = Y[i-1]
        # cur_eq = yeq[i]
        # delta = cur_eq - prev
        # # cond = 1 si acreción, 0 si erosión
        # k = kacr if prev < cur_eq else kero
        # # Y[i] = prev + k * dt[i-1] * delta
        # Y[i] = prev + k * dt[i-1] * delta

    return Y, yeq


class MD04Model:
    """Adapter for the Miller and Dean (2004) equilibrium shoreline model.

    The parameters match legacy usage:
       - raw[0] = log(kacr)      -> kacr = exp(raw[0])
       - raw[1] = log(kero)      -> kero = exp(raw[1])
       - raw[2] = DY0            -> DY0  = raw[2]

    Notes
    -----
    - This wrapper is black-box friendly: it simply calls `milerDean(...)` and returns
      the shoreline position time series on `dataset.time`.
    """

    def __init__(self, bounds_raw: np.ndarray | None = None):

        if bounds_raw is None:
            # kacr in [1e-4, 1e-2], kero in [1e-3, 1], DY0 in [-50, 50]

            bounds_raw = np.array(
                [
                    [np.log(1e-6), np.log(1e-2)],        # log(kacr)
                    [np.log(1e-4), np.log(1e-2)],         # log(kero)
                    [150.0, 250.0],                      # DY0
                ],
                dtype=float,
            )

        self._bounds_raw = np.asarray(bounds_raw, dtype=float)

        def pexp(x: np.ndarray) -> np.ndarray:
            return np.exp(np.asarray(x, dtype=float))
        
        self._parameters = [
            ParameterSpec("log_kacr", tuple(self._bounds_raw[0]), transform=pexp),
            ParameterSpec("log_kero", tuple(self._bounds_raw[1]), transform=pexp),
            ParameterSpec("DY0", tuple(self._bounds_raw[2]), transform=lambda x: np.asarray(x, dtype=float)),
        ]

    @property
    def parameters(self) -> list[ParameterSpec]:
        return list(self._parameters)

    def simulate(self, physical_params: np.ndarray, dataset: TimeSeriesDataset, y0: float | None = None) -> np.ndarray:
        
        kacr, kero, DY0 = map(float, physical_params)
        hb = np.asarray(dataset.forcings["hb"], dtype=float)
        depthb = np.asarray(dataset.forcings["depthb"], dtype=float)
        sl = np.asarray(dataset.forcings["sl"], dtype=float)
        wast = np.asarray(dataset.forcings["wast"], dtype=float)
        hberm = float(dataset.hberm) if dataset.hberm is not None else 1.5
        # DY0 = float(dataset.DY0) if dataset.DY0 is not None else 0.0
        dt = np.asarray(dataset.dt, dtype=float)
        if y0 is None:
            y0 = float(dataset.y0) if dataset.y0 is not None else float(dataset.obs[0])
        else:
            y0 = float(y0)

        y, _ = millerDean(hb, depthb, sl, wast, dt, hberm, DY0, kacr, kero, y0)
        return np.asarray(y, dtype=float)