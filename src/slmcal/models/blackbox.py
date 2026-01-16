from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from slmcal.data import TimeSeriesDataset
from slmcal.models.base import ParameterSpec


@dataclass
class BlackBoxModel:
    """Generic black-box model adapter (njit-friendly).

    Use this to wrap any forward model that looks like Yates09:

        y = f(E: ndarray, dt: ndarray, params_phys: ndarray, y0: float) -> ndarray

    The wrapped function can be a `numba.njit` function.

    Notes
    -----
    - The DE-MCMC sampler only ever calls `simulate(...)` as a black-box.
    - You control the raw->physical parameterization through `ParameterSpec.transform`.
    """

    name: str
    parameters: Sequence[ParameterSpec]
    forward: Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]

    def simulate(self, physical_params: np.ndarray, dataset: TimeSeriesDataset) -> np.ndarray:
        E = np.asarray(dataset.forcings["E"], dtype=float)
        dt = np.asarray(dataset.dt, dtype=float)
        y0 = float(dataset.y0) if dataset.y0 is not None else float(dataset.obs[0])
        return np.asarray(self.forward(E, dt, np.asarray(physical_params, dtype=float), y0), dtype=float)
