from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    """Definition of a model parameter.

    The workflow operates in *raw parameter space* (bounded) and converts to
    *physical parameter space* (what your model expects) before simulation.

    Examples
    --------
    - Sample/optimize `log(-a)` in raw space, then map to `a = -exp(raw)`.
    - Sample/optimize `b` directly with identity transform.
    """

    name: str
    bounds_raw: tuple[float, float]
    transform: Callable[[np.ndarray], np.ndarray] = lambda x: x


class ShorelineModel(Protocol):
    """Minimal interface a shoreline model must implement to be calibrated."""

    @property
    def parameters(self) -> list[ParameterSpec]:
        """List of parameter specifications in the *raw* space."""

    def simulate(
        self,
        physical_params: np.ndarray,
        dataset: "TimeSeriesDataset",
    ) -> np.ndarray:
        """Run the forward model.

        Parameters
        ----------
        physical_params
            1D array in the *physical* parameter space.
        dataset
            Forcings/observations/time grid.

        Returns
        -------
        y
            1D prediction on `dataset.time`.
        """


def bounds_matrix(parameters: list[ParameterSpec]) -> np.ndarray:
    """Return a (n_params, 2) bounds array from parameter specs."""

    return np.asarray([p.bounds_raw for p in parameters], dtype=float)


def transform_raw_to_physical(raw: np.ndarray, parameters: list[ParameterSpec]) -> np.ndarray:
    """Transform a raw parameter vector into physical parameters."""

    if raw.ndim != 1:
        raise ValueError("raw must be 1D")
    if len(raw) != len(parameters):
        raise ValueError("raw length does not match #parameters")

    out = np.empty_like(raw, dtype=float)
    for i, p in enumerate(parameters):
        # each transform must accept a 1D array and return a 1D array
        out[i] = np.asarray(p.transform(np.asarray([raw[i]], dtype=float)))[0]
    return out
