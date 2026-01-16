from __future__ import annotations

import numpy as np

from slmcal.models.base import ParameterSpec
from slmcal.data import TimeSeriesDataset


class Yates09Model:
    """Adapter for the Yates et al. (2009) equilibrium shoreline model.

    Raw parameterisation
    -------------------
    1) ``a_mode='linear'`` (Ensemble-style)
       - raw[0] = a_pos            -> a    = -raw[0]

    2) ``a_mode='log'`` (Bayes-style in some scripts)
       - raw[0] = log(a_pos)       -> a    = -exp(raw[0])

    The remaining parameters match legacy usage:
       - raw[1] = b                -> b    = raw[1]
       - raw[2] = log(|cacr|)      -> cacr = -exp(raw[2])
       - raw[3] = log(|cero|)      -> cero = -exp(raw[3])

    Notes
    -----
    - This wrapper is black-box friendly: it simply calls `yates09(...)` and returns
      the shoreline position time series on `dataset.time`.
    """

    def __init__(self, bounds_raw: np.ndarray | None = None, a_mode: str = "linear"):
        a_mode_l = str(a_mode).lower().strip()
        if a_mode_l not in {"linear", "log"}:
            raise ValueError("a_mode must be 'linear' or 'log'")

        if bounds_raw is None:
            # a_pos in [1e-4, 1e+1], b in [1e-3, 100], cacr/cero magnitudes in [1e-6, 1e-1]
            if a_mode_l == "linear":
                a_bounds = [1e-4, 1e1]
            else:
                a_bounds = [np.log(1e-4), np.log(1e1)]

            bounds_raw = np.array(
                [
                    a_bounds,
                    [1e-3, 100.0],                       # b
                    [np.log(1e-6), np.log(1e-1)],        # log(|cacr|)
                    [np.log(1e-6), np.log(1e-1)],        # log(|cero|)
                ],
                dtype=float,
            )

        self._bounds_raw = np.asarray(bounds_raw, dtype=float)

        def neg(x: np.ndarray) -> np.ndarray:
            return -np.asarray(x, dtype=float)

        def negexp(x: np.ndarray) -> np.ndarray:
            return -np.exp(np.asarray(x, dtype=float))

        if a_mode_l == "linear":
            a_transform = neg
            a_name = "a_pos"
        else:
            a_transform = negexp
            a_name = "log_a_pos"

        self._parameters = [
            ParameterSpec(a_name, tuple(self._bounds_raw[0]), transform=a_transform),
            ParameterSpec("b", tuple(self._bounds_raw[1]), transform=lambda x: np.asarray(x, dtype=float)),
            ParameterSpec("log_abs_cacr", tuple(self._bounds_raw[2]), transform=negexp),
            ParameterSpec("log_abs_cero", tuple(self._bounds_raw[3]), transform=negexp),
        ]

    @property
    def parameters(self) -> list[ParameterSpec]:
        return list(self._parameters)

    def simulate(self, physical_params: np.ndarray, dataset: TimeSeriesDataset) -> np.ndarray:
        try:
            from IHSetYates09 import yates09
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "IHSetYates09 is not installed. Install it (pip/conda) or provide your own model adapter."
            ) from e

        a, b, cacr, cero = map(float, physical_params)
        E = np.asarray(dataset.forcings["E"], dtype=float)
        dt = np.asarray(dataset.dt, dtype=float)
        y0 = float(dataset.y0) if dataset.y0 is not None else float(dataset.obs[0])

        y, _ = yates09(E, dt, a, b, cacr, cero, y0)
        return np.asarray(y, dtype=float)
