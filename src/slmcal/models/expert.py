from __future__ import annotations

"""Expert-model interface for Mixture-of-Experts (MoE) workflows.

The goal is to make adding new experts trivial: put a handler in
`slmcal.models` implementing the `Expert` protocol.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from slmcal.data import TimeSeriesDataset


@dataclass(frozen=True)
class ExpertPrediction:
    """Posterior/predictive draws for a single expert.

    Attributes
    ----------
    name
        Expert name.
    time
        1D time grid aligned with `draws` second dimension.
    draws
        Array of shape (n_draws, n_time).
    """

    name: str
    time: np.ndarray
    draws: np.ndarray

    def median(self) -> np.ndarray:
        return np.nanmedian(self.draws, axis=0)


class Expert(Protocol):
    """Minimal expert interface."""

    name: str

    def fit(self, dataset_cal: TimeSeriesDataset) -> "Expert":
        ...

    def predict_draws(
        self,
        dataset_full: TimeSeriesDataset,
        *,
        n_draws: int,
        random_seed: int = 42,
    ) -> ExpertPrediction:
        ...


def summarize_draws(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (p05, p10, p50, p90, p95, min, max) over draws."""

    d = np.asarray(draws, dtype=float)
    per5 = np.nanpercentile(d, 5, axis=0)
    per10 = np.nanpercentile(d, 10, axis=0)
    per50 = np.nanpercentile(d, 50, axis=0)
    per90 = np.nanpercentile(d, 90, axis=0)
    per95 = np.nanpercentile(d, 95, axis=0)
    mini = np.nanmin(d, axis=0)
    maxi = np.nanmax(d, axis=0)
    return per5, per10, per50, per90, per95, mini, maxi
