from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr


def open_precalibration_results(path: str | Path) -> xr.Dataset:
    """Open a pre-calibration results NetCDF file."""

    return xr.open_dataset(Path(path))


def open_calibration_results(path: str | Path) -> xr.Dataset:
    """Open a Bayesian calibration results NetCDF file."""

    return xr.open_dataset(Path(path))


def extract_prior_and_posterior_raw(
    precal: xr.Dataset | None,
    bayes: xr.Dataset,
    *,
    use_valid_prior: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (prior_raw, posterior_raw) arrays from results datasets."""

    if "posterior" not in bayes:
        raise KeyError("Calibration file does not contain variable 'posterior'.")

    posterior = np.asarray(bayes["posterior"].values, dtype=float)
    posterior_raw = posterior.reshape(-1, posterior.shape[-1])

    prior_raw = None
    if precal is not None:
        if use_valid_prior and "all_valid_individuals" in precal:
            prior_raw = np.asarray(precal["all_valid_individuals"].values, dtype=float)
        elif "all_individuals" in precal:
            prior_raw = np.asarray(precal["all_individuals"].values, dtype=float)

    if prior_raw is None:
        if use_valid_prior and "all_valid_individuals" in bayes:
            prior_raw = np.asarray(bayes["all_valid_individuals"].values, dtype=float)
        elif "all_individuals" in bayes:
            prior_raw = np.asarray(bayes["all_individuals"].values, dtype=float)
        else:
            raise KeyError(
                "Could not find prior individuals ('all_valid_individuals' or 'all_individuals')."
            )

    prior_raw = prior_raw[np.all(np.isfinite(prior_raw), axis=1)]

    return prior_raw, posterior_raw
