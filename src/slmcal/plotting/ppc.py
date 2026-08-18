"""PPC utilities.

This module provides a tiny helper to build a *PPC-like* object from an
ensemble of predictive draws (e.g., MoE ensemble), so it can be passed into the
existing plotting functions that expect `ppc.posterior_predictive`.

We intentionally keep this lightweight and dependency-minimal:
- Requires xarray (already used elsewhere in slmcal).
- Does not require ArviZ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import xarray as xr


@dataclass
class PPCLike:
    """Minimal object mimicking ArviZ/PyMC PPC output."""

    posterior_predictive: xr.Dataset


def ppc_like_from_draws(
    draws: np.ndarray,
    *,
    draw_axis: Optional[int] = None,
    var_name: str = "likelihood",
    obs_dim: str = "point",
    add_noise: bool = False,
    sigma: Optional[float] = None,
    noise_kind: str = "normal",  # "normal" | "studentt"
    nu: Optional[float] = None,
    random_seed: int = 0,
) -> PPCLike:
    """Build a PPC-like object from predictive draws.

    Parameters
    ----------
    draws:
        Predictive draws, shape (n_draw, n_obs) or (n_obs, n_draw).
    add_noise:
        If True, adds i.i.d. noise to each draw using `sigma`.
    sigma:
        Noise scale. Required when `add_noise=True`.
    noise_kind:
        "normal" or "studentt".
    nu:
        Degrees of freedom for Student-t noise (required if noise_kind="studentt").
    """
    arr = np.asarray(draws, dtype=float)
    if arr.ndim != 2:
        raise ValueError("draws must be 2D")

    # Accept either (draw, obs) or (obs, draw)
    if draw_axis is not None:
        if int(draw_axis) not in (0, 1):
            raise ValueError("draw_axis must be 0 or 1")
        dd = arr if int(draw_axis) == 0 else arr.T
    else:
        # Default heuristic: n_draw is typically >= n_obs, so the larger axis is assumed to be draws.
        # If shapes are equal, we assume the user passed (draw, obs).
        dd = arr if arr.shape[0] >= arr.shape[1] else arr.T

    if add_noise:
        if sigma is None:
            raise ValueError("sigma must be provided when add_noise=True")
        rng = np.random.default_rng(int(random_seed))
        if str(noise_kind).lower() in {"studentt", "student-t", "t"}:
            if nu is None:
                raise ValueError("nu must be provided for Student-t noise")
            eps = rng.standard_t(df=float(nu), size=dd.shape) * float(sigma)
        else:
            eps = rng.normal(loc=0.0, scale=float(sigma), size=dd.shape)
        dd = dd + eps

    n_draw, n_obs = dd.shape
    ds = xr.Dataset(
        {
            var_name: xr.DataArray(
                dd,
                dims=("draw", obs_dim),
                coords={"draw": np.arange(n_draw), obs_dim: np.arange(n_obs)},
            )
        }
    )
    return PPCLike(posterior_predictive=ds)
