"""Plotting utilities for IH-MOOSE examples.

These plots are designed to match the "full timeseries with posterior bands"
style you shared:
- per-transect shoreline distances with p1/p10/p50/p90/p99 bands;
- rotation signal (rot observations) vs modeled alpha(t) bands.

We keep this module lightweight and avoid dependencies beyond matplotlib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def _to_mpl_time(t: np.ndarray) -> np.ndarray:
    # matplotlib handles numpy datetime64 directly
    return np.asarray(t)


def plot_transect_timeseries_bands(
    *,
    out_path: str | Path,
    time: np.ndarray,
    p01: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    p99: np.ndarray,
    obs_time: np.ndarray,
    obs: np.ndarray,
    obs_mask: np.ndarray,
    split_date: Optional[str] = None,
    title: Optional[str] = None,
    ylabel: str = "Shoreline position (m)",
):
    """Single-transect plot."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t = _to_mpl_time(time)

    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.fill_between(t, p01, p99, alpha=0.15, label="P01–P99")
    ax.fill_between(t, p10, p90, alpha=0.25, label="P10–P90")
    ax.plot(t, p50, linewidth=2.0, label="P50")

    # observations
    obs_time = _to_mpl_time(obs_time)
    if obs_mask is None:
        good = np.isfinite(obs)
    else:
        good = ~np.asarray(obs_mask, dtype=bool)

    ax.scatter(obs_time[good], np.asarray(obs)[good], s=18, marker="o", alpha=0.85, label="Obs")

    if split_date is not None:
        split = np.datetime64(split_date)
        ax.axvline(split, linestyle="--", linewidth=1.5)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time")
    if title:
        ax.set_title(title)

    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_rotation_bands(
    *,
    out_path: str | Path,
    time: np.ndarray,
    p01: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    p99: np.ndarray,
    rot_obs_time: np.ndarray,
    rot_obs: np.ndarray,
    rot_obs_mask: np.ndarray,
    split_date: Optional[str] = None,
    title: str = "Rotation: observations vs modeled",
):
    """Rotation plot (rot obs + modeled alpha bands)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t = _to_mpl_time(time)

    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.fill_between(t, p01, p99, alpha=0.15, label="P01–P99")
    ax.fill_between(t, p10, p90, alpha=0.25, label="P10–P90")
    ax.plot(t, p50, linewidth=2.0, label="P50")

    rot_obs_time = _to_mpl_time(rot_obs_time)
    good = ~np.asarray(rot_obs_mask, dtype=bool) & np.isfinite(rot_obs)
    ax.scatter(rot_obs_time[good], np.asarray(rot_obs)[good], s=18, marker="o", alpha=0.85, label="rot obs")

    if split_date is not None:
        split = np.datetime64(split_date)
        ax.axvline(split, linestyle="--", linewidth=1.5)

    ax.set_ylabel("Rotation (deg)")
    ax.set_xlabel("Time")
    ax.set_title(title)

    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


__all__ = ["plot_transect_timeseries_bands", "plot_rotation_bands"]
