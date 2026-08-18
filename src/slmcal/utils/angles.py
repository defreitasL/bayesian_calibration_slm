from __future__ import annotations

"""Angle helpers (degrees).

The shoreline workflow mixes *nautical directions* (0..360) and *rotations*
that live on a circle. Using linear interpolation / arithmetic means on degrees
often introduces spurious jumps (e.g. 359 -> 1).

These helpers implement safe wrapping and circular means.
"""

import numpy as np


def wrap360_deg(x: np.ndarray | float) -> np.ndarray:
    """Wrap angles to [0, 360)."""
    a = np.asarray(x, dtype=float)
    out = np.mod(a, 360.0)
    out = np.where(out < 0.0, out + 360.0, out)
    return out


def wrap180_deg(x: np.ndarray | float) -> np.ndarray:
    """Wrap angles to (-180, 180]."""
    a = np.asarray(x, dtype=float)
    out = np.mod(a + 180.0, 360.0) - 180.0
    # Map -180 -> +180 to obtain (-180, 180]
    out = np.where(out <= -180.0, out + 360.0, out)
    return out


def circmean_deg(x: np.ndarray | float, *, nan_policy: str = "omit") -> float:
    """Circular mean of angles in degrees."""
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return float("nan")

    if nan_policy not in ("omit", "propagate"):
        raise ValueError("nan_policy must be 'omit' or 'propagate'")

    if nan_policy == "omit":
        a = a[np.isfinite(a)]
        if a.size == 0:
            return float("nan")
    else:
        if np.any(~np.isfinite(a)):
            return float("nan")

    ang = np.deg2rad(a)
    s = np.mean(np.sin(ang))
    c = np.mean(np.cos(ang))
    if not np.isfinite(s) or not np.isfinite(c) or (abs(s) < 1e-30 and abs(c) < 1e-30):
        return float("nan")

    mu = np.arctan2(s, c)
    return float(wrap360_deg(np.rad2deg(mu)))


def center_deg(x: np.ndarray | float, ref_deg: float) -> np.ndarray:
    """Center angles by subtracting a reference and wrapping to (-180, 180]."""
    return wrap180_deg(np.asarray(x, dtype=float) - float(ref_deg))


def uncenter_deg(x_centered: np.ndarray | float, ref_deg: float, *, wrap: str = "360") -> np.ndarray:
    """Undo centering (add reference) with optional wrapping."""
    out = np.asarray(x_centered, dtype=float) + float(ref_deg)
    w = str(wrap).lower().strip()
    if w in ("360", "wrap360"):
        return wrap360_deg(out)
    if w in ("180", "wrap180"):
        return wrap180_deg(out)
    if w in ("none", "raw", ""):
        return out
    raise ValueError("wrap must be one of: '360', '180', 'none'")


__all__ = [
    "wrap360_deg",
    "wrap180_deg",
    "circmean_deg",
    "center_deg",
    "uncenter_deg",
]
