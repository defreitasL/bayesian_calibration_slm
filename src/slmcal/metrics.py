from __future__ import annotations

import numpy as np


def _nan_filter(obs: np.ndarray, sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    m = np.isfinite(obs) & np.isfinite(sim)
    return obs[m], sim[m]


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    """Kling-Gupta Efficiency (KGE 2009). Higher is better (<=1)."""
    o, s = _nan_filter(obs, sim)
    if o.size < 2:
        return np.nan
    r = np.corrcoef(o, s)[0, 1]
    alpha = np.std(s) / (np.std(o) + 1e-12)
    beta = (np.mean(s) / (np.mean(o) + 1e-12))
    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


def spearman_r(obs: np.ndarray, sim: np.ndarray) -> float:
    """Spearman rank correlation (rho)."""
    o, s = _nan_filter(obs, sim)
    if o.size < 2:
        return np.nan
    # rank
    o_rank = o.argsort().argsort().astype(float)
    s_rank = s.argsort().argsort().astype(float)
    return float(np.corrcoef(o_rank, s_rank)[0, 1])


def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    """Percent bias (PBIAS). 0 is perfect. Sign indicates over/under estimation."""
    o, s = _nan_filter(obs, sim)
    denom = np.sum(o)
    if np.isclose(denom, 0.0):
        return np.nan
    return 100.0 * (np.sum(s - o) / denom)


def spbias(obs: np.ndarray, sim: np.ndarray) -> float:
    """Signed PBIAS magnitude. Minimization metric: abs(PBIAS)."""
    return abs(pbias(obs, sim))


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    o, s = _nan_filter(obs, sim)
    if o.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((s - o) ** 2)))


METRICS = {
    "kge": kge,
    "spearman": spearman_r,
    "pbias": pbias,
    "spbias": spbias,
    "rmse": rmse,
}


def objective_vector(obs: np.ndarray, sim: np.ndarray, metrics: tuple[str, ...]) -> np.ndarray:
    """Return an objective vector for minimization.

    Conventions
    -----------
    - kge: objective = 1 - KGE
    - spearman: objective = 1 - rho
    - spbias: objective = abs(PBIAS)
    - rmse: objective = RMSE
    """
    vals = []
    for m in metrics:
        if m not in METRICS:
            raise KeyError(f"Unknown metric '{m}'. Available: {sorted(METRICS)}")
        v = METRICS[m](obs, sim)
        if m in ("kge", "spearman"):
            vals.append(1.0 - v)
        else:
            vals.append(v)
    return np.asarray(vals, dtype=float)
