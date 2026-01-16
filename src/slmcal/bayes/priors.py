from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import gaussian_kde, norm

# -----------------------------
# Utilities (legacy-compatible)
# -----------------------------


def remove_identical_solutions(solutions: np.ndarray) -> np.ndarray:
    """Return indices of unique solutions (keeps first occurrence)."""
    _, idx = np.unique(np.asarray(solutions), axis=0, return_index=True)
    return idx


def maximum_dissimilarity_algorithm(data: np.ndarray, n_select: int, seed: int | None = None) -> np.ndarray:
    """Simple maximum dissimilarity selection (no sklearn dependency).

    Greedy farthest-point strategy on z-scored space.

    Parameters
    ----------
    data
        (N, D) array.
    n_select
        number of points to select.
    seed
        optional seed for initial pick.

    Returns
    -------
    idx
        indices of selected points in `data`.
    """
    X = np.asarray(data, dtype=float)
    N = X.shape[0]
    if n_select >= N:
        return np.arange(N, dtype=int)

    # Standardize (avoid scale dominance)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd

    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, N))
    selected = [first]

    # distances to selected set (keep min-dist to selected for each point)
    d2 = np.sum((Z - Z[first]) ** 2, axis=1)

    for _ in range(1, n_select):
        # pick farthest from selected set
        j = int(np.argmax(d2))
        selected.append(j)
        # update min distance to selected set
        d2 = np.minimum(d2, np.sum((Z - Z[j]) ** 2, axis=1))

    return np.asarray(selected, dtype=int)


def select_valid_individuals(
    all_individuals: np.ndarray,
    all_objectives: np.ndarray,
    thresholds: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    valid = where(((1-obj0) > thr0) & (obj1 < thr1) & ((1-obj2) > thr2))
    """
    thr0, thr1, thr2 = thresholds
    ind = np.asarray(all_individuals, dtype=float)
    obj = np.asarray(all_objectives, dtype=float)

    ii = remove_identical_solutions(ind)
    ind = ind[ii]
    obj = obj[ii]

    valid_idx = np.where(((1.0 - obj[:, 0]) > thr0) & (obj[:, 1] < thr1) & ((1.0 - obj[:, 2]) > thr2))[0]
    return ind[valid_idx], obj[valid_idx]


def subsample(
    X: np.ndarray,
    max_points: int | None,
    method: Literal["random", "max_dissimilarity"] = "random",
    seed: int | None = 42,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if max_points is None or X.shape[0] <= max_points:
        return X
    if method == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        return X[idx]
    idx = maximum_dissimilarity_algorithm(X, max_points, seed=seed)
    return X[idx]


# -----------------------------
# Priors
# -----------------------------


@dataclass(frozen=True)
class PriorMVN:
    """Multivariate normal prior in raw-parameter space."""

    mean: np.ndarray
    cov: np.ndarray

    def logpdf(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        # manual MVN logpdf (avoid importing scipy.stats.multivariate_normal for speed in loops)
        d = x.size
        diff = x - self.mean
        cov = self.cov
        # Regularize in case of numerical issues
        cov = cov + 1e-12 * np.eye(d)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            # fallback regularization
            cov = cov + 1e-6 * np.eye(d)
            sign, logdet = np.linalg.slogdet(cov)
        inv = np.linalg.inv(cov)
        quad = float(diff.T @ inv @ diff)
        return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)

    def rvs(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.multivariate_normal(self.mean, self.cov, size=int(n))


@dataclass(frozen=True)
class _MarginalKDEGrid:
    x_grid: np.ndarray
    pdf_grid: np.ndarray
    cdf_grid: np.ndarray

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self.x_grid, self.pdf_grid, left=1e-300, right=1e-300)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self.x_grid, self.cdf_grid, left=1e-12, right=1 - 1e-12)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-12, 1 - 1e-12)
        return np.interp(u, self.cdf_grid, self.x_grid, left=self.x_grid[0], right=self.x_grid[-1])


def _fit_1d_kde_grid(x: np.ndarray, grid_size: int = 1024) -> _MarginalKDEGrid:
    x = np.asarray(x, dtype=float)
    kde = gaussian_kde(x)
    lo = np.quantile(x, 0.001)
    hi = np.quantile(x, 0.999)
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    grid = np.linspace(lo - pad, hi + pad, grid_size)
    pdf = kde(grid)
    pdf = np.maximum(pdf, 1e-300)
    # cdf via cumulative trapezoid
    dx = np.diff(grid)
    cdf = np.empty_like(grid)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * dx)
    cdf = cdf / cdf[-1]
    cdf = np.clip(cdf, 1e-12, 1 - 1e-12)
    return _MarginalKDEGrid(x_grid=grid, pdf_grid=pdf, cdf_grid=cdf)


@dataclass(frozen=True)
class PriorCopulaKDE:
    """Gaussian-copula KDE prior (fast, scalable).

    Fits:
    - 1D KDE marginals per parameter
    - Gaussian copula correlation in normal-score space
    """

    marginals: tuple[_MarginalKDEGrid, ...]
    corr: np.ndarray  # (D, D)
    bounds_lower: np.ndarray | None = None
    bounds_upper: np.ndarray | None = None

    def logpdf(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        D = x.size

        # Optional hard bounds (recommended to prevent the prior becoming
        # effectively flat outside the tabulated KDE region).
        if self.bounds_lower is not None and np.any(x < self.bounds_lower):
            return float(-np.inf)
        if self.bounds_upper is not None and np.any(x > self.bounds_upper):
            return float(-np.inf)

        # marginals
        u = np.empty(D, dtype=float)
        log_f = 0.0
        for j, m in enumerate(self.marginals):
            fj = float(m.pdf(np.array([x[j]]))[0])
            fj = max(fj, 1e-300)
            log_f += np.log(fj)
            u[j] = float(m.cdf(np.array([x[j]]))[0])

        z = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))

        # copula density:
        # log c(u) = log N(z;0,R) - sum log phi(z_i)
        R = self.corr
        R = R + 1e-12 * np.eye(D)
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0:
            R = R + 1e-6 * np.eye(D)
            sign, logdet = np.linalg.slogdet(R)
        invR = np.linalg.inv(R)
        quad = float(z.T @ invR @ z)
        log_mvn = -0.5 * (D * np.log(2 * np.pi) + logdet + quad)
        log_phi = np.sum(norm.logpdf(z))
        log_copula = log_mvn - log_phi

        return float(log_copula + log_f)

    def rvs(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        D = len(self.marginals)
        z = rng.multivariate_normal(np.zeros(D), self.corr, size=int(n))
        u = norm.cdf(z)
        out = np.empty_like(z)
        for j, m in enumerate(self.marginals):
            out[:, j] = m.ppf(u[:, j])
        return out


# -----------------------------
# Fit priors from NSGA-II
# -----------------------------


def fit_mvn_prior_from_population(pop: np.ndarray) -> PriorMVN:
    pop = np.asarray(pop, dtype=float)
    mean = pop.mean(axis=0)
    cov = np.cov(pop, rowvar=False)
    # regularize
    cov = cov + 1e-12 * np.eye(cov.shape[0])
    return PriorMVN(mean=mean, cov=cov)


def fit_copula_kde_prior_from_population(
    pop: np.ndarray,
    max_points: int | None = 20000,
    subsample_method: Literal["random", "max_dissimilarity"] = "random",
    seed: int | None = 42,
    grid_size: int = 1024,
) -> PriorCopulaKDE:
    pop = subsample(pop, max_points=max_points, method=subsample_method, seed=seed)
    pop = np.asarray(pop, dtype=float)

    marginals = tuple(_fit_1d_kde_grid(pop[:, j], grid_size=grid_size) for j in range(pop.shape[1]))
    # transform to normal scores to estimate correlation
    U = np.column_stack([m.cdf(pop[:, j]) for j, m in enumerate(marginals)])
    Z = norm.ppf(np.clip(U, 1e-12, 1 - 1e-12))
    corr = np.corrcoef(Z, rowvar=False)
    # regularize / ensure SPD
    corr = 0.98 * corr + 0.02 * np.eye(corr.shape[0])
    bounds_lower = np.array([m.x_grid[0] for m in marginals], dtype=float)
    bounds_upper = np.array([m.x_grid[-1] for m in marginals], dtype=float)
    return PriorCopulaKDE(marginals=marginals, corr=corr, bounds_lower=bounds_lower, bounds_upper=bounds_upper)
