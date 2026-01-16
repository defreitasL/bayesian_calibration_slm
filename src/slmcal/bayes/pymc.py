from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slmcal.data import TimeSeriesDataset
from slmcal.models.base import ShorelineModel, transform_raw_to_physical


@dataclass(frozen=True)
class PriorMVN:
    """Multivariate normal prior in *raw* parameter space."""

    mean: np.ndarray
    cov: np.ndarray


@dataclass(frozen=True)
class PriorKDE:
    """Kernel density estimate prior in *raw* parameter space.

    Notes
    -----
    This is an *empirical* prior built directly from NSGA-II samples.
    It is intended for gradient-free samplers (e.g., DEMetropolisZ),
    which is what this workflow uses by default.
    """

    samples: np.ndarray  # (n_samples, n_params)
    bw_method: str | float | None = "scott"
    jitter: float = 1e-9  # small noise to avoid singular covariance in KDE fit
    bounds_lower: np.ndarray | None = None
    bounds_upper: np.ndarray | None = None


@dataclass(frozen=True)
class PriorCopulaKDE:
    """Gaussian-copula KDE prior in *raw* parameter space.

    Why this exists
    ---------------
    A full multivariate KDE in d dimensions is expensive and often unstable when
    d is large (even when you have many NSGA-II samples). A practical solution is
    to model:

      1) each marginal distribution with a 1D KDE, and
      2) the dependence structure with a Gaussian copula.

    This keeps cost approximately O(d) for marginals + O(d^2) (full) or O(dk^2)
    (factor) for the copula term, where k << d.

    Stored objects
    --------------
    We precompute a regular grid for each parameter and store PDF/CDF values on
    that grid, so that evaluating logp only requires cheap linear interpolation.

    The copula term can be represented either as:
      - method='full'  : Cholesky factor of a regularized correlation matrix
      - method='factor': low-rank factor model + diagonal (Woodbury identity)
    """

    # Marginal grids (shape: (d, m))
    grid: np.ndarray
    pdf: np.ndarray
    cdf: np.ndarray

    # Bounds in raw space (optional)
    bounds_lower: np.ndarray | None = None
    bounds_upper: np.ndarray | None = None

    # Copula representation
    method: str = "factor"  # 'factor' or 'full'

    # full: store cholesky + logdet
    chol_R: np.ndarray | None = None
    logdet_R: float | None = None

    # factor: store L (d,k), Dinv (d,), chol_middle (k,k), logdet
    L: np.ndarray | None = None
    Dinv: np.ndarray | None = None
    chol_middle: np.ndarray | None = None
    logdet_R_factor: float | None = None

    # Numeric stability
    eps_u: float = 1e-6
    eps_pdf: float = 1e-300


Prior = PriorMVN | PriorKDE | PriorCopulaKDE


def fit_mvn_prior_from_nsga2(individuals_raw: np.ndarray, cov_scale: float = 30.0) -> PriorMVN:
    """Fit a multivariate normal prior from NSGA-II samples.

    We use an empirical mean and covariance, and optionally inflate covariance
    """
    x = np.asarray(individuals_raw, dtype=float)
    mu = np.mean(x, axis=0)
    cov = np.cov(x, rowvar=False)
    return PriorMVN(mu, cov_scale * cov)


def fit_kde_prior_from_nsga2(
    individuals_raw: np.ndarray,
    bw_method: str | float | None = "scott",
    jitter: float = 1e-9,
    bounds_lower: np.ndarray | None = None,
    bounds_upper: np.ndarray | None = None,
) -> PriorKDE:
    """Fit a KDE prior from NSGA-II samples.

    Parameters are stored; the KDE object is constructed inside the PyMC model
    to keep this function lightweight and avoid importing SciPy unless needed.
    """
    x = np.asarray(individuals_raw, dtype=float)
    return PriorKDE(
        samples=x,
        bw_method=bw_method,
        jitter=jitter,
        bounds_lower=None if bounds_lower is None else np.asarray(bounds_lower, dtype=float),
        bounds_upper=None if bounds_upper is None else np.asarray(bounds_upper, dtype=float),
    )


def fit_copula_kde_prior_from_nsga2(
    individuals_raw: np.ndarray,
    bw_method: str | float | None = "scott",
    grid_size: int = 512,
    shrinkage: float = 0.05,
    method: str = "factor",
    rank: int | None = None,
    explained_var: float = 0.99,
    bounds_lower: np.ndarray | None = None,
    bounds_upper: np.ndarray | None = None,
    eps_u: float = 1e-6,
    eps_pdf: float = 1e-300,
    random_seed: int = 42,
) -> PriorCopulaKDE:
    """Fit a Gaussian-copula KDE prior from NSGA-II samples.

    This is a high-dimensional-friendly empirical prior.

    Approach
    --------
    1) Fit a 1D KDE for each parameter (marginals) and precompute PDF/CDF on a grid.
    2) Transform samples to uniforms using the marginal CDFs.
    3) Map uniforms to standard normals z = Phi^{-1}(u).
    4) Estimate a regularized correlation matrix R for z, then store either:
       - full Cholesky (method='full'), or
       - low-rank factor model (method='factor') for efficiency when d is large.

    Parameters
    ----------
    grid_size
        Number of grid points per dimension used to tabulate marginal PDF/CDF.
        Memory is O(d * grid_size).
    shrinkage
        Simple shrinkage toward identity for the copula correlation matrix.
        Helps ensure positive definiteness and robustness in large d.
    method
        'full' or 'factor'. Use 'factor' for large d (e.g., >50).
    rank / explained_var
        For method='factor', choose the number of factors k.
        If rank is None, choose the smallest k that explains `explained_var` of
        the variance (based on eigenvalues of the correlation matrix).
    """
    try:
        from scipy.stats import gaussian_kde, norm
        from scipy.integrate import cumulative_trapezoid
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Copula KDE prior requires SciPy. Install with: pip install -e '.[bayes]'"
        ) from e

    x = np.asarray(individuals_raw, dtype=float)
    # Defensive: avoid NaNs/Infs propagating into marginal grids and the copula.
    # (If present, KDE/CDF construction can yield NaN CDFs -> ndtri(NaN) -> NaN logp.)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim != 2:
        raise ValueError("individuals_raw must be 2D (n_samples, n_params)")
    n, d = x.shape
    if n < 20:
        raise ValueError("Too few samples to fit a stable copula prior")

    # ---- build marginal grids + pdf/cdf tables ----
    grid = np.empty((d, grid_size), dtype=float)
    pdf = np.empty((d, grid_size), dtype=float)
    cdf = np.empty((d, grid_size), dtype=float)

    rng = np.random.default_rng(random_seed)
    # (optional) tiny jitter avoids KDE singularities in degenerate dims
    xjitter = x + rng.normal(0.0, 1e-12, size=x.shape)

    for j in range(d):
        xj = xjitter[:, j]
        # Use robust range (quantiles) to avoid crazy tails dominating the grid
        lo = float(np.quantile(xj, 0.001))
        hi = float(np.quantile(xj, 0.999))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = float(np.min(xj))
            hi = float(np.max(xj))
            if lo == hi:
                lo -= 1.0
                hi += 1.0

        pad = 0.05 * (hi - lo)
        g = np.linspace(lo - pad, hi + pad, grid_size)
        kde = gaussian_kde(xj, bw_method=bw_method)
        p = kde.evaluate(g)
        # Robustify against any non-finite KDE outputs.
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        p = np.maximum(p, 0.0)

        # CDF from numerical integration (normalize to [0,1])
        c = cumulative_trapezoid(p, g, initial=0.0)
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(c[-1]) if c.size else 0.0
        if (not np.isfinite(total)) or total <= 0.0:
            # Fallback to a stable monotone CDF.
            c = np.linspace(0.0, 1.0, grid_size)
        else:
            c = c / total

        # Enforce monotonicity and strict bounds (important for interpolation stability)
        c = np.maximum.accumulate(c)

        grid[j, :] = g
        pdf[j, :] = p
        cdf[j, :] = np.clip(c, 0.0, 1.0)

    # Final sanity check: if anything is non-finite here, downstream logp will become NaN.
    if not (np.all(np.isfinite(grid)) and np.all(np.isfinite(pdf)) and np.all(np.isfinite(cdf))):
        raise ValueError(
            "Copula-KDE fit produced non-finite marginal tables. "
            "Check the NSGA-II population for extreme/invalid values, "
            "or reduce tails via stricter filtering/subsampling."
        )

    # If bounds are not provided, default to the marginal grid extents.
    # This is important: without bounds, the copula-kde potential can become
    # almost flat outside the tabulated region (because u and f are clamped),
    # allowing the sampler to drift to extreme raw values that can overflow
    # exponentials in the forward model and yield NaNs.
    if bounds_lower is None:
        bounds_lower = grid[:, 0].copy()
    if bounds_upper is None:
        bounds_upper = grid[:, -1].copy()

    # ---- transform samples -> uniforms -> normals for copula fit ----
    U = np.empty_like(x, dtype=float)
    for j in range(d):
        # interpolate CDF at sample locations
        U[:, j] = np.interp(x[:, j], grid[j, :], cdf[j, :], left=eps_u, right=1.0 - eps_u)
    U = np.nan_to_num(U, nan=0.5, posinf=1.0 - eps_u, neginf=eps_u)
    U = np.clip(U, eps_u, 1.0 - eps_u)
    Z = norm.ppf(U)

    # ---- correlation estimate + shrinkage ----
    R = np.corrcoef(Z, rowvar=False)
    if not np.all(np.isfinite(R)):
        R = np.eye(d)
    lam = float(np.clip(shrinkage, 0.0, 1.0))
    R = (1.0 - lam) * R + lam * np.eye(d)
    # Ensure symmetry
    R = 0.5 * (R + R.T)

    method_l = method.lower().strip()
    if method_l == "full":
        # store Cholesky + logdet once; per-eval cost O(d^2)
        try:
            chol = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            chol = np.linalg.cholesky(R + 1e-6 * np.eye(d))
        logdet = float(2.0 * np.sum(np.log(np.diag(chol))))
        return PriorCopulaKDE(
            grid=grid,
            pdf=pdf,
            cdf=cdf,
            bounds_lower=np.asarray(bounds_lower, dtype=float),
            bounds_upper=np.asarray(bounds_upper, dtype=float),
            method="full",
            chol_R=chol,
            logdet_R=logdet,
            eps_u=eps_u,
            eps_pdf=eps_pdf,
        )

    if method_l != "factor":
        raise ValueError("method must be 'full' or 'factor'")

    # ---- factor copula representation ----
    # Eigen decomposition of correlation
    evals, evecs = np.linalg.eigh(R)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    if rank is None:
        tot = float(np.sum(evals))
        csum = np.cumsum(evals) / (tot if tot > 0 else 1.0)
        k = int(np.searchsorted(csum, explained_var) + 1)
        k = int(np.clip(k, 1, d))
    else:
        k = int(np.clip(int(rank), 1, d))

    # Build factor loadings from top-k eigenpairs
    U_k = evecs[:, :k]
    S_k = np.maximum(evals[:k], 1e-12)
    L = U_k * np.sqrt(S_k)
    # Diagonal remainder to match unit diagonal
    diag_LL = np.sum(L * L, axis=1)
    D = 1.0 - diag_LL
    D = np.maximum(D, 1e-6)
    Dinv = 1.0 / D

    middle = np.eye(k) + (L.T * Dinv) @ L
    try:
        chol_mid = np.linalg.cholesky(middle)
    except np.linalg.LinAlgError:
        chol_mid = np.linalg.cholesky(middle + 1e-6 * np.eye(k))

    logdet = float(np.sum(np.log(D)) + 2.0 * np.sum(np.log(np.diag(chol_mid))))

    return PriorCopulaKDE(
        grid=grid,
        pdf=pdf,
        cdf=cdf,
        bounds_lower=np.asarray(bounds_lower, dtype=float),
        bounds_upper=np.asarray(bounds_upper, dtype=float),
        method="factor",
        L=L,
        Dinv=Dinv,
        chol_middle=chol_mid,
        logdet_R_factor=logdet,
        eps_u=eps_u,
        eps_pdf=eps_pdf,
    )


# Backwards-compatible alias
fit_prior_from_nsga2 = fit_mvn_prior_from_nsga2


def bayesian_calibrate(
    model: ShorelineModel,
    dataset: TimeSeriesDataset,
    prior: Prior,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    random_seed: int = 42,
    likelihood: str = "normal",
    sigma: float | None = None,
    estimate_sigma: bool = False,
    include_bias: bool = True,
    cores: int | None = None,
):
    """Run Bayesian calibration using a black-box forward model wrapper.

    This function requires `pymc` + `pytensor`.

    Returns
    -------
    trace, ppc
    """
    try:
        import pymc as pm
        import pytensor.tensor as pt
        from pytensor.graph.op import Op
        from pytensor.graph.basic import Apply
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "PyMC backend not installed. Install with: pip install -e '.[bayes]'"
        ) from e

    if sigma is None and not estimate_sigma:
        raise ValueError("Provide sigma or set estimate_sigma=True")

    # ---- PyTensor Op (black-box forward model) ----
    class ForwardOp(Op):
        itypes = [pt.dvector]
        otypes = [pt.dvector]

        def make_node(self, raw_par):
            raw_par = pt.as_tensor_variable(raw_par)
            if raw_par.ndim != 1:
                raise ValueError("raw_par must be a vector")
            return Apply(self, [raw_par], [pt.dvector()])

        def perform(self, node, inputs, outputs):
            (raw_par,) = inputs
            physical = transform_raw_to_physical(np.asarray(raw_par, dtype=float), model.parameters)
            y = model.simulate(physical, dataset)
            y_obs = y[dataset.idx_obs]
            outputs[0][0] = np.asarray(y_obs, dtype="float64")

    fwd_op = ForwardOp()

    with pm.Model() as pm_model:
        
        # ---- prior on raw parameters ----
        if isinstance(prior, PriorMVN):
            raw_par = pm.MvNormal(
                "raw_par",
                mu=prior.mean,
                cov=prior.cov,
                shape=prior.mean.size,
            )
        elif isinstance(prior, PriorKDE):
            try:
                from scipy.stats import gaussian_kde
            except Exception as e:  # pragma: no cover
                raise ImportError(
                    "KDE prior requires SciPy. Install with: pip install -e '.[bayes]'"
                ) from e

            x = np.asarray(prior.samples, dtype=float)
            if prior.jitter and prior.jitter > 0:
                x = x + np.random.default_rng(random_seed).normal(0.0, prior.jitter, size=x.shape)

            kde = gaussian_kde(x.T, bw_method=prior.bw_method)

            class KDELogPOp(Op):
                itypes = [pt.dvector]
                otypes = [pt.dscalar]

                def perform(self, node, inputs, outputs):
                    (v,) = inputs
                    v = np.asarray(v, dtype=float)

                    if prior.bounds_lower is not None and np.any(v < prior.bounds_lower):
                        outputs[0][0] = np.array(-np.inf, dtype="float64")
                        return
                    if prior.bounds_upper is not None and np.any(v > prior.bounds_upper):
                        outputs[0][0] = np.array(-np.inf, dtype="float64")
                        return

                    # gaussian_kde may not expose logpdf in older SciPy -> use evaluate()
                    dens = float(kde.evaluate(v[:, None])[0])
                    outputs[0][0] = np.array(np.log(dens + 1e-300), dtype="float64")

            raw_par = pm.Flat("raw_par", shape=prior.samples.shape[1])
            pm.Potential("kde_prior", KDELogPOp()(raw_par))
        elif isinstance(prior, PriorCopulaKDE):
            # High-dimensional-friendly empirical prior: 1D KDE marginals + Gaussian copula
            try:
                from scipy.special import ndtri  # inverse standard normal CDF
                from scipy.linalg import solve_triangular
            except Exception as e:  # pragma: no cover
                raise ImportError(
                    "Copula KDE prior requires SciPy. Install with: pip install -e '.[bayes]'"
                ) from e

            grid = np.asarray(prior.grid, dtype=float)
            pdf = np.asarray(prior.pdf, dtype=float)
            cdf = np.asarray(prior.cdf, dtype=float)

            d = grid.shape[0]
            eps_u = float(prior.eps_u)
            eps_pdf = float(prior.eps_pdf)

            # Precompute constants for normal logpdf
            LOG2PI = float(np.log(2.0 * np.pi))

            if prior.method == "full":
                if prior.chol_R is None or prior.logdet_R is None:
                    raise ValueError("PriorCopulaKDE(full) missing chol_R/logdet_R")
                chol_R = np.asarray(prior.chol_R, dtype=float)
                logdet_R = float(prior.logdet_R)
            else:
                if prior.L is None or prior.Dinv is None or prior.chol_middle is None or prior.logdet_R_factor is None:
                    raise ValueError("PriorCopulaKDE(factor) missing L/Dinv/chol_middle/logdet")
                L = np.asarray(prior.L, dtype=float)
                Dinv = np.asarray(prior.Dinv, dtype=float)
                chol_mid = np.asarray(prior.chol_middle, dtype=float)
                logdet_R = float(prior.logdet_R_factor)

            class CopulaKDELogPOp(Op):
                itypes = [pt.dvector]
                otypes = [pt.dscalar]

                def perform(self, node, inputs, outputs):
                    (v,) = inputs
                    v = np.asarray(v, dtype=float)

                    # bounds check (raw space)
                    if prior.bounds_lower is not None and np.any(v < prior.bounds_lower):
                        outputs[0][0] = np.array(-np.inf, dtype="float64")
                        return
                    if prior.bounds_upper is not None and np.any(v > prior.bounds_upper):
                        outputs[0][0] = np.array(-np.inf, dtype="float64")
                        return

                    # ---- marginals: interpolate u=F(x) and f(x) ----
                    u = np.empty(d, dtype=float)
                    logf = 0.0
                    for j in range(d):
                        gj = grid[j]
                        cj = cdf[j]
                        pj = pdf[j]

                        # np.interp is ok here (m is modest); keep it stable
                        uj = float(np.interp(v[j], gj, cj, left=eps_u, right=1.0 - eps_u))
                        if not np.isfinite(uj):
                            # If the marginal CDF table is somehow ill-formed, fail safe.
                            uj = 0.5
                        uj = float(np.clip(uj, eps_u, 1.0 - eps_u))
                        u[j] = uj

                        fj = float(np.interp(v[j], gj, pj, left=eps_pdf, right=eps_pdf))
                        if (not np.isfinite(fj)) or (fj <= 0.0):
                            fj = eps_pdf
                        logf += float(np.log(fj + eps_pdf))

                    # z = Phi^{-1}(u)
                    # z = Phi^{-1}(u) (guard against NaNs)
                    u = np.nan_to_num(u, nan=0.5, posinf=1.0 - eps_u, neginf=eps_u)
                    u = np.clip(u, eps_u, 1.0 - eps_u)
                    z = ndtri(u)
                    z = np.asarray(z, dtype=float)

                    # standard normal logpdf sum (for the copula correction)
                    z2 = float(np.dot(z, z))
                    logphi = -0.5 * (d * LOG2PI + z2)

                    # ---- copula term ----
                    if prior.method == "full":
                        # Compute z^T R^{-1} z using the Cholesky factor
                        y = solve_triangular(chol_R, z, lower=True, check_finite=False)
                        zTRinvz = float(np.dot(y, y))
                    else:
                        # Factor model: R ≈ L L^T + diag(D), with Dinv provided
                        zD = Dinv * z
                        # temp = L^T D^{-1} z
                        temp = L.T @ zD
                        # solve middle * a = temp using Cholesky
                        y = solve_triangular(chol_mid, temp, lower=True, check_finite=False)
                        a = solve_triangular(chol_mid.T, y, lower=False, check_finite=False)
                        zTRinvz = float(np.dot(z, zD) - np.dot(temp, a))

                    # log copula density: log c(u) = -0.5 logdet(R) -0.5 (z^T(R^{-1}-I)z)
                    logc = -0.5 * logdet_R - 0.5 * (zTRinvz - z2)

                    # full prior: log p(x) = log c(u) + sum log f_j(x_j)
                    outputs[0][0] = np.array(logc + logf, dtype="float64")

            raw_par = pm.Flat("raw_par", shape=d)
            pm.Potential("copula_kde_prior", CopulaKDELogPOp()(raw_par))
        else:  # pragma: no cover
            raise TypeError(f"Unsupported prior type: {type(prior)}")

        yhat = fwd_op(raw_par)

        if include_bias:
            sigma_bias = pm.HalfNormal("sigma_bias", sigma=5.0)
            mu_bias = pm.Normal("mu_bias", mu=0.0, sigma=5.0)
            bias = pm.Normal("bias", mu=mu_bias, sigma=sigma_bias)
        else:
            bias = 0.0

        if estimate_sigma:
            sigma_rv = pm.HalfNormal("sigma", sigma=2.0 * float(sigma or 30))
        else:
            sigma_rv = float(sigma)

        if likelihood.lower() == "normal":
            pm.Normal("likelihood", mu=yhat + bias, sigma=sigma_rv, observed=dataset.obs)
        elif likelihood.lower() in {"studentt", "student-t", "t"}:
            nu = pm.Exponential("nu", 1 / 30.0) + 1
            pm.StudentT("likelihood", mu=yhat + bias, sigma=sigma_rv, nu=nu, observed=dataset.obs)
        else:
            raise ValueError(f"Unknown likelihood='{likelihood}'")

        step = pm.DEMetropolisZ()
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            step=step,
            random_seed=random_seed,
            cores=cores,
            progressbar=True,
        )

        ppc = pm.sample_posterior_predictive(trace, var_names=["likelihood"])

    return trace, ppc
