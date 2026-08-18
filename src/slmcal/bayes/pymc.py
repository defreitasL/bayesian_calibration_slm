from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slmcal.data import TimeSeriesDataset
from slmcal.models.base import ShorelineModel, transform_raw_to_physical


@dataclass(frozen=True)
class PriorMVN:
    """Multivariate normal prior in *raw* parameter space.

    Optionally carries an ``init_pool`` (e.g. top NSGA-II solutions) to initialize
    MCMC chains with diverse starting points (important for DEMetropolisZ).
    """

    mean: np.ndarray
    cov: np.ndarray
    init_pool: np.ndarray | None = None



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

    # Optional small pool of NSGA-II samples used to initialize chains
    # (critical for DEMetropolisZ, which needs diverse chain starting points).
    init_pool: np.ndarray | None = None


Prior = PriorMVN | PriorKDE | PriorCopulaKDE


def fit_mvn_prior_from_nsga2(
    individuals_raw: np.ndarray,
    cov_scale: float = 30.0,
    *,
    init_pool: np.ndarray | None = None,
    min_std: float = 1e-3,
) -> PriorMVN:
    """Fit a multivariate normal prior from NSGA-II samples (raw space).

    Parameters
    ----------
    individuals_raw
        Array of NSGA-II individuals in raw parameter space (n, d).
    cov_scale
        Multiplicative inflation applied to the empirical covariance.
    init_pool
        Optional small pool of points (e.g. top-K) used for chain initialisation.
    min_std
        Minimum marginal standard deviation enforced (prevents degenerate MVNs).
    """
    x = np.asarray(individuals_raw, dtype=float)
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError("individuals_raw must have shape (n, d) with n>=1")

    # Remove exact duplicates (common when NSGA-II converges tightly)
    try:
        x_u = np.unique(x, axis=0)
    except Exception:
        x_u = x

    mu = np.mean(x_u, axis=0)
    d = int(mu.size)

    if x_u.shape[0] < 2:
        cov = (float(min_std) ** 2) * np.eye(d, dtype=float)
    else:
        cov = np.cov(x_u, rowvar=False)
        cov = np.asarray(cov, dtype=float)
        if cov.ndim == 0:  # d == 1 special-case
            cov = np.array([[float(cov)]], dtype=float)

        # Ensure PSD + enforce minimum variance
        try:
            w, V = np.linalg.eigh(cov)
            w = np.maximum(w, float(min_std) ** 2)
            cov = (V * w) @ V.T
        except Exception:
            cov = cov + (float(min_std) ** 2) * np.eye(d, dtype=float)

    cov = float(cov_scale) * cov

    pool = None
    if init_pool is not None:
        pool = np.asarray(init_pool, dtype=float)
        if pool.ndim != 2 or pool.shape[1] != d:
            pool = None

    return PriorMVN(mu, cov, init_pool=pool)


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
    init_pool_max: int = 5000,
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

    # Small pool for chain initialisation (avoid DEMetropolisZ collapse when all chains start equal)
    n_pool = int(min(max(init_pool_max, 1), x.shape[0]))
    rng = np.random.default_rng(int(random_seed))
    if n_pool >= x.shape[0]:
        init_pool = x.copy()
    else:
        init_pool = x[rng.choice(x.shape[0], size=n_pool, replace=False)]

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
        p = np.maximum(p, 0.0)
        # CDF from numerical integration (normalize to [0,1])
        c = cumulative_trapezoid(p, g, initial=0.0)
        if c[-1] <= 0:
            c = np.linspace(0.0, 1.0, grid_size)
        else:
            c = c / c[-1]

        grid[j, :] = g
        pdf[j, :] = p
        cdf[j, :] = np.clip(c, 0.0, 1.0)

    # ---- transform samples -> uniforms -> normals for copula fit ----
    U = np.empty_like(x, dtype=float)
    for j in range(d):
        # interpolate CDF at sample locations
        U[:, j] = np.interp(x[:, j], grid[j, :], cdf[j, :])
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
            init_pool=init_pool,
            bounds_lower=None if bounds_lower is None else np.asarray(bounds_lower, dtype=float),
            bounds_upper=None if bounds_upper is None else np.asarray(bounds_upper, dtype=float),
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
        init_pool=init_pool,
        bounds_lower=None if bounds_lower is None else np.asarray(bounds_lower, dtype=float),
        bounds_upper=None if bounds_upper is None else np.asarray(bounds_upper, dtype=float),
        method="factor",
        L=L,
        Dinv=Dinv,
        chol_middle=chol_mid,
        logdet_R_factor=logdet,
        eps_u=eps_u,
        eps_pdf=eps_pdf,
    )


def estimate_studentt_nu(
    residuals: np.ndarray,
    *,
    sigma: float | None = None,
    nu_bounds: tuple[float, float] = (2.0, 50.0),
) -> float:
    """Estimate a reasonable Student-t degrees of freedom (nu) from residuals.

    This is an empirical-Bayes convenience: you fit ``nu`` outside the sampler
    (typically from residuals of a good deterministic run) and then pass it as
    a fixed value to ``bayesian_calibrate`` to improve mixing.

    Parameters
    ----------
    residuals
        1D array of residuals (obs - model) at observation times.
    sigma
        Optional scale used to standardize residuals. If None, a robust MAD
        estimate is used.
    nu_bounds
        Lower/upper bounds for ``nu``. Values <=2 imply infinite variance; the
        default lower bound keeps variance finite and mixing stable.

    Returns
    -------
    nu_hat
        Estimated degrees of freedom within ``nu_bounds``.
    """
    r = np.asarray(residuals, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 20:
        return float(max(nu_bounds[0], min(nu_bounds[1], 10.0)))

    # Robust centering
    r = r - np.median(r)

    # Robust scale if not provided
    if sigma is None:
        mad = np.median(np.abs(r - np.median(r)))
        sigma = float(1.4826 * mad) if mad > 0 else float(np.std(r))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(np.std(r))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    z = r / float(sigma)

    lo, hi = float(nu_bounds[0]), float(nu_bounds[1])
    if hi <= lo:
        raise ValueError('nu_bounds must satisfy hi > lo')

    # Try MLE via SciPy if available
    try:
        from scipy.special import gammaln
        from scipy.optimize import minimize_scalar

        def nll(nu: float) -> float:
            # Student-t logpdf for standardized residuals with scale 1
            nu = float(nu)
            if nu <= 2.0 or not np.isfinite(nu):
                return 1e100
            # logpdf = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*log(nu*pi) - ((nu+1)/2)*log(1 + z^2/nu)
            a = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi)
            b = -((nu + 1.0) / 2.0) * np.log1p((z * z) / nu)
            ll = a + b
            s = -float(np.sum(ll))
            return s if np.isfinite(s) else 1e100

        res = minimize_scalar(nll, bounds=(max(lo, 2.01), hi), method='bounded')
        if res.success and np.isfinite(res.x):
            return float(np.clip(res.x, lo, hi))
    except Exception:
        pass

    # Fallback: method-of-moments from excess kurtosis (valid for nu > 4)
    m2 = float(np.mean(z * z))
    if not np.isfinite(m2) or m2 <= 0:
        return float(max(lo, min(hi, 10.0)))
    zc = z - float(np.mean(z))
    m4 = float(np.mean(zc ** 4))
    if not np.isfinite(m4) or m4 <= 0:
        return float(max(lo, min(hi, 10.0)))
    kurt = m4 / (float(np.mean(zc ** 2)) ** 2 + 1e-12) - 3.0
    if not np.isfinite(kurt) or kurt <= 0:
        # near-normal
        return float(max(lo, min(hi, 30.0)))
    nu = 4.0 + 6.0 / kurt
    return float(np.clip(nu, lo, hi))


# Backwards-compatible alias
fit_prior_from_nsga2 = fit_mvn_prior_from_nsga2


@dataclass(frozen=True)
class PriorProduct:
    """Product of independent priors on parameter blocks.

    This is used to combine a cross-shore prior (from NSGA-II on the reference transect)
    with a rotation prior (from NSGA-II on the rotation signal).
    """

    priors: tuple[object, ...]
    dims: tuple[int, ...]  # number of parameters in each block


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
    estimate_initial_position: bool = False,
    nu_fixed: float | None = None,
    nu_bounds: tuple[float, float] = (2.0, 50.0),
    cores: int | None = None,
):
    """Run Bayesian calibration using a black-box forward model wrapper.

    Notes
    -----
    - This backend uses PyMC + PyTensor but keeps the *forward model* as a pure
      black-box call (compatible with @njit functions).
    - When using empirical priors implemented via ``pm.Potential`` (KDE / Copula-KDE)
      together with ``DEMetropolisZ``, chains must start at *different* initial points.
      Otherwise all chains can collapse to the same value (often zeros) and never move.
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

    # Chain-specific initvals (only used for empirical priors)
    initvals = None

        # Precompute observed vector and observation indices once (critical for speed)
    _obs_flat = getattr(dataset, "obs_flat", None)
    _idx_obs_attr = getattr(dataset, "idx_obs", None)

    if _obs_flat is not None and _idx_obs_attr is not None:
        _observed_vec = np.asarray(_obs_flat, dtype=float)
        _idx_obs = np.asarray(_idx_obs_attr, dtype=np.int64)
        _is2d = True
    else:
        _obs_1d = getattr(dataset, "obs", None)
        if _obs_1d is None:
            _obs_1d = getattr(dataset, "rot", None)
        if _obs_1d is None:
            raise ValueError("Dataset must provide 'obs' or ('obs_flat' and 'idx_obs'), or 'rot' for rotation-only calibration.")
        _observed_vec = np.asarray(_obs_1d, dtype=float)
        _idx_obs = np.asarray(_idx_obs_attr, dtype=np.int64) if _idx_obs_attr is not None else None
        _is2d = False

    _n_obs = int(_observed_vec.size)
    _use_simulate_obs = bool(_is2d and hasattr(model, "simulate_obs"))

# ---- PyTensor Op (black-box forward model) ----
    class ForwardOp(Op):
        itypes = [pt.dvector, pt.dscalar]
        otypes = [pt.dvector]

        def make_node(self, raw_par, y0):
            raw_par = pt.as_tensor_variable(raw_par)
            y0 = pt.as_tensor_variable(y0)

            if raw_par.ndim != 1:
                raise ValueError("raw_par must be a vector")
            if y0.ndim != 0:
                raise ValueError("y0 must be a scalar")

            return Apply(self, [raw_par, y0], [pt.dvector()])

        def perform(self, node, inputs, outputs):
            raw_par, y0 = inputs
            physical = transform_raw_to_physical(np.asarray(raw_par, dtype=float), model.parameters)

            try:
                # Fast path for 2D likelihoods (IH-MOOSE): evaluate only needed points
                if _use_simulate_obs:
                    y_obs = model.simulate_obs(physical, dataset, float(y0))
                else:
                    y = model.simulate(physical, dataset, float(y0))
                    if _idx_obs is None:
                        y_obs = np.asarray(y, dtype=float).reshape(-1)
                    else:
                        if _is2d:
                            y_obs = np.asarray(y, dtype=float).reshape(-1)[_idx_obs]
                        else:
                            y_obs = np.asarray(y, dtype=float)[_idx_obs]
            except Exception:
                # Any numerical failure of the forward model should *not* kill a chain.
                # Returning a huge misfit makes the logp ~ -inf so the proposal is rejected.
                y_obs = np.full(max(1, _n_obs), 1e20, dtype=float)

            y_obs = np.asarray(y_obs, dtype=float).reshape(-1)
            if y_obs.size != max(1, _n_obs):
                # Shape mismatch (can happen if a forward returns full series unexpectedly)
                y_obs = np.full(max(1, _n_obs), 1e20, dtype=float)

            # Guard against NaNs/Infs produced by unstable proposals
            if not np.all(np.isfinite(y_obs)):
                y_obs = np.where(np.isfinite(y_obs), y_obs, 1e20)

            outputs[0][0] = np.asarray(y_obs, dtype="float64")
    fwd_op = ForwardOp()

    # For PriorProduct we keep a reference to each raw_par block RV to enable blocked sampling
    raw_blocks_rv = None

    with pm.Model() as pm_model:
        # ---- prior on raw parameters ----
        if isinstance(prior, PriorMVN):
            raw_par = pm.MvNormal(
                "raw_par",
                mu=prior.mean,
                cov=prior.cov,
                shape=prior.mean.size,
            )

            # Diverse starting points per chain (important for DEMetropolisZ)
            try:
                rng = np.random.default_rng(int(random_seed))
                if prior.init_pool is not None and np.asarray(prior.init_pool).ndim == 2:
                    pool = np.asarray(prior.init_pool, dtype=float)
                    replace = pool.shape[0] < int(chains)
                    idx0 = rng.choice(pool.shape[0], size=int(chains), replace=replace)
                    initvals = [{"raw_par": pool[i].astype("float64")} for i in idx0]
                else:
                    starts = rng.multivariate_normal(prior.mean, prior.cov, size=int(chains))
                    initvals = [{"raw_par": starts[i].astype("float64")} for i in range(int(chains))]
            except Exception:
                initvals = None

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

                    dens = float(kde.evaluate(v[:, None])[0])
                    if not np.isfinite(dens) or dens <= 0.0:
                        outputs[0][0] = np.array(-np.inf, dtype="float64")
                        return
                    outputs[0][0] = np.array(np.log(dens + 1e-300), dtype="float64")

            raw_par = pm.Flat("raw_par", shape=prior.samples.shape[1])
            pm.Potential("kde_prior", KDELogPOp()(raw_par))

            # Diverse starting points per chain (DEMetropolisZ needs cross-chain differences)
            pool = np.asarray(prior.samples, dtype=float)
            rng = np.random.default_rng(int(random_seed))
            replace = pool.shape[0] < int(chains)
            idx0 = rng.choice(pool.shape[0], size=int(chains), replace=replace)
            initvals = [{"raw_par": pool[i].astype("float64")} for i in idx0]

        elif isinstance(prior, PriorCopulaKDE):
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

            d = int(grid.shape[0])
            eps_u = float(prior.eps_u)
            eps_pdf = float(prior.eps_pdf)
            LOG2PI = float(np.log(2.0 * np.pi))

            if prior.method == "full":
                if prior.chol_R is None or prior.logdet_R is None:
                    raise ValueError("PriorCopulaKDE(full) missing chol_R/logdet_R")
                chol_R = np.asarray(prior.chol_R, dtype=float)
                logdet_R = float(prior.logdet_R)
            else:
                if (
                    prior.L is None
                    or prior.Dinv is None
                    or prior.chol_middle is None
                    or prior.logdet_R_factor is None
                ):
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

                    if prior.bounds_lower is not None and np.any(v < prior.bounds_lower):
                        outputs[0][0] = np.array(-np.inf, dtype="float64")
                        return
                    if prior.bounds_upper is not None and np.any(v > prior.bounds_upper):
                        outputs[0][0] = np.array(-np.inf, dtype="float64")
                        return

                    u = np.empty(d, dtype=float)
                    logf = 0.0
                    for j in range(d):
                        gj = grid[j]
                        cj = cdf[j]
                        pj = pdf[j]

                        uj = float(np.interp(v[j], gj, cj))
                        if not np.isfinite(uj):
                            uj = 0.5
                        uj = min(max(uj, eps_u), 1.0 - eps_u)
                        u[j] = uj

                        fj = float(np.interp(v[j], gj, pj))
                        if (not np.isfinite(fj)) or fj <= 0.0:
                            fj = eps_pdf
                        logf += float(np.log(fj + eps_pdf))

                    z = ndtri(u)
                    z = np.asarray(z, dtype=float)
                    if not np.all(np.isfinite(z)):
                        outputs[0][0] = np.array(-np.inf, dtype="float64")
                        return

                    z2 = float(np.dot(z, z))

                    if prior.method == "full":
                        y = solve_triangular(chol_R, z, lower=True, check_finite=False)
                        zTRinvz = float(np.dot(y, y))
                    else:
                        zD = Dinv * z
                        temp = L.T @ zD
                        y = solve_triangular(chol_mid, temp, lower=True, check_finite=False)
                        a = solve_triangular(chol_mid.T, y, lower=False, check_finite=False)
                        zTRinvz = float(np.dot(z, zD) - np.dot(temp, a))

                    logc = -0.5 * logdet_R - 0.5 * (zTRinvz - z2)
                    outputs[0][0] = np.array(logc + logf, dtype="float64")

            raw_par = pm.Flat("raw_par", shape=d)
            pm.Potential("copula_kde_prior", CopulaKDELogPOp()(raw_par))

            # Diverse starting points per chain
            if prior.init_pool is not None and np.asarray(prior.init_pool).ndim == 2:
                pool = np.asarray(prior.init_pool, dtype=float)
                rng = np.random.default_rng(int(random_seed))
                replace = pool.shape[0] < int(chains)
                idx0 = rng.choice(pool.shape[0], size=int(chains), replace=replace)
                initvals = [{"raw_par": pool[i].astype("float64")} for i in idx0]
            else:
                mid = np.array([grid[j, grid.shape[1] // 2] for j in range(d)], dtype=float)
                initvals = [{"raw_par": mid.astype("float64")} for _ in range(int(chains))]

        elif isinstance(prior, PriorProduct):
            # Product prior implemented for MVN blocks (robust and fast).
            if len(prior.priors) != len(prior.dims):
                raise ValueError("PriorProduct: priors and dims length mismatch")

            raw_blocks = []
            for bi, (pblock, dblock) in enumerate(zip(prior.priors, prior.dims)):
                if not isinstance(pblock, PriorMVN):
                    raise TypeError("PriorProduct currently supports PriorMVN blocks only")
                raw_blocks.append(
                    pm.MvNormal(f"raw_par_{bi}", mu=pblock.mean, cov=pblock.cov, shape=int(dblock))
                )

            # Keep block RVs for blocked sampling (helps mixing for IH-MOOSE)
            raw_blocks_rv = list(raw_blocks)

            raw_par = pt.concatenate(raw_blocks)

            # Diverse per-chain initvals: draw each block from its init_pool if available
            try:
                rng = np.random.default_rng(int(random_seed))
                initvals = []
                for c in range(int(chains)):
                    d = {}
                    for bi, (pblock, dblock) in enumerate(zip(prior.priors, prior.dims)):
                        if getattr(pblock, "init_pool", None) is not None:
                            pool = np.asarray(pblock.init_pool, dtype=float)
                            replace = pool.shape[0] < 1
                            ii = int(rng.choice(pool.shape[0], size=1, replace=replace)[0])
                            d[f"raw_par_{bi}"] = pool[ii].astype("float64")
                        else:
                            d[f"raw_par_{bi}"] = rng.multivariate_normal(pblock.mean, pblock.cov).astype("float64")
                    initvals.append(d)
            except Exception:
                initvals = None

        else:  # pragma: no cover
            raise TypeError(f"Unsupported prior type: {type(prior)}")

        # baseline initial value for y0
        if getattr(dataset, "y0", None) is not None:
            y0_det = float(dataset.y0)
        elif getattr(dataset, "alpha0", None) is not None:
            y0_det = float(dataset.alpha0)
        elif getattr(dataset, "obs", None) is not None and np.asarray(dataset.obs).size:
            y0_det = float(np.asarray(dataset.obs)[0])
        elif getattr(dataset, "rot", None) is not None and np.asarray(dataset.rot).size:
            y0_det = float(np.asarray(dataset.rot)[0])
        else:
            raise ValueError("Dataset has no y0/alpha0/obs/rot to define initial value.")

        if estimate_initial_position:
            # If you know your shoreline measurement error (~7 m), you can even FIX this instead of sampling
            # sigma_y0 = pm.HalfNormal("sigma_y0", sigma=20.0)
            # mu_y0 = pm.Normal("mu_y0", mu=y0_det, sigma=20.0)

            sigma_y0 = 10.0

            # y0_delta = pm.Normal("y0_delta", 0.0, 1.0)
            # y0 = pm.Deterministic("y0", y0_det + sigma_y0 * y0_delta)

            # centered parametrization around deterministic baseline
            # y0_delta = pm.Normal("y0_delta", mu=0.0, sigma=sigma_y0)
            # y0 = pm.Deterministic("y0", y0_det + y0_delta)
            # y0 = pm.Normal("y0", mu=mu_y0, sigma=sigma_y0)
            y0 = pm.Normal("y0", mu=y0_det, sigma=sigma_y0)

        else:
            y0 = pt.constant(y0_det, dtype="float64")

        yhat = fwd_op(raw_par, y0)

        # observed vector must match yhat length
        if getattr(dataset, "obs_flat", None) is not None and getattr(dataset, "idx_obs", None) is not None:
            observed_vec = np.asarray(dataset.obs_flat, dtype=float)   # 2D likelihood
        else:
            if getattr(dataset, "obs", None) is not None:
                observed_vec = np.asarray(dataset.obs, dtype=float)
            elif getattr(dataset, "rot", None) is not None:
                observed_vec = np.asarray(dataset.rot, dtype=float)
            else:
                raise ValueError("1D calibration requires dataset.obs or dataset.rot.")

        bias_rv = None
        if include_bias:
            SIGMA_BIAS = 5.0
            bias = pm.Normal("bias", mu=0.0, sigma=SIGMA_BIAS)
            bias_rv = bias
        else:
            bias = 0.0

        log_sigma_rv = None
        if estimate_sigma:
            # sigma_rv = pm.HalfNormal("sigma", sigma=2.0 * float(sigma or 30))
            log_sigma = pm.Normal("log_sigma", mu=np.log(float(sigma or 20)), sigma=0.5)
            log_sigma_rv = log_sigma
            sigma_rv = pm.Deterministic("sigma", pm.math.exp(log_sigma))

        else:
            sigma_rv = float(sigma)

        if likelihood.lower() == "normal":
            pm.Normal("likelihood", mu=yhat + bias, sigma=sigma_rv, observed=observed_vec)
        elif likelihood.lower() in {"studentt", "student-t", "t"}:
            # Student-t often mixes poorly when nu is free with random-walk samplers.
            # Allow fixing nu (empirical Bayes) for speed, otherwise sample a bounded nu.
            if nu_fixed is not None:
                nu = pm.Deterministic("nu", pt.constant(float(nu_fixed), dtype="float64"))
            else:
                nu_raw = pm.Normal("nu_raw", mu=0.0, sigma=1.0)
                lo, hi = float(nu_bounds[0]), float(nu_bounds[1])
                if hi <= lo:
                    raise ValueError("nu_bounds must satisfy hi > lo")
                nu = pm.Deterministic("nu", lo + (hi - lo) * pm.math.sigmoid(nu_raw))

            pm.StudentT("likelihood", mu=yhat + bias, sigma=sigma_rv, nu=nu, observed=observed_vec)
        else:
            raise ValueError(f"Unknown likelihood='{likelihood}'")




        # NOTE: Using multiple step methods (e.g., several DEMetropolisZ blocks)
        # can trigger a PyMC backend error:
        #   ValueError: Sampler statistic accepted appears with different types.
        # because different blocks emit an "accepted" stat with different shapes.
        # For robustness across PyMC versions, we default to a single DEMetropolisZ.
        # (We still benefit from better priors + chain-specific initvals.)
        step = pm.DEMetropolisZ()


        import inspect

        sample_kwargs = dict(
            draws=draws,
            tune=tune,
            chains=chains,
            step=step,
            random_seed=random_seed,
            cores=cores,
            progressbar=True,
        )

        # Store log-likelihood in InferenceData when supported (enables likelihood diagnostics plots).
        try:
            sig_sample = inspect.signature(pm.sample)
            if "idata_kwargs" in sig_sample.parameters:
                sample_kwargs["idata_kwargs"] = {"log_likelihood": True}
        except Exception:
            pass
        if initvals is not None:
            sig = inspect.signature(pm.sample)
            if "initvals" in sig.parameters:
                sample_kwargs["initvals"] = initvals
            elif "initval" in sig.parameters:
                sample_kwargs["initval"] = initvals

        trace = pm.sample(**sample_kwargs)
        print(pm.summary(trace))
        ppc = pm.sample_posterior_predictive(trace, var_names=["likelihood"])

    return trace, ppc


def bayesian_calibrate_hdsc(
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
    sigma_covariates: np.ndarray | None = None,        # (n_obs, k) at obs times
    sigma_covariate_names: list[str] | None = None,    # optional, for bookkeeping
    sigma_beta_sd: float = 0.3,                        # regularization strength
    sigma_floor: float = 1e-6,                         # avoid sigma -> 0
    standardize_sigma_covariates: bool = True,
    include_bias: bool = True,
    estimate_initial_position: bool = False,
    nu_fixed: float | None = None,
    nu_bounds: tuple[float, float] = (2.0, 50.0),
    cores: int | None = None,
):
    """Run Bayesian calibration using a black-box forward model wrapper.

    Notes
    -----
    - This backend uses PyMC + PyTensor but keeps the *forward model* as a pure
      black-box call (compatible with @njit functions).
    - When using empirical priors implemented via ``pm.Potential`` (KDE / Copula-KDE)
      together with ``DEMetropolisZ``, chains must start at *different* initial points.
      Otherwise all chains can collapse to the same value (often zeros) and never move.
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

    # Chain-specific initvals (only used for empirical priors)
    initvals = None

    # ---- PyTensor Op (black-box forward model) ----
    class ForwardOp(Op):
        itypes = [pt.dvector, pt.dscalar]
        otypes = [pt.dvector]

        def make_node(self, raw_par, y0):
            raw_par = pt.as_tensor_variable(raw_par)
            y0 = pt.as_tensor_variable(y0)

            if raw_par.ndim != 1:
                raise ValueError("raw_par must be a vector")
            if y0.ndim != 0:
                raise ValueError("y0 must be a scalar")

            return Apply(self, [raw_par, y0], [pt.dvector()])

        def perform(self, node, inputs, outputs):
            raw_par, y0 = inputs
            physical = transform_raw_to_physical(np.asarray(raw_par, dtype=float), model.parameters)

            try:
                y = model.simulate(physical, dataset, float(y0))

                # Extract model predictions at observation points (supports 1D and 2D obs)
                if getattr(dataset, "obs_flat", None) is not None and getattr(dataset, "idx_obs", None) is not None:
                    y_flat = np.asarray(y, dtype=float).reshape(-1)
                    y_obs = y_flat[np.asarray(dataset.idx_obs, dtype=np.int64)]
                else:
                    y_obs = np.asarray(y[np.asarray(dataset.idx_obs, dtype=np.int64)], dtype=float)
            except Exception:
                try:
                    if getattr(dataset, "obs_flat", None) is not None:
                        n = int(np.asarray(dataset.obs_flat).size)
                    else:
                        n = int(np.asarray(getattr(dataset, "idx_obs", [])).size)
                except Exception:
                    n = 0
                if n <= 0 and getattr(dataset, "obs", None) is not None:
                    n = int(np.asarray(dataset.obs).size)
                y_obs = np.full(max(1, n), 1e20, dtype=float)

            # Guard against NaNs/Infs produced by unstable proposals
            if not np.all(np.isfinite(y_obs)):
                y_obs = np.where(np.isfinite(y_obs), y_obs, 1e20)

            outputs[0][0] = y_obs.astype("float64")

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

                    dens = float(kde.evaluate(v[:, None])[0])
                    if not np.isfinite(dens) or dens <= 0.0:
                        outputs[0][0] = np.array(-np.inf, dtype="float64")
                        return
                    outputs[0][0] = np.array(np.log(dens + 1e-300), dtype="float64")

            raw_par = pm.Flat("raw_par", shape=prior.samples.shape[1])
            pm.Potential("kde_prior", KDELogPOp()(raw_par))

            # Diverse starting points per chain (DEMetropolisZ needs cross-chain differences)
            pool = np.asarray(prior.samples, dtype=float)
            rng = np.random.default_rng(int(random_seed))
            replace = pool.shape[0] < int(chains)
            idx0 = rng.choice(pool.shape[0], size=int(chains), replace=replace)
            initvals = [{"raw_par": pool[i].astype("float64")} for i in idx0]

        elif isinstance(prior, PriorCopulaKDE):
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

            d = int(grid.shape[0])
            eps_u = float(prior.eps_u)
            eps_pdf = float(prior.eps_pdf)
            LOG2PI = float(np.log(2.0 * np.pi))

            if prior.method == "full":
                if prior.chol_R is None or prior.logdet_R is None:
                    raise ValueError("PriorCopulaKDE(full) missing chol_R/logdet_R")
                chol_R = np.asarray(prior.chol_R, dtype=float)
                logdet_R = float(prior.logdet_R)
            else:
                if (
                    prior.L is None
                    or prior.Dinv is None
                    or prior.chol_middle is None
                    or prior.logdet_R_factor is None
                ):
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

                    if prior.bounds_lower is not None and np.any(v < prior.bounds_lower):
                        outputs[0][0] = np.array(-np.inf, dtype="float64")
                        return
                    if prior.bounds_upper is not None and np.any(v > prior.bounds_upper):
                        outputs[0][0] = np.array(-np.inf, dtype="float64")
                        return

                    u = np.empty(d, dtype=float)
                    logf = 0.0
                    for j in range(d):
                        gj = grid[j]
                        cj = cdf[j]
                        pj = pdf[j]

                        uj = float(np.interp(v[j], gj, cj))
                        if not np.isfinite(uj):
                            uj = 0.5
                        uj = min(max(uj, eps_u), 1.0 - eps_u)
                        u[j] = uj

                        fj = float(np.interp(v[j], gj, pj))
                        if (not np.isfinite(fj)) or fj <= 0.0:
                            fj = eps_pdf
                        logf += float(np.log(fj + eps_pdf))

                    z = ndtri(u)
                    z = np.asarray(z, dtype=float)
                    if not np.all(np.isfinite(z)):
                        outputs[0][0] = np.array(-np.inf, dtype="float64")
                        return

                    z2 = float(np.dot(z, z))

                    if prior.method == "full":
                        y = solve_triangular(chol_R, z, lower=True, check_finite=False)
                        zTRinvz = float(np.dot(y, y))
                    else:
                        zD = Dinv * z
                        temp = L.T @ zD
                        y = solve_triangular(chol_mid, temp, lower=True, check_finite=False)
                        a = solve_triangular(chol_mid.T, y, lower=False, check_finite=False)
                        zTRinvz = float(np.dot(z, zD) - np.dot(temp, a))

                    logc = -0.5 * logdet_R - 0.5 * (zTRinvz - z2)
                    outputs[0][0] = np.array(logc + logf, dtype="float64")

            raw_par = pm.Flat("raw_par", shape=d)
            pm.Potential("copula_kde_prior", CopulaKDELogPOp()(raw_par))

            # Diverse starting points per chain
            if prior.init_pool is not None and np.asarray(prior.init_pool).ndim == 2:
                pool = np.asarray(prior.init_pool, dtype=float)
                rng = np.random.default_rng(int(random_seed))
                replace = pool.shape[0] < int(chains)
                idx0 = rng.choice(pool.shape[0], size=int(chains), replace=replace)
                initvals = [{"raw_par": pool[i].astype("float64")} for i in idx0]
            else:
                mid = np.array([grid[j, grid.shape[1] // 2] for j in range(d)], dtype=float)
                initvals = [{"raw_par": mid.astype("float64")} for _ in range(int(chains))]

        elif isinstance(prior, PriorProduct):
            # Product prior implemented for MVN blocks (robust and fast).
            if len(prior.priors) != len(prior.dims):
                raise ValueError("PriorProduct: priors and dims length mismatch")
            raw_blocks = []
            for bi, (pblock, dblock) in enumerate(zip(prior.priors, prior.dims)):
                if not isinstance(pblock, PriorMVN):
                    raise TypeError("PriorProduct currently supports PriorMVN blocks only")
                raw_blocks.append(pm.MvNormal(f"raw_par_{bi}", mu=pblock.mean, cov=pblock.cov, shape=int(dblock)))
            raw_par = pt.concatenate(raw_blocks)

        else:  # pragma: no cover
            raise TypeError(f"Unsupported prior type: {type(prior)}")

        y0_det = float(getattr(dataset, "y0", dataset.obs[0]))

        if estimate_initial_position:
            # If you know your shoreline measurement error (~7 m), you can even FIX this instead of sampling
            # sigma_y0 = pm.HalfNormal("sigma_y0", sigma=20.0)
            # mu_y0 = pm.Normal("mu_y0", mu=y0_det, sigma=20.0)

            sigma_y0 = 10.0

            # y0_delta = pm.Normal("y0_delta", 0.0, 1.0)
            # y0 = pm.Deterministic("y0", y0_det + sigma_y0 * y0_delta)

            # centered parametrization around deterministic baseline
            # y0_delta = pm.Normal("y0_delta", mu=0.0, sigma=sigma_y0)
            # y0 = pm.Deterministic("y0", y0_det + y0_delta)
            # y0 = pm.Normal("y0", mu=mu_y0, sigma=sigma_y0)
            y0 = pm.Normal("y0", mu=y0_det, sigma=sigma_y0)

        else:
            y0 = pt.constant(y0_det, dtype="float64")

        yhat = fwd_op(raw_par, y0)

        if include_bias:
            SIGMA_BIAS = 5.0
            bias = pm.Normal("bias", mu=0.0, sigma=SIGMA_BIAS)
        else:
            bias = 0.0

        # -------------------------
        # Heteroscedastic sigma(t)
        # -------------------------
        Xsig = None
        Xsig_mu = None
        Xsig_sd = None

        if sigma_covariates is not None:
            Xsig = np.asarray(sigma_covariates, dtype="float64")
            if Xsig.ndim == 1:
                Xsig = Xsig[:, None]
            if Xsig.shape[0] != len(dataset.obs):
                raise ValueError(
                    f"sigma_covariates must have n_obs rows = len(dataset.obs)={len(dataset.obs)}, "
                    f"got {Xsig.shape[0]}"
                )

            if standardize_sigma_covariates:
                Xsig_mu = Xsig.mean(axis=0)
                Xsig_sd = Xsig.std(axis=0)
                Xsig_sd = np.where(Xsig_sd > 0, Xsig_sd, 1.0)
                Xsig = (Xsig - Xsig_mu) / Xsig_sd

            Xsig_pt = pt.constant(Xsig, dtype="float64")
            ksig = int(Xsig.shape[1])

            # Baseline scale prior center: use provided sigma if available, else a safe default
            sigma0 = float(sigma) if sigma is not None else 30.0

            sigma_alpha = pm.Normal("sigma_alpha", mu=np.log(sigma0), sigma=0.5)
            sigma_beta = pm.Normal("sigma_beta", mu=0.0, sigma=float(sigma_beta_sd), shape=ksig)

            log_sigma_t = pm.Deterministic("log_sigma_t", sigma_alpha + pt.dot(Xsig_pt, sigma_beta))
            sigma_rv = pm.Deterministic("sigma", pm.math.exp(log_sigma_t) + float(sigma_floor))

        else:
            # ---- original homoscedastic behavior ----
            if sigma is None and not estimate_sigma:
                raise ValueError("Provide sigma or set estimate_sigma=True")

            if estimate_sigma:
                log_sigma = pm.Normal("log_sigma", mu=np.log(float(sigma or 30)), sigma=0.5)
                sigma_rv = pm.Deterministic("sigma", pm.math.exp(log_sigma))
            else:
                sigma_rv = float(sigma)

        if likelihood.lower() == "normal":
            pm.Normal("likelihood", mu=yhat + bias, sigma=sigma_rv, observed=observed_vec)
        elif likelihood.lower() in {"studentt", "student-t", "t"}:
            # Student-t often mixes poorly when nu is free with random-walk samplers.
            # Allow fixing nu (empirical Bayes) for speed, otherwise sample a bounded nu.
            if nu_fixed is not None:
                nu = pm.Deterministic("nu", pt.constant(float(nu_fixed), dtype="float64"))
            else:
                nu_raw = pm.Normal("nu_raw", mu=0.0, sigma=1.0)
                lo, hi = float(nu_bounds[0]), float(nu_bounds[1])
                if hi <= lo:
                    raise ValueError("nu_bounds must satisfy hi > lo")
                nu = pm.Deterministic("nu", lo + (hi - lo) * pm.math.sigmoid(nu_raw))

            pm.StudentT("likelihood", mu=yhat + bias, sigma=sigma_rv, nu=nu, observed=observed_vec)
        else:
            raise ValueError(f"Unknown likelihood='{likelihood}'")

        step = pm.DEMetropolisZ()

        import inspect

        sample_kwargs = dict(
            draws=draws,
            tune=tune,
            chains=chains,
            step=step,
            random_seed=random_seed,
            cores=cores,
            progressbar=True,
        )

        # Store log-likelihood in InferenceData when supported (enables likelihood diagnostics plots).
        try:
            sig_sample = inspect.signature(pm.sample)
            if "idata_kwargs" in sig_sample.parameters:
                sample_kwargs["idata_kwargs"] = {"log_likelihood": True}
        except Exception:
            pass
        if initvals is not None:
            sig = inspect.signature(pm.sample)
            if "initvals" in sig.parameters:
                sample_kwargs["initvals"] = initvals
            elif "initval" in sig.parameters:
                sample_kwargs["initval"] = initvals

        trace = pm.sample(**sample_kwargs)
        ppc = pm.sample_posterior_predictive(trace, var_names=["likelihood"])

    return trace, ppc
