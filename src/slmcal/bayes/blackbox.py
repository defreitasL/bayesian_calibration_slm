from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from slmcal.data import TimeSeriesDataset
from slmcal.models.base import ShorelineModel, transform_raw_to_physical
from slmcal.bayes.priors import PriorMVN, PriorCopulaKDE

try:
    from scipy.stats import norm
except Exception:  # pragma: no cover
    norm = None  # type: ignore


@dataclass(frozen=True)
class BlackBoxDEMCConfig:
    chains: int = 20
    draws: int = 1000
    tune: int = 20000
    gamma: float | None = None  # if None use 2.38/sqrt(2*d)
    jitter: float = 1e-6
    random_seed: int = 42
    include_bias: bool = False
    estimate_sigma: bool = False
    sigma_fixed: float = 50.0

    # bias priors
    mu_bias_sd: float = 5.0
    sigma_bias_sd: float = 5.0


@dataclass
class BlackBoxResult:
    posterior_raw: np.ndarray          # (chains, draws, n_params_model)
    sigma: np.ndarray                  # (chains, draws)
    logp: np.ndarray                   # (chains, draws)
    accept_rate: np.ndarray            # (chains,)
    # Optional bias outputs
    mu_bias: Optional[np.ndarray] = None        # (chains, draws)
    sigma_bias: Optional[np.ndarray] = None     # (chains, draws)
    bias: Optional[np.ndarray] = None           # (chains, draws)


def _log_norm(x: np.ndarray, mu: np.ndarray, sigma: float) -> float:
    """Sum log N(x | mu, sigma) computed defensively.

    In early tuning/proposal phases sigma can become extremely small, which makes
    r = (x-mu)/sigma huge and can overflow when squaring. We clamp sigma to a
    tiny positive value and return -inf if the result is non-finite.
    """
    sig = float(sigma)
    if not np.isfinite(sig) or sig <= 0.0:
        return -np.inf

    # Avoid pathological underflow/overflow
    sig = max(sig, 1e-12)

    r = (np.asarray(x, dtype=float) - np.asarray(mu, dtype=float)) / sig
    # More stable than sum(r*r) and avoids an intermediate array
    ss = float(np.dot(r, r))
    if not np.isfinite(ss):
        return -np.inf

    n = int(r.size)
    return float(-0.5 * ss - n * np.log(sig) - 0.5 * n * np.log(2.0 * np.pi))


def _log_halfnorm(x: float, sd: float) -> float:
    # HalfNormal(sd) for x>0
    if x <= 0:
        return -np.inf
    return float(np.log(np.sqrt(2) / (sd * np.sqrt(np.pi))) - (x * x) / (2 * sd * sd))


def _log_prior_bias(mu_bias: float, sigma_bias: float, bias: float, mu_sd: float, sigma_sd: float) -> float:
    # mu_bias ~ N(0, mu_sd)
    lp = float(-0.5 * (mu_bias / mu_sd) ** 2 - np.log(mu_sd) - 0.5 * np.log(2 * np.pi))
    # sigma_bias ~ HalfNormal(sigma_sd)
    lp += _log_halfnorm(sigma_bias, sigma_sd)
    # bias ~ N(mu_bias, sigma_bias)
    lp += float(-0.5 * ((bias - mu_bias) / sigma_bias) ** 2 - np.log(sigma_bias) - 0.5 * np.log(2 * np.pi))
    return lp


def _log_likelihood(
    model: ShorelineModel,
    dataset: TimeSeriesDataset,
    theta_raw_model: np.ndarray,
    bias: float,
    sigma: float,
) -> float:
    phys = transform_raw_to_physical(theta_raw_model, model.parameters)
    y_full = model.simulate(phys, dataset)
    idx = np.asarray(dataset.idx_obs, dtype=int)
    y = np.asarray(y_full, dtype=float)[idx]
    obs = np.asarray(dataset.obs, dtype=float)
    mu = y + bias
    return _log_norm(obs, mu, float(sigma))


def blackbox_demcmc(
    model: ShorelineModel,
    dataset: TimeSeriesDataset,
    prior: PriorMVN | PriorCopulaKDE,
    cfg: BlackBoxDEMCConfig,
    init_population: np.ndarray | None = None,
) -> BlackBoxResult:
    """Black-box DE-MCMC (DE-Metropolis) sampler.

    - calls the model as a black-box (works with @njit forward models)
    - supports optional scalar bias term (with hierarchical priors)
    - supports fixed or estimated sigma
    """
    rng = np.random.default_rng(int(cfg.random_seed))

    n_model = len(model.parameters)
    include_bias = bool(cfg.include_bias)
    estimate_sigma = bool(cfg.estimate_sigma)

    # Parameter layout for the internal sampler vector:
    # [raw_model (n_model), (mu_bias, sigma_bias, bias)? , (log_sigma)?]
    idx_mu_bias = n_model
    idx_sigma_bias = n_model + 1
    idx_bias = n_model + 2
    idx_log_sigma = n_model + (3 if include_bias else 0)

    d_total = n_model + (3 if include_bias else 0) + (1 if estimate_sigma else 0)

    gamma = cfg.gamma
    if gamma is None:
        gamma = 2.38 / np.sqrt(2 * d_total)

    # Initialize chains
    if init_population is not None:
        init_population = np.asarray(init_population, dtype=float)
        if init_population.ndim != 2 or init_population.shape[1] < n_model:
            raise ValueError("init_population must be (N, n_model) in raw space")
        # pick random rows for chains
        idx = rng.choice(init_population.shape[0], size=cfg.chains, replace=init_population.shape[0] < cfg.chains)
        theta0_model = init_population[idx, :n_model]
    else:
        # prior-based init
        theta0_model = prior.rvs(cfg.chains, rng=rng)

    theta0 = np.zeros((cfg.chains, d_total), dtype=float)
    theta0[:, :n_model] = theta0_model

    if include_bias:
        theta0[:, idx_mu_bias] = rng.normal(0.0, cfg.mu_bias_sd, size=cfg.chains)
        theta0[:, idx_sigma_bias] = np.abs(rng.normal(cfg.sigma_bias_sd * 0.5, cfg.sigma_bias_sd * 0.25, size=cfg.chains)) + 1e-3
        theta0[:, idx_bias] = rng.normal(theta0[:, idx_mu_bias], theta0[:, idx_sigma_bias])

    if estimate_sigma:
        theta0[:, idx_log_sigma] = np.log(cfg.sigma_fixed)

    def unpack(theta_vec: np.ndarray):
        raw_model = theta_vec[:n_model]
        if include_bias:
            mu_b = float(theta_vec[idx_mu_bias])
            sig_b = float(theta_vec[idx_sigma_bias])
            b = float(theta_vec[idx_bias])
        else:
            mu_b, sig_b, b = 0.0, 0.0, 0.0

        if estimate_sigma:
            sigma = float(np.exp(theta_vec[idx_log_sigma]))
        else:
            sigma = float(cfg.sigma_fixed)
        return raw_model, mu_b, sig_b, b, sigma

    def logp(theta_vec: np.ndarray) -> float:
        raw_model, mu_b, sig_b, b, sigma = unpack(theta_vec)
        lp = float(prior.logpdf(raw_model))
        if include_bias:
            lp += _log_prior_bias(mu_b, sig_b, b, cfg.mu_bias_sd, cfg.sigma_bias_sd)
        if estimate_sigma:
            # sigma ~ HalfNormal(sigma_fixed) (weakly informative around cfg.sigma_fixed)
            lp += _log_halfnorm(sigma, cfg.sigma_fixed)
            # add jacobian for log_sigma transform: sigma = exp(log_sigma)
            lp += float(np.log(sigma))
        lp += _log_likelihood(model, dataset, raw_model, b if include_bias else 0.0, sigma)
        return lp

    # Evaluate initial logp
    current = theta0.copy()
    current_lp = np.asarray([logp(current[c]) for c in range(cfg.chains)], dtype=float)

    total_iters = int(cfg.tune) + int(cfg.draws)
    out_model = np.empty((cfg.chains, cfg.draws, n_model), dtype=float)
    out_sigma = np.empty((cfg.chains, cfg.draws), dtype=float)
    out_logp = np.empty((cfg.chains, cfg.draws), dtype=float)

    out_mu_bias = np.empty((cfg.chains, cfg.draws), dtype=float) if include_bias else None
    out_sigma_bias = np.empty((cfg.chains, cfg.draws), dtype=float) if include_bias else None
    out_bias = np.empty((cfg.chains, cfg.draws), dtype=float) if include_bias else None

    accepted = np.zeros(cfg.chains, dtype=int)
    stored = 0

    for it in range(total_iters):
        for c in range(cfg.chains):
            # pick r1, r2 distinct from c
            choices = [i for i in range(cfg.chains) if i != c]
            r1, r2 = rng.choice(choices, size=2, replace=False)
            proposal = current[c] + gamma * (current[r1] - current[r2]) + rng.normal(0.0, cfg.jitter, size=d_total)

            lp_prop = logp(proposal)
            if np.isfinite(lp_prop):
                if np.log(rng.random()) < (lp_prop - current_lp[c]):
                    current[c] = proposal
                    current_lp[c] = lp_prop
                    if it >= cfg.tune:
                        accepted[c] += 1

        if it >= cfg.tune:
            # store
            for c in range(cfg.chains):
                raw_model, mu_b, sig_b, b, sigma = unpack(current[c])
                out_model[c, stored, :] = raw_model
                out_sigma[c, stored] = sigma
                out_logp[c, stored] = current_lp[c]
                if include_bias:
                    out_mu_bias[c, stored] = mu_b
                    out_sigma_bias[c, stored] = sig_b
                    out_bias[c, stored] = b
            stored += 1

    accept_rate = accepted / max(1, cfg.draws)

    return BlackBoxResult(
        posterior_raw=out_model,
        sigma=out_sigma,
        logp=out_logp,
        accept_rate=accept_rate,
        mu_bias=out_mu_bias,
        sigma_bias=out_sigma_bias,
        bias=out_bias,
    )
