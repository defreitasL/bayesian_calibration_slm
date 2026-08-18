from __future__ import annotations

"""Utilities to save Bayesian samplers and propagate identifiable uncertainty sources.

The calibration workflow estimates physical model parameters together with an
optional initial shoreline position ``y0`` and nuisance/error terms such as
``bias`` and ``sigma``.  Because ``y0`` and an additive bias both control the
absolute shoreline level, they are not generally identifiable as independent
uncertainty sources from a single shoreline record.  The propagation routines
therefore use three interpretable sources:

- dynamic/physical-parameter uncertainty;
- reference-position uncertainty, combining ``y0`` and additive ``bias``;
- residual predictive uncertainty represented by ``sigma * z``.

The deterministic reference is ``f(theta_median, y0_median) + bias_median``.
For every posterior draw, the compact effects satisfy exactly

``full = deterministic_reference + parameter_effect
        + reference_position_effect + residual_noise``.

The interaction between the physical parameters and ``y0`` is allocated
symmetrically with a two-factor Shapley decomposition before the centered bias
is added to the reference-position effect.
"""

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Mapping

import numpy as np

from slmcal.data import TimeSeriesDataset
from slmcal.models.base import ShorelineModel, transform_raw_to_physical


@dataclass(frozen=True)
class PosteriorSampler:
    raw: np.ndarray
    physical: np.ndarray
    param_names: list[str]
    draw_index: np.ndarray
    sigma: np.ndarray | None = None
    bias: np.ndarray | None = None
    y0: np.ndarray | None = None
    nu: np.ndarray | None = None


def flatten_raw_parameter_samples(trace) -> np.ndarray:
    """Return posterior raw-parameter samples with shape ``(n_draws, n_params)``."""
    post = trace.posterior
    if "raw_par" in post:
        arr = np.asarray(post["raw_par"].values)
        if arr.ndim != 3:
            raise ValueError("trace.posterior['raw_par'] must have dims (chain, draw, param)")
        return arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2]).astype(float)

    blocks: list[np.ndarray] = []
    bi = 0
    while f"raw_par_{bi}" in post:
        b = np.asarray(post[f"raw_par_{bi}"].values)
        if b.ndim != 3:
            raise ValueError(f"trace.posterior['raw_par_{bi}'] must have dims (chain, draw, param)")
        blocks.append(b.reshape(b.shape[0] * b.shape[1], b.shape[2]).astype(float))
        bi += 1
    if not blocks:
        raise KeyError("Trace does not contain 'raw_par' or 'raw_par_i' parameter blocks.")
    return np.concatenate(blocks, axis=1)


def _flat_var(trace, name: str) -> np.ndarray | None:
    try:
        if name not in trace.posterior:
            return None
        return np.asarray(trace.posterior[name].values, dtype=float).reshape(-1)
    except Exception:
        return None


def _posterior_nuisance(trace, n_expected: int, *, sigma_fixed: float | None = None) -> dict[str, np.ndarray | None]:
    sigma = _flat_var(trace, "sigma")
    if sigma is None:
        log_sigma = _flat_var(trace, "log_sigma")
        if log_sigma is not None:
            sigma = np.exp(log_sigma)
    if sigma is None and sigma_fixed is not None:
        sigma = np.full(n_expected, float(sigma_fixed), dtype=float)

    out = {
        "sigma": sigma,
        "bias": _flat_var(trace, "bias"),
        "y0": _flat_var(trace, "y0"),
        "nu": _flat_var(trace, "nu"),
    }
    # Keep only arrays aligned with raw posterior draws.
    for k, v in list(out.items()):
        if v is not None and v.size != n_expected:
            out[k] = None
    return out


def _choose_draws(n_avail: int, n_draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.choice(int(n_avail), size=int(n_draws), replace=(int(n_draws) > int(n_avail))).astype(int)


def posterior_parameter_sampler(
    trace,
    model: ShorelineModel,
    *,
    n_draws: int | None = None,
    seed: int = 42,
    sigma_fixed: float | None = None,
) -> PosteriorSampler:
    """Extract an aligned posterior sampler for parameters and nuisance terms."""
    raw_all = flatten_raw_parameter_samples(trace)
    n_avail = int(raw_all.shape[0])
    if n_draws is None or int(n_draws) >= n_avail:
        idx = np.arange(n_avail, dtype=int)
    else:
        idx = _choose_draws(n_avail, int(n_draws), int(seed))

    raw = raw_all[idx]
    physical = np.vstack([transform_raw_to_physical(r, model.parameters) for r in raw]).astype(float)
    nuis = _posterior_nuisance(trace, n_avail, sigma_fixed=sigma_fixed)

    def take(v: np.ndarray | None) -> np.ndarray | None:
        return None if v is None else np.asarray(v[idx], dtype=float)

    return PosteriorSampler(
        raw=raw,
        physical=physical,
        param_names=[p.name for p in model.parameters],
        draw_index=idx,
        sigma=take(nuis["sigma"]),
        bias=take(nuis["bias"]),
        y0=take(nuis["y0"]),
        nu=take(nuis["nu"]),
    )


def sample_prior_raw(prior: object, n_samples: int, *, seed: int = 42) -> np.ndarray | None:
    """Draw/sample prior raw parameters from the supported empirical prior objects."""
    rng = np.random.default_rng(int(seed))

    if prior is None:
        return None

    if hasattr(prior, "priors") and hasattr(prior, "dims"):
        blocks = []
        for i, p in enumerate(getattr(prior, "priors")):
            s = sample_prior_raw(p, n_samples, seed=seed + 101 * (i + 1))
            if s is None:
                return None
            blocks.append(s)
        return np.concatenate(blocks, axis=1)

    if hasattr(prior, "mean") and hasattr(prior, "cov"):
        mu = np.asarray(getattr(prior, "mean"), dtype=float)
        cov = np.asarray(getattr(prior, "cov"), dtype=float)
        try:
            return rng.multivariate_normal(mu, cov, size=int(n_samples)).astype(float)
        except Exception:
            cov = cov + 1e-8 * np.eye(mu.size)
            return rng.multivariate_normal(mu, cov, size=int(n_samples)).astype(float)

    if hasattr(prior, "samples"):
        samples = np.asarray(getattr(prior, "samples"), dtype=float)
        if samples.ndim == 2 and samples.shape[0] > 0:
            idx = rng.choice(samples.shape[0], size=int(n_samples), replace=(samples.shape[0] < int(n_samples)))
            return samples[idx].astype(float)

    # PriorCopulaKDE stores tabulated marginals but not an exact sampler.  For a
    # lightweight diagnostic prior-vs-posterior plot, sample each marginal
    # independently from the stored grid/CDF.  This intentionally ignores the
    # copula dependence, so downstream code should prefer init_pool when present.
    if hasattr(prior, "grid") and hasattr(prior, "cdf"):
        if getattr(prior, "init_pool", None) is not None:
            pool = np.asarray(getattr(prior, "init_pool"), dtype=float)
            idx = rng.choice(pool.shape[0], size=int(n_samples), replace=(pool.shape[0] < int(n_samples)))
            return pool[idx].astype(float)
        grid = np.asarray(getattr(prior, "grid"), dtype=float)
        cdf = np.asarray(getattr(prior, "cdf"), dtype=float)
        d = int(grid.shape[0])
        out = np.empty((int(n_samples), d), dtype=float)
        u = rng.uniform(0.0, 1.0, size=(int(n_samples), d))
        for j in range(d):
            out[:, j] = np.interp(u[:, j], cdf[j], grid[j])
        return out

    return None


def raw_to_physical_samples(raw: np.ndarray | None, model: ShorelineModel) -> np.ndarray | None:
    if raw is None:
        return None
    r = np.asarray(raw, dtype=float)
    return np.vstack([transform_raw_to_physical(v, model.parameters) for v in r]).astype(float)


def _median_or_none(v: np.ndarray | None) -> float | None:
    if v is None or np.asarray(v).size == 0:
        return None
    x = np.asarray(v, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return float(np.median(x))


def _simulate_one(model: ShorelineModel, physical: np.ndarray, dataset: TimeSeriesDataset, y0: float | None) -> np.ndarray:
    if y0 is not None:
        try:
            return np.asarray(model.simulate(physical, dataset, y0=float(y0)), dtype=float)
        except TypeError:
            pass
    return np.asarray(model.simulate(physical, dataset), dtype=float)


def _cap_draws_for_memory(n_draws: int, dataset: TimeSeriesDataset, dtype, max_bytes: int, n_components: int = 3) -> int:
    n_t = int(np.asarray(dataset.time).size)
    item = np.dtype(dtype).itemsize
    est = int(n_draws) * n_t * int(n_components) * item
    if est <= int(max_bytes):
        return int(n_draws)
    return max(1, int(max_bytes // max(1, n_t * int(n_components) * item)))


def summarise_draws(draws: np.ndarray, q: tuple[float, ...] = (1, 5, 10, 50, 90, 95, 99)) -> dict[str, np.ndarray]:
    d = np.asarray(draws, dtype=float)
    out = {f"p{int(qq):02d}": np.nanpercentile(d, qq, axis=0) for qq in q}
    out["min"] = np.nanmin(d, axis=0)
    out["max"] = np.nanmax(d, axis=0)
    out["mean"] = np.nanmean(d, axis=0)
    out["std"] = np.nanstd(d, axis=0)
    out["emv"] = out["max"] - out["min"]
    return out


def propagate_uncertainty_components(
    trace,
    model: ShorelineModel,
    dataset: TimeSeriesDataset,
    *,
    n_draws: int = 1000,
    seed: int = 42,
    sigma_fixed: float | None = None,
    include_bias: bool = True,
    include_noise: bool = True,
    dtype=np.float32,
    max_bytes: int = 500_000_000,
    materialize_components: bool = True,
) -> dict[str, Any]:
    """Propagate three identifiable posterior uncertainty sources.

    The returned draw-level effects have shape ``(n_draws, n_time)``:

    ``parameter_effect``
        Symmetric Shapley contribution of the physical model parameters.

    ``reference_position_effect``
        Symmetric Shapley contribution of ``y0`` plus the centered additive
        bias, ``bias_i - bias_median``.

    ``residual_noise``
        Gaussian residual realization ``sigma_i * z_i(t)``.

    Four deterministic simulations are used to allocate the interaction between
    physical parameters and ``y0``:

    ``f00 = f(theta_median, y0_median)``
    ``f10 = f(theta_i,      y0_median)``
    ``f01 = f(theta_median, y0_i)``
    ``f11 = f(theta_i,      y0_i)``

    with

    ``parameter_effect = 0.5 * [(f10 - f00) + (f11 - f01)]``
    ``y0_effect        = 0.5 * [(f01 - f00) + (f11 - f10)]``.

    The deterministic reference includes the posterior-median bias:

    ``deterministic_reference = f00 + bias_median``

    and

    ``reference_position_effect = y0_effect + (bias_i - bias_median)``.

    Consequently, draw by draw,

    ``full = deterministic_reference + parameter_effect
            + reference_position_effect + residual_noise``.

    Notes
    -----
    ``sigma`` is retained as a combined residual predictive term.  Unless an
    external observation-error model is supplied in the likelihood, it cannot
    be interpreted as structural uncertainty alone.

    Parameters
    ----------
    materialize_components:
        If True, also return absolute ensembles for interactive use.  The saved
        artifact remains compact and stores only the deterministic reference and
        the three effects.
    """
    # Three persisted matrices plus one temporary absolute matrix used while
    # calculating summaries.
    n_components_for_cap = 8 if materialize_components else 4
    n_draws = _cap_draws_for_memory(
        int(n_draws),
        dataset,
        dtype,
        int(max_bytes),
        n_components=n_components_for_cap,
    )

    sampler = posterior_parameter_sampler(
        trace,
        model,
        n_draws=n_draws,
        seed=seed,
        sigma_fixed=sigma_fixed,
    )

    rng = np.random.default_rng(int(seed) + 991)
    n = int(sampler.raw.shape[0])
    n_t = int(np.asarray(dataset.time).size)

    # Reference values come from the complete posterior, not only from the
    # propagation subset.
    raw_all = flatten_raw_parameter_samples(trace)
    raw_median = np.nanmedian(raw_all, axis=0)
    physical_median = transform_raw_to_physical(raw_median, model.parameters)

    y0_all = _flat_var(trace, "y0")
    y0_median = _median_or_none(y0_all)

    bias_all = _flat_var(trace, "bias") if include_bias else None
    bias_median = 0.0 if bias_all is None else float(np.nanmedian(bias_all))

    f00 = _simulate_one(
        model,
        physical_median,
        dataset,
        y0_median,
    ).astype(float)

    deterministic_reference = f00 + bias_median

    parameter_effect = np.empty((n, n_t), dtype=float)
    y0_effect = np.empty((n, n_t), dtype=float)

    for i in range(n):
        y0_i = y0_median if sampler.y0 is None else float(sampler.y0[i])

        f10 = _simulate_one(
            model,
            sampler.physical[i],
            dataset,
            y0_median,
        ).astype(float)

        if sampler.y0 is None:
            f01 = f00
        else:
            f01 = _simulate_one(
                model,
                physical_median,
                dataset,
                y0_i,
            ).astype(float)

        f11 = _simulate_one(
            model,
            sampler.physical[i],
            dataset,
            y0_i,
        ).astype(float)

        parameter_effect[i] = 0.5 * (
            (f10 - f00)
            + (f11 - f01)
        )
        y0_effect[i] = 0.5 * (
            (f01 - f00)
            + (f11 - f10)
        )

    centered_bias = np.zeros(n, dtype=float)
    if include_bias and sampler.bias is not None:
        centered_bias = (
            np.asarray(sampler.bias, dtype=float)
            - bias_median
        )

    reference_position_effect = (
        y0_effect
        + centered_bias[:, None]
    )

    residual_noise = np.zeros((n, n_t), dtype=float)
    if include_noise and sampler.sigma is not None:
        sigma_values = np.maximum(
            np.asarray(sampler.sigma, dtype=float),
            0.0,
        )
        residual_noise = (
            sigma_values[:, None]
            * rng.normal(0.0, 1.0, size=(n, n_t))
        )

    deterministic_out = deterministic_reference.astype(dtype)
    parameter_effect_out = parameter_effect.astype(dtype)
    reference_effect_out = reference_position_effect.astype(dtype)
    residual_noise_out = residual_noise.astype(dtype)

    def absolute_from_effects(*effects: np.ndarray) -> np.ndarray:
        out = np.broadcast_to(
            deterministic_out[None, :],
            (n, n_t),
        ).copy()
        for effect in effects:
            out += effect
        return out.astype(dtype, copy=False)

    summary: dict[str, dict[str, np.ndarray]] = {}

    virtual_components = {
        "physical_parameter_only": (parameter_effect_out,),
        "reference_position_only": (reference_effect_out,),
        "latent_shoreline": (
            parameter_effect_out,
            reference_effect_out,
        ),
        "residual_only": (residual_noise_out,),
        "full": (
            parameter_effect_out,
            reference_effect_out,
            residual_noise_out,
        ),
    }

    for name, effects in virtual_components.items():
        temp = absolute_from_effects(*effects)
        summary[name] = summarise_draws(temp)
        del temp

    summary["parameter_effect"] = summarise_draws(parameter_effect_out)
    summary["reference_position_effect"] = summarise_draws(
        reference_effect_out
    )
    summary["residual_noise"] = summarise_draws(residual_noise_out)

    result: dict[str, Any] = {
        "time": np.asarray(dataset.time),
        "sampler": sampler,
        "deterministic": deterministic_out,
        "deterministic_reference": deterministic_out,
        "parameter_effect": parameter_effect_out,
        "reference_position_effect": reference_effect_out,
        "residual_noise": residual_noise_out,
        "bias_reference": float(bias_median),
        "y0_reference": (
            None if y0_median is None else float(y0_median)
        ),
        "summary": summary,
    }

    if materialize_components:
        result["physical_parameter_only"] = absolute_from_effects(
            parameter_effect_out
        )
        result["reference_position_only"] = absolute_from_effects(
            reference_effect_out
        )
        result["latent_shoreline"] = absolute_from_effects(
            parameter_effect_out,
            reference_effect_out,
        )
        result["model_only"] = result["latent_shoreline"]
        result["residual_only"] = absolute_from_effects(
            residual_noise_out
        )
        result["full"] = absolute_from_effects(
            parameter_effect_out,
            reference_effect_out,
            residual_noise_out,
        )

    return result


def _save_npz(path: Path, **kwargs) -> None:
    clean = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        clean[k] = v
    np.savez_compressed(path, **clean)


def save_bayesian_artifacts(
    *,
    out_dir: str | Path,
    trace,
    ppc=None,
    prior: object | None,
    model: ShorelineModel,
    dataset_predict: TimeSeriesDataset,
    n_draws: int = 1000,
    seed: int = 42,
    sigma_fixed: float | None = None,
    include_bias: bool = True,
    include_noise: bool = True,
    dtype=np.float32,
    max_bytes: int = 500_000_000,
    save_trace: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Save aligned posterior samples and compact three-source propagation.

    The uncertainty propagator stores:

    - ``parameter_effect``;
    - ``reference_position_effect`` combining ``y0`` and centered bias;
    - ``residual_noise`` equal to ``sigma * z``;
    - the posterior-median deterministic reference.

    Absolute ensembles are reconstructed exactly by the plotting and analysis
    utilities.  This avoids persisting duplicate matrices and permits more
    posterior draws within the same memory budget.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if save_trace:
        try:
            trace_path = out / "bayesian_trace.nc"
            trace.to_netcdf(trace_path)
            written["trace"] = trace_path
        except Exception:
            pass
        if ppc is not None:
            try:
                ppc_path = out / "posterior_predictive.nc"
                ppc.to_netcdf(ppc_path)
                written["ppc"] = ppc_path
            except Exception:
                pass

    prop = propagate_uncertainty_components(
        trace,
        model,
        dataset_predict,
        n_draws=n_draws,
        seed=seed,
        sigma_fixed=sigma_fixed,
        include_bias=include_bias,
        include_noise=include_noise,
        dtype=dtype,
        max_bytes=max_bytes,
        materialize_components=False,
    )

    sampler = prop["sampler"]
    n_saved = int(sampler.raw.shape[0])

    prior_raw = (
        sample_prior_raw(prior, n_saved, seed=int(seed) + 123)
        if prior is not None
        else None
    )
    prior_physical = (
        raw_to_physical_samples(prior_raw, model)
        if prior_raw is not None
        else None
    )

    param_path = out / "bayesian_parameter_sampler.npz"
    _save_npz(
        param_path,
        posterior_raw=sampler.raw,
        posterior_physical=sampler.physical,
        prior_raw=prior_raw,
        prior_physical=prior_physical,
        param_names=np.asarray(sampler.param_names, dtype=object),
        draw_index=sampler.draw_index,
        sigma=sampler.sigma,
        bias=sampler.bias,
        y0=sampler.y0,
        nu=sampler.nu,
    )
    written["parameter_sampler"] = param_path

    prop_path = out / "uncertainty_propagator.npz"
    _save_npz(
        prop_path,
        artifact_format_version=np.asarray(4, dtype=np.int16),
        time=np.asarray(prop["time"]),
        draw_index=sampler.draw_index,
        deterministic=np.asarray(prop["deterministic_reference"]),
        deterministic_reference=np.asarray(
            prop["deterministic_reference"]
        ),
        parameter_effect=np.asarray(prop["parameter_effect"]),
        reference_position_effect=np.asarray(
            prop["reference_position_effect"]
        ),
        residual_noise=np.asarray(prop["residual_noise"]),
        bias_reference=np.asarray(
            prop["bias_reference"],
            dtype=float,
        ),
        y0_reference=(
            None
            if prop["y0_reference"] is None
            else np.asarray(prop["y0_reference"], dtype=float)
        ),
    )
    written["uncertainty_propagator"] = prop_path

    summary_kwargs: dict[str, np.ndarray] = {
        "artifact_format_version": np.asarray(4, dtype=np.int16),
        "time": np.asarray(prop["time"]),
    }
    for comp, summ in prop["summary"].items():
        for name, arr in summ.items():
            summary_kwargs[f"{comp}_{name}"] = np.asarray(arr)
    summary_path = out / "uncertainty_propagator_summary.npz"
    _save_npz(summary_path, **summary_kwargs)
    written["uncertainty_summary"] = summary_path

    # Human-readable spread summary.  Use real newline characters so the file
    # can be read directly by pandas and spreadsheet software.
    csv_path = out / "uncertainty_spread_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("component,mean_std,mean_emv,median_emv,p95_emv\n")
        for comp, summ in prop["summary"].items():
            std = np.asarray(summ["std"], dtype=float)
            emv = np.asarray(summ["emv"], dtype=float)
            f.write(
                f"{comp},{np.nanmean(std):.8g},{np.nanmean(emv):.8g},"
                f"{np.nanmedian(emv):.8g},"
                f"{np.nanpercentile(emv, 95):.8g}\n"
            )
    written["spread_metrics"] = csv_path

    meta = {
        "artifact_format_version": 4,
        "model_class": model.__class__.__name__,
        "param_names": sampler.param_names,
        "n_draws_requested": int(n_draws),
        "n_draws_saved": n_saved,
        "include_bias": bool(include_bias),
        "include_noise": bool(include_noise),
        "sigma_fixed": None if sigma_fixed is None else float(sigma_fixed),
        "deterministic_definition": (
            "f(theta_median, y0_median) + bias_median"
        ),
        "parameter_effect_definition": (
            "symmetric Shapley effect of physical parameters"
        ),
        "reference_position_effect_definition": (
            "symmetric Shapley effect of y0 plus "
            "(bias - bias_median)"
        ),
        "residual_noise_definition": "sigma * standard_normal",
        "full_definition": (
            "deterministic_reference + parameter_effect + "
            "reference_position_effect + residual_noise"
        ),
        "interpretation_note": (
            "y0 and additive bias are grouped because a single shoreline "
            "record generally identifies their sum more robustly than either "
            "term separately. sigma is residual predictive uncertainty and "
            "is not structural uncertainty alone unless observation error is "
            "modelled independently."
        ),
        "virtual_components": {
            "physical_parameter_only": (
                "deterministic_reference + parameter_effect"
            ),
            "reference_position_only": (
                "deterministic_reference + reference_position_effect"
            ),
            "latent_shoreline": (
                "deterministic_reference + parameter_effect + "
                "reference_position_effect"
            ),
            "residual_only": (
                "deterministic_reference + residual_noise"
            ),
            "full": (
                "deterministic_reference + parameter_effect + "
                "reference_position_effect + residual_noise"
            ),
        },
    }
    if metadata:
        meta.update(dict(metadata))

    meta_path = out / "metadata.json"
    meta_path.write_text(
        json.dumps(meta, indent=2, default=str),
        encoding="utf-8",
    )
    written["metadata"] = meta_path

    return written


def load_bayesian_artifacts(out_dir: str | Path) -> dict[str, Any]:
    """Load the compact artifacts written by :func:`save_bayesian_artifacts`."""
    out = Path(out_dir)
    data: dict[str, Any] = {}
    for key, filename in {
        "parameter_sampler": "bayesian_parameter_sampler.npz",
        "propagator": "uncertainty_propagator.npz",
        "summary": "uncertainty_propagator_summary.npz",
    }.items():
        path = out / filename
        if path.exists():
            data[key] = np.load(path, allow_pickle=True)
    meta = out / "metadata.json"
    if meta.exists():
        data["metadata"] = json.loads(meta.read_text(encoding="utf-8"))
    return data
