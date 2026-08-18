from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, overload

import numpy as np
import pandas as pd
import xarray as xr

from slmcal.preprocess import PreprocessResult
from slmcal.optimization.nsga2_fastopt import FastOptNSGA2Result
import gc


def _get_any(ds: xr.Dataset, name: str) -> np.ndarray:
    """Get an array from coords or data_vars (legacy files store most fields as coords)."""
    if name in ds.coords:
        return np.asarray(ds.coords[name].values)
    if name in ds.data_vars:
        return np.asarray(ds[name].values)
    raise KeyError(
        f"Legacy dataset is missing '{name}'. Available coords={list(ds.coords)}; data_vars={list(ds.data_vars)}"
    )


def _to_datetime64_ns(x: np.ndarray) -> np.ndarray:
    """Robust conversion to numpy datetime64[ns] (handles strings/cftime-like objects)."""
    try:
        # Already datetime64
        if np.issubdtype(np.asarray(x).dtype, np.datetime64):
            return np.asarray(x).astype("datetime64[ns]")
    except Exception:
        pass
    return pd.to_datetime(np.asarray(x)).to_numpy(dtype="datetime64[ns]")


def _to_datetime64_scalar(x: np.ndarray) -> np.datetime64:
    """Robust conversion of a scalar-like value to numpy datetime64."""
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("Cannot convert empty array to datetime64")
    v = x.reshape(-1)[0]
    return np.datetime64(pd.to_datetime(v))


def save_precalibration_legacy(
    out_file: str | Path,
    all_individuals: np.ndarray,
    all_objectives: np.ndarray,
    prep: PreprocessResult,
    e_best_par: np.ndarray,
    e_best_fit: np.ndarray,
    engine: str = "netcdf4",
) -> None:
    """Save results NSGA-II legacy file."""
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    all_individuals = np.asarray(all_individuals, dtype=float)
    all_objectives = np.asarray(all_objectives, dtype=float)

    ds = xr.Dataset(
        {
            "all_individuals": (["solutions", "parameters"], all_individuals),
            "all_objectives": (["solutions", "metrics"], all_objectives),
        },
        coords={
            "solutions": range(all_individuals.shape[0]),
            "parameters": range(all_individuals.shape[1]),
            "metrics": range(all_objectives.shape[1]),
            "idx_obs": np.asarray(prep.idx_obs, dtype=int),
            "idx_obs_splited": np.asarray(prep.idx_obs_splited, dtype=int),
            "Obs_splited": np.asarray(prep.Obs_splited, dtype=float),
            "e_best_par": np.asarray(e_best_par, dtype=float),
            "e_best_fit": np.asarray(e_best_fit, dtype=float),
            "time": prep.time,
            "time_splited": prep.time_splited,
            "time_obs": prep.time_obs,
            "time_obs_splited": prep.time_obs_splited,
            "E": np.asarray(prep.E, dtype=float),
            "E_splited": np.asarray(prep.E_splited, dtype=float),
            "dt": np.asarray(prep.dt, dtype=float),
            "dt_splited": np.asarray(prep.dt_splited, dtype=float),
            "Obs": np.asarray(prep.Obs, dtype=float),
            "start_date": np.asarray(prep.start_date),
            "end_date": np.asarray(prep.end_date),
        },
        attrs={
            "variables characteristics": "none, none, exp, exp, none",
        },
    )

    ds.to_netcdf(out_file, engine=engine)
    ds.close()
    del ds
    gc.collect()


@overload
def load_precalibration_legacy(
    path: str | Path,
    *,
    metrics: Sequence[str] | None = None,
    return_preprocess: bool = False,
    engine: str | None = None,
) -> FastOptNSGA2Result: ...


@overload
def load_precalibration_legacy(
    path: str | Path,
    *,
    metrics: Sequence[str] | None = None,
    return_preprocess: bool = True,
    engine: str | None = None,
) -> Tuple[FastOptNSGA2Result, PreprocessResult]: ...


def load_precalibration_legacy(
    path: str | Path,
    *,
    metrics: Sequence[str] | None = None,
    return_preprocess: bool = False,
    engine: str | None = None,
) -> FastOptNSGA2Result | Tuple[FastOptNSGA2Result, PreprocessResult]:
    """Load an existing legacy `results_Yates.nc` (NSGA-II pre-calibration output).

    This avoids re-running NSGA-II when you are iterating on Bayesian settings.

    Parameters
    ----------
    path
        Path to the legacy NetCDF produced by `save_precalibration_legacy`
    metrics
        Optional metric names to attach to the returned result object. Legacy files
        do not store metric names, only the objective values.
    return_preprocess
        If True, also reconstruct and return the legacy preprocessing arrays
        as a :class:`slmcal.preprocess.PreprocessResult`.
    engine
        Optional xarray engine override.

    Returns
    -------
    FastOptNSGA2Result
        With `.individuals_raw`, `.objectives`, `.best_individual_raw`, `.best_objectives`.
    (FastOptNSGA2Result, PreprocessResult)
        If `return_preprocess=True`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    ds = xr.open_dataset(path, engine=engine)
    try:
        all_ind = np.asarray(ds["all_individuals"].values, dtype=float)
        all_obj = np.asarray(ds["all_objectives"].values, dtype=float)

        best_ind = np.asarray(_get_any(ds, "e_best_par"), dtype=float)
        best_fit = np.asarray(_get_any(ds, "e_best_fit"), dtype=float)

        m = tuple(metrics) if metrics is not None else tuple(f"m{i}" for i in range(all_obj.shape[1]))

        res = FastOptNSGA2Result(
            metrics=m,
            best_individual_raw=best_ind,
            best_objectives=best_fit,
            individuals_raw=all_ind,
            objectives=all_obj,
        )

        if not return_preprocess:
            return res

        prep = PreprocessResult(
            time=_to_datetime64_ns(_get_any(ds, "time")),
            E=np.asarray(_get_any(ds, "E"), dtype=float),
            dt=np.asarray(_get_any(ds, "dt"), dtype=float),
            time_obs=_to_datetime64_ns(_get_any(ds, "time_obs")),
            Obs=np.asarray(_get_any(ds, "Obs"), dtype=float),
            start_date=_to_datetime64_scalar(_get_any(ds, "start_date")),
            end_date=_to_datetime64_scalar(_get_any(ds, "end_date")),
            time_splited=_to_datetime64_ns(_get_any(ds, "time_splited")),
            E_splited=np.asarray(_get_any(ds, "E_splited"), dtype=float),
            dt_splited=np.asarray(_get_any(ds, "dt_splited"), dtype=float),
            time_obs_splited=_to_datetime64_ns(_get_any(ds, "time_obs_splited")),
            Obs_splited=np.asarray(_get_any(ds, "Obs_splited"), dtype=float),
            idx_obs=np.asarray(_get_any(ds, "idx_obs"), dtype=int),
            idx_obs_splited=np.asarray(_get_any(ds, "idx_obs_splited"), dtype=int),
        )

        return res, prep
    finally:
        ds.close()


def save_bayes_legacy(
    out_file: str | Path,
    *,
    label: str,
    all_individuals: np.ndarray,
    all_objectives: np.ndarray,
    all_valid_individuals: np.ndarray,
    all_valid_objectives: np.ndarray,
    posterior_raw: np.ndarray,  # (chains, draws, parameters)
    sigma: np.ndarray,          # (chains, draws)
    Obs: np.ndarray,
    E: np.ndarray,
    new_lines: np.ndarray,      # (time, samples)
    per50: np.ndarray,
    per5: np.ndarray,
    per95: np.ndarray,
    per1: np.ndarray,
    per99: np.ndarray,
    mini: np.ndarray,
    maxi: np.ndarray,
    uncertainty_lower: np.ndarray,
    uncertainty_upper: np.ndarray,
    time: np.ndarray,
    dt: np.ndarray,
    time_obs: np.ndarray,
    idx_obs_len: int,
    Obs_splited_len: int,
    start_date: np.datetime64,
    end_date: np.datetime64,
    e_best_par_len: int = 4,
    e_best_fit_len: int = 2,
    mu_bias: Optional[np.ndarray] = None,
    sigma_bias: Optional[np.ndarray] = None,
    bias: Optional[np.ndarray] = None,
    engine: str = "netcdf4",
) -> None:
    """Save `results_{label}.nc`."""
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    all_individuals = np.asarray(all_individuals, dtype=float)
    all_objectives = np.asarray(all_objectives, dtype=float)
    all_valid_individuals = np.asarray(all_valid_individuals, dtype=float)
    all_valid_objectives = np.asarray(all_valid_objectives, dtype=float)

    posterior_raw = np.asarray(posterior_raw, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    chains, draws, parameters = posterior_raw.shape
    total_samples = int(new_lines.shape[1])

    data_vars = {
        "all_individuals": (["solutions", "parameters"], all_individuals),
        "all_objectives": (["solutions", "metrics"], all_objectives),
        "posterior": (["chains", "draws", "parameters"], posterior_raw),
        "sigma": (["chains", "draws"], sigma),
        "all_valid_individuals": (["valid_solutions", "parameters"], all_valid_individuals),
        "all_valid_objectives": (["valid_solutions", "metrics"], all_valid_objectives),
        "Obs": (["time_obs"], np.asarray(Obs, dtype=float)),
        "E": (["time"], np.asarray(E, dtype=float)),
        "new_lines": (["time", "samples"], np.asarray(new_lines, dtype=float)),
        "per50": (["time"], np.asarray(per50, dtype=float)),
        "per5": (["time"], np.asarray(per5, dtype=float)),
        "per95": (["time"], np.asarray(per95, dtype=float)),
        "per1": (["time"], np.asarray(per1, dtype=float)),
        "per99": (["time"], np.asarray(per99, dtype=float)),
        "mini": (["time"], np.asarray(mini, dtype=float)),
        "maxi": (["time"], np.asarray(maxi, dtype=float)),
        "uncertainty_lower": (["time_obs"], np.asarray(uncertainty_lower, dtype=float)),
        "uncertainty_upper": (["time_obs"], np.asarray(uncertainty_upper, dtype=float)),
    }

    # optional bias 
    if mu_bias is not None and sigma_bias is not None and bias is not None:
        data_vars["mu_bias"] = (["chains", "draws"], np.asarray(mu_bias, dtype=float))
        data_vars["sigma_bias"] = (["chains", "draws"], np.asarray(sigma_bias, dtype=float))
        data_vars["bias"] = (["chains", "draws"], np.asarray(bias, dtype=float))

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "solutions": range(all_individuals.shape[0]),
            "valid_solutions": range(all_valid_individuals.shape[0]),
            "parameters": range(all_individuals.shape[1]),  # overwritten below (matches legacy)
            "metrics": range(all_objectives.shape[1]),
            "idx_obs": range(int(idx_obs_len)),
            "Obs_splited": range(int(Obs_splited_len)),
            "e_best_par": range(int(e_best_par_len)),
            "e_best_fit": range(int(e_best_fit_len)),
            "time": time,
            "dt": dt,
            "time_obs": time_obs,
            "total_samples": range(total_samples),
            "chains": range(chains),
            "draws": range(draws),
            "parameters": range(parameters),  # final overwrite (matches legacy behaviour)
            "start_date": np.asarray(start_date),
            "end_date": np.asarray(end_date),
        },
        attrs={
            "description": "Bayesian Yates 2009 model results",
        },
    )

    ds.to_netcdf(out_file, engine=engine)
    ds.close()
    del ds
    gc.collect()
