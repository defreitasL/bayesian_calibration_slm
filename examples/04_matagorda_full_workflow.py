"""
Matagorda
========
Author: Lucas de Freitas Pereira

Run
---
python examples/04_matagorda_full_workflow.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import xarray as xr

# Allow running this example without installing the package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slmcal.data import TimeSeriesDataset
from slmcal.models.yates09 import Yates09Model
from slmcal.optimization.nsga2_fastopt import FastOptNSGA2Config, run_nsga2_fastopt
from slmcal.preprocess import preprocess_legacy_yates
from slmcal.io.legacy import save_precalibration_legacy, save_bayes_legacy
from slmcal.bayes.priors import select_valid_individuals, subsample
from slmcal.bayes.pymc import (
    bayesian_calibrate,
    fit_mvn_prior_from_nsga2,
    fit_copula_kde_prior_from_nsga2,
)
from slmcal.models.base import transform_raw_to_physical


# -----------------------------
# User-configurable variables
# -----------------------------

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "data" / "Matagorda_data.nc"

OUT_DIR = HERE / ".." / "outputs" / "Matagorda" / "Y09"
OUT_DIR = OUT_DIR.resolve()

START_DATE = "1990-01-01"
END_DATE = "2024-01-01"

# Yates raw-parameterisation for "a":
#   - "linear": a = -raw[0]        
#   - "log"   : a = -exp(raw[0])   (ensures positivity)
A_MODE = "linear"

# NSGA-II
METRICS = ("kge", "pbias", "spearman")
NSGA_CFG = FastOptNSGA2Config(
    num_generations=150,      # increase for real runs (e.g., 150)
    population_size=2000,     # increase for real runs (e.g., 2000)
    cross_prob=0.8,
    mutation_rate=0.2,
    regeneration_rate=0.15,
    pressure=2,
    kstop=100,
    pcento=0.001,
    peps=1e-4,
    n_restarts=20,            # increase for real runs (e.g., 30)
    random_seed=42,
)

# Prior kind: "mvn" or "copula_kde"
PRIOR_KIND = "copula_kde"
VALIDITY_THRESHOLDS = (0.0, 20.0, 0.4)  # [kge, pbias, spearman] thresholds (legacy)

# Bayesian calibration (PyMC + black-box Op)
LABEL = "Matagorda_Y09"
CHAINS = 4
DRAWS = 1000
TUNE = 20000
RANDOM_SEED = 42

SIGMA_FIXED = 50.0
ESTIMATE_SIGMA = True

INCLUDE_BIAS = False  # <---- option bias on/off
CORES = 8

# Copula-KDE subsampling (keeps it fast even with huge Pareto fronts)
COPULA_MAX_POINTS = 20000
COPULA_SUBSAMPLE_METHOD = "random"  # or "max_dissimilarity"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------
    # Load data + legacy preprocessing
    # -----------------------------------
    ds = xr.open_dataset(DATA_FILE)
    prep = preprocess_legacy_yates(ds, start_date=START_DATE, end_date=END_DATE)

    # split dataset for calibration (legacy window)
    ds_cal = TimeSeriesDataset(
        time=prep.time_splited,
        forcings={"E": prep.E_splited},
        obs_time=prep.time_obs_splited,
        obs=prep.Obs_splited,
        y0=float(prep.Obs_splited[0]),
        idx_obs=prep.idx_obs_splited,
        dt=prep.dt_splited,
    )

    # full dataset for posterior predictive
    ds_full = TimeSeriesDataset(
        time=prep.time,
        forcings={"E": prep.E},
        obs_time=prep.time_obs,
        obs=prep.Obs,
        y0=float(prep.Obs[0]),
        idx_obs=None,
        dt=prep.dt,
    )

    model = Yates09Model(a_mode=A_MODE)

    # -----------------------------------
    # 1) NSGA-II pre-calibration
    # -----------------------------------
    nsga_res = run_nsga2_fastopt(model=model, dataset=ds_cal, metrics=METRICS, cfg=NSGA_CFG)

    # Legacy file name
    precal_file = OUT_DIR / "results_Yates.nc"
    save_precalibration_legacy(
        precal_file,
        all_individuals=nsga_res.individuals_raw,
        all_objectives=nsga_res.objectives,
        prep=prep,
        e_best_par=nsga_res.best_individual_raw,
        e_best_fit=nsga_res.best_objectives,
    )
    print(f"[OK] Saved pre-calibration: {precal_file}")

    # -----------------------------------
    # 2) Fit empirical prior from Pareto set
    # -----------------------------------
    valid_ind, valid_obj = select_valid_individuals(
        nsga_res.individuals_raw, nsga_res.objectives, thresholds=VALIDITY_THRESHOLDS
    )

    if valid_ind.shape[0] < 50:
        # fallback to all individuals if thresholds are too strict
        valid_ind = nsga_res.individuals_raw
        valid_obj = nsga_res.objectives

    if PRIOR_KIND.lower() == "mvn":
        prior = fit_mvn_prior_from_nsga2(valid_ind, cov_scale=30.0)
    elif PRIOR_KIND.lower() in ("copula_kde", "copula-kde", "kde"):
        valid_ind_sub = subsample(
            valid_ind,
            max_points=COPULA_MAX_POINTS,
            method=COPULA_SUBSAMPLE_METHOD,  # type: ignore[arg-type]
            seed=42,
        )
        prior = fit_copula_kde_prior_from_nsga2(
            valid_ind_sub,
            bw_method="scott",
            grid_size=1024,
            method="factor",
            explained_var=0.99,
            random_seed=RANDOM_SEED,
        )
    else:
        raise ValueError(f"Unknown PRIOR_KIND: {PRIOR_KIND}")

    # -----------------------------------
    # 3) Bayesian calibration (PyMC + black-box forward model)
    # -----------------------------------
    trace, ppc = bayesian_calibrate(
        model=model,
        dataset=ds_cal,
        prior=prior,
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        random_seed=RANDOM_SEED,
        sigma=SIGMA_FIXED,
        estimate_sigma=ESTIMATE_SIGMA,
        include_bias=INCLUDE_BIAS,
        cores=CORES,
    )

    posterior_raw = trace.posterior["raw_par"].values  # (chains, draws, n_params)
    if ESTIMATE_SIGMA:
        sigma_arr = trace.posterior["sigma"].values
    else:
        sigma_arr = np.full(posterior_raw.shape[:2], float(SIGMA_FIXED), dtype=float)

    if INCLUDE_BIAS:
        mu_bias = trace.posterior["mu_bias"].values
        sigma_bias = trace.posterior["sigma_bias"].values
        bias = trace.posterior["bias"].values
    else:
        mu_bias = sigma_bias = bias = None

    # -----------------------------------
    # 4) Posterior predictive on full period
    # -----------------------------------
    pars = posterior_raw  # (chains, draws, 4)
    chains, draws, npar = pars.shape
    flat = pars.reshape(chains * draws, npar)

    new_lines = []
    for p in flat:
        # run on full time vector
        phys = transform_raw_to_physical(p, model.parameters)
        y = model.simulate(phys, ds_full)
        new_lines.append(y)

    new_lines = np.asarray(new_lines, dtype=float).T  # (time, samples)
    per5 = np.percentile(new_lines, 5, axis=1)
    per50 = np.percentile(new_lines, 50, axis=1)
    per95 = np.percentile(new_lines, 95, axis=1)
    mini = np.min(new_lines, axis=1)
    maxi = np.max(new_lines, axis=1)

    sigma_mean = float(np.mean(sigma_arr))
    uncertainty_lower = prep.Obs - sigma_mean
    uncertainty_upper = prep.Obs + sigma_mean

    # -----------------------------------
    # 5) Save legacy-compatible Bayes output
    # -----------------------------------
    bayes_file = OUT_DIR / f"results_{LABEL}.nc"
    save_bayes_legacy(
        bayes_file,
        label=LABEL,
        all_individuals=nsga_res.individuals_raw,
        all_objectives=nsga_res.objectives,
        all_valid_individuals=valid_ind,
        all_valid_objectives=valid_obj,
        posterior_raw=posterior_raw,
        sigma=sigma_arr,
        Obs=prep.Obs,
        E=prep.E,
        new_lines=new_lines,
        per50=per50,
        per5=per5,
        per95=per95,
        mini=mini,
        maxi=maxi,
        uncertainty_lower=uncertainty_lower,
        uncertainty_upper=uncertainty_upper,
        time=prep.time,
        dt=prep.dt,
        time_obs=prep.time_obs,
        idx_obs_len=int(prep.idx_obs.shape[0]),
        Obs_splited_len=int(prep.Obs_splited.shape[0]),
        start_date=prep.start_date,
        end_date=prep.end_date,
        mu_bias=mu_bias,
        sigma_bias=sigma_bias,
        bias=bias,
    )
    print(f"[OK] Saved bayesian calibration: {bayes_file}")


if __name__ == "__main__":
    main()
