"""
Angourie
========
Author: Lucas de Freitas Pereira
 
Run
---
python examples/02_angourie_full_workflow.py
"""
 
from __future__ import annotations
 
from pathlib import Path
import sys
 
import numpy as np
import xarray as xr
import arviz as az
 
# Allow running this example without installing the package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
 
from slmcal.data import TimeSeriesDataset
from slmcal.models.yates09 import Yates09Model
from slmcal.models.vitousek21 import Vitousek21YatesModel
from slmcal.optimization.nsga2_fastopt import FastOptNSGA2Config, run_nsga2_fastopt
from slmcal.preprocess import preprocess_legacy_yates
from slmcal.io.legacy import save_precalibration_legacy
from slmcal.bayes.priors import select_valid_individuals, subsample
from slmcal.bayes.pymc import (
    bayesian_calibrate,
    fit_mvn_prior_from_nsga2,
    fit_copula_kde_prior_from_nsga2,
    estimate_studentt_nu,
)
from slmcal.models.base import transform_raw_to_physical
from slmcal.plotting import make_default_calibration_plots
from slmcal.bayes.outputs import save_bayesian_artifacts
from slmcal.preprocess_examples import sample_posterior_draws
 
# -----------------------------
# User-configurable variables
# -----------------------------
 
HERE = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "Angourie_CoastSat_NSWWaves_default.nc"
 
OUT_DIR = HERE / ".." / "outputs" / "Angourie" / "Vitousek21Yates"
OUT_DIR = OUT_DIR.resolve()
 
START_DATE = "1990-01-01"
# NOTE: preprocessing now uses `split_date` as the calibration cutoff, and `end_date=None` means end of record.
# Calibration/validation split:
#   calibration: [START_DATE, SPLIT_DATE)
#   validation:  [SPLIT_DATE, end of record]
SPLIT_DATE = "2010-01-01"
 
# Optional hard end-date crop for the forcing (None = use end of record)
END_DATE = None
 
# Yates raw-parameterisation for "a":
#   - "linear": a = -raw[0]        (matches Ensemble_Yates.py)
#   - "log"   : a = -exp(raw[0])   (used in some Bayes scripts)
A_MODE = "linear"

# Model selector:
#   - "yates09"    : original a/b/C+/- parameterisation
#   - "vitousek21" : Vitousek et al. (2021) physically interpretable reformulation
MODEL_KIND = "vitousek21"
VITOUSEK_SPLIT_TIMESCALES = False  # True keeps separate DeltaT_acc/DeltaT_ero
VITOUSEK_INTEGRATION = "exact"
 
# NSGA-II
USE_PRECALIBRATED_DS = False  # set to False to re-run NSGA-II pre-calibration
METRICS = ("kge", "pbias", "spearman")
NSGA_CFG = FastOptNSGA2Config(
    num_generations=50,      # increase for real runs (e.g., 150)
    population_size=500,     # increase for real runs (e.g., 2000)
    cross_prob=0.8, 
    mutation_rate=0.2,
    regeneration_rate=0.15,
    pressure=2,
    kstop=100,
    pcento=0.001,
    peps=1e-4,
    n_restarts=8,            # increase for real runs (e.g., 30)
    random_seed=42,
)
 
# Prior kind: "mvn" or "copula_kde"
PRIOR_KIND = "copula_kde"
VALIDITY_THRESHOLDS = (0.35, 2.0, 0.5)  # [kge, pbias, spearman] thresholds (legacy)
 
# Bayesian calibration (PyMC + black-box Op)
LIKELIHOOD_KIND = "normal"  # "normal" or "studentt"
AUTO_NU_FROM_NSGA2_BEST = False  # empirical-Bayes nu from NSGA2-best residuals
NU_FIXED: float | None = None  # set a number to force nu, else auto if enabled
NU_BOUNDS = (2.0, 50.0)
 
LABEL = "Angourie_Vitousek21Yates"
CHAINS = 6
DRAWS = 15000
TUNE = 15000
RANDOM_SEED = 42
 
SIGMA_FIXED = 15.0
ESTIMATE_SIGMA = True
 
INCLUDE_BIAS = True  # <---- option bias on/off
CORES = 6
 
ESTIMATE_INITIAL_POSITION = True  # whether to estimate y0 in Bayesian calibration
 
# Copula-KDE subsampling (keeps it fast even with huge Pareto fronts)
COPULA_MAX_POINTS = 40000
COPULA_SUBSAMPLE_METHOD = "random"  # or "max_dissimilarity"
 
MAKE_DEFAULT_PLOTS = True
SAVE_BAYESIAN_ARTIFACTS = True
N_PRED_DRAWS = 1000  # bounded predictive sample for full-period envelopes / diagnostics
PRED_DTYPE = np.float32
PRED_MAX_BYTES = 8_000_000_000
 
 
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
 
    ds = xr.open_dataset(DATA_FILE)
    prep = preprocess_legacy_yates(ds, start_date=START_DATE, split_date=SPLIT_DATE, end_date=END_DATE)
 
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
 
    model_kind = str(MODEL_KIND).lower().strip()
    if model_kind in {"vitousek21", "v21", "vitousek_yates", "vitousek21_yates"}:
        model = Vitousek21YatesModel(
            split_timescales=VITOUSEK_SPLIT_TIMESCALES,
            integration=VITOUSEK_INTEGRATION,
        )
        print("[INFO] Using Vitousek21YatesModel")
    elif model_kind in {"yates09", "y09", "legacy"}:
        model = Yates09Model(a_mode=A_MODE)
        print("[INFO] Using legacy Yates09Model")
    else:
        raise ValueError(f"Unknown MODEL_KIND={MODEL_KIND!r}")
 
    # -----------------------------------
    # 1) NSGA-II pre-calibration
    # -----------------------------------
    if USE_PRECALIBRATED_DS:
        from slmcal.io.legacy import load_precalibration_legacy
 
        nsga_res = load_precalibration_legacy(
            OUT_DIR / "results_Yates.nc",
            metrics=METRICS,
        )
        print(f"[OK] Loaded NSGA-II results from file: {OUT_DIR / 'results_Yates.nc'}")
    else:
        nsga_res = run_nsga2_fastopt(model=model, dataset=ds_cal, metrics=METRICS, cfg=NSGA_CFG)
 
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
 
    if valid_ind.shape[0] < 10:
        # fallback to all individuals if thresholds are too strict
        valid_ind = nsga_res.individuals_raw
        valid_obj = nsga_res.objectives
 
    if PRIOR_KIND.lower() == "mvn":
        prior = fit_mvn_prior_from_nsga2(valid_ind, cov_scale=5.0)
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
    nu_for_bayes = None
    if LIKELIHOOD_KIND.lower() in {"studentt", "student-t", "t"}:
        if NU_FIXED is not None:
            nu_for_bayes = float(NU_FIXED)
        elif AUTO_NU_FROM_NSGA2_BEST:
            best_raw = getattr(nsga_res, "best_individual_raw", None)
            if best_raw is None:
                best_raw = valid_ind[0]
            phys_best = transform_raw_to_physical(best_raw, model.parameters)
 
            y_best = model.simulate(phys_best, ds_cal)
            y_best_obs = np.asarray(y_best[ds_cal.idx_obs], dtype=float)
 
            resid = np.asarray(ds_cal.obs, dtype=float) - y_best_obs
            resid = resid - np.median(resid)  # remove bias-like offset
            nu_for_bayes = estimate_studentt_nu(resid, nu_bounds=NU_BOUNDS)
            print(f"[INFO] Using fixed nu from NSGA2-best residuals: {nu_for_bayes:.3f}")
 
    trace, ppc = bayesian_calibrate(
        model=model,
        dataset=ds_cal,
        prior=prior,
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        random_seed=RANDOM_SEED,
        likelihood=LIKELIHOOD_KIND,
        nu_fixed=nu_for_bayes,
        nu_bounds=NU_BOUNDS,
        sigma=SIGMA_FIXED,
        estimate_sigma=ESTIMATE_SIGMA,
        include_bias=INCLUDE_BIAS,
        estimate_initial_position=ESTIMATE_INITIAL_POSITION,
        cores=CORES,
    )
 
 
    posterior_raw = trace.posterior["raw_par"].values  # (chains, draws, n_params)
    if ESTIMATE_SIGMA:
        sigma_arr = trace.posterior["sigma"].values
        print(az.summary(trace, var_names=['sigma', 'log_sigma']))
    else:
        sigma_arr = np.full(posterior_raw.shape[:2], float(SIGMA_FIXED), dtype=float)
 
    if INCLUDE_BIAS:
        mu_bias = sigma_bias = None
        bias = trace.posterior["bias"].values
        print(az.summary(trace, var_names=['bias']))
    else:
        mu_bias = sigma_bias = bias = None
 
    if ESTIMATE_INITIAL_POSITION:
        y0_arr = trace.posterior["y0"].values
        print(az.summary(trace, var_names=['y0']))
 
    # -----------------------------------
    # 4) Bounded posterior predictive on full period
    # -----------------------------------
    y_draws, _ = sample_posterior_draws(
        trace,
        model,
        ds_full,
        n_draws=N_PRED_DRAWS,
        seed=RANDOM_SEED,
        dtype=PRED_DTYPE,
        max_bytes=PRED_MAX_BYTES,
        verbose=True,
        progress_every=100,
        mode="ppc",
        sigma_fixed=(SIGMA_FIXED if not ESTIMATE_SIGMA else None),
        include_bias=INCLUDE_BIAS,
        noise_at_obs_only=False,
    )
 
    per1 = np.nanpercentile(y_draws, 1, axis=0)
    per5 = np.nanpercentile(y_draws, 5, axis=0)
    per10 = np.nanpercentile(y_draws, 10, axis=0)
    per50 = np.nanpercentile(y_draws, 50, axis=0)
    per90 = np.nanpercentile(y_draws, 90, axis=0)
    per95 = np.nanpercentile(y_draws, 95, axis=0)
    per99 = np.nanpercentile(y_draws, 99, axis=0)
    mini = np.nanmin(y_draws, axis=0)
    maxi = np.nanmax(y_draws, axis=0)

    if SAVE_BAYESIAN_ARTIFACTS:
        artifacts_dir = OUT_DIR / "bayesian_artifacts"
        written = save_bayesian_artifacts(
            out_dir=artifacts_dir,
            trace=trace,
            ppc=ppc,
            prior=prior,
            model=model,
            dataset_predict=ds_full,
            n_draws=N_PRED_DRAWS,
            seed=RANDOM_SEED,
            sigma_fixed=(SIGMA_FIXED if not ESTIMATE_SIGMA else None),
            include_bias=INCLUDE_BIAS,
            include_noise=True,
            dtype=PRED_DTYPE,
            max_bytes=PRED_MAX_BYTES,
            metadata={
                "label": LABEL,
                "model_kind": MODEL_KIND,
                "split_date": SPLIT_DATE,
                "start_date": START_DATE,
                "end_date": END_DATE,
            },
        )
        print("[OK] Saved Bayesian artifacts:")
        for key, path in written.items():
            print(f"  - {key}: {path}")
 
    if MAKE_DEFAULT_PLOTS:
        prior_raw_for_plot = valid_ind_sub if 'valid_ind_sub' in locals() else valid_ind
        make_default_calibration_plots(
            out_dir=OUT_DIR,
            prior_raw=prior_raw_for_plot,
            posterior_raw=posterior_raw.reshape(-1, posterior_raw.shape[-1]),
            time=prep.time,
            per5=per5,
            per10=per10,
            per90=per90,
            per50=per50,
            per95=per95,
            per1=per1,
            per99=per99,
            mini=mini,
            maxi=maxi,
            draws=y_draws,
            obs_time=prep.time_obs,
            obs=prep.Obs,
            # Validation support: SPLIT_DATE is the calibration cutoff; validation goes to end of record
            obs_mask_cal=getattr(prep, "mask_cal", None),
            obs_mask_val=getattr(prep, "mask_val", None),
            split_date=np.datetime64(SPLIT_DATE),
            trace=trace,
            ppc=ppc,
            param_names=[p.name for p in model.parameters],
            label=LABEL,
        )
        print(f"[OK] Saved default plots to: {OUT_DIR}")
 
 
if __name__ == "__main__":
    main()