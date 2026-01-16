"""Template: calibrate the Yates09 model with your own dataset.

Expected inputs
---------------
- A model time axis `time` and wave energy `E` aligned with `time`
- An observation time axis `time_obs` and shoreline observations `obs`

For example, you can build `E` from wave height: `E ~ Hs**2`.
"""

from pathlib import Path
import sys

import numpy as np
import xarray as xr
import pandas as pd

# Allow running this example without installing the package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slmcal.data import TimeSeriesDataset
from slmcal.core.workflow import CalibrationWorkflow
from slmcal.models.yates09 import Yates09Model
from slmcal.optimization.nsga2 import NSGA2Config


def load_private_dataset(path_nc: str, transect_index: int = 0) -> TimeSeriesDataset:
    ds = xr.open_dataset(path_nc)
    # TODO: adjust variable names to your netCDF conventions
    time = pd.to_datetime(ds["time"].values)
    time_obs = pd.to_datetime(ds["time_obs"].values)

    # Example: energy forcing
    # If you only have Hs and Tp:
    # E = ds["hs"].values**2
    # or: P = E * ds["tp"].values
    E = ds["E"].values

    # shoreline observations (choose one transect or an average)
    obs = ds["obs"].values
    if obs.ndim == 2:
        obs = obs[:, transect_index]

    ds.close()

    return TimeSeriesDataset(
        time=np.asarray(time),
        forcings={"E": np.asarray(E, dtype=float)},
        obs_time=np.asarray(time_obs),
        obs=np.asarray(obs, dtype=float),
        y0=float(obs[0]),
    )


def main():
    # --- user settings ---
    path_nc = "REPLACE_WITH_YOUR_NETCDF.nc"

    dataset = load_private_dataset(path_nc, transect_index=0)
    model = Yates09Model()  # adjust bounds if needed

    wf = CalibrationWorkflow(model=model, dataset=dataset)

    # 1) NSGA-II pre-calibration
    nsga = wf.precalibrate_nsga2(
        cfg=NSGA2Config(n_generations=80, pop_size=200, n_restarts=10, random_seed=42),
        progress=True,
    )

    wf.build_prior_from_nsga2(cov_scale=30.0, use_valid_only=True)

    # 2) Bayesian calibration (requires pymc)
    trace, ppc = wf.bayesian_calibrate(
        draws=2000,
        tune=2000,
        chains=4,
        sigma=7.0,  # set to your observation error or use estimate_sigma=True
        likelihood="normal",
        include_bias=True,
    )

    # TODO: save `trace`, diagnostics, and ensemble predictions
    print(trace)


if __name__ == "__main__":
    main()
