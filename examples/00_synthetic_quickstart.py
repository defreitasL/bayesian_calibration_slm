from pathlib import Path
import sys

import numpy as np

# Allow running this example without installing the package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slmcal.data import TimeSeriesDataset
from slmcal.optimization.nsga2 import NSGA2Config
from slmcal.core.workflow import CalibrationWorkflow
from slmcal.models.base import ParameterSpec


class ToyLinearModel:
    """y_{t+1} = y_t + dt*(b + c*E_t)."""

    def __init__(self):
        self._parameters = [
            ParameterSpec("b", (-2.0, 2.0), transform=lambda x: x),
            ParameterSpec("c", (-2.0, 2.0), transform=lambda x: x),
        ]

    @property
    def parameters(self):
        return self._parameters

    def simulate(self, physical_params: np.ndarray, dataset: TimeSeriesDataset) -> np.ndarray:
        b, c = map(float, physical_params)
        E = np.asarray(dataset.forcings["E"], dtype=float)
        dt = np.asarray(dataset.dt, dtype=float)

        y = np.empty_like(E, dtype=float)
        y[0] = float(dataset.y0)
        for i in range(len(dt)):
            y[i + 1] = y[i] + dt[i] * (b + c * E[i])
        return y


def main():
    rng = np.random.default_rng(7)

    # time axis (hours)
    n = 365
    time = np.arange(n, dtype=float)
    E = 0.5 + 0.3 * np.sin(2 * np.pi * time / 30.0) + 0.1 * rng.standard_normal(n)

    # ground truth
    b_true, c_true = 0.05, -0.08
    y0 = 100.0
    dt = np.ones(n - 1)
    y = np.empty(n)
    y[0] = y0
    for i in range(n - 1):
        y[i + 1] = y[i] + dt[i] * (b_true + c_true * E[i])

    # observations (every 7 steps)
    obs_idx = np.arange(0, n, 7)
    obs_time = time[obs_idx]
    obs = y[obs_idx] + 0.5 * rng.standard_normal(obs_idx.size)

    ds = TimeSeriesDataset(
        time=time,
        forcings={"E": E},
        obs_time=obs_time,
        obs=obs,
        y0=y0,
    )

    model = ToyLinearModel()
    wf = CalibrationWorkflow(model=model, dataset=ds)

    print("Running NSGA-II pre-calibration...")
    nsga = wf.precalibrate_nsga2(
        metrics=("kge", "spbias", "spearman"),
        cfg=NSGA2Config(
            n_generations=40,
            pop_size=120,
            n_restarts=5,
            random_seed=1,
        ),
        progress=True,
    )

    valid, _ = nsga.select_valid(kge_min=0.0, spearman_min=0.0, spbias_max=50.0)
    print(f"Total solutions: {nsga.individuals_raw.shape[0]} | Valid: {valid.shape[0]}")

    print("Building empirical prior from NSGA-II...")
    wf.build_prior_from_nsga2(kind="mvn", cov_scale=10.0, use_valid_only=True, validity_thresholds={"spbias_max": 50.0})
    # Or use the empirical KDE prior (requires SciPy):
    # wf.build_prior_from_nsga2(kind="kde", bw_method="scott")
    print("Prior summary:", wf.prior_summary())

    print("Attempting Bayesian calibration (requires pymc)...")
    try:
        trace, _ = wf.bayesian_calibrate(
            draws=800,
            tune=800,
            chains=2,
            sigma=0.5,
            likelihood="normal",
        )
        print(trace.posterior["raw_par"].mean(dim=("chain", "draw")).values)
    except ImportError as e:
        print(str(e))


if __name__ == "__main__":
    main()
