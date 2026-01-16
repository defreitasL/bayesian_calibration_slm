import numpy as np

from slmcal.data import TimeSeriesDataset
from slmcal.models.base import ParameterSpec
from slmcal.optimization.nsga2 import run_nsga2, NSGA2Config


class Toy:
    def __init__(self):
        self._parameters = [
            ParameterSpec("b", (-1.0, 1.0), transform=lambda x: x),
            ParameterSpec("c", (-1.0, 1.0), transform=lambda x: x),
        ]

    @property
    def parameters(self):
        return self._parameters

    def simulate(self, physical_params, dataset):
        b, c = map(float, physical_params)
        E = np.asarray(dataset.forcings["E"], dtype=float)
        dt = np.asarray(dataset.dt, dtype=float)
        y = np.empty_like(E)
        y[0] = dataset.y0
        for i in range(len(dt)):
            y[i + 1] = y[i] + dt[i] * (b + c * E[i])
        return y


def test_nsga2_runs():
    n = 50
    time = np.arange(n)
    E = np.ones(n)
    obs_idx = np.arange(0, n, 5)
    obs_time = time[obs_idx]
    obs = np.linspace(0, 1, obs_idx.size)

    ds = TimeSeriesDataset(time=time, forcings={"E": E}, obs_time=obs_time, obs=obs, y0=0.0)

    res = run_nsga2(
        model=Toy(),
        dataset=ds,
        metrics=("kge", "spbias"),
        cfg=NSGA2Config(n_generations=3, pop_size=20, n_restarts=2, random_seed=0),
        progress=False,
    )

    assert res.individuals_raw.shape[1] == 2
    assert res.objectives.shape[1] == 2
