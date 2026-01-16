from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from slmcal.data import TimeSeriesDataset
from slmcal.models.base import ShorelineModel, bounds_matrix, transform_raw_to_physical

try:
    from fast_optimization.objectives_functions import multi_obj_indexes
except Exception as e:  # pragma: no cover
    multi_obj_indexes = None  # type: ignore


@dataclass(frozen=True)
class FastOptNSGA2Config:
    """
    Notes
    -----
    This backend delegates objective calculation to `fast_optimization` and uses
    the same index-based metric selection pattern you already use
    (`indexes = multi_obj_indexes(metrics)`).
    """

    num_generations: int = 150
    population_size: int = 2000
    cross_prob: float = 0.8
    mutation_rate: float = 0.2
    regeneration_rate: float = 0.15
    pressure: int = 2
    kstop: int = 100
    pcento: float = 0.001
    peps: float = 1e-4
    n_restarts: int = 30
    random_seed: int | None = None


@dataclass
class FastOptNSGA2Result:
    metrics: tuple[str, ...]
    best_individual_raw: np.ndarray
    best_objectives: np.ndarray
    individuals_raw: np.ndarray
    objectives: np.ndarray


def _default_initialize_population(bounds: np.ndarray) -> Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Create an initializer with the same signature expected by nsgaii_algorithm_ts.

    Returns
    -------
    initialize_population(population_size) -> (population, lower_bounds, upper_bounds)
    """
    lb = bounds[:, 0].astype(float)
    ub = bounds[:, 1].astype(float)

    def init(population_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pop = np.empty((population_size, bounds.shape[0]), dtype=float)
        for i in range(bounds.shape[0]):
            pop[:, i] = np.random.uniform(lb[i], ub[i], population_size)
        return pop, lb, ub

    return init


def _simulate_at_obs(model: ShorelineModel, dataset: TimeSeriesDataset, raw_params: np.ndarray) -> np.ndarray:
    """Generic helper: raw -> physical -> model.simulate -> select obs indices."""
    physical = transform_raw_to_physical(raw_params, model.parameters)
    y_full = model.simulate(physical, dataset)
    return np.asarray(y_full, dtype=float)[np.asarray(dataset.idx_obs, dtype=int)]


def run_nsga2_fastopt(
    model: ShorelineModel,
    dataset: TimeSeriesDataset,
    metrics: Sequence[str] = ("kge", "pbias", "spearman"),
    cfg: FastOptNSGA2Config = FastOptNSGA2Config(),
    initialize_population: Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> FastOptNSGA2Result:
    """
    Parameters
    ----------
    model
        Shoreline model implementing :class:`slmcal.models.base.ShorelineModel`.
    dataset
        TimeSeriesDataset with `obs` and `idx_obs` filled.
    metrics
        List of metric names recognized by `fast_optimization` (e.g., 'kge', 'pbias', 'spearman').
        These are mapped to metric indexes through `fast_optimization.objectives_functions.multi_obj_indexes`.
    cfg
        NSGA-II hyperparameters.
    initialize_population
        Optional custom initializer; use this if you want custom parameterizations (e.g., log-space)
        `initialize_population(population_size) -> (population, lower_bounds, upper_bounds)`.

    Returns
    -------
    FastOptNSGA2Result
        Includes the full set of Pareto solutions (individuals_raw, objectives) plus best solution.
    """
    if multi_obj_indexes is None:  # pragma: no cover
        raise ImportError(
            "fast_optimization is required for run_nsga2_fastopt(). "
            "Install it (e.g., pip install 'fast_optimization @ git+https://github.com/defreitasL/fast_optimization')."
        )

    from slmcal.optimization._nsga2_ts_legacy import nsgaii_algorithm_ts

    if cfg.random_seed is not None:
        np.random.seed(int(cfg.random_seed))

    bounds = bounds_matrix(model.parameters)
    init = initialize_population or _default_initialize_population(bounds)

    indexes = multi_obj_indexes(list(metrics))

    def model_simulation(par: np.ndarray) -> np.ndarray:
        return _simulate_at_obs(model, dataset, np.asarray(par, dtype=float))

    best_ind, best_fit, all_ind, all_obj = nsgaii_algorithm_ts(
        model_simulation=model_simulation,
        Obs=np.asarray(dataset.obs, dtype=float),
        initialize_population=init,
        num_generations=int(cfg.num_generations),
        population_size=int(cfg.population_size),
        cross_prob=float(cfg.cross_prob),
        mutation_rate=float(cfg.mutation_rate),
        pressure=int(cfg.pressure),
        regeneration_rate=float(cfg.regeneration_rate),
        kstop=int(cfg.kstop),
        pcento=float(cfg.pcento),
        peps=float(cfg.peps),
        index_metrics=np.asarray(indexes, dtype=np.int64),
        n_restarts=int(cfg.n_restarts),
    )

    return FastOptNSGA2Result(
        metrics=tuple(metrics),
        best_individual_raw=np.asarray(best_ind, dtype=float),
        best_objectives=np.asarray(best_fit, dtype=float),
        individuals_raw=np.asarray(all_ind, dtype=float),
        objectives=np.asarray(all_obj, dtype=float),
    )
