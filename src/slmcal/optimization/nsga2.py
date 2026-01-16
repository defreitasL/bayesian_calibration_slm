from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from slmcal.metrics import objective_vector
from slmcal.models.base import ShorelineModel, bounds_matrix, transform_raw_to_physical
from slmcal.data import TimeSeriesDataset


@dataclass(frozen=True)
class NSGA2Config:
    n_generations: int = 80
    pop_size: int = 200
    crossover_prob: float = 0.9
    mutation_prob: float = 0.02
    tournament_pressure: int = 2
    regeneration_rate: float = 0.25
    n_restarts: int = 10
    random_seed: int | None = 42


@dataclass
class NSGA2Result:
    metrics: tuple[str, ...]
    individuals_raw: np.ndarray  # (n_solutions, n_params)
    objectives: np.ndarray       # (n_solutions, n_obj)

    def select_valid(
        self,
        kge_min: float | None = 0.0,
        spearman_min: float | None = 0.0,
        spbias_max: float | None = 25.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter solutions using simple thresholds.

        This reproduces the common pattern: keep solutions with KGE>0, Spearman>0, and
        |PBIAS| < 25.

        Notes
        -----
        Objectives are stored in minimization form. We invert them where needed.
        """
        obj = self.objectives
        keep = np.ones(obj.shape[0], dtype=bool)

        if kge_min is not None and "kge" in self.metrics:
            i = self.metrics.index("kge")
            kge_val = 1.0 - obj[:, i]
            keep &= kge_val >= kge_min

        if spearman_min is not None and "spearman" in self.metrics:
            i = self.metrics.index("spearman")
            rho_val = 1.0 - obj[:, i]
            keep &= rho_val >= spearman_min

        if spbias_max is not None and "spbias" in self.metrics:
            i = self.metrics.index("spbias")
            keep &= obj[:, i] <= spbias_max

        return self.individuals_raw[keep], self.objectives[keep]


def run_nsga2(
    model: ShorelineModel,
    dataset: TimeSeriesDataset,
    metrics: tuple[str, ...],
    cfg: NSGA2Config = NSGA2Config(),
    progress: bool = True,
) -> NSGA2Result:
    """Run a lightweight NSGA-II pre-calibration.

    The optimizer operates in *raw parameter space* within the bounds defined by
    `model.parameters[i].bounds_raw`.

    Returns
    -------
    NSGA2Result
        Stacked solutions from all restarts.
    """

    bounds = bounds_matrix(model.parameters)
    n_params = bounds.shape[0]

    rng = np.random.default_rng(cfg.random_seed)

    all_ind = []
    all_obj = []

    for r in range(cfg.n_restarts):
        seed_r = rng.integers(0, 2**32 - 1)
        rrng = np.random.default_rng(int(seed_r))

        pop = _init_population(rrng, cfg.pop_size, bounds)
        obj = _evaluate_population(model, dataset, pop, metrics)

        for g in range(cfg.n_generations):
            ranks = _fast_non_dominated_sort(obj)
            crowd = _crowding_distance(obj, ranks)

            parents_idx = _tournament_select(rrng, ranks, crowd, cfg.pop_size, cfg.tournament_pressure)
            mating_pool = pop[parents_idx]

            offspring = _sbx_crossover(rrng, mating_pool, cfg.crossover_prob, bounds)
            offspring = _polynomial_mutation(rrng, offspring, cfg.mutation_prob, bounds)

            # regeneration
            n_regen = int(np.ceil(cfg.regeneration_rate * cfg.pop_size))
            regen = _init_population(rrng, n_regen, bounds)
            offspring[:n_regen] = regen

            pop = offspring
            obj = _evaluate_population(model, dataset, pop, metrics)

            if progress and (g % max(1, cfg.n_generations // 10) == 0):
                best = _best_by_rank_and_crowding(ranks, crowd)
                msg = f"restart {r+1}/{cfg.n_restarts} | gen {g}/{cfg.n_generations} | "
                msg += " ".join([f"{m}={obj[best, i]:.3g}" for i, m in enumerate(metrics)])
                print(msg)

        all_ind.append(pop)
        all_obj.append(obj)

    individuals = np.vstack(all_ind)
    objectives = np.vstack(all_obj)

    return NSGA2Result(metrics=metrics, individuals_raw=individuals, objectives=objectives)


# ------------------------- internals -------------------------

def _init_population(rng: np.random.Generator, n: int, bounds: np.ndarray) -> np.ndarray:
    low = bounds[:, 0]
    high = bounds[:, 1]
    return rng.uniform(low, high, size=(n, bounds.shape[0])).astype(float)


def _evaluate_population(
    model: ShorelineModel,
    dataset: TimeSeriesDataset,
    pop_raw: np.ndarray,
    metrics: tuple[str, ...],
) -> np.ndarray:
    obj = np.zeros((pop_raw.shape[0], len(metrics)), dtype=float)

    for i in range(pop_raw.shape[0]):
        physical = transform_raw_to_physical(pop_raw[i], model.parameters)
        y = model.simulate(physical, dataset)
        y_obs = y[dataset.idx_obs]
        obj[i] = objective_vector(dataset.obs, y_obs, metrics)

        # guard against NaNs
        if not np.all(np.isfinite(obj[i])):
            obj[i] = np.inf

    return obj


def _fast_non_dominated_sort(objectives: np.ndarray) -> np.ndarray:
    """Return Pareto rank per individual (0 = best front)."""
    n = objectives.shape[0]
    ranks = np.full(n, -1, dtype=int)
    S = [list() for _ in range(n)]
    n_dom = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if np.all(objectives[p] <= objectives[q]) and np.any(objectives[p] < objectives[q]):
                S[p].append(q)
            elif np.all(objectives[q] <= objectives[p]) and np.any(objectives[q] < objectives[p]):
                n_dom[p] += 1

        if n_dom[p] == 0:
            ranks[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    ranks[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return ranks


def _crowding_distance(objectives: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    n, m = objectives.shape
    d = np.zeros(n, dtype=float)

    for r in np.unique(ranks):
        idx = np.where(ranks == r)[0]
        if idx.size == 0:
            continue
        if idx.size <= 2:
            d[idx] = np.inf
            continue

        front = objectives[idx]
        dist = np.zeros(idx.size)

        for j in range(m):
            order = np.argsort(front[:, j])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf

            vmin = front[order[0], j]
            vmax = front[order[-1], j]
            denom = (vmax - vmin) + 1e-12

            for k in range(1, idx.size - 1):
                dist[order[k]] += (front[order[k + 1], j] - front[order[k - 1], j]) / denom

        d[idx] = dist

    return d


def _tournament_select(
    rng: np.random.Generator,
    ranks: np.ndarray,
    crowd: np.ndarray,
    n_select: int,
    pressure: int,
) -> np.ndarray:
    """Tournament selection using rank first, then crowding distance."""
    n = ranks.size
    out = np.empty(n_select, dtype=int)

    for i in range(n_select):
        cand = rng.integers(0, n, size=pressure)
        best = cand[0]
        for c in cand[1:]:
            if ranks[c] < ranks[best]:
                best = c
            elif ranks[c] == ranks[best] and crowd[c] > crowd[best]:
                best = c
        out[i] = best

    return out


def _best_by_rank_and_crowding(ranks: np.ndarray, crowd: np.ndarray) -> int:
    best = np.argmin(ranks)
    # tie break
    best_rank = ranks[best]
    same = np.where(ranks == best_rank)[0]
    if same.size > 1:
        best = same[np.argmax(crowd[same])]
    return int(best)


def _sbx_crossover(
    rng: np.random.Generator,
    parents: np.ndarray,
    p_crossover: float,
    bounds: np.ndarray,
    eta: float = 15.0,
) -> np.ndarray:
    """Simulated binary crossover (SBX)."""
    n, d = parents.shape
    off = parents.copy()

    for i in range(0, n - 1, 2):
        if rng.random() > p_crossover:
            continue
        p1, p2 = parents[i].copy(), parents[i + 1].copy()
        u = rng.random(d)

        beta = np.empty(d)
        beta[u <= 0.5] = (2 * u[u <= 0.5]) ** (1.0 / (eta + 1.0))
        beta[u > 0.5] = (1.0 / (2 * (1 - u[u > 0.5]))) ** (1.0 / (eta + 1.0))

        c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)

        off[i] = _clip(c1, bounds)
        off[i + 1] = _clip(c2, bounds)

    return off


def _polynomial_mutation(
    rng: np.random.Generator,
    pop: np.ndarray,
    p_mut: float,
    bounds: np.ndarray,
    eta: float = 20.0,
) -> np.ndarray:
    n, d = pop.shape
    out = pop.copy()

    for i in range(n):
        for j in range(d):
            if rng.random() > p_mut:
                continue
            x = out[i, j]
            xl, xu = bounds[j]
            if xl == xu:
                continue

            delta1 = (x - xl) / (xu - xl)
            delta2 = (xu - x) / (xu - xl)
            u = rng.random()
            mut_pow = 1.0 / (eta + 1.0)

            if u < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta + 1.0))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta + 1.0))
                deltaq = 1.0 - val ** mut_pow

            x = x + deltaq * (xu - xl)
            out[i, j] = np.clip(x, xl, xu)

    return out


def _clip(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, bounds[:, 0]), bounds[:, 1])
