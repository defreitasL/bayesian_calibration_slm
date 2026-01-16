from __future__ import annotations

import math
from typing import Callable, Literal

import numpy as np
from numba import jit

from fast_optimization.objectives_functions import multi_obj_func, select_best_solution
from fast_optimization.metrics import backtot


CaptureMode = Literal["pareto_front", "best_only", "filter"]


def nsgaii_algorithm_ts(
    model_simulation: Callable[[np.ndarray], np.ndarray],
    Obs: np.ndarray,
    initialize_population: Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_generations: int,
    population_size: int,
    cross_prob: float,
    mutation_rate: float,
    pressure: int,
    regeneration_rate: float,
    kstop: int,
    pcento: float,
    peps: float,
    index_metrics: np.ndarray,
    n_restarts: int = 5,
    capture: CaptureMode = "pareto_front",
    solution_filter: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """NSGA-II with tournament selection + crowding distance.

    - Objective evaluation via `fast_optimization.objectives_functions.multi_obj_func`
    - "Best solution" selection via `select_best_solution`
    - Uses numba-jitted primitives for sorting/crowding/crossover/mutation.

    Differences vs the legacy script
    -------------------------------
    This version supports:

    - capture='pareto_front' (default): store all rank-0 (non-dominated) solutions each generation
    - capture='best_only': store only the best solution each generation
    - capture='filter': store solutions where `solution_filter(objectives)` is True (user-defined)

    Parameters
    ----------
    model_simulation
        Callable that maps a parameter vector -> simulated shoreline at observation times.
    Obs
        Observations aligned with the output of `model_simulation`.
    initialize_population
        Callable: `initialize_population(population_size) -> (population, lower_bounds, upper_bounds)`
    index_metrics
        Metric indices returned by `fast_optimization.objectives_functions.multi_obj_indexes(metrics)`.
    solution_filter
        Only used if capture='filter'. Must return a boolean mask of shape (population_size,).

    Returns
    -------
    best_individual, best_fitness, all_individuals, all_objectives
        `all_*` contains the union of captured solutions across generations and restarts.
    """
    print("Precompilation done!")
    print(f"Starting NSGA-II (tournament selection) with {n_restarts} restarts...")

    metrics_name_list, mask = backtot()
    metrics_name_list = [metrics_name_list[int(k)] for k in index_metrics]
    mask = [mask[int(k)] for k in index_metrics]

    # accumulate across restarts
    n_params = int(initialize_population(1)[0].shape[1])
    n_obj = int(len(index_metrics))
    all_individuals = np.zeros((0, n_params), dtype=float)
    all_objectives = np.zeros((0, n_obj), dtype=float)

    new_seed = np.random.randint(0, 1_000_000)

    for restart in range(int(n_restarts)):
        np.random.seed(int(new_seed + restart))
        print(f"Starting restart {restart+1}/{n_restarts}")

        # Initialize the population
        population, lower_bounds, upper_bounds = initialize_population(int(population_size))
        npar = int(population.shape[1])
        objectives = np.zeros((int(population_size), n_obj), dtype=float)

        # Evaluate initial population
        for i in range(int(population_size)):
            simulation = model_simulation(population[i])
            objectives[i] = multi_obj_func(Obs, simulation, index_metrics)

        # Collect solutions for this restart
        rest_individuals = np.zeros((0, npar), dtype=float)
        rest_objectives = np.zeros((0, n_obj), dtype=float)

        # Number of individuals to regenerate each generation
        num_to_regenerate = int(np.ceil(float(regeneration_rate) * int(population_size)))

        best_fitness_history: list[np.ndarray] = []

        report_every = max(1, int(num_generations) // 10)

        for generation in range(int(num_generations)):
            ranks, _, _ = fast_non_dominated_sort(objectives)
            crowding_distances = crowd_distance(objectives, ranks)

            # Tournament selection with pressure
            next_population_indices = tournament_selection_with_crowding(
                ranks, crowding_distances, int(pressure)
            )

            # Create mating pool and generate the next generation
            mating_pool = population[next_population_indices.astype(np.int32)]

            min_cross_prob = 0.5
            adaptive_cross_prob = max(float(cross_prob) * (1.0 - generation / float(num_generations)), min_cross_prob)

            offspring = crossover(mating_pool, npar, adaptive_cross_prob, lower_bounds, upper_bounds)

            min_mutation_rate = 0.001
            adaptive_mutation_rate = max(float(mutation_rate) * (1.0 - generation / float(num_generations)), min_mutation_rate)

            offspring = polynomial_mutation(offspring, adaptive_mutation_rate, npar, lower_bounds, upper_bounds)

            # Reintroduce new individuals to maintain diversity
            new_individuals, _, _ = initialize_population(num_to_regenerate)
            offspring = np.vstack((offspring, new_individuals))

            # Evaluate offspring
            new_objectives = np.zeros_like(objectives)
            for i in range(int(population_size)):
                simulation = model_simulation(offspring[i])
                new_objectives[i] = multi_obj_func(Obs, simulation, index_metrics)

            population = offspring
            objectives = new_objectives

            # Track best solution for early stopping
            ii_best = int(select_best_solution(objectives)[0])
            best_fitness = objectives[ii_best].copy()
            best_fitness_history.append(best_fitness)

            # Capture solutions
            if capture == "best_only":
                rest_individuals = np.vstack((rest_individuals, population[[ii_best]]))
                rest_objectives = np.vstack((rest_objectives, objectives[[ii_best]]))

            elif capture == "filter":
                if solution_filter is None:
                    raise ValueError("capture='filter' requires solution_filter")
                keep = solution_filter(objectives)
                rest_individuals = np.vstack((rest_individuals, population[keep]))
                rest_objectives = np.vstack((rest_objectives, objectives[keep]))

            else:  # 'pareto_front'
                keep = ranks == 0
                rest_individuals = np.vstack((rest_individuals, population[keep]))
                rest_objectives = np.vstack((rest_objectives, objectives[keep]))

            # Early stopping based on improvement of mean normalized objective sum
            if generation > int(kstop) and len(best_fitness_history) >= int(kstop):
                recent = np.asarray(best_fitness_history[-int(kstop):], dtype=float)
                # normalized within the recent window
                mn = recent.min(axis=0)
                mx = recent.max(axis=0)
                norm = (recent - mn) / (mx - mn + 1e-10)
                mean_norm = float(np.mean(np.sum(norm, axis=1)))
                # Compare last half-window to first half-window
                half = max(1, int(kstop) // 2)
                prev = float(np.mean(np.sum(norm[:half], axis=1)))
                imp = (prev - mean_norm) / (abs(prev) + 1e-12)
                if imp < float(pcento):
                    print(f"Converged at generation {generation} (improvement criteria).")
                    break

            # Early stopping based on parameter space contraction
            epsilon = 1e-10
            gnrng = float(np.exp(np.mean(np.log((np.max(population, axis=0) - np.min(population, axis=0) + epsilon) /
                                               (upper_bounds - lower_bounds + epsilon)))))
            if gnrng < float(peps):
                print(f"Converged at generation {generation} (parameter space criteria).")
                break

            if generation % report_every == 0:
                print(f"Generation {generation} / {num_generations}")
                for j in range(n_obj):
                    if bool(mask[j]):
                        print(f"{metrics_name_list[j]}: {best_fitness[j]:.3f}")
                    else:
                        print(f"{metrics_name_list[j]}: {(1.0 - best_fitness[j]):.3f}")

        # Add restart captures to global pool
        all_individuals = np.vstack((all_individuals, rest_individuals))
        all_objectives = np.vstack((all_objectives, rest_objectives))

    # Best across all captured solutions
    ii = int(select_best_solution(all_objectives)[0])
    best_individual = all_individuals[ii]
    best_fitness = all_objectives[ii]

    print("NSGA-II completed.")
    print("Best fitness found:")
    for j in range(n_obj):
        if bool(mask[j]):
            print(f"{metrics_name_list[j]}: {best_fitness[j]:.3f}")
        else:
            print(f"{metrics_name_list[j]}: {(1.0 - best_fitness[j]):.3f}")

    return best_individual, best_fitness, all_individuals, all_objectives


@jit(nopython=True, cache=True)
def fast_non_dominated_sort(objectives):
    population_size = objectives.shape[0]
    domination_count = np.zeros(population_size, dtype=np.int32)
    dominated_solutions = np.full((population_size, population_size), -1, dtype=np.int32)
    current_counts = np.zeros(population_size, dtype=np.int32)
    ranks = np.zeros(population_size, dtype=np.int32)

    front_indices = np.full((population_size, population_size), -1, dtype=np.int32)
    front_sizes = np.zeros(population_size, dtype=np.int32)

    for p in range(population_size):
        for q in range(population_size):
            if np.all(objectives[p] <= objectives[q]) and np.any(objectives[p] < objectives[q]):
                dominated_solutions[p, current_counts[p]] = q
                current_counts[p] += 1
            elif np.all(objectives[q] <= objectives[p]) and np.any(objectives[q] < objectives[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            ranks[p] = 0
            front_indices[0, front_sizes[0]] = p
            front_sizes[0] += 1

    i = 0
    while front_sizes[i] > 0:
        next_front_size = 0
        for j in range(front_sizes[i]):
            p = front_indices[i, j]
            for k in range(current_counts[p]):
                q = dominated_solutions[p, k]
                if q == -1:
                    break
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    ranks[q] = i + 1
                    front_indices[i + 1, next_front_size] = q
                    next_front_size += 1
        front_sizes[i + 1] = next_front_size
        i += 1

    return ranks, front_indices, front_sizes


@jit(nopython=True, cache=True)
def crowd_distance(objectives, ranks):
    population_size = objectives.shape[0]
    nobj = objectives.shape[1]
    distances = np.zeros(population_size, dtype=np.float64)

    max_rank = 0
    for i in range(population_size):
        if ranks[i] > max_rank:
            max_rank = ranks[i]

    for rank in range(max_rank + 1):
        # gather indices in this front
        count = 0
        for i in range(population_size):
            if ranks[i] == rank:
                count += 1
        if count == 0:
            continue

        front = np.empty(count, dtype=np.int32)
        k = 0
        for i in range(population_size):
            if ranks[i] == rank:
                front[k] = i
                k += 1

        for m in range(nobj):
            # sort indices by objective m
            # simple argsort for numba
            sorted_idx = front.copy()
            # insertion sort
            for i in range(1, sorted_idx.size):
                key = sorted_idx[i]
                key_val = objectives[key, m]
                j = i - 1
                while j >= 0 and objectives[sorted_idx[j], m] > key_val:
                    sorted_idx[j + 1] = sorted_idx[j]
                    j -= 1
                sorted_idx[j + 1] = key

            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf

            min_value = objectives[sorted_idx[0], m]
            max_value = objectives[sorted_idx[-1], m]
            denom = max_value - min_value
            if denom == 0.0:
                continue

            for i in range(1, sorted_idx.size - 1):
                distances[sorted_idx[i]] += (
                    (objectives[sorted_idx[i + 1], m] - objectives[sorted_idx[i - 1], m]) / denom
                )

    return distances


@jit(nopython=True, cache=True)
def tournament_selection_with_crowding(ranks, crowding_distances, pressure):
    n_select = len(ranks)
    n_random = n_select * pressure
    n_perms = math.ceil(n_random / len(ranks))

    P = np.empty((n_random,), dtype=np.int32)
    for i in range(n_perms):
        P[i * len(ranks):(i + 1) * len(ranks)] = np.random.permutation(len(ranks))
    P = P[:n_random].reshape(n_select, pressure)

    selected_indices = np.full(n_select, -1, dtype=np.int32)
    for i in range(n_select):
        best = P[i, 0]
        for j in range(1, pressure):
            cand = P[i, j]
            if ranks[cand] < ranks[best]:
                best = cand
            elif ranks[cand] == ranks[best]:
                if crowding_distances[cand] > crowding_distances[best]:
                    best = cand
        selected_indices[i] = best

    return selected_indices


@jit(nopython=True, cache=True)
def crossover(population, num_vars, crossover_prob, lower_bounds, upper_bounds):
    n_pop = population.shape[0]
    child_population = population.copy()

    for i in range(n_pop):
        if np.random.random() < crossover_prob:
            p1 = np.random.randint(0, n_pop)
            p2 = np.random.randint(0, n_pop)
            if num_vars > 1:
                point = np.random.randint(1, num_vars)
                for j in range(num_vars):
                    if j < point:
                        child_population[i, j] = population[p1, j]
                    else:
                        child_population[i, j] = population[p2, j]
            else:
                child_population[i, 0] = 0.5 * (population[p1, 0] + population[p2, 0])

            # clip bounds
            for j in range(num_vars):
                if child_population[i, j] < lower_bounds[j]:
                    child_population[i, j] = lower_bounds[j]
                elif child_population[i, j] > upper_bounds[j]:
                    child_population[i, j] = upper_bounds[j]

    return child_population


@jit(nopython=True, cache=True)
def polynomial_mutation(population, mutation_rate, num_vars, lower_bounds, upper_bounds, eta_mut=10):
    X = population.copy()
    Y = X.copy()
    do_mutation = np.random.random(X.shape) < mutation_rate

    for i in range(X.shape[0]):
        for j in range(num_vars):
            if do_mutation[i, j]:
                xl = lower_bounds[j]
                xu = upper_bounds[j]
                x = X[i, j]

                delta1 = (x - xl) / (xu - xl)
                delta2 = (xu - x) / (xu - xl)
                mut_pow = 1.0 / (eta_mut + 1.0)
                rand = np.random.random()

                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_mut + 1.0))
                    deltaq = (val ** mut_pow) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_mut + 1.0))
                    deltaq = 1.0 - (val ** mut_pow)

                mutated_value = x + deltaq * (xu - xl)
                if mutated_value < xl:
                    mutated_value = xl
                elif mutated_value > xu:
                    mutated_value = xu
                Y[i, j] = mutated_value

    return Y
