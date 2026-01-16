import numpy as np
from numba import jit
import math
from fast_optimization.objectives_functions import multi_obj_func, select_best_solution
from fast_optimization.metrics import backtot

def nsgaii_algorithm_ts(model_simulation, Obs, initialize_population, num_generations, population_size, cross_prob, mutation_rate, pressure, regeneration_rate, kstop, pcento, peps, index_metrics, n_restarts = 5):
    """
    NSGA-II Algorithm with Tournament Selection.
    This algorithm aims to optimize a multi-objective function using NSGA-II with tournament selection to balance between exploration and exploitation.
    
    Parameters:
    - model_simulation: Function that simulates the model given an input vector of parameters.
    - Obs: Observed data used for calculating the fitness.
    - initialize_population: Function to initialize the population.
    - num_generations: Number of generations to evolve.
    - population_size: Size of the population.
    - cross_prob: Probability of crossover.
    - mutation_rate: Probability of mutation.
    - pressure: Selection pressure for tournament selection.
    - regeneration_rate: Proportion of the population to be regenerated each generation.
    - kstop: Number of past evolution loops to assess convergence.
    - pcento: Percentage improvement allowed in the past kstop loops for convergence.
    - peps: Threshold for the normalized geometric range of parameters to determine convergence.
    - index_metrics: Index used for multi-objective evaluation.

    Returns:
    - best_individual: The best solution found.
    - best_fitness: Fitness value of the best solution.
    - best_fitness_history: History of the best fitness values over generations.
    """
    print('Precompilation done!')
    print(f'Starting NSGA-II with tournament selection algorithm with {n_restarts} restarts...')

    metrics_name_list, mask = backtot()
    metrics_name_list = [metrics_name_list[k] for k in index_metrics]
    mask = [mask[k] for k in index_metrics]

    all_individuals = np.zeros((0, len(initialize_population(1)[0][0])))
    all_objectives = np.zeros((0, len(index_metrics)))



    new_seed = np.random.randint(0, 1000000)

    for restart in range(n_restarts):

        rest_individuals = np.zeros((0, len(initialize_population(1)[0][0])))
        rest_objectives = np.zeros((0, len(index_metrics)))

        np.random.seed(new_seed + restart)

        print(f'Starting {restart+1}/{n_restarts}')

        best_fitness_history = []
        best_individuals = []
        
        # Initialize the population
        population, lower_bounds, upper_bounds = initialize_population(population_size)

        npar = population.shape[1]
        nobj = len(index_metrics)
        objectives = np.zeros((population_size, nobj))  # Objectives for multi-objective evaluation

        # Evaluate initial population
        for i in range(population_size):
            simulation = model_simulation(population[i])
            objectives[i] = multi_obj_func(Obs, simulation, index_metrics)

        # Number of individuals to regenerate each generation
        num_to_regenerate = int(np.ceil(regeneration_rate * population_size))

        # Main loop of NSGA-II algorithm
        for generation in range(num_generations):
            ranks, front_indices, front_sizes = fast_non_dominated_sort(objectives)
            crowding_distances = crowd_distance(objectives, ranks)
            
            # Tournament selection with pressure
            next_population_indices = tournament_selection_with_crowding(ranks, crowding_distances, pressure)

            # Create mating pool and generate the next generation
            mating_pool = population[next_population_indices.astype(np.int32)]
            
            # Crossover operation
            min_cross_prob = 0.5  # Minimum crossover probability to maintain diversity
            adaptive_cross_prob = max(cross_prob * (1 - generation / num_generations), min_cross_prob)

            offspring = crossover(mating_pool, npar, adaptive_cross_prob, lower_bounds, upper_bounds)
            
            # Mutation operation
            min_mutation_rate = 0.001  # Minimum mutation rate to prevent premature convergence
            adaptive_mutation_rate = max(mutation_rate * (1 - generation / num_generations), min_mutation_rate)

            offspring = polynomial_mutation(offspring, adaptive_mutation_rate, npar, lower_bounds, upper_bounds)

            # Reintroduce new individuals to maintain genetic diversity
            new_individuals, _, _ = initialize_population(num_to_regenerate)
            offspring = np.vstack((offspring, new_individuals))

            # Evaluate the new offspring population
            new_objectives = np.zeros_like(objectives)
            for i in range(population_size):
                simulation = model_simulation(offspring[i])
                new_objectives[i] = multi_obj_func(Obs, simulation, index_metrics)

            # Prepare the next generation
            population = offspring
            objectives = new_objectives

            # Early stopping based on improvement criteria
            ii = select_best_solution(objectives)[0]
            current_best_fitness = objectives[ii]
            best_fitness_history.append(current_best_fitness)
            best_individuals.append(population[ii])

            ii_filt = np.where(((1-objectives[:, 0]) > 0) & ((objectives[:, 1]) < 25)& ((1-objectives[:, 2]) > 0))[0]


            rest_individuals = np.vstack((rest_individuals, population[ii_filt]))
            rest_objectives = np.vstack((rest_objectives, objectives[ii_filt]))

            if generation > kstop:
                # Normalize objectives for proper comparison
                normalized_objectives = (objectives - objectives.min(axis=0)) / (objectives.max(axis=0) - objectives.min(axis=0) + 1e-10)
                mean_normalized_fitness = np.mean(np.sum(normalized_objectives, axis=1))
                previous_mean_fitness = np.mean(np.sum((best_fitness_history[-kstop]), axis=0)) if len(best_fitness_history) >= kstop else mean_normalized_fitness
                recent_improvement = (previous_mean_fitness - mean_normalized_fitness) / abs(previous_mean_fitness)
                if recent_improvement < pcento:
                    print(f"Converged at generation {generation} based on improvement criteria.")
                    break

            # Early stopping based on parameter space convergence
            epsilon = 1e-10
            gnrng = np.exp(np.mean(np.log((np.max(population, axis=0) - np.min(population, axis=0) + epsilon) / (upper_bounds - lower_bounds))))
            if gnrng < peps:
                print(f"Converged at generation {generation} based on parameter space convergence.")
                break

            if generation % (num_generations // (num_generations/10)) == 0:
                print(f"Generation {generation} of {num_generations} completed")

                print(f'Last 6 pars:{best_individuals[-1][-6:]}')

                current_best_fitness
                for j in range(nobj):
                    if mask[j]:
                        print(f"{metrics_name_list[j]}: {current_best_fitness[j]:.3f}")
                    else:
                        print(f"{metrics_name_list[j]}: {(1 - current_best_fitness[j]):.3f}")
        # Select the best final solution
        if restart == 0:
            total_objectives = np.vstack((rest_objectives, np.array(best_fitness_history)))
            total_individuals = np.vstack((rest_individuals, np.array(best_individuals)))
            best_index = select_best_solution(total_objectives)[0]
            best_fitness = total_objectives[best_index]
            best_individual = total_individuals[best_index]
            all_individuals = np.vstack((all_individuals, total_individuals))
            all_objectives = np.vstack((all_objectives, total_objectives))
            # print(all_individuals.shape)
            # print('-----------------------------')
            # print(total_individuals.shape)
            # print('-----------------------------')

        else:
            total_objectives = np.vstack((rest_objectives, np.array(best_fitness_history)))
            total_individuals = np.vstack((rest_individuals, np.array(best_individuals)))
            total_objectives = np.vstack((total_objectives, np.array([best_fitness])))
            total_individuals = np.vstack((total_individuals, np.array([best_individual])))
            best_index = select_best_solution(total_objectives)[0]
            best_fitness = total_objectives[best_index]
            best_individual = total_individuals[best_index]
            all_individuals = np.vstack((all_individuals, total_individuals))
            all_objectives = np.vstack((all_objectives, total_objectives))
            # print(all_individuals.shape)
            # print('-----------------------------')
            # print(total_individuals.shape)
            # print('-----------------------------')


    print(f'NSGA-II with tournament selection algorithm completed after {n_restarts} restarts.')
    print('Best fitness found:')
    for j in range(nobj):
        if mask[j]:
            print(f"{metrics_name_list[j]}: {best_fitness[j]:.3f}")
        else:
            print(f"{metrics_name_list[j]}: {(1 - best_fitness[j]):.3f}")

    ii = select_best_solution(all_objectives)[0]
    best_individual = all_individuals[ii]
    best_fitness = all_objectives[ii]

    return best_individual, best_fitness, all_individuals, all_objectives


@jit(nopython=True)
def fast_non_dominated_sort(objectives):
    """
    Perform fast non-dominated sorting on a set of objectives to create fronts.
    
    Parameters:
    - objectives: Objective values for each individual.
    
    Returns:
    - ranks: Ranks of individuals based on non-domination.
    - front_indices: Indices of individuals in each front.
    - front_sizes: Sizes of each front.
    """
    population_size = objectives.shape[0]
    domination_count = np.zeros(population_size, dtype=np.int32)
    dominated_solutions = np.full((population_size, population_size), -1, dtype=np.int32)
    current_counts = np.zeros(population_size, dtype=np.int32)
    ranks = np.zeros(population_size, dtype=np.int32)

    # Array to store fronts as indices
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
                    front_indices[i+1, next_front_size] = q
                    next_front_size += 1
        front_sizes[i+1] = next_front_size
        i += 1

    return ranks, front_indices, front_sizes

@jit(nopython=True)
def crowd_distance(objectives, ranks):
    """
    Calculate crowding distance for each individual in the population.
    
    Parameters:
    - objectives: Objective values for each individual.
    - ranks: Ranks of individuals based on fitness.
    
    Returns:
    - distances: Crowding distance for each individual.
    """
    population_size = objectives.shape[0]
    nobj = objectives.shape[1]
    distances = np.zeros(population_size, dtype=np.float64)

    for rank in range(np.max(ranks) + 1):
        front = np.where(ranks == rank)[0]
        if len(front) == 0:
            continue

        for m in range(nobj):
            sorted_indices = front[np.argsort(objectives[front, m])]
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            min_value = objectives[sorted_indices[0], m]
            max_value = objectives[sorted_indices[-1], m]

            if max_value - min_value == 0:
                continue

            for i in range(1, len(sorted_indices) - 1):
                distances[sorted_indices[i]] += (
                    (objectives[sorted_indices[i + 1], m] - objectives[sorted_indices[i - 1], m])
                    / (max_value - min_value)
                )

    return distances

@jit(nopython=True)
def tournament_selection_with_crowding(ranks, crowding_distances, pressure):
    """
    Perform tournament selection with crowding distance as a tiebreaker.
    
    Parameters:
    - ranks: Ranks of individuals based on fitness.
    - crowding_distances: Crowding distances for each individual.
    - pressure: Selection pressure for tournament selection.
    
    Returns:
    - selected_indices: Indices of the individuals selected for the mating pool.
    """
    n_select = len(ranks)
    n_random = n_select * pressure
    n_perms = math.ceil(n_random / len(ranks))

    P = np.empty((n_random,), dtype=np.int32)
    for i in range(n_perms):
        P[i * len(ranks):(i + 1) * len(ranks)] = np.random.permutation(len(ranks))
    P = P[:n_random].reshape(n_select, pressure)

    selected_indices = np.full(n_select, -1, dtype=np.int32)
    for i in range(n_select):
        a, b = P[i]
        if ranks[a] < ranks[b]:
            selected_indices[i] = a
        elif ranks[a] > ranks[b]:
            selected_indices[i] = b
        else:  # Tie: use crowding distance
            if crowding_distances[a] > crowding_distances[b]:
                selected_indices[i] = a
            else:
                selected_indices[i] = b

    return selected_indices

@jit(nopython=True)
def crossover(population, num_vars, crossover_prob, lower_bounds, upper_bounds):
    """
    Perform crossover operation on the population.
    
    Parameters:
    - population: The current population of individuals.
    - num_vars: Number of variables in each individual.
    - crossover_prob: Probability of crossover.
    - lower_bounds: Lower bounds for each variable.
    - upper_bounds: Upper bounds for each variable.
    
    Returns:
    - child_population: The new population after crossover.
    """
    n_pop = population.shape[0]
    cross_probability = np.random.random(n_pop)
    do_cross = cross_probability < crossover_prob
    R = np.random.randint(0, n_pop, (n_pop, 2))
    parents = R[do_cross]
    child_population = population.copy()

    if num_vars > 1:
        # General case: num_vars > 1
        cross_point = np.random.randint(1, num_vars, len(parents))
        for i in range(len(parents)):
            parent1, parent2 = parents[i]
            point = cross_point[i]
            # Concatenate parts of parents to create offspring
            child = np.concatenate((population[parent1, :point], population[parent2, point:]))
            
            # Ensure bounds are respected
            for j in range(num_vars):
                child[j] = min(max(child[j], lower_bounds[j]), upper_bounds[j])
            
            # Update child in the population
            child_population[do_cross][i] = child
    else:
        # Special case: num_vars = 1
        for i in range(len(parents)):
            parent1, parent2 = parents[i]
            # For a single variable, offspring is a weighted average of parents
            child = (population[parent1] + population[parent2]) / 2.0
            
            # Ensure bounds are respected
            child = min(max(child[0], lower_bounds[0]), upper_bounds[0])
            
            # Update child in the population
            child_population[do_cross][i] = child

    return child_population

@jit(nopython=True)
def polynomial_mutation(population, mutation_rate, num_vars, lower_bounds, upper_bounds, eta_mut=10):
    """
    Perform polynomial mutation on the population.
    
    Parameters:
    - population: The current population of individuals.
    - mutation_rate: Probability of mutation.
    - num_vars: Number of variables in each individual.
    - lower_bounds: Lower bounds for each variable.
    - upper_bounds: Upper bounds for each variable.
    - eta_mut: Mutation parameter (default is 10).
    
    Returns:
    - Y: The new population after mutation.
    """
    X = population.copy()
    Y = np.full(X.shape, np.inf)
    do_mutation = np.random.random(X.shape) < mutation_rate
    Y[:, :] = X

    for i in range(len(population)):
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
                mutated_value = max(xl, min(mutated_value, xu))
                Y[i, j] = mutated_value

    return Y
