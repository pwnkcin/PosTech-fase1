"""Generic, algorithm-agnostic genetic algorithm for hyperparameter search.

Operates purely on chromosomes (gene_name -> value dicts) and a caller-supplied
fitness function -- it knows nothing about sklearn or ML. `hyperparam_spaces`
supplies the domain knowledge (valid ranges/choices, decode-to-kwargs); this
module supplies selection, crossover, mutation and the generational loop.
"""

import logging

from joblib import Parallel, delayed

from src.hyperparam_spaces import ALGORITHM_SPACES, random_chromosome

logger = logging.getLogger(__name__)


def tournament_selection(population: list[dict], tournament_size: int, rng) -> dict:
    """Picks the fittest chromosome among `tournament_size` random contenders."""
    contenders = rng.sample(population, tournament_size)
    winner = max(contenders, key=lambda individual: individual["fitness"])
    return winner["chromosome"]


def crossover(parent_a: dict, parent_b: dict, rng) -> dict:
    """Uniform crossover: each gene independently inherited from either parent."""
    return {
        gene_name: (parent_a[gene_name] if rng.random() < 0.5 else parent_b[gene_name])
        for gene_name in parent_a
    }


def mutate(chromosome: dict, algorithm: str, rate: float, rng) -> dict:
    """Each gene independently resampled from its full domain with probability `rate`."""
    space = ALGORITHM_SPACES[algorithm]
    mutated = dict(chromosome)
    for gene_name, spec in space.items():
        if rng.random() < rate:
            if spec["type"] == "choice":
                mutated[gene_name] = rng.choice(spec["choices"])
            elif spec["type"] == "int":
                mutated[gene_name] = rng.randint(spec["low"], spec["high"])
            else:
                mutated[gene_name] = rng.uniform(spec["low"], spec["high"])
    return mutated


def _evaluate_population(chromosomes: list[dict], fitness_fn, n_jobs: int) -> list[float]:
    """Evaluates fitness for a batch of chromosomes, parallelized across CPU
    cores so throughput scales with available hardware (joblib/loky)."""
    return Parallel(n_jobs=n_jobs)(delayed(fitness_fn)(chromosome) for chromosome in chromosomes)


def run(
    algorithm: str,
    fitness_fn,
    population_size: int,
    generations: int,
    mutation_rate: float,
    crossover_rate: float,
    tournament_size: int,
    rng,
    elitism: int = 1,
    n_jobs: int = -1,
) -> dict:
    """Runs the GA and returns the best chromosome plus per-generation history."""
    chromosomes = [random_chromosome(algorithm, rng) for _ in range(population_size)]
    fitnesses = _evaluate_population(chromosomes, fitness_fn, n_jobs)
    population = [
        {"chromosome": c, "fitness": f} for c, f in zip(chromosomes, fitnesses, strict=True)
    ]

    history = []
    best_overall = max(population, key=lambda individual: individual["fitness"])

    for generation in range(generations):
        ranked = sorted(population, key=lambda individual: individual["fitness"], reverse=True)
        next_population = ranked[:elitism]

        children = []
        while len(next_population) + len(children) < population_size:
            parent_a = tournament_selection(population, tournament_size, rng)
            parent_b = tournament_selection(population, tournament_size, rng)
            child = (
                crossover(parent_a, parent_b, rng) if rng.random() < crossover_rate else parent_a
            )
            children.append(mutate(child, algorithm, mutation_rate, rng))

        child_fitnesses = _evaluate_population(children, fitness_fn, n_jobs)
        next_population.extend(
            {"chromosome": c, "fitness": f}
            for c, f in zip(children, child_fitnesses, strict=True)
        )

        population = next_population
        gen_best = max(population, key=lambda individual: individual["fitness"])
        if gen_best["fitness"] > best_overall["fitness"]:
            best_overall = gen_best

        fitnesses = [individual["fitness"] for individual in population]
        mean_fitness = sum(fitnesses) / len(fitnesses)
        variance = sum((f - mean_fitness) ** 2 for f in fitnesses) / len(fitnesses)
        history.append(
            {
                "generation": generation,
                "best": best_overall["fitness"],
                "mean": mean_fitness,
                "std": variance**0.5,
            }
        )
        logger.info(
            "algorithm=%s gen=%d best=%.4f mean=%.4f std=%.4f",
            algorithm,
            generation,
            best_overall["fitness"],
            mean_fitness,
            variance**0.5,
        )

    return {
        "best_chromosome": best_overall["chromosome"],
        "best_fitness": best_overall["fitness"],
        "history": history,
    }
