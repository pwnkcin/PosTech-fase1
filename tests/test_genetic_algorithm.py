import random

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from src.genetic_algorithm import crossover, mutate, run, tournament_selection
from src.hyperparam_spaces import decode, random_chromosome

ALGORITHM = "Arvore de Decisao"


def test_tournament_selection_prefers_fitter_individuals_over_many_trials():
    rng = random.Random(0)
    population = [
        {"chromosome": {"id": "low"}, "fitness": 0.1},
        {"chromosome": {"id": "mid"}, "fitness": 0.5},
        {"chromosome": {"id": "high"}, "fitness": 0.9},
    ]

    picks = [
        tournament_selection(population, tournament_size=2, rng=rng)["id"] for _ in range(500)
    ]

    assert picks.count("high") > picks.count("low")


def test_crossover_child_genes_each_come_from_one_parent():
    rng = random.Random(0)
    parent_a = {"max_depth": 1, "min_samples_split": 2, "min_samples_leaf": 1, "criterion": "gini"}
    parent_b = {"max_depth": 20, "min_samples_split": 20, "min_samples_leaf": 10, "criterion": "entropy"}

    child = crossover(parent_a, parent_b, rng)

    for gene_name, value in child.items():
        assert value in (parent_a[gene_name], parent_b[gene_name])


def test_mutate_with_full_rate_changes_at_least_one_gene_and_stays_in_bounds():
    rng = random.Random(0)
    original = random_chromosome(ALGORITHM, rng)

    mutated = mutate(original, ALGORITHM, rate=1.0, rng=rng)
    decode(ALGORITHM, mutated)  # raises if any gene is out of the declared domain

    assert mutated != original


def test_mutate_with_zero_rate_never_changes_the_chromosome():
    rng = random.Random(0)
    original = random_chromosome(ALGORITHM, rng)

    mutated = mutate(original, ALGORITHM, rate=0.0, rng=rng)

    assert mutated == original


def test_run_finds_hyperparameters_at_least_as_good_as_a_deliberately_bad_baseline():
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=6, random_state=42
    )

    def fitness_fn(chromosome):
        kwargs = decode(ALGORITHM, chromosome)
        model = DecisionTreeClassifier(random_state=42, **kwargs)
        return cross_val_score(model, X, y, cv=3, scoring="f1").mean()

    bad_baseline = cross_val_score(
        DecisionTreeClassifier(max_depth=1, random_state=42), X, y, cv=3, scoring="f1"
    ).mean()

    result = run(
        algorithm=ALGORITHM,
        fitness_fn=fitness_fn,
        population_size=12,
        generations=6,
        mutation_rate=0.2,
        crossover_rate=0.8,
        tournament_size=3,
        rng=random.Random(42),
        n_jobs=1,  # tiny workload: avoid process-pool spin-up overhead in tests
    )

    assert result["best_fitness"] >= bad_baseline
    assert len(result["history"]) == 6
    best_per_gen = [gen["best"] for gen in result["history"]]
    assert best_per_gen == sorted(best_per_gen)  # elitism: never regresses
