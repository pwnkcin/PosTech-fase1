import random

from src.hyperparam_spaces import ALGORITHM_SPACES, decode, random_chromosome

ALGORITHMS = ("Regressao Logistica", "Arvore de Decisao", "Random Forest")


def test_random_chromosome_decodes_to_kwargs_within_declared_bounds():
    rng = random.Random(0)

    for algorithm in ALGORITHMS:
        space = ALGORITHM_SPACES[algorithm]
        for _ in range(100):
            chromosome = random_chromosome(algorithm, rng)
            kwargs = decode(algorithm, chromosome)

            for gene_name, gene_spec in space.items():
                value = kwargs[gene_spec["kwarg"]]
                if gene_spec["type"] == "choice":
                    assert value in gene_spec["choices"]
                else:
                    assert gene_spec["low"] <= value <= gene_spec["high"]


def test_decoded_logistic_regression_never_pairs_l1_with_an_incompatible_solver():
    rng = random.Random(1)
    incompatible = {"lbfgs", "newton-cg", "sag"}

    for _ in range(200):
        chromosome = random_chromosome("Regressao Logistica", rng)
        kwargs = decode("Regressao Logistica", chromosome)

        if kwargs["penalty"] == "l1":
            assert kwargs["solver"] not in incompatible
