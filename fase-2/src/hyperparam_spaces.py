"""Hyperparameter search spaces for the GA, one per Modulo 1 algorithm.

Each gene is either a numeric range ("int"/"float") or a fixed "choice" list.
`decode()` turns a chromosome (gene_name -> raw value) into constructor
kwargs for the corresponding sklearn estimator. Derived/coupled kwargs (e.g.
the LogisticRegression solver, which must match the chosen penalty) are not
independent genes -- they're computed in decode() so an invalid combination
can never be encoded in the first place.
"""

ALGORITHM_SPACES = {
    "Regressao Logistica": {
        "C": {"kwarg": "C", "type": "float", "low": 0.001, "high": 100.0},
        "penalty": {"kwarg": "penalty", "type": "choice", "choices": ["l1", "l2"]},
        "class_weight": {"kwarg": "class_weight", "type": "choice", "choices": [None, "balanced"]},
    },
    "Arvore de Decisao": {
        "max_depth": {"kwarg": "max_depth", "type": "int", "low": 1, "high": 20},
        "min_samples_split": {"kwarg": "min_samples_split", "type": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"kwarg": "min_samples_leaf", "type": "int", "low": 1, "high": 10},
        "criterion": {"kwarg": "criterion", "type": "choice", "choices": ["gini", "entropy"]},
    },
    "Random Forest": {
        "n_estimators": {"kwarg": "n_estimators", "type": "int", "low": 50, "high": 300},
        "max_depth": {"kwarg": "max_depth", "type": "int", "low": 1, "high": 30},
        "min_samples_split": {"kwarg": "min_samples_split", "type": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"kwarg": "min_samples_leaf", "type": "int", "low": 1, "high": 10},
        "max_features": {"kwarg": "max_features", "type": "choice", "choices": ["sqrt", "log2", None]},
    },
}

_PENALTY_TO_SOLVER = {"l1": "liblinear", "l2": "lbfgs"}


def random_chromosome(algorithm: str, rng) -> dict:
    chromosome = {}
    for gene_name, spec in ALGORITHM_SPACES[algorithm].items():
        if spec["type"] == "choice":
            chromosome[gene_name] = rng.choice(spec["choices"])
        elif spec["type"] == "int":
            chromosome[gene_name] = rng.randint(spec["low"], spec["high"])
        else:
            chromosome[gene_name] = rng.uniform(spec["low"], spec["high"])
    return chromosome


def decode(algorithm: str, chromosome: dict) -> dict:
    kwargs = {}
    for gene_name, spec in ALGORITHM_SPACES[algorithm].items():
        kwargs[spec["kwarg"]] = chromosome[gene_name]

    if algorithm == "Regressao Logistica":
        kwargs["solver"] = _PENALTY_TO_SOLVER[kwargs["penalty"]]

    return kwargs
