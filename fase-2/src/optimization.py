from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.genetic_algorithm import run as run_ga
from src.hyperparam_spaces import decode

RANDOM_STATE = 42

_ESTIMATOR_CLASSES = {
    "Regressao Logistica": LogisticRegression,
    "Arvore de Decisao": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
}


def _build_estimator(algorithm: str, chromosome: dict):
    kwargs = decode(algorithm, chromosome)
    estimator_cls = _ESTIMATOR_CLASSES[algorithm]
    if estimator_cls is LogisticRegression:
        kwargs["max_iter"] = 10000
    return estimator_cls(random_state=RANDOM_STATE, **kwargs)


def compare_to_baseline(baseline_metrics: dict, optimized_metrics: dict) -> dict:
    return {
        metric: {
            "baseline": baseline_metrics[metric],
            "optimized": optimized_metrics[metric],
            "delta": round(optimized_metrics[metric] - baseline_metrics[metric], 6),
        }
        for metric in baseline_metrics
    }


def run_experiment(
    algorithm: str,
    X_train,
    y_train,
    pipeline_steps: list,
    population_size: int,
    generations: int,
    mutation_rate: float,
    crossover_rate: float,
    tournament_size: int,
    cv: int,
    rng,
    n_jobs: int = -1,
) -> dict:
    """Optimizes hyperparameters via GA using CV fitness on X_train only, then
    fits the best-found configuration on the full X_train for later, separate
    evaluation on a held-out test set."""

    def fitness_fn(chromosome: dict) -> float:
        estimator = _build_estimator(algorithm, chromosome)
        pipeline = Pipeline(pipeline_steps + [("clf", estimator)])
        return cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1").mean()

    ga_result = run_ga(
        algorithm=algorithm,
        fitness_fn=fitness_fn,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        tournament_size=tournament_size,
        rng=rng,
        n_jobs=n_jobs,
    )

    best_estimator = _build_estimator(algorithm, ga_result["best_chromosome"])
    fitted_model = Pipeline(pipeline_steps + [("clf", best_estimator)])
    fitted_model.fit(X_train, y_train)

    return {"ga_result": ga_result, "fitted_model": fitted_model}
