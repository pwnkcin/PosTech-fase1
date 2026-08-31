import random

from sklearn.model_selection import train_test_split

from src.baseline import evaluate
from src.data import build_pipeline, load_data
from src.optimization import compare_to_baseline, run_experiment


def test_compare_to_baseline_computes_deltas_for_each_metric():
    baseline = {"accuracy": 0.80, "recall": 0.70, "f1": 0.75}
    optimized = {"accuracy": 0.85, "recall": 0.65, "f1": 0.78}

    comparison = compare_to_baseline(baseline, optimized)

    assert comparison["accuracy"]["delta"] == round(0.85 - 0.80, 6)
    assert comparison["recall"]["delta"] == round(0.65 - 0.70, 6)
    assert comparison["f1"]["delta"] == round(0.78 - 0.75, 6)


def test_run_experiment_produces_a_test_set_evaluated_model_never_trained_on_test_data():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    result = run_experiment(
        algorithm="Arvore de Decisao",
        X_train=X_train,
        y_train=y_train,
        pipeline_steps=build_pipeline().steps,
        population_size=6,
        generations=3,
        mutation_rate=0.2,
        crossover_rate=0.8,
        tournament_size=2,
        cv=3,
        rng=random.Random(7),
        n_jobs=1,  # tiny workload: avoid process-pool spin-up overhead in tests
    )

    test_metrics = evaluate(result["fitted_model"], X_test, y_test)

    assert set(test_metrics) == {"accuracy", "recall", "f1"}
    for value in test_metrics.values():
        assert 0.0 <= value <= 1.0
    assert len(result["ga_result"]["history"]) == 3
