from src.baseline import ALGORITHMS, select_champion, train_baseline_models


def test_train_baseline_models_returns_one_result_per_algorithm_with_valid_metrics():
    results = train_baseline_models()

    assert set(results.keys()) == set(ALGORITHMS)
    for result in results.values():
        for metric in ("accuracy", "recall", "f1"):
            assert 0.0 <= result["metrics"][metric] <= 1.0


def test_champion_has_the_highest_f1_among_all_trained_models():
    results = train_baseline_models()

    champion_name, champion = select_champion(results)

    all_f1s = [r["metrics"]["f1"] for r in results.values()]
    assert champion["metrics"]["f1"] == max(all_f1s)
    assert champion_name in results
