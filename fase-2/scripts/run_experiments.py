"""Runs the Modulo 1 baseline + 3 GA hyperparameter-optimization experiments
across all 3 algorithms (Regressao Logistica, Arvore de Decisao, Random
Forest). Logs per-generation convergence, saves convergence plots, and writes
experiments/summary.json consumed by scripts/present_results.py.

Usage: python -m scripts.run_experiments
"""

import csv
import json
import logging
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.baseline import ALGORITHMS, evaluate, train_baseline_models
from src.data import build_pipeline, load_data
from src.optimization import compare_to_baseline, run_experiment

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"
LOGS_DIR = ROOT / "logs"

# 3 distinct GA configurations, per the Fase 2 requirement of >=3 experiments
# with different population/mutation settings.
GA_CONFIGS = {
    "A_rapido": dict(
        population_size=15, generations=10, mutation_rate=0.10, crossover_rate=0.80, tournament_size=3
    ),
    "B_completo": dict(
        population_size=30, generations=20, mutation_rate=0.05, crossover_rate=0.90, tournament_size=4
    ),
    "C_exploratorio": dict(
        population_size=10, generations=8, mutation_rate=0.35, crossover_rate=0.70, tournament_size=2
    ),
}
CV_FOLDS = 5
SEED = 42


def _setup_logging() -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=LOGS_DIR / "ga_optimization.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filemode="w",
    )


def _save_history_csv(config_name: str, algorithm: str, history: list[dict]) -> Path:
    path = EXPERIMENTS_DIR / f"history_{config_name}_{algorithm.replace(' ', '_')}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["generation", "best", "mean", "std"])
        writer.writeheader()
        writer.writerows(history)
    return path


def _plot_convergence(config_name: str, per_algorithm_history: dict[str, list[dict]]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for algorithm, history in per_algorithm_history.items():
        ax.plot(
            [h["generation"] for h in history],
            [h["best"] for h in history],
            label=algorithm,
            marker="o",
            markersize=3,
        )
    ax.set_xlabel("Geracao")
    ax.set_ylabel("Melhor Fitness (F1 - validacao cruzada)")
    ax.set_title(f"Convergencia do GA - Configuracao {config_name}")
    ax.legend()
    fig.tight_layout()
    path = EXPERIMENTS_DIR / f"convergence_{config_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    _setup_logging()
    EXPERIMENTS_DIR.mkdir(exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    pipeline_steps = build_pipeline().steps

    print("Treinando modelos baseline (Modulo 1)...")
    baseline_results = train_baseline_models()
    for name, result in baseline_results.items():
        m = result["metrics"]
        print(f"  {name:22s}: Acc={m['accuracy']:.4f} | Recall={m['recall']:.4f} | F1={m['f1']:.4f}")

    summary = {
        "baseline": {name: r["metrics"] for name, r in baseline_results.items()},
        "experiments": {},
    }

    for config_name, config in GA_CONFIGS.items():
        print(f"\n=== Experimento {config_name}: {config} ===")
        per_algorithm_history = {}
        summary["experiments"][config_name] = {"config": config, "algorithms": {}}

        for algorithm in ALGORITHMS:
            result = run_experiment(
                algorithm=algorithm,
                X_train=X_train,
                y_train=y_train,
                pipeline_steps=pipeline_steps,
                cv=CV_FOLDS,
                rng=random.Random(SEED),
                **config,
            )
            test_metrics = evaluate(result["fitted_model"], X_test, y_test)
            comparison = compare_to_baseline(baseline_results[algorithm]["metrics"], test_metrics)

            _save_history_csv(config_name, algorithm, result["ga_result"]["history"])
            per_algorithm_history[algorithm] = result["ga_result"]["history"]

            summary["experiments"][config_name]["algorithms"][algorithm] = {
                "best_chromosome": result["ga_result"]["best_chromosome"],
                "cv_fitness": result["ga_result"]["best_fitness"],
                "test_metrics": test_metrics,
                "comparison_vs_baseline": comparison,
            }
            print(
                f"  {algorithm:22s}: CV F1={result['ga_result']['best_fitness']:.4f} "
                f"| Test F1={test_metrics['f1']:.4f} "
                f"(baseline {baseline_results[algorithm]['metrics']['f1']:.4f}, "
                f"delta {comparison['f1']['delta']:+.4f})"
            )

        _plot_convergence(config_name, per_algorithm_history)

    summary_path = EXPERIMENTS_DIR / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResumo salvo em {summary_path}")


if __name__ == "__main__":
    main()
