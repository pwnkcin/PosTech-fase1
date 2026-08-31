"""Video-demo script: prints a clean, narration-ready summary of the GA
optimization results (baseline vs each experiment vs best-per-algorithm) and,
if ANTHROPIC_API_KEY is set, generates one live LLM diagnosis explanation and
one live LLM optimization explanation.

Run after scripts/run_experiments.py.
Usage: python -m scripts.present_results
"""

import json
from pathlib import Path

from dotenv import load_dotenv

from src.llm_explainer import explain_diagnosis, explain_optimization, get_client

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = ROOT / "experiments" / "summary.json"


def _print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _best_experiment_per_algorithm(summary: dict) -> dict:
    best = {}
    for config_name, exp in summary["experiments"].items():
        for algorithm, data in exp["algorithms"].items():
            f1 = data["test_metrics"]["f1"]
            if algorithm not in best or f1 > best[algorithm]["test_metrics"]["f1"]:
                best[algorithm] = {**data, "config_name": config_name}
    return best


def print_summary(summary: dict) -> dict:
    _print_header("MODULO 1 - BASELINE (sem otimizacao)")
    for algorithm, metrics in summary["baseline"].items():
        print(
            f"  {algorithm:22s}: Acc={metrics['accuracy']:.4f} | "
            f"Recall={metrics['recall']:.4f} | F1={metrics['f1']:.4f}"
        )

    _print_header("3 EXPERIMENTOS DE ALGORITMO GENETICO (avaliados no conjunto de teste)")
    for config_name, exp in summary["experiments"].items():
        print(f"\n[{config_name}] {exp['config']}")
        for algorithm, data in exp["algorithms"].items():
            c = data["comparison_vs_baseline"]
            print(
                f"  {algorithm:22s}: F1 {c['f1']['baseline']:.4f} -> {c['f1']['optimized']:.4f} "
                f"({c['f1']['delta']:+.4f}) | "
                f"Recall {c['recall']['baseline']:.4f} -> {c['recall']['optimized']:.4f} "
                f"({c['recall']['delta']:+.4f})"
            )

    _print_header("MELHOR CONFIGURACAO POR ALGORITMO")
    best = _best_experiment_per_algorithm(summary)
    for algorithm, data in best.items():
        c = data["comparison_vs_baseline"]
        print(
            f"  {algorithm:22s}: melhor config = {data['config_name']} | "
            f"F1 {c['f1']['baseline']:.4f} -> {c['f1']['optimized']:.4f} ({c['f1']['delta']:+.4f})"
        )
        print(f"    hiperparametros otimizados: {data['best_chromosome']}")
    return best


def demo_llm_explanations(summary: dict, best: dict) -> None:
    load_dotenv()
    try:
        client = get_client()
    except RuntimeError as exc:
        print(f"\n(Explicacoes via LLM puladas: {exc})")
        return

    _print_header("EXPLICACAO LLM - PACIENTE DE EXEMPLO")
    explanation = explain_diagnosis(
        client=client,
        biomarkers={"AMH": 8.2, "beta_HCG_I": 1.1, "beta_HCG_II": 0.5},
        prediction_label="Com PCOS",
        probability=0.87,
        top_shap_feature="AMH",
    )
    print(explanation)

    algorithm = "Random Forest"
    if algorithm in best:
        _print_header(f"EXPLICACAO LLM - RESULTADO DA OTIMIZACAO ({algorithm})")
        data = best[algorithm]
        opt_explanation = explain_optimization(
            client=client,
            algorithm=algorithm,
            baseline_metrics=summary["baseline"][algorithm],
            optimized_metrics=data["test_metrics"],
            ga_config=summary["experiments"][data["config_name"]]["config"],
        )
        print(opt_explanation)


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise SystemExit(
            f"{SUMMARY_PATH} nao encontrado. Rode 'python -m scripts.run_experiments' primeiro."
        )
    summary = json.loads(SUMMARY_PATH.read_text())
    best = print_summary(summary)
    demo_llm_explanations(summary, best)


if __name__ == "__main__":
    main()
