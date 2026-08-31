"""Turns model predictions + SHAP values into physician-facing explanations
via the Anthropic API. Prompt-building is pure (no network); the network call
is isolated in `_call_llm` so it's the only thing tests need to mock."""

import os

MODEL = "claude-sonnet-5"
MAX_TOKENS = 2048

SYSTEM_PROMPT = """Voce e um assistente que traduz resultados de um modelo de \
machine learning para deteccao de PCOS (Sindrome dos Ovarios Policisticos) em \
linguagem clara para profissionais de saude. Estruture SEMPRE a resposta em \
quatro secoes: Resumo, Fatores Determinantes, Recomendacao, Limitacoes. \
Termine sempre com o aviso: esta e uma ferramenta de apoio, o diagnostico \
final e responsabilidade do profissional de saude habilitado. Nunca afirme \
certeza absoluta."""


def get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY nao configurada. Copie .env.example para .env "
            "e preencha a chave antes de gerar explicacoes."
        )
    import anthropic

    return anthropic.Anthropic(api_key=api_key)


def _call_llm(client, user_prompt: str) -> str:
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as exc:
        raise RuntimeError(f"Falha ao chamar a API da Anthropic: {exc}") from exc
    return response.content[0].text


def build_diagnosis_prompt(
    biomarkers: dict, prediction_label: str, probability: float, top_shap_feature: str
) -> str:
    biomarker_lines = "\n".join(f"- {name}: {value}" for name, value in biomarkers.items())
    return f"""Dados da paciente:
{biomarker_lines}

Predicao do modelo: {prediction_label} (probabilidade: {probability:.2f})
Biomarker de maior impacto na predicao (SHAP): {top_shap_feature}

Explique este resultado para um(a) medico(a), destacando o papel do \
biomarker de maior impacto."""


def explain_diagnosis(
    client, biomarkers: dict, prediction_label: str, probability: float, top_shap_feature: str
) -> str:
    prompt = build_diagnosis_prompt(biomarkers, prediction_label, probability, top_shap_feature)
    return _call_llm(client, prompt)


def build_optimization_prompt(
    algorithm: str, baseline_metrics: dict, optimized_metrics: dict, ga_config: dict
) -> str:
    return f"""Um algoritmo genetico otimizou os hiperparametros do modelo \
"{algorithm}" usado para deteccao de PCOS.

Configuracao do algoritmo genetico: {ga_config}
Metricas do modelo original: {baseline_metrics}
Metricas do modelo otimizado: {optimized_metrics}

Escreva um resumo, em linguagem natural para gestores de um hospital, sobre \
o que essa otimizacao significa na pratica."""


def explain_optimization(
    client, algorithm: str, baseline_metrics: dict, optimized_metrics: dict, ga_config: dict
) -> str:
    prompt = build_optimization_prompt(algorithm, baseline_metrics, optimized_metrics, ga_config)
    return _call_llm(client, prompt)
