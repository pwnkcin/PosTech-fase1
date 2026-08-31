# FemHealth AI — Detecção de PCOS

**Tech Challenge FIAP PosTech — Fase 1**

Sistema de suporte ao diagnóstico em saúde feminina utilizando Machine Learning para detecção da Síndrome dos Ovários Policísticos (PCOS) a partir de biomarkers hormonais.

---

## Dataset

**PCOS Disease Dataset** — `PCOS_infertility.csv` (incluso no repositório)

- 541 amostras | 3 features (biomarkers hormonais) | 1 valor inconsistente tratado no pipeline
- **Features:** `AMH (ng/mL)`, `beta-HCG I (mIU/mL)`, `beta-HCG II (mIU/mL)`
- **Variável alvo:** `PCOS (Y/N)` — 0 = Sem PCOS, 1 = Com PCOS
- Fonte: [Kaggle — Prasoon Kottarathil](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos)

---

## Pré-requisitos

- Python 3.12+
- pip

---

## Instalação

```bash
# Cria ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# ou: .venv\Scripts\activate     # Windows

# Instala dependências
pip install -r requirements.txt
```

---

## Execução

```bash
# Abre o Jupyter Lab
jupyter lab

# Abra o arquivo: analysis.ipynb
# Execute todas as células (Run → Run All Cells)
```

O modelo campeão será salvo em `models/pcos_champion.pkl`.

---

## Docker

```bash
# Build
docker build -t femhealth-ai .

# Run
docker run -p 8888:8888 femhealth-ai

# Acesse: http://localhost:8888
```

---

## Estrutura do Projeto

```
TechChallenge/
├── analysis.ipynb          ← Notebook principal (EDA + Modelagem + SHAP)
├── PCOS_infertility.csv    ← Dataset
├── requirements.txt        ← Dependências Python
├── Dockerfile              ← Container Docker
├── README.md               ← Este arquivo
└── models/                 ← Modelos salvos após execução do notebook
```

---

## Conteúdo do Notebook

| Seção | Conteúdo |
|---|---|
| 1. Configuração | Imports e setup |
| 2. Carregamento | Leitura do CSV, inspeção, tipos |
| 3. Pré-processamento | Limpeza do valor 'a', renomeação de colunas, split |
| 4. EDA | Distribuição de classes, boxplots, violins, pair plot, correlação |
| 5. Pipeline | Imputação (mediana) + StandardScaler |
| 6. Modelagem | Regressão Logística, Árvore de Decisão, Random Forest |
| 7. Avaliação | Relatórios, Matrizes de Confusão, Comparação, Validação Cruzada |
| 8. Explicabilidade | Feature Importance, SHAP Beeswarm, SHAP Waterfall |
| 9. Conclusões | Discussão crítica sobre uso clínico e limitações |

---

## Algoritmos e Métricas

**Algoritmos:** Regressão Logística · Árvore de Decisão · Random Forest

**Métrica prioritária:** **Recall** — falsos negativos (PCOS não detectado) têm consequências clínicas graves para a paciente.

---

## Aviso Legal

> Este sistema é uma **ferramenta de apoio ao diagnóstico**. O diagnóstico final é responsabilidade exclusiva do profissional de saúde habilitado. Não substitui consulta médica, exames clínicos ou avaliação especializada.

---

## Fase 2 — Otimização via Algoritmo Genético + LLM

Projeto 1 do Tech Challenge Fase 2: otimização dos hiperparâmetros dos 3 modelos acima via algoritmo genético, e interpretação dos resultados via LLM (Anthropic Claude). Todo o código, testes e o relatório técnico ficam em [`fase-2/`](fase-2/) — ver [`fase-2/RELATORIO_FASE2.md`](fase-2/RELATORIO_FASE2.md).

```bash
cd fase-2

# Instala as dependências novas (anthropic, python-dotenv, pytest) — usa o
# requirements.txt da raiz do TechChallenge, compartilhado com a Fase 1
pip install -r ../requirements.txt

# Roda a suíte de testes (TDD)
python -m pytest tests/ -v

# Roda o baseline + 3 experimentos de GA (3 algoritmos cada) — gera experiments/summary.json
python -m scripts.run_experiments

# Apresenta os resultados formatados para o vídeo de demonstração;
# se ANTHROPIC_API_KEY estiver configurada (copie .env.example para .env),
# também gera explicações da LLM ao vivo.
python -m scripts.present_results
```

Estrutura de `fase-2/`:

```
fase-2/
├── src/
│   ├── data.py                 ← extraído do analysis.ipynb (mesma limpeza/pipeline)
│   ├── baseline.py              ← os 3 modelos Módulo 1, como no notebook
│   ├── hyperparam_spaces.py     ← espaço de busca por algoritmo + decode()
│   ├── genetic_algorithm.py     ← GA genérico (seleção, cruzamento, mutação, execução paralela via joblib)
│   ├── optimization.py          ← orquestra GA + comparação com baseline
│   └── llm_explainer.py         ← prompts + chamada à API da Anthropic
├── scripts/
│   ├── run_experiments.py       ← roda baseline + 3 configs de GA × 3 algoritmos
│   └── present_results.py       ← resumo pronto para o vídeo (+ explicações LLM ao vivo)
├── tests/                        ← 1 arquivo de teste por módulo acima
├── experiments/                  ← histórico de convergência (CSV), gráficos (PNG), summary.json
├── logs/ga_optimization.log      ← log de monitoramento gerado por run_experiments.py
├── .env.example                  ← copie para .env e preencha ANTHROPIC_API_KEY
└── RELATORIO_FASE2.md            ← relatório técnico da Fase 2
```
