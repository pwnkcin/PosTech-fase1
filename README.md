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
