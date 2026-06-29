# Relatório Técnico — FemHealth AI: Detecção de PCOS

**Tech Challenge FIAP PosTech — Fase 1**
**Autor:** Nicolas Gomes e William Coelho
**Data:** Junho de 2026

---

## 1. Introdução e Contextualização do Problema

A Síndrome dos Ovários Policísticos (PCOS/SOP) é a endocrinopatia mais comum em mulheres em idade reprodutiva, com prevalência estimada entre 6% e 15%. Caracteriza-se por disfunção ovulatória, hiperandrogenismo e morfologia policística dos ovários, sendo a principal causa de infertilidade anovulatória.

O diagnóstico tardio é frequente: muitas mulheres passam anos sem diagnóstico correto, agravando consequências reprodutivas e aumentando o risco de diabetes tipo 2, síndrome metabólica e doenças cardiovasculares.

**Objetivo do projeto:** Construir modelos de Machine Learning para classificação binária de PCOS a partir de biomarkers hormonais, comparar o desempenho de três algoritmos e selecionar o modelo campeão com base em métricas clínicas prioritárias.

---

## 2. Dataset

**PCOS Disease Dataset** — `PCOS_infertility.csv`
Fonte: [Kaggle — Prasoon Kottarathil](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos)

| Atributo | Valor |
|---|---|
| Total de amostras | 541 |
| Features utilizadas | 3 (biomarkers hormonais) |
| Variável alvo | `PCOS (Y/N)` — 0 = Sem PCOS, 1 = Com PCOS |
| Distribuição de classes | 364 Sem PCOS (67,3%) / 177 Com PCOS (32,7%) |
| Dado inconsistente | Linha 307: AMH = `'a'` — erro de digitação |

### Biomarkers utilizados

| Feature | Descrição Clínica |
|---|---|
| **AMH (ng/mL)** | Anti-Müllerian Hormone — produzido pelos folículos antrais. Valores elevados indicam maior reserva ovariana, característica central da PCOS. AMH > 3,4 ng/mL é frequentemente utilizado como critério diagnóstico auxiliar. |
| **beta-HCG I (mIU/mL)** | Gonadotrofina coriônica humana — primeira medição. Utilizada no contexto de diagnóstico diferencial de infertilidade. |
| **beta-HCG II (mIU/mL)** | Segunda medição de beta-HCG. Alta correlação com a primeira medição (esperado: mesmo hormônio em momentos distintos). |

### Estatísticas descritivas

| Feature | Mínimo | Mediana | Máximo | Desvio Padrão |
|---|---|---|---|---|
| beta_HCG_I | 1,30 | 20,00 | 32.460,97 | 3.348,92 |
| beta_HCG_II | 0,11 | 1,99 | 25.000,00 | 1.603,83 |
| AMH | 0,10 | 3,70 | 66,00 | 5,88 |

**Observação:** beta-HCG apresenta distribuição fortemente assimétrica com outliers extremos, característica esperada: a maioria das pacientes não grávidas tem valores próximos ao limite inferior de detecção, enquanto gestantes apresentam valores muito elevados.

---

## 3. Pré-processamento de Dados

### 3.1 Qualidade dos Dados

A coluna `AMH(ng/mL)` foi lida como tipo `object` pelo pandas, indicando a presença de um valor não numérico. Inspeção identificou a linha 307 com o valor literal `'a'`, provavelmente erro de digitação no prontuário.

**Estratégia adotada:**
- Conversão para numérico com `pd.to_numeric(errors='coerce')` — transforma `'a'` em `NaN`
- Imputação pela mediana no pipeline (antes de qualquer ajuste nos dados de teste)
- Resultado: 1 valor NaN em AMH, zero NaNs nas demais features

### 3.2 Divisão Treino/Teste

- **Split:** 80% treino (432 amostras) / 20% teste (109 amostras)
- **Estratificação:** `stratify=y` — preserva a proporção de classes em ambos os conjuntos
- Treino: 67,4% Sem PCOS / 32,6% Com PCOS
- Teste: 67,0% Sem PCOS / 33,0% Com PCOS

### 3.3 Pipeline de Transformação

```
SimpleImputer(strategy='median') → StandardScaler()
```

| Etapa | Transformação | Justificativa |
|---|---|---|
| `SimpleImputer(median)` | Substitui NaN pela mediana do treino | Mediana é robusta a outliers extremos de beta-HCG |
| `StandardScaler` | Centraliza (μ=0) e normaliza (σ=1) | Essencial para Regressão Logística; beneficia demais modelos |

**Anti-data-leakage:** O pipeline é ajustado (`fit`) exclusivamente nos dados de treino e aplicado (`transform`) ao teste, garantindo que nenhuma informação do conjunto de teste influencie o pré-processamento.

Verificação pós-pipeline: média = 0,0000, desvio padrão = 1,0000, NaNs = 0.

---

## 4. Análise Exploratória de Dados (EDA)

### 4.1 Distribuição das Classes

O dataset apresenta desbalanceamento moderado: 67,3% Sem PCOS e 32,7% Com PCOS. Condizente com a prevalência clínica esperada em populações com suspeita de infertilidade.

### 4.2 Correlação com o Target

AMH apresenta a maior correlação positiva com o diagnóstico de PCOS, corroborando seu reconhecimento como biomarcador central na literatura clínica. beta-HCG I e II apresentam alta correlação entre si e menor correlação com o target.

### 4.3 Insights dos Boxplots e Violins

- **AMH:** Separação mais clara entre as classes — pacientes Com PCOS têm valores consistentemente mais elevados. Isso é biologicamente esperado: ovários policísticos contêm mais folículos antrais, que produzem AMH em maior quantidade.
- **beta-HCG:** Grande variabilidade em ambos os grupos com outliers extremos. A separação entre classes é menos evidente, indicando menor poder discriminativo isolado.

### 4.4 Correlação entre Features

beta-HCG I e II apresentam alta correlação mútua (esperado). AMH tem baixa correlação com beta-HCGs, indicando que os três biomarcadores contribuem com informações complementares ao modelo.

---

## 5. Modelagem

Foram treinados e comparados três algoritmos, cada um embutido em um Pipeline completo (pré-processamento + classificador):

| Algoritmo | Configuração | Justificativa |
|---|---|---|
| **Regressão Logística** | `max_iter=10000` | Baseline interpretável; coeficientes revelam direção e magnitude de cada biomarcador |
| **Árvore de Decisão** | `max_depth=5` | Regras auditáveis (ex: "se AMH > X então PCOS provável"); facilita revisão clínica |
| **Random Forest** | `n_estimators=200` | Ensemble robusto; feature importance nativa; melhor generalização ao reduzir variância |

---

## 6. Resultados

### 6.1 Métricas no Conjunto de Teste

| Modelo | Accuracy | Recall | F1-Score |
|---|---|---|---|
| Regressão Logística | 0,6422 | 0,1111 | 0,1702 |
| Árvore de Decisão | 0,6881 | 0,2778 | 0,3704 |
| **Random Forest** | **0,6606** | **0,3889** | **0,4308** |

### 6.2 Relatório de Classificação — Random Forest (Modelo Campeão)

|  | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Sem PCOS | 0,72 | 0,79 | 0,76 | 73 |
| Com PCOS | 0,48 | 0,39 | 0,43 | 36 |
| **Accuracy** | | | **0,66** | 109 |
| Macro avg | 0,60 | 0,59 | 0,59 | 109 |
| Weighted avg | 0,64 | 0,66 | 0,65 | 109 |

### 6.3 Validação Cruzada — 5-fold (dataset completo)

| Modelo | F1-Score (média ± std) | Recall (média ± std) |
|---|---|---|
| Regressão Logística | 0,2809 ± 0,0505 | 0,1922 ± 0,0639 |
| Árvore de Decisão | 0,2590 ± 0,1068 | 0,2156 ± 0,1281 |
| **Random Forest** | **0,3077 ± 0,0615** | **0,2663 ± 0,0639** |

A validação cruzada revela que a Árvore de Decisão apresenta maior variabilidade (std alto), indicando sensibilidade ao split específico. O Random Forest tem o melhor equilíbrio entre desempenho e estabilidade.

### 6.4 Justificativa das Métricas

**Recall** foi adotado como métrica prioritária por razão clínica: um falso negativo (PCOS não detectada) pode atrasar o tratamento por anos, agravando infertilidade e aumentando riscos metabólicos. O custo de um diagnóstico perdido supera o custo de um falso positivo.

**F1-Score** complementa o Recall ao penalizar excessos de falsos positivos — que também têm custo: exames desnecessários, ansiedade e sobrecarga do sistema de saúde.

**Accuracy** foi evitada como critério primário: com 33% de positivos, um modelo que sempre prediz "Sem PCOS" teria accuracy de 67% — maior que todos os modelos treinados, porém clinicamente inútil.

---

## 7. Explicabilidade dos Modelos

A explicabilidade é tão crítica quanto a acurácia em contexto clínico: o profissional de saúde precisa entender **por que** o modelo fez determinada predição para aceitá-la ou questioná-la.

### 7.1 Feature Importance — Random Forest

O Random Forest computa a importância de cada feature como a redução média de impureza (Gini) ao longo de todas as árvores. AMH emerge como o feature de maior importância, alinhado com o conhecimento clínico estabelecido sobre seu papel como biomarcador de PCOS.

### 7.2 SHAP Values — Explicabilidade Global (Beeswarm)

O gráfico beeswarm mostra, para cada paciente do conjunto de teste, o quanto cada biomarker empurrou a predição para "Com PCOS" (valores SHAP positivos) ou "Sem PCOS" (negativos). A cor indica o valor da feature: vermelho = alto, azul = baixo.

**Leitura esperada:** AMH elevado (vermelho) deve aparecer com SHAP positivo alto, indicando que valores altos de AMH aumentam a probabilidade de PCOS — padrão consistente com a literatura.

### 7.3 SHAP Waterfall — Explicabilidade Individual

Para cada predição individual, o gráfico waterfall mostra como os biomarkers de uma paciente específica desviam a predição a partir do valor base (média do modelo no conjunto de teste). Barras vermelhas aumentam o risco estimado de PCOS; azuis reduzem.

Esse nível de explicabilidade permite ao médico verificar se a predição é coerente clinicamente: por exemplo, se uma paciente foi classificada como "Com PCOS" principalmente por AMH elevado, o raciocínio é clinicamente válido e auditável.

---

## 8. Modelo Campeão e Artefato

**Modelo selecionado:** Random Forest
**Critério:** Maior F1-Score no conjunto de teste (0,4308) com Recall de 0,3889

O modelo campeão foi serializado com `joblib` e salvo em `models/pcos_champion.pkl`, pronto para integração em aplicações de triagem clínica.

---

## 9. Conclusões e Discussão Crítica

### Resumo dos Resultados

| Aspecto | Resultado |
|---|---|
| Dataset | 541 amostras, 3 biomarkers hormonais |
| Dado inconsistente | 1 valor inválido em AMH — tratado no pipeline |
| Desbalanceamento | 67,3% Sem PCOS / 32,7% Com PCOS |
| Biomarker mais relevante | AMH (maior correlação e feature importance) |
| Melhor modelo | Random Forest — F1=0,4308, Recall=0,3889 |

### Pode ser utilizado na prática?

**Como ferramenta de triagem, sim — com ressalvas importantes.**

**Pontos fortes:**
- AMH é clinicamente reconhecido como um dos melhores biomarcadores isolados para PCOS
- O modelo integra automaticamente os três marcadores, ponderando-os de forma não-linear
- Explicabilidade via SHAP permite auditoria de cada predição individual
- Pipeline reprodutível e versionado garante consistência entre ambientes

**Limitações críticas:**

1. **Apenas 3 features:** O diagnóstico de PCOS pelos Critérios de Rotterdam exige avaliação de anovulação, hiperandrogenismo clínico/laboratorial e morfologia ultrassonográfica ovariana. Este modelo cobre apenas aspectos bioquímicos parciais.

2. **Dataset pequeno (541 amostras):** Insuficiente para capturar a variabilidade clínica completa da PCOS. Validação em dataset maior e multicêntrico é necessária antes de qualquer uso clínico real.

3. **Contexto do beta-HCG:** Valores elevados de beta-HCG refletem gestação, não PCOS diretamente. Incluir essa variável sem contexto gestacional pode introduzir confusão em algumas pacientes.

4. **Viés de seleção:** Dados originados em clínica de infertilidade não representam a população geral de mulheres com PCOS, limitando a generalização.

### Recomendação de uso

Utilizar como ferramenta de **triagem de segundo nível**: após coleta de AMH e beta-HCG, o modelo sinaliza casos que devem ser priorizados para avaliação clínica completa (ultrassom transvaginal e dosagem de andrógenos). O diagnóstico final é sempre responsabilidade do médico especialista.

---

## 10. Estrutura do Projeto

```
TechChallenge/
├── analysis.ipynb          ← Notebook principal (EDA + Modelagem + SHAP)
├── PCOS_infertility.csv    ← Dataset
├── requirements.txt        ← Dependências Python
├── Dockerfile              ← Container Docker (Jupyter Lab)
├── README.md               ← Instruções de instalação e execução
├── RELATORIO.md            ← Este relatório técnico
└── models/                 ← Modelos salvos após execução do notebook
    └── pcos_champion.pkl   ← Random Forest serializado
```

---

## Aviso Legal

> **Este sistema é uma ferramenta de apoio ao diagnóstico.** O diagnóstico final de PCOS requer avaliação clínica completa por ginecologista ou endocrinologista, incluindo exame físico, exames laboratoriais e ultrassonografia. Não substitui consulta médica ou avaliação especializada. **O médico sempre deve ter a palavra final.**
