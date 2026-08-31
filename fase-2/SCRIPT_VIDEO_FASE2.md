# Script do Vídeo — Fase 2, Projeto 1: Otimização de Modelos via Algoritmo Genético + LLM

**Tech Challenge FIAP PosTech — Fase 2**
**Apresentador:** Nicolas Gomes
**Duração estimada:** ~13–14 minutos (limite: 15 min)

> Este script cobre os 4 pontos obrigatórios do vídeo pedidos no PDF da Fase 2: demonstração do sistema em execução, explicação dos componentes da solução, apresentação dos resultados da otimização via GA, e demonstração da integração com LLM.

---

## [ABERTURA] — ~1 min

Olá! Sou Nicolas Gomes, aluno do PosTech FIAP, e este é o vídeo de apresentação do Projeto 1 da Fase 2 do Tech Challenge.

Na Fase 1, construímos o **FemHealth AI**: modelos de Machine Learning para detecção de PCOS a partir de 3 biomarkers hormonais — Regressão Logística, Árvore de Decisão e Random Forest, todos com hiperparâmetros escolhidos manualmente.

Na Fase 2, Projeto 1, o desafio foi duplo: **otimizar automaticamente os hiperparâmetros** desses modelos usando um **algoritmo genético**, e **integrar uma LLM** para traduzir os resultados técnicos em explicações clínicas para profissionais de saúde.

Vou mostrar a arquitetura da solução, o algoritmo genético rodando de verdade, os resultados de 3 experimentos com configurações diferentes, e a integração com a LLM gerando explicações ao vivo.

---

## [BLOCO 1 — ARQUITETURA DA SOLUÇÃO] — ~1,5 min

*(Tela: abrir `fase-2/RELATORIO_FASE2.md`, seção 1.1, mostrar o diagrama)*

Todo o código da Fase 2 vive em `TechChallenge/fase-2/`, separado do que já foi entregue e avaliado na Fase 1.

A arquitetura tem 3 camadas:

Primeiro, `src/data.py` e `src/baseline.py` — que reaproduzem exatamente a lógica do notebook da Fase 1: mesma limpeza de dados, mesmo pipeline de imputação e normalização, mesmos 3 modelos com os mesmos hiperparâmetros fixos. Isso garante que a comparação depois seja justa.

Segundo, o núcleo do algoritmo genético: `hyperparam_spaces.py`, que define o espaço de busca de cada algoritmo, e `genetic_algorithm.py`, que implementa seleção, cruzamento, mutação e o loop de gerações. `optimization.py` orquestra os dois e calcula a fitness.

Terceiro, `llm_explainer.py`, que integra a API da Anthropic para gerar as explicações clínicas.

Dois scripts amarram tudo: `run_experiments.py` roda o baseline e os experimentos de GA; `present_results.py` — que vou usar daqui a pouco — apresenta os resultados e chama a LLM ao vivo.

---

## [BLOCO 2 — ALGORITMO GENÉTICO: CODIFICAÇÃO E OPERADORES] — ~2 min

*(Tela: abrir `src/hyperparam_spaces.py`, depois `src/genetic_algorithm.py`)*

Cada indivíduo do algoritmo genético é um **cromossomo**: um dicionário onde cada gene é um hiperparâmetro do modelo. Por exemplo, para a Regressão Logística, os genes são `C`, `penalty` e `class_weight`. Para o Random Forest, são 5 genes: número de árvores, profundidade máxima, mínimo de amostras por split e por folha, e número de features consideradas em cada divisão.

Um detalhe de engenharia importante: o `solver` da Regressão Logística **não é um gene independente** — ele é derivado do `penalty` escolhido. Isso garante que o algoritmo genético nunca gaste avaliações tentando uma combinação de hiperparâmetros que o scikit-learn simplesmente rejeitaria.

Os operadores genéticos, em `genetic_algorithm.py`:

- **Seleção por torneio** — sorteio de alguns indivíduos, vence o de maior fitness.
- **Cruzamento uniforme** — cada gene do filho vem aleatoriamente de um dos dois pais.
- **Mutação** — cada gene pode ser re-sorteado de todo o seu domínio, o que favorece exploração, já que nosso espaço de busca é pequeno.
- **Elitismo** — o melhor indivíduo de cada geração sempre sobrevive, então o fitness nunca piora ao longo das gerações.

E um ponto metodológico crítico: a **fitness é calculada só com validação cruzada sobre o conjunto de treino**, nunca sobre o teste. Se o algoritmo genético "visse" o conjunto de teste durante a busca, a comparação final contra o baseline seria enviesada — um erro clássico de vazamento de dados que decidimos evitar desde o design.

---

## [BLOCO 3 — DEMONSTRAÇÃO AO VIVO: TESTES E EXPERIMENTOS] — ~2 min

*(Tela: terminal, dentro de `fase-2/`)*

Vamos ver o sistema rodando de verdade. Primeiro, a suíte de testes:

```bash
cd fase-2
python -m pytest tests/ -v
```

*(Mostrar os 16 testes passando)*

São 16 testes cobrindo o comportamento observável do algoritmo genético — por exemplo, que a seleção por torneio realmente favorece indivíduos mais aptos, que o cruzamento sempre herda genes válidos dos pais, que a mutação nunca produz um hiperparâmetro fora do domínio permitido — além do espaço de busca e da integração com a LLM.

Agora, o script que roda o baseline e os 3 experimentos de algoritmo genético:

```bash
python -m scripts.run_experiments
```

*(Mostrar a saída rodando — ou, se já tiver rodado antes por questão de tempo, mostrar a saída salva e explicar: "isso já rodei antes porque leva alguns minutos, mas é exatamente este comando")*

Esse comando treina o baseline, roda 3 configurações diferentes de algoritmo genético para os 3 algoritmos — 9 execuções no total — e salva tudo em `experiments/`: histórico de convergência em CSV, gráficos, e um resumo em `summary.json`. A avaliação de fitness de cada geração é paralelizada entre os núcleos do processador via `joblib`, o que é a forma que demos ao requisito de escalabilidade automática do enunciado.

---

## [BLOCO 4 — RESULTADOS DOS 3 EXPERIMENTOS DE GA] — ~2,5 min

*(Tela: mostrar os gráficos de convergência em `experiments/convergence_*.png`, depois a tabela da Seção 4 do relatório)*

Rodamos 3 configurações de algoritmo genético — uma rápida, uma completa e uma exploratória, variando tamanho de população, número de gerações e taxa de mutação — cada uma para os 3 algoritmos.

Os resultados, comparando o modelo original da Fase 1 contra o melhor modelo encontrado pelo GA, no mesmo conjunto de teste:

| Modelo | F1 antes | F1 depois | Recall antes | Recall depois |
|---|---|---|---|---|
| Regressão Logística | 0,17 | **0,42** | 0,11 | 0,39 |
| Árvore de Decisão | 0,37 | **0,45** | 0,28 | 0,44 |
| Random Forest | 0,43 | 0,38 | 0,39 | 0,33 |

Na Regressão Logística, o ganho foi enorme. O motivo: o baseline usava os hiperparâmetros default do scikit-learn, que ignoram o desbalanceamento das classes. O algoritmo genético descobriu sozinho que usar `class_weight="balanced"` resolve a maior parte do problema — o Recall mais que triplicou.

Na Árvore de Decisão, o GA encontrou uma árvore bem mais rasa que a original — praticamente um único split — que generaliza melhor. Com só 3 features e 541 amostras, uma árvore mais profunda tende a decorar ruído.

E aqui vou ser honesto: no **Random Forest, o resultado piorou** em todas as 3 configurações. Investigamos o motivo: o modelo original usa profundidade ilimitada, mas o espaço de busca que definimos para o GA limita a profundidade máxima a 30. Mesmo a configuração que chegou mais perto, com profundidade 26, não alcançou o baseline. Isso não é uma falha do algoritmo genético — ele convergiu de forma consistente para o melhor ponto dentro do espaço definido — é uma limitação de escopo da busca, e está documentada assim no relatório, sem maquiagem.

---

## [BLOCO 5 — DEMONSTRAÇÃO AO VIVO: INTEGRAÇÃO COM LLM] — ~2,5 min

*(Tela: terminal)*

Agora a parte de linguagem natural. Rodando:

```bash
python -m scripts.present_results
```

*(Mostrar a saída até a seção "EXPLICACAO LLM - PACIENTE DE EXEMPLO" rodando ao vivo)*

Esse comando lê o resumo dos experimentos, formata os resultados, e — porque configurei a chave da API da Anthropic — chama a LLM ao vivo para gerar duas explicações: uma para um caso de paciente exemplo, outra interpretando o resultado da otimização do Random Forest.

*(Ler trechos da explicação gerada na tela)*

Note que a resposta segue uma estrutura fixa que definimos no prompt: Resumo, Fatores Determinantes, Recomendação e Limitações, sempre terminando com o aviso de que é uma ferramenta de apoio, não um substituto do profissional de saúde. A LLM identificou corretamente o AMH como o biomarker de maior impacto — informação que veio do SHAP, calculado no modelo, não inventada pela LLM — e relacionou isso com os critérios clínicos de Rotterdam.

Na segunda explicação, sobre a piora do Random Forest, a LLM foi direta: recomendou **não substituir** o modelo em produção, e apontou o recall mais baixo como o dado mais preocupante — exatamente a leitura clínica correta, considerando que falsos negativos em PCOS têm custo alto.

Como qualidade de texto gerado por LLM é subjetiva, avaliamos manualmente com um checklist: menciona o biomarker certo, tem as 4 seções, termina com o aviso, evita afirmar certeza absoluta. As duas respostas passaram em todos os itens — isso está documentado no relatório.

---

## [BLOCO 6 — DESAFIOS TÉCNICOS] — ~1,5 min

Alguns desafios que valem mencionar:

Vazamento de dados na busca de hiperparâmetros — resolvido fixando a fitness do GA à validação cruzada sobre o treino, nunca o teste.

Combinações inválidas de hiperparâmetros — resolvido tornando o solver da Regressão Logística derivado do penalty, e não um gene independente.

E um bug real que encontramos testando com a API de verdade: a primeira explicação gerada pela LLM veio cortada no meio de uma frase. Investigamos e o `stop_reason` da resposta confirmava: `max_tokens`. O limite de tokens que tínhamos configurado era baixo demais para a explicação mais longa. Corrigimos e voltamos a testar — as explicações que mostrei há pouco já são da versão corrigida.

---

## [BLOCO 7 — CONCLUSÕES E LIMITAÇÕES] — ~1 min

Para fechar: o algoritmo genético trouxe ganhos reais e explicáveis para Regressão Logística e Árvore de Decisão, e uma regressão honesta e diagnosticada no Random Forest — que documentamos como limitação do espaço de busca, não escondemos ajustando os números depois.

A integração com LLM cumpriu o que o enunciado pedia: traduzir resultados técnicos em insights acionáveis para profissionais de saúde, com prompt engineering estruturado, e prepara a base para a integração com dados textuais que vem no Módulo 3.

A limitação de fundo continua sendo a mesma da Fase 1: um dataset pequeno, com poucas features. Otimizar hiperparâmetros tem um teto — a maior alavanca de performance aqui seria mais dados e mais features clínicas, não mais geração de algoritmo genético.

---

## [ENCERRAMENTO] — ~30 seg

O repositório completo — código, testes, experimentos e o relatório técnico da Fase 2 — está disponível no GitHub, na pasta `TechChallenge/fase-2/`.

Obrigado pela atenção — Nicolas Gomes, PosTech FIAP, Fase 2.

---

*Duração estimada por bloco:*

| Bloco | Conteúdo | Tempo |
|---|---|---|
| Abertura | Apresentação | ~1 min |
| Bloco 1 | Arquitetura da solução | ~1,5 min |
| Bloco 2 | GA: codificação e operadores | ~2 min |
| Bloco 3 | Demo ao vivo: testes + experimentos | ~2 min |
| Bloco 4 | Resultados dos 3 experimentos de GA | ~2,5 min |
| Bloco 5 | Demo ao vivo: integração com LLM | ~2,5 min |
| Bloco 6 | Desafios técnicos | ~1,5 min |
| Bloco 7 | Conclusões e limitações | ~1 min |
| Encerramento | | ~30 seg |
| **Total** | | **~14 min** |
