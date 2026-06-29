# Arquitetura do Sistema — Predição de Burnout

## Visão Geral

O sistema é um **pipeline sequencial de ciência de dados** composto por cinco componentes funcionais executados em ordem. Cada componente tem responsabilidade única, consome artefatos produzidos pelo anterior e persiste seus próprios artefatos em diretórios dedicados.

Não há servidor, API nem interface web. O produto final é um conjunto de scripts Python modulares, modelos persistidos, relatórios visuais e uma função de scoring reutilizável.

---

## Dataset

- **Nome:** Remote Work Health Impact Survey 2025
- **Fonte:** https://www.kaggle.com/datasets/pratyushpuri/remote-work-health-impact-survey-2025
- **Autor:** Pratyush Puri (Kaggle, junho de 2025)
- **Registros:** 3.157 (sem nulos no alvo)
- **Colunas originais:** 14
- **Variável alvo:** `Burnout_Level` (categórica ordinal: Low=0, Medium=1, High=2)
- **Natureza:** survey auto-reportado, contexto pós-pandemia, regimes remoto/híbrido/presencial

---

## Diagrama do Pipeline

```
datasets/survey_2025.csv
        │
        ▼
┌───────────────────┐
│  Componente 1     │  src/eda.py
│  Explorador EDA   │──► reports/figures/ (5 gráficos)
└────────┬──────────┘──► reports/results/eda_summary.txt
         │ diagnóstico de qualidade
         ▼
┌───────────────────┐
│  Componente 2     │  src/preprocessing.py
│  Transformador    │──► data/processed/ (X_train, X_test, y_train, y_test)
└────────┬──────────┘──► models/ (imputer.pkl, scaler.pkl)
         │ dados limpos + split 80/20
         ▼
┌───────────────────┐
│  Componente 3     │  src/training.py
│  Treinador        │──► models/ (linear_regression.pkl, random_forest.pkl,
└────────┬──────────┘            xgboost_model.pkl)
         │ três modelos serializados
         ▼
┌───────────────────┐
│  Componente 4     │  src/evaluation.py
│  Avaliador        │──► reports/figures/ (7 gráficos)
└────────┬──────────┘──► reports/results/ (model_comparison.csv, conclusion.txt)
         │ melhor modelo identificado
         ▼
┌───────────────────┐
│  Componente 5     │  src/scoring.py
│  Scoring          │──► predict_burnout(employee) → (score, risco)
└───────────────────┘
```

---

## Componentes

### Componente 1 — Explorador de Dados (`src/eda.py`)
- **Entrada:** `datasets/survey_2025.csv`
- **Saída:** 5 visualizações em `reports/figures/` + `reports/results/eda_summary.txt`
  - `burnout_distribution.png` — distribuição de Burnout_Level (barras)
  - `correlation_heatmap.png` — correlações de Pearson entre variáveis numéricas
  - `burnout_by_work_location.png` — distribuição por regime de trabalho
  - `burnout_by_gender.png` — distribuição por gênero
  - `burnout_by_industry.png` — percentual de burnout alto por setor
- **Dependências:** nenhuma (ponto de entrada do pipeline)

### Componente 2 — Transformador de Dados (`src/preprocessing.py`)
- **Entrada:** `datasets/survey_2025.csv`
- **Saída:** `data/processed/` (4 CSVs) + `models/imputer.pkl` + `models/scaler.pkl`
- **Transformações:**
  1. Encoding do alvo ordinal (Low=0, Medium=1, High=2)
  2. Derivação de `has_physical_issue` (binário a partir de `Physical_Health_Issues`)
  3. Preenchimento de `Mental_Health_Status` NaN → "None"
  4. Descarte de `Survey_Date` e `Physical_Health_Issues`
  5. Encoding categórico via `ENCODING_MAP` (ordinal + label)
  6. Split 80/20 com `SEED=42` (2.525 treino, 632 teste)
  7. Imputação por mediana (fit somente no treino)
  8. MinMaxScaler (fit somente no treino)
- **Decisão arquitetural:** não há `days_since_joining` — o dataset não contém data de admissão

### Componente 3 — Treinador de Modelos (`src/training.py`)
- **Entrada:** `data/processed/X_train.csv`, `data/processed/y_train.csv`
- **Saída:** `models/linear_regression.pkl`, `models/random_forest.pkl`, `models/xgboost_model.pkl`
- **Modelos:** LinearRegression (baseline) → RandomForestRegressor (100 árvores) → XGBRegressor (GridSearchCV, 60 fits)
- **Melhores hiperparâmetros XGBoost:** `learning_rate=0.05`, `max_depth=3`, `n_estimators=100`

### Componente 4 — Avaliador de Modelos (`src/evaluation.py`)
- **Entrada:** `data/processed/` (todos os splits) + `models/` (3 modelos)
- **Saída:** 7 gráficos em `reports/figures/` + 3 arquivos em `reports/results/`
- **Métricas:** MAE, RMSE, R² no teste + K-Fold k=5 no treino + análise de equidade por Gender e Work_Arrangement
- **Resultados:** R²≈0,04 em todos os modelos (sinal fraco em dados de survey)

### Componente 5 — Função de Scoring (`src/scoring.py`)
- **Entrada:** dict com atributos brutos do funcionário + artefatos de `models/`
- **Saída:** `(score: float, risco: str)` onde `score = clamp(raw_prediction / 2, 0, 1)`
- **Dependências:** Componentes 2 e 3 (artefatos de pré-processamento e modelo XGBoost)
- **Features de entrada (12):** Age, Gender, Region, Industry, Job_Role, Work_Arrangement, Hours_Per_Week, Mental_Health_Status, Work_Life_Balance_Score, has_physical_issue, Social_Isolation_Score, Salary_Range

---

## Módulo de Configuração (`src/config.py`)

Centraliza todas as constantes compartilhadas:
- `SEED = 42` — semente global para reprodutibilidade
- `DATASET_PATH` — caminho para `datasets/survey_2025.csv`
- Caminhos de todos os diretórios e arquivos
- `TARGET_COLUMN = "Burnout_Level"`, `TARGET_ENCODING = {"Low": 0, "Medium": 1, "High": 2}`
- `ENCODING_MAP` — mapeamentos completos de todas as variáveis categóricas
- `FEATURE_COLUMNS` — lista ordenada das 12 features finais
- `XGB_PARAM_GRID` — espaço de busca de hiperparâmetros

---

## Estrutura de Diretórios

```
ia_final/
├── datasets/          # Dataset original (não versionado)
│   └── survey_2025.csv
├── data/
│   └── processed/     # Artefatos do Componente 2
├── models/            # Artefatos dos Componentes 2 e 3
├── reports/
│   ├── figures/       # Artefatos dos Componentes 1 e 4
│   └── results/       # Artefatos do Componente 4
├── src/               # Scripts do pipeline
│   ├── config.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   └── scoring.py
├── tests/             # Testes por módulo
├── docs/              # Documentação
│   ├── architecture.md
│   └── limitations.md
├── infra/             # Configuração de infraestrutura
│   └── setup.md
├── main.py            # Orquestrador do pipeline completo
└── prever.py          # Interface interativa de terminal
```

---

## Decisões Arquiteturais Relevantes

| Decisão | Justificativa |
|---|---|
| Splits salvos como CSVs separados | Permite que Componentes 3, 4 e 5 sejam re-executados independentemente sem re-fazer o split |
| `imputer.pkl` e `scaler.pkl` salvos em `models/` | Necessários para Componente 5 replicar exatamente as transformações do treino |
| Cache de artefatos em `scoring.py` | Evita I/O repetido em chamadas múltiplas à função de scoring |
| Sem `days_since_joining` | Dataset não contém data de admissão; `Survey_Date` (todos em junho/2025) não tem valor preditivo |
| `has_physical_issue` como binário | `Physical_Health_Issues` é texto multi-label complexo; binarização preserva informação com mínima complexidade |
| `Mental_Health_Status` NaN → "None" | NaN indica ausência de diagnóstico reportado — categoria distinta, não informação faltante aleatória |
| Score = clamp(raw/2, 0, 1) | Target ordinal 0-2; divisão por 2 normaliza para [0,1] mantendo compatibilidade com as faixas de risco (RN-02) |
| MinMaxScaler em vez de StandardScaler | Mantém features em [0,1]; escala consistente com score normalizado do alvo |
