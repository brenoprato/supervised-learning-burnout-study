# Feature 2 — Pré-processamento dos Dados

## Descrição

Esta feature transforma o dataset bruto em artefatos limpos, padronizados e prontos para treinamento dos modelos de predição de burnout. Todas as transformações são rastreáveis, reprodutíveis e livres de vazamento de dados (data leakage).

Corresponde à **SPEC 2.2** e ao **Componente 2 — Transformador de Dados** do PLAN.

---

## Requisitos Atendidos

| Critério de Aceitação (SPEC 2.2) | Status |
|---|---|
| Valores ausentes tratados por imputação estatística (média ou mediana) | ✅ Atendido |
| Variáveis categóricas convertidas para representação numérica | ✅ Atendido |
| Atributo derivado de tempo de empresa criado a partir da data de admissão | ✅ Atendido |
| Atributos numéricos normalizados por escalonamento | ✅ Atendido |
| Dataset resultante sem valores ausentes e pronto para treinamento | ✅ Atendido |

---

## Transformações Aplicadas

### 1. Remoção de linhas com target ausente

Linhas onde `Burn Rate` é nulo são removidas antes de qualquer outra operação (1.124 linhas, ~4,94%). A variável alvo não pode ser imputada para regressão supervisionada — imputá-la introduziria ruído artificial no treinamento.

Dataset após remoção: **21.626 registros**.

### 2. Remoção do identificador

A coluna `Employee ID` é descartada por ser um identificador opaco sem valor preditivo.

### 3. Feature engineering — `days_since_joining` (RN-05)

A coluna `Date of Joining` é convertida para o atributo derivado `days_since_joining`:

```
days_since_joining = REFERENCE_DATE - Date of Joining
```

**Data de referência fixa:** `2016-01-01`
Escolhida por ser posterior à maior data de admissão do dataset, garantindo valores sempre positivos e reprodutibilidade em qualquer execução futura.

A coluna original `Date of Joining` é removida após a derivação.

### 4. Encoding binário de variáveis categóricas

Todas as variáveis categóricas são binárias; o mapeamento direto é suficiente e mais interpretável que LabelEncoder:

| Coluna | Mapeamento |
|---|---|
| `Gender` | Female → 0, Male → 1 |
| `Company Type` | Product → 0, Service → 1 |
| `WFH Setup Available` | No → 0, Yes → 1 |

**Justificativa:** OneHotEncoding não é necessário para variáveis binárias e adicionaria colunas redundantes; o mapeamento direto preserva a semântica e é compatível com todos os modelos.

### 5. Divisão treino/teste — 80% / 20% (RN-06)

- **Treino:** 17.300 registros (80,0%)
- **Teste:** 4.326 registros (20,0%)
- **Semente:** `SEED = 42` (definida em `src/config.py`)

A divisão ocorre **antes** do fit de imputação e escalonamento para prevenir data leakage.

### 6. Imputação por mediana

Aplicada apenas nas colunas com valores ausentes. O `SimpleImputer` é **ajustado exclusivamente no conjunto de treino** e depois aplicado ao conjunto de teste.

| Coluna | Mediana (treino) | Nulos antes |
|---|---|---|
| `Resource Allocation` | 4,0 | ~6,07% |
| `Mental Fatigue Score` | 5,9 | ~9,31% |

**Justificativa da mediana sobre a média:** as distribuições dessas colunas são aproximadamente simétricas (verificado na EDA), então mediana e média são próximas. A mediana é mais robusta a possíveis outliers e produz valores consistentes com a escala original (1–10).

### 7. Escalonamento — MinMaxScaler

O `MinMaxScaler` é **ajustado exclusivamente no conjunto de treino** e aplicado a ambos os conjuntos.

**Justificativa do MinMaxScaler:** mantém todas as features no intervalo [0, 1], consistente com o `Burn Rate` (target também em [0, 1]). Adequado para Regressão Linear, Random Forest e XGBoost. `StandardScaler` foi descartado por pressupor distribuição gaussiana, não garantida neste dataset.

---

## Features do Dataset Final

Ordem das colunas em `X_train.csv` e `X_test.csv`:

| # | Feature | Tipo original | Transformação |
|---|---|---|---|
| 1 | `days_since_joining` | Data | Derivada + MinMaxScaler |
| 2 | `Gender` | Categórica binária | Encoding 0/1 + MinMaxScaler |
| 3 | `Company Type` | Categórica binária | Encoding 0/1 + MinMaxScaler |
| 4 | `WFH Setup Available` | Categórica binária | Encoding 0/1 + MinMaxScaler |
| 5 | `Designation` | Numérica ordinal | MinMaxScaler |
| 6 | `Resource Allocation` | Numérica contínua | Imputação mediana + MinMaxScaler |
| 7 | `Mental Fatigue Score` | Numérica contínua | Imputação mediana + MinMaxScaler |

---

## Artefatos Gerados

| Artefato | Caminho | Descrição |
|---|---|---|
| Features treino | `data/processed/X_train.csv` | 17.300 × 7 — features escalonadas |
| Features teste | `data/processed/X_test.csv` | 4.326 × 7 — features escalonadas |
| Target treino | `data/processed/y_train.csv` | 17.300 valores de Burn Rate |
| Target teste | `data/processed/y_test.csv` | 4.326 valores de Burn Rate |
| Imputador | `models/imputer.pkl` | `SimpleImputer(strategy="median")` ajustado no treino |
| Escalonador | `models/scaler.pkl` | `MinMaxScaler()` ajustado no conjunto de treino |

O `imputer.pkl` e o `scaler.pkl` são necessários para a **Feature 5 (Scoring Individual)** aplicar as mesmas transformações a dados novos.

---

## Como Re-executar

```bash
python src/preprocessing.py
```

Requer que `datasets/train.csv` exista (baixado na Fase 0).

---

## Como Rodar os Testes

```bash
python -m pytest tests/test_preprocessing.py -v
```

Cobertura dos checkpoints do PLAN:

| Checkpoint | Teste(s) |
|---|---|
| 1 — Estratégia de imputação aplicada | `test_no_nulls_after_preprocessing` |
| 2 — Dataset livre de nulos | `test_no_nulls_after_preprocessing` |
| 3 — Encoding aplicado e verificado | `test_encoding_*`, `test_encoded_columns_are_numeric` |
| 4 — `days_since_joining` derivado | `test_days_since_joining_*` |
| 5 — Normalização sem data leakage | `test_no_data_leakage_scaler`, `test_scaler_fit_only_on_train` |
| 6 — Divisão 80/20 com semente fixa | `test_train_test_split_proportions`, `test_split_is_reproducible` |
| 7 — Artefatos persistidos | `test_processed_files_exist`, `test_model_artifacts_exist` |

**Resultado esperado:** 19 testes passando.
