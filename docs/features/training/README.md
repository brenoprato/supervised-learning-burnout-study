# Feature 3 — Treinamento dos Modelos de Predição

## Descrição

Esta feature treina os três modelos de regressão supervisionada em ordem crescente de complexidade, utilizando os dados pré-processados produzidos pela Feature 2. Todos os modelos são persistidos em disco para uso pelas features de avaliação e scoring.

Corresponde à **SPEC 2.3** e ao **Componente 3 — Treinador de Modelos** do PLAN.

---

## Requisitos Atendidos

| Critério de Aceitação (SPEC 2.3) | Status |
|---|---|
| Regressão Linear treinada e usada como linha de base comparativa | ✅ Atendido |
| Random Forest Regressor treinado com importância relativa de atributos | ✅ Atendido |
| XGBoost Regressor treinado com otimização de hiperparâmetros | ✅ Atendido |
| Os três modelos treinados salvos no repositório | ✅ Atendido |
| Divisão 80/20 com semente fixa (herdada da Feature 2) | ✅ Atendido |

---

## Modelos Treinados

### Modelo 1 — Regressão Linear (baseline)

- **Classe:** `sklearn.linear_model.LinearRegression`
- **Parâmetros:** padrões da biblioteca (sem regularização)
- **Justificativa:** modelo mais simples possível; serve como piso de comparação. Se os modelos complexos não superarem a regressão linear, há algo errado com os dados ou com o pipeline.
- **Interpretação dos coeficientes aprendidos:**

| Feature | Coeficiente |
|---|---|
| Mental Fatigue Score | +0.6698 |
| Resource Allocation | +0.2818 |
| Designation | +0.0437 |
| WFH Setup Available | −0.0170 |
| Gender | +0.0063 |
| days_since_joining | +0.0024 |
| Company Type | +0.0003 |

`Mental Fatigue Score` domina com folga — coerente com a análise de correlação da EDA.

---

### Modelo 2 — Random Forest Regressor

- **Classe:** `sklearn.ensemble.RandomForestRegressor`
- **Parâmetros:** `n_estimators=100`, `random_state=42`
- **Justificativa:** ensemble de árvores robusto a outliers e relações não-lineares; expõe importância de features nativamente.
- **Importância das features aprendida:**

| Feature | Importância |
|---|---|
| Mental Fatigue Score | 0.8194 |
| Resource Allocation | 0.0926 |
| days_since_joining | 0.0573 |
| Designation | 0.0127 |
| WFH Setup Available | 0.0062 |
| Gender | 0.0060 |
| Company Type | 0.0058 |

O Random Forest confirma que `Mental Fatigue Score` explica ~82% da variância preditiva — dominância ainda mais pronunciada do que na regressão linear.

---

### Modelo 3 — XGBoost Regressor (hiperparâmetros otimizados)

- **Classe:** `xgboost.XGBRegressor`
- **Otimização:** `GridSearchCV` com 5-fold cross-validation, métrica `neg_mean_squared_error`
- **Espaço de busca:**

| Hiperparâmetro | Valores testados |
|---|---|
| `n_estimators` | 100, 200 |
| `max_depth` | 3, 5 |
| `learning_rate` | 0.05, 0.1, 0.2 |

Total: 12 combinações × 5 folds = **60 fits**

- **Melhores hiperparâmetros encontrados:**

| Hiperparâmetro | Valor escolhido |
|---|---|
| `learning_rate` | 0.05 |
| `max_depth` | 5 |
| `n_estimators` | 200 |

- **Melhor MSE (CV):** 0.003700
- **Justificativa do XGBoost:** gradient boosting sequencial é estado da arte para dados tabulares; captura relações não-lineares complexas e corrige erros residuais iterativamente.

---

## Artefatos Gerados

| Artefato | Caminho | Descrição |
|---|---|---|
| Regressão Linear | `models/linear_regression.pkl` | Modelo baseline serializado |
| Random Forest | `models/random_forest.pkl` | Ensemble de 100 árvores serializado |
| XGBoost | `models/xgboost_model.pkl` | Modelo com melhores hiperparâmetros serializado |

---

## Como Recarregar os Modelos Sem Re-treinar

```python
import joblib

lr  = joblib.load("models/linear_regression.pkl")
rf  = joblib.load("models/random_forest.pkl")
xgb = joblib.load("models/xgboost_model.pkl")

# Predição (X deve ter as 7 colunas na ordem de FEATURE_COLUMNS)
preds = xgb.predict(X_test)
```

---

## Como Re-executar o Treinamento

```bash
python src/training.py
```

Requer que `data/processed/X_train.csv` e `data/processed/y_train.csv` existam (Feature 2).

---

## Como Rodar os Testes

```bash
python -m pytest tests/test_training.py -v
```

Cobertura dos checkpoints do PLAN:

| Checkpoint | Teste(s) |
|---|---|
| 1 — Linear Regression treinada e serializada | `test_linear_regression_*` |
| 2 — Random Forest com feature importance | `test_random_forest_*` |
| 3 — Espaço de busca XGBoost e GridSearchCV | `test_xgboost_best_params_within_search_space` |
| 4 — XGBoost com melhores params serializado | `test_xgboost_file_created`, `test_saved_xgboost_loads` |
| 5 — Três arquivos presentes em models/ | `test_all_three_model_files_exist` |
| 6 — Semente fixa e reprodutibilidade | `test_random_forest_uses_fixed_seed`, `test_training_is_reproducible` |

**Resultado esperado:** 20 testes passando.
