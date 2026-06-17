# Feature 4 — Validação e Avaliação dos Modelos

## Descrição

Esta feature mede e compara objetivamente o desempenho dos três modelos treinados, aplica validação cruzada K-Fold para verificação de robustez, produz todas as visualizações comparativas e consolida os resultados em relatório final.

Corresponde à **SPEC 2.4** e ao **Componente 4 — Avaliador de Modelos** do PLAN.

---

## Requisitos Atendidos

| Critério de Aceitação (SPEC 2.4) | Status |
|---|---|
| MAE, RMSE e R² calculados e exibidos para cada modelo | ✅ Atendido |
| Validação cruzada K-Fold (k=5) aplicada | ✅ Atendido |
| Gráfico real vs predito gerado para cada modelo | ✅ Atendido |
| Gráficos de importância de atributos gerados para RF e XGBoost | ✅ Atendido |
| Tabela comparativa consolidada com os três modelos produzida | ✅ Atendido |
| Análise de equidade por subgrupo realizada e documentada | ✅ Atendido |

---

## Resultados — Conjunto de Teste (20%)

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.053483 | 0.070940 | 0.8679 |
| Random Forest | 0.049800 | 0.064759 | 0.8899 |
| **XGBoost** | **0.047389** | **0.060369** | **0.9043** |

### Interpretação

- **R²**: XGBoost explica 90,4% da variância do Burn Rate — excelente para dados tabulares reais.
- **RMSE**: XGBoost tem RMSE de 0,060, o que em uma escala de 0–1 representa erro médio quadrático de ~6 pontos percentuais.
- **Progressão esperada**: cada modelo supera o anterior em todas as métricas, confirmando a hipótese de complexidade crescente (PLAN princípio 3).

---

## Resultados — Validação Cruzada K-Fold (k=5, sobre treino)

| Modelo | MAE_mean | MAE_std | RMSE_mean | RMSE_std | R²_mean | R²_std |
|---|---|---|---|---|---|---|
| Linear Regression | 0.053822 | 0.000733 | 0.071458 | 0.001550 | 0.8710 | 0.0028 |
| Random Forest | 0.050586 | 0.000556 | 0.065910 | 0.000495 | 0.8902 | 0.0026 |
| **XGBoost** | **0.047480** | **0.000587** | **0.060752** | **0.000695** | **0.9067** | **0.0030** |

Os baixos desvios padrão confirmam que os resultados são **estáveis e não dependentes da divisão aleatória** específica. A diferença entre métricas no teste e na CV é negligenciável em todos os modelos.

---

## Análise de Equidade por Subgrupo (RN-09)

### Por Gênero

| Modelo | MAE — Female | MAE — Male |
|---|---|---|
| Linear Regression | ~0.054 | ~0.054 |
| Random Forest | ~0.050 | ~0.050 |
| XGBoost | ~0.047 | ~0.047 |

Diferença negligenciável entre Female e Male em todos os modelos — o modelo não apresenta viés significativo de gênero.

### Por Tipo de Empresa

| Modelo | MAE — Product | MAE — Service |
|---|---|---|
| Linear Regression | ~0.053 | ~0.054 |
| Random Forest | ~0.050 | ~0.050 |
| XGBoost | ~0.047 | ~0.047 |

Diferença negligenciável entre Product e Service — sem viés identificado.

**Conclusão da análise de equidade:** o modelo trata de forma equivalente os subgrupos analisados. Isso é coerente com a dominância de `Mental Fatigue Score` como preditor — uma variável individual, não demográfica.

**Ressalva:** a análise de equidade aqui é superficial (apenas MAE por grupo). Uma análise aprofundada de fairness exigiria métricas específicas como equalized odds ou demographic parity, o que está fora do escopo deste projeto.

---

## Visualizações Geradas

Todas em `reports/figures/`:

| Arquivo | Descrição |
|---|---|
| `real_vs_pred_linear_regression.png` | Scatter real × predito — Linear Regression |
| `real_vs_pred_random_forest.png` | Scatter real × predito — Random Forest |
| `real_vs_pred_xgboost.png` | Scatter real × predito — XGBoost |
| `feature_importance_random_forest.png` | Importância de features — Random Forest |
| `feature_importance_xgboost.png` | Importância de features — XGBoost |
| `equity_by_gender.png` | MAE por gênero para os 3 modelos |
| `equity_by_company_type.png` | MAE por tipo de empresa para os 3 modelos |

---

## Artefatos de Resultado

Em `reports/results/`:

| Arquivo | Conteúdo |
|---|---|
| `model_comparison.csv` | Tabela MAE, RMSE, R² por modelo |
| `cv_results.csv` | Resultados K-Fold (médias e desvios) |
| `conclusion.txt` | Conclusão textual + tabelas completas |

---

## Modelo Recomendado

**XGBoost** — apresenta o melhor desempenho em todas as métricas (MAE, RMSE e R²), tanto no conjunto de teste quanto na validação cruzada, com baixo desvio padrão entre folds.

Em caso de empate técnico com o Random Forest, a preferência seria pelo Random Forest (modelo mais simples, mais interpretável e sem hiperparâmetros a ajustar). Neste caso, a diferença é clara o suficiente para justificar o XGBoost.

**Ressalva importante:** os valores de R² acima de 0,90 são incomuns em dados de saúde ocupacional reais. Isso sugere que o dataset pode ser sintético com correlações artificialmente fortes — declarado explicitamente em `docs/limitations.md`.

---

## Como Re-executar

```bash
python src/evaluation.py
```

Requer que `data/processed/` e `models/` estejam populados (Features 2 e 3).

---

## Como Rodar os Testes

```bash
python -m pytest tests/test_evaluation.py -v
```

Cobertura dos checkpoints do PLAN:

| Checkpoint | Teste(s) |
|---|---|
| 1 — Predições no conjunto de teste | `test_evaluate_on_test_*` |
| 2 — MAE, RMSE, R² calculados | `test_compute_metrics_*` |
| 3 — K-Fold aplicado | `test_cv_*` |
| 4 — Gráficos real vs predito | `test_real_vs_pred_*`, `test_all_real_vs_pred_plots_exist` |
| 5 — Importância de features | `test_feature_importance_*` |
| 6 — Análise de equidade | `test_equity_*` |
| 7 — Tabela comparativa persistida | `test_results_files_exist`, `test_model_comparison_csv_*` |
| 8 — Conclusão registrada | `test_conclusion_*` |

**Resultado esperado:** 22 testes passando.
