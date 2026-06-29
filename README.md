# Predição de Burnout em Funcionários

Sistema de predição de risco de esgotamento profissional (burnout) baseado em aprendizado de máquina. Dado um conjunto de atributos de um funcionário, o sistema retorna um score contínuo de 0 a 1 e uma classificação de risco (baixo, moderado ou alto).

**Dataset:** [Remote Work Health Impact Survey 2025](https://www.kaggle.com/datasets/pratyushpuri/remote-work-health-impact-survey-2025) — Pratyush Puri (Kaggle, junho/2025) — 3.157 registros, contexto pós-pandemia, regimes remoto/híbrido/presencial.

---

## Alinhamento com os ODS da ONU

| ODS | Conexão |
|---|---|
| **ODS 3 — Saúde e Bem-Estar** | Identificação precoce de risco de burnout, permitindo intervenções preventivas antes do agravamento |
| **ODS 8 — Trabalho Decente** | Apoio à manutenção de condições laborais saudáveis, reduzindo rotatividade e absenteísmo |
| **ODS 10 — Redução das Desigualdades** | Análise de equidade por gênero e regime de trabalho para identificar disparidades nas predições |

---

## Resultados Obtidos

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| **Regressão Linear (baseline)** | **0,635** | **0,764** | **0,040** |
| Random Forest | 0,665 | 0,784 | −0,011 |
| XGBoost | 0,640 | 0,766 | 0,036 |

> **Nota:** O alvo é ordinal (Low=0, Medium=1, High=2). MAE na escala 0–2; score normalizado = predição/2 para interface com usuário. R² baixo é esperado em dados de survey auto-reportados com sinal preditivo fraco (max correlação Pearson ≈ 0,044).

Validação cruzada K-Fold (k=5): Regressão Linear R²_mean=0,021 (dp=0,021); XGBoost MAE_mean=0,596 (dp=0,009).

**Modelo em uso no scoring:** XGBoost com `learning_rate=0.05`, `max_depth=3`, `n_estimators=100`.

**Feature mais importante (RF):** `Age` (16,5%) e `Hours_Per_Week` (15,3%). No XGBoost: `Work_Arrangement` (29,3%).

**Achado exploratório:** Trabalhadores em regime *remote* têm burnout médio 1,33 vs. 0,96 no presencial.

---

## Configuração do Ambiente

```bash
# 1. Criar e ativar ambiente virtual
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 2. Baixar o dataset
# Faça download do survey_2025.csv em:
# https://www.kaggle.com/datasets/pratyushpuri/remote-work-health-impact-survey-2025
# E coloque em: datasets/survey_2025.csv

# 3. Instalar dependências
pip install -r requirements.txt
```

---

## Execução do Pipeline

### Pipeline completo

```bash
python main.py
```

### Etapas individuais

```bash
python src/eda.py            # Feature 1 — Análise Exploratória
python src/preprocessing.py  # Feature 2 — Pré-processamento
python src/training.py       # Feature 3 — Treinamento dos Modelos
python src/evaluation.py     # Feature 4 — Avaliação e Comparação
python src/scoring.py        # Feature 5 — Demonstração do Scoring
```

---

## Uso da Função de Scoring

```python
import sys
sys.path.insert(0, "src")
from scoring import predict_burnout

score, risco = predict_burnout({
    "Age":                     40,
    "Gender":                  "Male",
    "Region":                  "North America",
    "Industry":                "Finance",
    "Job_Role":                "Project Manager",
    "Work_Arrangement":        "Hybrid",
    "Hours_Per_Week":          50,
    "Mental_Health_Status":    "Anxiety",
    "Work_Life_Balance_Score": 3,
    "has_physical_issue":      0,
    "Social_Isolation_Score":  3,
    "Salary_Range":            "$60K-80K",
})
print(f"Score: {score:.4f} | Risco: {risco}")
# Score: 0.5630 | Risco: moderado
```

---

## Interface Interativa

```bash
python prever.py
```

Coleta 13 campos via menus numerados (gênero, cargo, regime, horas, saúde mental, etc.) e exibe resultado:

```
==================================================
   RESULTADO — FUNCIONÁRIO
==================================================
  Score    : 0.5630  [###########         ]
  Risco    : MODERADO
  Orientação: Atenção recomendada. Acompanhar de perto.
==================================================
```

---

## Testes

```bash
python -m pytest tests/ -v
```

**111 testes passando** cobrindo todas as 5 features.

---

## Estrutura do Projeto

```
ia_final/
├── datasets/              # Dataset original (não versionado)
│   └── survey_2025.csv    # Remote Work Health Impact Survey 2025
├── data/processed/        # Dados pré-processados (X_train, X_test, y_train, y_test)
├── models/                # Modelos e transformadores serializados
├── reports/
│   ├── figures/           # Gráficos gerados (EDA + avaliação)
│   └── results/           # Métricas e conclusão final
├── src/                   # Scripts do pipeline
│   ├── config.py          # Constantes globais
│   ├── eda.py             # Feature 1
│   ├── preprocessing.py   # Feature 2
│   ├── training.py        # Feature 3
│   ├── evaluation.py      # Feature 4
│   └── scoring.py         # Feature 5
├── tests/                 # Testes unitários e de integração
├── docs/                  # Documentação
│   ├── architecture.md
│   └── limitations.md
├── infra/                 # Configuração de infraestrutura
│   └── setup.md           # Instruções de setup
├── main.py                # Orquestrador
├── prever.py              # Interface interativa de terminal
├── requirements.txt
```
---

## Limitações
Ver [docs/limitations.md](docs/limitations.md) para declaração completa. Resumo:

- Dataset de survey auto-reportado — R²≈0,04 é típico para dados reais de saúde ocupacional; correlações fracas limitam o poder preditivo
- Score é preditivo, **não diagnóstico** — não substitui avaliação profissional de saúde
- Análise de equidade superficial — viés não pode ser descartado sem auditoria formal de *fairness*
- Sem atualização do modelo — padrões de burnout mudam ao longo do tempo
