# Feature 5 — Função de Scoring Individual

## Descrição

Esta feature disponibiliza uma função autossuficiente que recebe os atributos brutos de um único funcionário e retorna o score contínuo de risco de burnout (0–1) acompanhado da classificação de risco correspondente. A função pode ser importada e chamada de forma independente, sem necessidade de re-executar o pipeline completo.

Corresponde à **SPEC 2.5** e ao **Componente 5 — Função de Scoring Individual** do PLAN.

---

## Requisitos Atendidos

| Critério de Aceitação (SPEC 2.5) | Status |
|---|---|
| Função aceita atributos de um funcionário como entrada | ✅ Atendido |
| Função retorna valor numérico contínuo entre 0 e 1 | ✅ Atendido |
| Função retorna classificação de risco (baixo / moderado / alto) conforme RN-02 | ✅ Atendido |
| Função documentada e reutilizável de forma independente | ✅ Atendido |

---

## Interface da Função

```python
from src.scoring import predict_burnout

score, risco = predict_burnout(employee: dict) -> tuple[float, str]
```

### Atributos de entrada (`employee`)

| Campo | Tipo | Valores válidos | Obrigatório |
|---|---|---|---|
| `"Date of Joining"` | str | `"YYYY-MM-DD"` | Sim |
| `"Gender"` | str | `"Female"` \| `"Male"` | Sim |
| `"Company Type"` | str | `"Product"` \| `"Service"` | Sim |
| `"WFH Setup Available"` | str | `"Yes"` \| `"No"` | Sim |
| `"Designation"` | float | 0.0 – 5.0 | Sim |
| `"Resource Allocation"` | float | 1.0 – 10.0 | Não (imputado pela mediana) |
| `"Mental Fatigue Score"` | float | 0.0 – 10.0 | Não (imputado pela mediana) |

### Saída

| Campo | Tipo | Descrição |
|---|---|---|
| `score` | float | Valor contínuo em [0, 1] — RN-01 |
| `risco` | str | `"baixo"` \| `"moderado"` \| `"alto"` — RN-02 |

---

## Fluxo Interno

1. **Derivação de `days_since_joining`** — calcula `REFERENCE_DATE - Date of Joining` em dias (mesma lógica da Feature 2, RN-05).
2. **Encoding categórico** — aplica o mapeamento binário definido em `config.CATEGORICAL_ENCODING` (Female→0, Male→1 etc.).
3. **Construção do vetor de features** — monta DataFrame de 1 linha na ordem exata de `FEATURE_COLUMNS`.
4. **Imputação** — aplica `imputer.pkl` (mediana do treino) a campos ausentes; evita valores `NaN` no modelo.
5. **Escalonamento** — aplica `scaler.pkl` (MinMaxScaler ajustado no treino); garante a mesma escala usada durante o treinamento.
6. **Predição** — chama `xgboost_model.pkl`; o valor bruto é clampado para `[0, 1]` via `np.clip`.
7. **Classificação de risco** — mapeia o score para a faixa conforme RN-02.

Os artefatos (`imputer.pkl`, `scaler.pkl`, `xgboost_model.pkl`) são carregados do disco uma única vez por processo e armazenados em cache para chamadas subsequentes.

---

## Faixas de Risco (RN-02)

| Faixa | Classificação | Ação recomendada |
|---|---|---|
| score ≤ 0,3 (inclusive) | `"baixo"` | Nenhuma ação imediata |
| 0,3 < score < 0,6 | `"moderado"` | Atenção recomendada |
| score ≥ 0,6 (inclusive) | `"alto"` | Intervenção necessária |

---

## Casos de Teste por Faixa de Risco

### Caso 1 — Risco Baixo

**Entrada:**
- Date of Joining: 2012-06-01
- Gender: Female, Company Type: Service, WFH: Yes
- Designation: 2.0, Resource Allocation: 3.0, Mental Fatigue Score: **2.0**

**Saída esperada:** score ≈ 0.14 → classificação `"baixo"`

### Caso 2 — Risco Moderado

**Entrada:**
- Date of Joining: 2012-06-01
- Gender: Male, Company Type: Service, WFH: No
- Designation: 3.0, Resource Allocation: 5.0, Mental Fatigue Score: **5.5**

**Saída esperada:** score ≈ 0.45 → classificação `"moderado"`

### Caso 3 — Risco Alto

**Entrada:**
- Date of Joining: 2012-06-01
- Gender: Male, Company Type: Product, WFH: No
- Designation: 4.0, Resource Allocation: 8.0, Mental Fatigue Score: **8.0**

**Saída esperada:** score ≈ 0.75 → classificação `"alto"`

---

## Pré-requisitos para Uso

Os seguintes artefatos devem existir em `models/` (gerados pelas Features 2 e 3):

- `models/imputer.pkl`
- `models/scaler.pkl`
- `models/xgboost_model.pkl`

---

## Como Usar de Forma Independente

```python
import sys
sys.path.insert(0, "src")          # ajustar conforme diretório de execução
from scoring import predict_burnout

score, risco = predict_burnout({
    "Date of Joining":      "2015-03-20",
    "Gender":               "Male",
    "Company Type":         "Service",
    "WFH Setup Available":  "No",
    "Designation":          3.0,
    "Resource Allocation":  6.0,
    "Mental Fatigue Score": 7.5,
})

print(f"Score: {score:.4f} | Risco: {risco}")
```

## Como Re-executar a Demonstração

```bash
python src/scoring.py
```

---

## Como Rodar os Testes

```bash
python -m pytest tests/test_scoring.py -v
```

Cobertura dos checkpoints do PLAN:

| Checkpoint | Teste(s) |
|---|---|
| 1 — Assinatura clara da função | `test_returns_tuple`, `test_score_is_float` |
| 2 — Fluxo interno documentado | Este README (seção "Fluxo Interno") |
| 3 — Casos de teste por faixa | `test_low_risk_case`, `test_moderate_risk_case`, `test_high_risk_case` |
| 4 — Resultados conferidos contra RN-02 | `TestClassifyRisk` (10 testes de borda) |
| 5 — Função independente e importável | `test_function_importable_independently`, `test_models_loaded_from_disk` |

**Resultado esperado:** 28 testes passando.

---

## Limitações

- O score é um **indicador preditivo**, não um diagnóstico clínico (RN-08 / RNF 3.5).
- Funcionários com data de admissão posterior a `2016-01-01` produzirão `days_since_joining` negativo — o escalonador extrapolará fora do intervalo de treino. Registrar como limitação conhecida.
- O modelo foi treinado em um dataset possivelmente sintético; aplicação em dados reais pode apresentar desempenho inferior (ver `docs/limitations.md`).
