# Feature 1 — Análise Exploratória dos Dados (EDA)

## Descrição

Esta feature implementa a etapa de Análise Exploratória dos Dados (EDA) do pipeline de predição de burnout. O objetivo é compreender a estrutura, qualidade e distribuição do dataset bruto antes de qualquer transformação, gerando diagnóstico completo que orienta as decisões das etapas seguintes.

Corresponde à **SPEC 2.1** e ao **Componente 1 — Explorador de Dados** do PLAN.

---

## Requisitos Atendidos

| Critério de Aceitação (SPEC 2.1) | Status |
|---|---|
| Distribuição da variável alvo `Burn Rate` apresentada graficamente | ✅ Atendido |
| Mapa de calor de correlação entre todas as variáveis numéricas gerado e salvo | ✅ Atendido |
| Todos os valores ausentes identificados e documentados | ✅ Atendido |
| Análises por subgrupo (gênero, tipo de empresa, disponibilidade de home office) produzidas e exibidas | ✅ Atendido |

---

## Achados Principais

### Estrutura do Dataset

- **22.750 registros** × **9 colunas**
- Colunas numéricas: `Designation`, `Resource Allocation`, `Mental Fatigue Score`, `Burn Rate`
- Colunas categóricas: `Gender`, `Company Type`, `WFH Setup Available`
- Colunas especiais: `Employee ID` (identificador), `Date of Joining` (data — usada na Feature 2 para derivar tempo de empresa)

### Valores Ausentes

| Coluna | Ausentes | Percentual |
|---|---|---|
| `Resource Allocation` | 1.381 | 6,07% |
| `Mental Fatigue Score` | 2.117 | 9,31% |
| `Burn Rate` | 1.124 | 4,94% |

Apenas colunas numéricas apresentam valores ausentes. Colunas categóricas e o identificador estão completos.

### Distribuição da Variável Alvo

O `Burn Rate` apresenta distribuição aproximadamente uniforme entre 0 e 1, com:
- Média: **0,452**
- Desvio padrão: **0,198**
- Mínimo: 0,0 — Máximo: 1,0

### Correlações Relevantes

`Mental Fatigue Score` apresenta a maior correlação positiva com `Burn Rate`, indicando ser o atributo mais preditivo. `Resource Allocation` também mostra correlação positiva relevante.

### Análise por Subgrupo

| Subgrupo | Média Burn Rate |
|---|---|
| Gênero — Female | 0,423 |
| Gênero — Male | 0,484 |
| Tipo de Empresa — Product | 0,451 |
| Tipo de Empresa — Service | 0,453 |
| WFH — Não disponível | 0,518 |
| WFH — Disponível | 0,396 |

**Achado relevante:** Funcionários **sem** disponibilidade de home office apresentam Burn Rate médio 12 pontos percentuais maior do que aqueles com WFH disponível. Homens apresentam média levemente superior à de mulheres. O tipo de empresa tem diferença negligenciável.

---

## Visualizações Geradas

Todos os gráficos são salvos em `reports/figures/`:

| Arquivo | Descrição |
|---|---|
| `burnrate_distribution.png` | Histograma e curva KDE da variável alvo `Burn Rate` |
| `correlation_heatmap.png` | Mapa de calor de correlação entre variáveis numéricas |
| `burnrate_by_gender.png` | Boxplot de Burn Rate por gênero (Female/Male) |
| `burnrate_by_company_type.png` | Boxplot de Burn Rate por tipo de empresa (Product/Service) |
| `burnrate_by_wfh_setup_available.png` | Boxplot de Burn Rate por disponibilidade de home office |

---

## Decisões de Design

- **Backend Agg do Matplotlib:** usado para execução não-interativa em scripts (sem exibição de janelas). Obrigatório para execução headless e em testes automatizados.
- **Função `run_eda` parametrizável:** aceita `data_path` e `figures_dir` opcionais, o que permite o uso em testes com caminhos temporários sem alterar a lógica principal.
- **Retorno do DataFrame:** `run_eda` retorna o DataFrame carregado para facilitar integração com outros módulos do pipeline.
- **Separação de responsabilidades:** cada tipo de visualização está em função dedicada, facilitando testes unitários independentes.

---

## Decisões com Impacto nas Etapas Seguintes

- Os valores ausentes em `Resource Allocation`, `Mental Fatigue Score` e `Burn Rate` demandam imputação na Feature 2. A estratégia escolhida (média ou mediana) deve considerar a distribuição de cada coluna.
- A coluna `Date of Joining` não é usada diretamente; na Feature 2 será convertida em `days_since_joining` (tempo de empresa em dias), conforme RN-05.
- O `Employee ID` é uma coluna de identificação e deve ser removida antes do treinamento.
- As colunas categóricas (`Gender`, `Company Type`, `WFH Setup Available`) precisam de encoding na Feature 2.

---

## Como Re-executar

A partir do diretório raiz do projeto:

```bash
python src/eda.py
```

O script imprime o diagnóstico completo no stdout e salva os cinco gráficos em `reports/figures/`.

---

## Como Rodar os Testes

```bash
python -m pytest tests/test_eda.py -v
```

Os testes validam todos os checkpoints do PLAN e os critérios de aceitação da SPEC 2.1:

- Checkpoint 1: dimensões do dataset (22.750 registros × 9 colunas)
- Checkpoint 2: tipos de dados corretos por coluna
- Checkpoint 3: valores ausentes identificados nas colunas corretas
- Checkpoints 4–6: geração de todos os gráficos exigidos
- Testes de borda: arquivo inexistente, DataFrame sem nulos

**Resultado esperado:** 19 testes passando.
