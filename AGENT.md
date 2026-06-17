# PROMPT DE AGENTE — Migração de Dataset e Refatoração Completa do Projeto de Predição de Burnout

## CONTEXTO

Você é um agente de desenvolvimento Python e LaTeX. Você está operando **dentro da pasta raiz do projeto `ia_final/`**. Você tem acesso completo ao sistema de arquivos e deve executar, criar e modificar arquivos conforme necessário.

O projeto era baseado no dataset **"Are Your Employees Burning Out?"** (Kaggle, 22.750 registros sintéticos, variável alvo `Burn Rate` contínua 0–1).

O dataset foi trocado para o **"Remote Work Health Impact Survey 2025"**:
- **URL:** `https://www.kaggle.com/datasets/pratyushpuri/remote-work-health-impact-survey-2025`
- **Autor:** Pratyush Puri
- **Coletado:** junho de 2025
- **Contexto:** dados pós-pandemia, regime remoto/híbrido/presencial

O novo arquivo CSV já está disponível em `datasets/` (renomeie-o para `datasets/survey_2025.csv` se necessário).

Sua missão é **refazer completamente o projeto** — todos os scripts Python, documentação, e o relatório LaTeX — para que tudo reflita o novo dataset, suas colunas reais, seus resultados reais e sua narrativa correta.

### Arquivos de referência disponíveis

Você pode (e deve) ler os seguintes arquivos antes de modificar qualquer coisa:

| Arquivo | Quando ler |
|---------|-----------|
| `SPEC.md` | Antes de qualquer modificação — contém regras de negócio (RN), requisitos funcionais e não funcionais que devem ser preservados |
| `PLAN.md` | Para entender a estrutura planejada do pipeline e garantir que a refatoração não quebre dependências entre componentes |
| `docs/architecture.md` | Para entender os contratos de entrada/saída de cada componente |
| `docs/limitations.md` | Para atualizar as limitações declaradas conforme RNF 3.5 da SPEC |

**Fluxo recomendado:** leia `SPEC.md` → `PLAN.md` → inspecione o dataset (PASSO 0) → refatore.

---

## PASSO 0 — INSPEÇÃO OBRIGATÓRIA DO NOVO DATASET

Antes de qualquer coisa, execute:

```python
import pandas as pd
df = pd.read_csv('datasets/survey_2025.csv')
print(df.shape)
print(df.dtypes)
print(df.head(3))
print(df.isnull().sum())
print(df.describe())
for col in df.select_dtypes(include='object').columns:
    print(f"\n{col}:", df[col].unique()[:15])
```

Com base nessa inspeção:
1. Identifique a **variável alvo** de burnout (procure colunas como `Burnout_Level`, `Mental_Health_Condition`, `burnout`, `Stress_Level` — pode ser categórica ordinal ou contínua)
2. Liste todas as features disponíveis
3. Decida se o problema é **regressão** (alvo contínuo/ordinal numérico) ou **classificação** (alvo categórico como Alto/Médio/Baixo)
4. Se a variável alvo for categórica ordinal (ex: "Low", "Medium", "High"), converta para numérica (0, 1, 2) e trate como **regressão ordinal** — mantendo o mesmo pipeline de regressão do projeto original
5. Documente em comentários todas as decisões tomadas

---

## PASSO 1 — ATUALIZAR `src/config.py`

Reescreva completamente `src/config.py` com:

```python
# src/config.py — gerado pelo agente após inspeção do dataset survey_2025.csv

import os
from pathlib import Path

# ── Reprodutibilidade ──────────────────────────────────────────────────────
SEED = 42

# ── Caminhos ───────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
DATASET_PATH    = BASE_DIR / "datasets" / "survey_2025.csv"
PROCESSED_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR      = BASE_DIR / "models"
FIGURES_DIR     = BASE_DIR / "reports" / "figures"
RESULTS_DIR     = BASE_DIR / "reports" / "results"

# ── Dataset — preencher após inspeção (PASSO 0) ────────────────────────────
TARGET_COL      = "???"          # ex: "Burnout_Level" — substituir pelo nome real
DROP_COLS       = ["???"]        # colunas a descartar (IDs, texto livre, etc.)
NUMERIC_COLS    = ["???"]        # features numéricas
CATEGORICAL_COLS = ["???"]       # features categóricas a encodar

# ── Encoding das categóricas — preencher com valores reais ─────────────────
ENCODING_MAP = {
    # Exemplo: "Work_Location": {"Remote": 0, "Hybrid": 1, "Onsite": 2},
    # Exemplo: "Gender": {"Male": 0, "Female": 1, "Non-binary": 2},
}

# ── Se alvo for categórico ordinal, mapeamento de conversão ───────────────
TARGET_ENCODING = {
    # Exemplo: {"Low": 0, "Medium": 1, "High": 2}
    # Se alvo já for numérico, deixar como {}
}

# ── Split ──────────────────────────────────────────────────────────────────
TEST_SIZE = 0.20

# ── XGBoost GridSearch ─────────────────────────────────────────────────────
XGB_PARAM_GRID = {
    "n_estimators":  [100, 200],
    "max_depth":     [3, 5],
    "learning_rate": [0.05, 0.10, 0.20],
}
```

**Substitua todos os `"???"` pelos valores reais descobertos no PASSO 0.**

---

## PASSO 2 — REESCREVER `src/eda.py`

Reescreva completamente o script EDA para o novo dataset. O script deve:

1. Carregar `datasets/survey_2025.csv`
2. Imprimir shape, dtypes e valores ausentes
3. Gerar e salvar em `reports/figures/` os seguintes gráficos (nomes de arquivo fixos para o LaTeX):
   - `burnout_distribution.png` — distribuição da variável alvo (histograma + KDE se contínua, barras se categórica)
   - `correlation_heatmap.png` — heatmap de correlação de Pearson entre variáveis numéricas (incluindo alvo se numérico)
   - `burnout_by_work_location.png` — boxplot ou barras do alvo por regime de trabalho (remoto/híbrido/presencial)
   - `burnout_by_gender.png` — distribuição do alvo por gênero
   - `burnout_by_industry.png` — distribuição por setor/indústria (ou outra variável categórica relevante disponível)
4. Imprimir no terminal os achados principais: médias por subgrupo, correlações mais altas
5. Salvar `reports/results/eda_summary.txt` com os achados em texto

**Se alguma das colunas usadas nos gráficos não existir no dataset, adapte para as colunas reais disponíveis e ajuste os nomes dos arquivos PNG de forma consistente com o LaTeX.**

---

## PASSO 3 — REESCREVER `src/preprocessing.py`

Reescreva completamente o pré-processamento para o novo dataset:

1. Carregar o dataset
2. **Remover linhas onde o alvo está ausente** (nunca impute o alvo)
3. **Converter alvo** para numérico se necessário (via `TARGET_ENCODING` de config.py)
4. **Descartar colunas** de `DROP_COLS` (identificadores, texto livre, colunas redundantes)
5. **Feature engineering**: se houver coluna de data de admissão/tempo de empresa, derivar `days_since_joining` com `REFERENCE_DATE` fixo; se não houver, pular esta etapa e documentar
6. **Encoding das categóricas** via mapeamento direto (`ENCODING_MAP`) — preferir encoding ordinal para variáveis com ordem natural (ex: nível de estresse Baixo/Médio/Alto), OneHot apenas se necessário para nominais sem ordem
7. **Split 80/20** com `SEED` — **antes** de ajustar qualquer transformador (prevenção de data leakage)
8. **Imputação por mediana** (`SimpleImputer`) ajustada apenas no treino, aplicada em treino e teste separadamente
9. **MinMaxScaler** ajustado apenas no treino
10. Salvar em `data/processed/`: `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
11. Salvar em `models/`: `imputer.pkl`, `scaler.pkl`
12. Imprimir resumo: shape final treino/teste, features usadas, nulos tratados

---

## PASSO 4 — REESCREVER `src/training.py`

Mantém a mesma estrutura (3 modelos em complexidade crescente), mas adapte se necessário:

- **Se o alvo for numérico/ordinal contínuo:** manter `LinearRegression`, `RandomForestRegressor`, `XGBRegressor` (regressão)
- **Se o alvo for estritamente binário ou multiclasse sem ordem:** trocar para `LogisticRegression`, `RandomForestClassifier`, `XGBClassifier` (classificação) e ajustar métricas no PASSO 5

Para regressão (caso padrão):
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
```

GridSearchCV no XGBoost com `XGB_PARAM_GRID` de config.py, `cv=5`, `scoring='neg_mean_squared_error'`.

Salvar os três modelos em `models/`: `linear_regression.pkl`, `random_forest.pkl`, `xgboost_model.pkl`.

Imprimir melhores hiperparâmetros do XGBoost e MSE de validação cruzada.

---

## PASSO 5 — REESCREVER `src/evaluation.py`

Calcule e salve todas as métricas e gráficos:

### Métricas (regressão):
- MAE, RMSE, R² no conjunto de teste para cada modelo
- K-Fold (k=5) no treino: MAE_mean, MAE_std, RMSE_mean, RMSE_std, R²_mean, R²_std

### Se for classificação, usar em vez disso:
- Accuracy, F1-macro, ROC-AUC no teste
- K-Fold com F1-macro

### Gráficos a gerar em `reports/figures/` (nomes fixos):
- `real_vs_pred_linear_regression.png`
- `real_vs_pred_random_forest.png`
- `real_vs_pred_xgboost.png`
- `feature_importance_random_forest.png`
- `feature_importance_xgboost.png`
- `equity_by_gender.png` — MAE (ou F1) por gênero nos 3 modelos
- `equity_by_work_location.png` — MAE (ou F1) por regime de trabalho nos 3 modelos

### Arquivos a salvar em `reports/results/`:
- `model_comparison.csv` — tabela com todas as métricas dos 3 modelos
- `cv_results.csv` — resultados do K-Fold
- `conclusion.txt` — texto com o melhor modelo e justificativa

---

## PASSO 6 — REESCREVER `src/scoring.py`

Reescreva a função `predict_burnout(employee: dict) -> tuple[float, str]`:

1. Recebe um dicionário com as features brutas do funcionário (os mesmos campos do novo dataset)
2. Aplica exatamente as mesmas transformações do `preprocessing.py` (usando artefatos `.pkl` carregados em cache)
3. Retorna `(score: float, risco: str)` onde:
   - Score entre 0 e 1 (normalizado se necessário)
   - Risco: `"baixo"` (0–0.33), `"moderado"` (0.33–0.66), `"alto"` (0.66–1.0)
4. Atualizar o exemplo de uso nos comentários/docstring com um funcionário fictício usando as colunas **reais do novo dataset**

---

## PASSO 7 — REESCREVER `prever.py`

Reescreva completamente `prever.py` para coletar os campos **do novo dataset**. O script é a interface de terminal usada por profissionais de RH sem conhecimento técnico — mantenha este princípio.

### Regras obrigatórias:

1. **Campos de entrada**: devem corresponder exatamente às features usadas em `src/scoring.py` (as mesmas colunas do novo dataset após o PASSO 0). Não peça colunas que foram descartadas no pré-processamento.

2. **Validação de entrada**: para cada campo, valide o tipo e o intervalo esperado. Em caso de entrada inválida, exiba mensagem clara e solicite novamente (loop `while True`).

3. **Menus numerados** para variáveis categóricas:
   - Ex: se `Work_Location` tem valores `Remote`, `Hybrid`, `Onsite`, mostre:
     ```
     Regime de trabalho:
       [1] Remote
       [2] Hybrid
       [3] Onsite
     Escolha (1-3): ___
     ```
   - Adapte para todas as categóricas reais do novo dataset.

4. **Campos numéricos**: exiba o intervalo válido entre parênteses. Ex: `Horas trabalhadas por semana (1-80): ___`

5. **Saída visual** — mantenha exatamente este formato, adaptando os rótulos:
   ```
   ==================================================
      RESULTADO — [NOME DO FUNCIONÁRIO]
   ==================================================
     Score    : 0.XXXX  [######            ]
     Risco    : [BAIXO / MODERADO / ALTO]
     Orientação: [mensagem correspondente]
   ==================================================
   ```
   - Barra de progresso: 20 caracteres, `#` proporcional ao score
   - Mensagens por faixa:
     - Baixo (0–0.33): `"Nenhuma ação imediata necessária."`
     - Moderado (0.33–0.66): `"Atenção recomendada. Acompanhar de perto."`
     - Alto (0.66–1.0): `"Intervenção necessária. Acionar equipe de saúde ocupacional."`

6. **Internamente**: o script deve construir o dicionário de features e chamar `predict_burnout()` de `src/scoring.py` — sem duplicar lógica de predição.

7. **Header informativo** no início da execução:
   ```
   ========================================================
     Sistema de Predição de Risco de Burnout
     Dataset: Remote Work Health Impact Survey 2025
     Modelo: XGBoost (melhor desempenho)
   ========================================================
   ```

8. **Casos de referência**: ao final do script (em comentário), inclua 3 exemplos de entrada e saída esperada cobrindo as três faixas de risco (baixo, moderado, alto), usando valores plausíveis das colunas reais.

### Exemplo de estrutura base (adaptar para colunas reais após PASSO 0):

```python
#!/usr/bin/env python3
"""
prever.py — Interface interativa de terminal para predição de risco de burnout.
Dataset: Remote Work Health Impact Survey 2025
Fonte: https://www.kaggle.com/datasets/pratyushpuri/remote-work-health-impact-survey-2025
"""

import sys
sys.path.insert(0, "src")
from scoring import predict_burnout

def _barra(score: float, width: int = 20) -> str:
    filled = round(score * width)
    return "#" * filled + " " * (width - filled)

def _input_opcao(prompt: str, opcoes: dict) -> str:
    """Exibe menu numerado e retorna o valor correspondente."""
    while True:
        print(prompt)
        for k, v in opcoes.items():
            print(f"  [{k}] {v}")
        escolha = input("Escolha: ").strip()
        if escolha in opcoes:
            return opcoes[escolha]
        print("  ✗ Opção inválida. Tente novamente.\n")

def _input_numero(prompt: str, minimo: float, maximo: float) -> float:
    """Solicita número em intervalo válido."""
    while True:
        try:
            valor = float(input(f"{prompt} ({minimo}–{maximo}): ").strip())
            if minimo <= valor <= maximo:
                return valor
            print(f"  ✗ Valor fora do intervalo [{minimo}, {maximo}].\n")
        except ValueError:
            print("  ✗ Digite um número válido.\n")

def main():
    print("\n" + "=" * 56)
    print("  Sistema de Predição de Risco de Burnout")
    print("  Dataset: Remote Work Health Impact Survey 2025")
    print("  Modelo: XGBoost (melhor desempenho)")
    print("=" * 56 + "\n")

    nome = input("Nome do funcionário (apenas para exibição): ").strip() or "Funcionário"

    # ── ADAPTAR OS CAMPOS ABAIXO PARA AS COLUNAS REAIS DO DATASET ──
    # Exemplo (substituir com base no PASSO 0):
    employee = {}
    # employee["Work_Location"] = _input_opcao(
    #     "\nRegime de trabalho:",
    #     {"1": "Remote", "2": "Hybrid", "3": "Onsite"}
    # )
    # employee["Hours_Worked_Per_Week"] = _input_numero("\nHoras trabalhadas por semana", 1, 80)
    # [... outros campos ...]

    score, risco = predict_burnout(employee)

    msg = {
        "baixo":    "Nenhuma ação imediata necessária.",
        "moderado": "Atenção recomendada. Acompanhar de perto.",
        "alto":     "Intervenção necessária. Acionar equipe de saúde ocupacional.",
    }

    print("\n" + "=" * 50)
    print(f"   RESULTADO — {nome.upper()}")
    print("=" * 50)
    print(f"  Score    : {score:.4f}  [{_barra(score)}]")
    print(f"  Risco    : {risco.upper()}")
    print(f"  Orientação: {msg[risco]}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()

# ── Casos de referência (atualizar com valores reais após treinamento) ──────
# Caso 1 — Baixo risco:   [...campos...] → score ≈ 0.15, risco = baixo
# Caso 2 — Moderado:      [...campos...] → score ≈ 0.50, risco = moderado
# Caso 3 — Alto risco:    [...campos...] → score ≈ 0.80, risco = alto
```

---

## PASSO 8 — ATUALIZAR `docs/architecture.md`

Atualize o arquivo com:
- Novo nome do dataset, fonte e URL completo: `https://www.kaggle.com/datasets/pratyushpuri/remote-work-health-impact-survey-2025`
- Novas colunas/features
- Nova variável alvo e sua natureza (contínua, ordinal, etc.)
- Qualquer mudança no tipo de tarefa (regressão → classificação ou vice-versa)
- Manter a estrutura de 5 componentes e o diagrama ASCII

---

## PASSO 9 — ATUALIZAR `docs/limitations.md`

Substitua a seção "1. Possível Natureza Sintética do Dataset" por:

```markdown
## 1. Dataset Pós-Pandemia com Contexto de Trabalho Remoto

O dataset "Remote Work Health Impact Survey 2025" (Kaggle, coletado em junho de 2025)
representa uma amostra diversificada de trabalhadores globais em contexto pós-pandemia,
cobrindo regimes remoto, híbrido e presencial. Embora mais representativo do cenário
atual do que datasets sintéticos, trata-se de dados de survey auto-reportados, sujeitos
a viés de resposta e seleção amostral. Os resultados devem ser interpretados com essa
limitação em mente.
```

Atualize também a Limitação 4 (Extrapolação Temporal) para refletir as novas features de data, se houver.

Mantenha as demais limitações (caráter preditivo não diagnóstico, equidade superficial, ausência de atualização do modelo).

---

## PASSO 10 — ATUALIZAR `README.md`

Atualize:
- Descrição do dataset: nome completo, URL `https://www.kaggle.com/datasets/pratyushpuri/remote-work-health-impact-survey-2025`, tamanho, colunas principais
- Tabela de resultados (MAE/RMSE/R² ou Accuracy/F1/AUC — valores reais obtidos)
- Exemplo de uso da função `predict_burnout()` com campos **do novo dataset**
- Manter toda a estrutura de pastas, ODS, e instruções de execução

---

## PASSO 11 — ATUALIZAR `relatorio.tex` (CRÍTICO)

Este é o passo mais importante. Atualize o arquivo `relatorio.tex` com precisão cirúrgica, mantendo toda a estrutura LaTeX, formatação IEEEtran, pacotes, macros e estilo visual. **Não quebre o LaTeX.**

### 11.1 — Abstract (linhas 79–98)
Substitua completamente o abstract. O novo deve mencionar:
- O novo dataset: "Remote Work Health Impact Survey 2025" (Kaggle, coletado junho 2025)
- O número real de registros do novo dataset
- As novas features principais (regime de trabalho, horas semanais, etc.)
- Os novos resultados reais de MAE/RMSE/R² (ou Accuracy/F1/AUC)
- A nova feature mais importante identificada

### 11.2 — Seção `\section{Dataset}` (linhas 162–223)
- Subseção 2.1: Trocar nome, fonte, número de registros e colunas
- Atualizar `\caption` e conteúdo da `tab:features` com as colunas reais do novo dataset
- Atualizar os percentuais de valores ausentes (valores reais)
- Subseção 2.2 (Observação sobre natureza): Substituir pelo texto abaixo, adaptando com fatos reais:

```latex
\subsection{Natureza e Contexto do Dataset}

O \textit{dataset} \textit{``Remote Work Health Impact Survey 2025''}
\cite{kaggle_dataset} foi coletado em junho de 2025 e representa uma
amostra global de trabalhadores em regime remoto, híbrido e presencial
no período pós-pandemia. Os dados são auto-reportados por meio de
survey anônimo, cobrindo [NÚMERO REAL] respondentes de múltiplas
regiões, setores e faixas etárias. O uso de dados reais de survey
— em contraste com datasets sintéticos — confere maior validade
ecológica às conclusões, embora introduza viés de resposta e
seleção amostral inerentes à metodologia de survey.
```

### 11.3 — Seção EDA (linhas 288–356)
- Atualizar estatísticas descritivas da variável alvo (média, desvio padrão) com valores reais
- Atualizar achados por subgrupo (agora: regime de trabalho, gênero, setor) com valores reais
- Atualizar nomes dos arquivos de figura nos `\includegraphics{}` para os novos nomes
- Atualizar os `\caption{}` das figuras

### 11.4 — Seção Pré-processamento (linhas 359–439)
- Atualizar o número de linhas removidas (alvo ausente)
- Atualizar os nomes das colunas com nulos e seus percentuais
- Atualizar o encoding binário/ordinal para refletir as novas categóricas
- Atualizar medianas imputadas com valores reais
- Se não houver `Date of Joining`, remover a subseção de feature engineering de data e substituir por outra transformação realizada

### 11.5 — Seção Modelos (linhas 442–564)
- Manter a estrutura teórica (equações de Regressão Linear, Random Forest, XGBoost)
- Atualizar o coeficiente da Regressão Linear (mencionar a feature mais influente com valor real)
- Atualizar a lista de importância das features do Random Forest com valores reais
- Atualizar a tabela `tab:xgb_grid` com os melhores hiperparâmetros reais encontrados

### 11.6 — Seção Avaliação (linhas 566–697)
- Atualizar `tab:test`: valores reais de MAE, RMSE, R² (ou métricas de classificação)
- Atualizar `tab:cv`: valores reais do K-Fold
- Atualizar o tamanho do conjunto de teste com o número real
- Atualizar o texto analítico ("XGBoost explica X% da variância...")
- Atualizar as figuras referenciadas nos `\includegraphics{}`

### 11.7 — Seção Equidade (linhas 699–750)
- Atualizar `tab:equity`: valores reais de MAE por subgrupo (gênero, regime de trabalho)
- Atualizar o texto analítico com o achado real (há ou não viés?)
- Substituir `Company Type` por `Work Location` (ou a variável categórica relevante do novo dataset)

### 11.8 — Seção Scoring (linhas 752–850)
- Atualizar o exemplo de código em `lstlisting` com as novas features do dataset
- Atualizar os valores de saída do exemplo (score e risco com valores plausíveis)
- Atualizar os campos interativos do `prever.py` mencionados no texto

### 11.9 — Seção Limitações (linhas 852–879)
- Substituir limitação 1 (dataset sintético) pela nova limitação de survey auto-reportado
- Manter limitações 2, 3, 4

### 11.10 — Seção Conclusão (linhas 881–908)
- Atualizar os valores numéricos de R², MAE
- Atualizar o achado exploratório de maior impacto prático (antes: WFH 12 p.p.; agora: o achado real)
- Manter a estrutura e o tom acadêmico

### 11.11 — Referências (linhas 916–977)
- Atualizar `\bibitem{kaggle_dataset}` com:
```latex
\bibitem{kaggle_dataset}
  PURI, Pratyush.
  \textbf{Remote Work Health Impact Survey 2025}. [S.l.]: Kaggle, 2025.
  Disponível em:
  \url{https://www.kaggle.com/datasets/pratyushpuri/remote-work-health-impact-survey-2025}.
  Acesso em: jun.~2026.
```
- Manter todas as outras referências (Maslach, Salvagioni, ONU ODS, Breiman, Chen, sklearn, xgboost)

### 11.12 — Verificação global de URL e nome do dataset

Após todas as edições, faça uma busca global no `relatorio.tex` por:
- `blurredmachine` → deve não existir
- `Are Your Employees` → deve não existir
- `22.750` → deve não existir (substituído pelo número real)
- `pratyushpuri` → deve aparecer exatamente uma vez (na referência)
- `Remote Work Health Impact Survey 2025` → deve aparecer no abstract, seção Dataset e referência

---

## PASSO 12 — ATUALIZAR `tests/`

Para cada arquivo de teste existente em `tests/`:
1. Atualizar os nomes de colunas usados nos fixtures/mocks para as novas colunas
2. Atualizar o dicionário de exemplo passado para `predict_burnout()` com as novas features
3. Verificar se os testes passam com `python -m pytest tests/ -v`
4. Corrigir qualquer teste quebrado pela mudança de dataset

---

## PASSO 13 — EXECUTAR E VALIDAR O PIPELINE COMPLETO

Execute em ordem:
```bash
python src/eda.py
python src/preprocessing.py
python src/training.py
python src/evaluation.py
python src/scoring.py
python -m pytest tests/ -v
```

Se qualquer etapa falhar, corrija o erro antes de prosseguir.

Após execução bem-sucedida:
1. Colete os valores reais de métricas de `reports/results/model_comparison.csv`
2. Volte ao `relatorio.tex` e substitua todos os placeholders `[VALOR REAL]` com os números obtidos
3. Verifique que os arquivos `.png` em `reports/figures/` foram gerados corretamente
4. Compile o LaTeX para verificar que não há erros: `xelatex relatorio.tex` (ou `pdflatex` se o ambiente não tiver XeLaTeX)

---

## REGRAS CRÍTICAS PARA O AGENTE

1. **NUNCA use os valores numéricos do dataset antigo** (22.750 registros, R²=0.904, MAE=0.047, RMSE=0.060, Mental Fatigue Score 81.94%, etc.). Todos os números no relatório devem vir da execução real com o novo dataset.

2. **NUNCA mencione "Are Your Employees Burning Out?" ou "blurredmachine"** em nenhum arquivo após a migração.

3. **Mantenha SEED=42** em todas as operações aleatórias para reprodutibilidade.

4. **Preserve a estrutura LaTeX** do `relatorio.tex` — pacotes, macros `\fonte`, `\instituicao`, nomes dos autores (Breno Porto Pinheiro do Prado RA 185196, Guilherme Bento Ramos RA 185226), classe IEEEtran, todas as equações matemáticas.

5. **Adapte o tipo de tarefa** (regressão vs classificação) com base no que o dataset realmente permite — mas documente a decisão no relatório.

6. **Se o dataset não tiver `Date of Joining`**, remova completamente a feature `days_since_joining`, a equação correspondente no LaTeX e a referência a `REFERENCE_DATE`.

7. **Nomes dos arquivos PNG devem ser consistentes** entre os scripts Python e os `\includegraphics{}` do LaTeX.

8. **Não altere**: os pacotes LaTeX, o estilo de citação, as equações de MAE/RMSE/R², as seções de teoria (Regressão Linear, Random Forest, XGBoost), as referências bibliográficas (exceto a do dataset).

9. Ao final, liste todos os arquivos modificados com um resumo de cada mudança realizada.

---

## CHECKLIST FINAL

Antes de declarar a tarefa concluída, confirme:

- [ ] `SPEC.md` e `PLAN.md` lidos antes de começar
- [ ] `src/config.py` — colunas reais do novo dataset
- [ ] `src/eda.py` — gera 5 PNGs com nomes corretos para o LaTeX
- [ ] `src/preprocessing.py` — pipeline sem data leakage, artefatos salvos
- [ ] `src/training.py` — 3 modelos treinados, .pkl salvos
- [ ] `src/evaluation.py` — métricas reais calculadas, gráficos gerados
- [ ] `src/scoring.py` — `predict_burnout()` funciona com colunas novas
- [ ] `prever.py` — menus numerados, validação de entrada, saída visual, docstring com URL do dataset, 3 casos de referência comentados
- [ ] `docs/architecture.md` — novo dataset com URL completo documentado
- [ ] `docs/limitations.md` — limitação de dataset sintético removida, nova limitação de survey inserida
- [ ] `README.md` — URL do dataset, resultados reais, exemplo de uso atualizado
- [ ] `relatorio.tex` — TODOS os valores numéricos, nomes de colunas, figuras e referência de dataset atualizados
- [ ] `relatorio.tex` — busca global confirma ausência de "blurredmachine", "Are Your Employees", "22.750"
- [ ] `relatorio.tex` — `\bibitem{kaggle_dataset}` com URL completo de pratyushpuri
- [ ] `tests/` — testes passando com as novas colunas
- [ ] Pipeline completo executado sem erros (`python main.py`)
- [ ] LaTeX compilado sem erros (`xelatex relatorio.tex`)
