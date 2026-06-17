"""
Objetivo: Transformar o dataset bruto em artefatos limpos e padronizados para treinamento.
Entradas: datasets/survey_2025.csv — Remote Work Health Impact Survey 2025.
Saídas:
  - data/processed/X_train.csv — features de treino (escalonadas, sem nulos)
  - data/processed/X_test.csv  — features de teste  (escalonadas, sem nulos)
  - data/processed/y_train.csv — target de treino (0=Low, 1=Medium, 2=High)
  - data/processed/y_test.csv  — target de teste
  - models/imputer.pkl         — SimpleImputer ajustado no conjunto de treino
  - models/scaler.pkl          — MinMaxScaler   ajustado no conjunto de treino

Decisões de design:
  - Survey_Date descartada: todos os registros são de junho/2025, sem valor preditivo.
  - Physical_Health_Issues é multi-label; substituída por has_physical_issue (0/1).
  - Mental_Health_Status tem 25% de NaN → preenchido com "None" antes do encoding.
  - Não há data de admissão → feature days_since_joining NÃO é derivada.
  - Target ordinal: Low=0, Medium=1, High=2; mantém pipeline de regressão.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_PATH,
    DROP_COLS,
    ENCODING_MAP,
    FEATURE_COLUMNS,
    MODELS_DIR,
    SEED,
    TARGET_COLUMN,
    TARGET_ENCODING,
)

# ── Funções de transformação ──────────────────────────────────────────────────


def _drop_target_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas onde o alvo está ausente (nunca imputa o target)."""
    before = len(df)
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    removed = before - len(df)
    print(f"  Linhas removidas (target ausente): {removed}")
    return df


def _encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Converte Burnout_Level categórico em inteiro ordinal (0/1/2)."""
    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(TARGET_ENCODING)
    print(f"  Encoding do target: {TARGET_ENCODING}")
    return df


def _derive_physical_issue_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria feature binária has_physical_issue:
      1 se o funcionário reportou algum problema físico, 0 caso contrário (NaN).
    A coluna original Physical_Health_Issues é removida após a derivação.
    """
    df = df.copy()
    df["has_physical_issue"] = df["Physical_Health_Issues"].notna().astype(int)
    print(
        f"  'has_physical_issue' criado "
        f"(1=tem problema, 0=sem problema/ausente). "
        f"Distribuição: {df['has_physical_issue'].value_counts().to_dict()}"
    )
    return df


def _fill_mental_health_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preenche NaN em Mental_Health_Status com 'None' antes do encoding.
    NaN indica ausência de diagnóstico relatado — codificado como categoria distinta.
    """
    df = df.copy()
    n_before = df["Mental_Health_Status"].isnull().sum()
    df["Mental_Health_Status"] = df["Mental_Health_Status"].fillna("None")
    print(f"  Mental_Health_Status: {n_before} NaN → 'None'")
    return df


def _drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas sem valor preditivo."""
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"  Colunas removidas: {cols_to_drop}")
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica mapeamento direto a todas as colunas categóricas via ENCODING_MAP."""
    df = df.copy()
    for col, mapping in ENCODING_MAP.items():
        if col not in df.columns:
            continue
        df[col] = df[col].map(mapping)
        print(f"  '{col}': {len(mapping)} categorias encodadas")
    return df


# ── Orquestrador principal ─────────────────────────────────────────────────────


def run_preprocessing(
    data_path=None,
    processed_dir=None,
    models_dir=None,
):
    if data_path is None:
        data_path = DATA_RAW_PATH
    if processed_dir is None:
        processed_dir = DATA_PROCESSED_DIR
    if models_dir is None:
        models_dir = MODELS_DIR

    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PRÉ-PROCESSAMENTO — Remote Work Health Impact Survey 2025")
    print("=" * 60 + "\n")

    df = pd.read_csv(data_path)
    print(f"Dataset carregado: {df.shape[0]} registros × {df.shape[1]} colunas")

    # 1 — Remove linhas sem target
    print("\n[1] Remoção de linhas com target ausente")
    df = _drop_target_nulls(df)

    # 2 — Converte alvo ordinal para numérico
    print("\n[2] Encoding do target (Low=0, Medium=1, High=2)")
    df = _encode_target(df)

    # 3 — Deriva has_physical_issue ANTES de descartar Physical_Health_Issues
    print("\n[3] Feature engineering — has_physical_issue")
    df = _derive_physical_issue_flag(df)

    # 4 — Preenche NaN em Mental_Health_Status com "None"
    print("\n[4] Preenchimento de NaN em Mental_Health_Status → 'None'")
    df = _fill_mental_health_nulls(df)

    # 5 — Descarta Survey_Date e Physical_Health_Issues
    print("\n[5] Remoção de colunas sem valor preditivo")
    df = _drop_columns(df)

    # 6 — Encoding das variáveis categóricas
    print("\n[6] Encoding de variáveis categóricas")
    df = _encode_categoricals(df)

    # 7 — Separação features / target e split 80/20
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    print(f"\n[7] Divisão treino/teste (seed={SEED})")
    print(
        f"  Treino : {X_train.shape[0]} registros  ({X_train.shape[0] / len(X) * 100:.1f}%)"
    )
    print(
        f"  Teste  : {X_test.shape[0]} registros  ({X_test.shape[0] / len(X) * 100:.1f}%)"
    )

    # 8 — Imputação por mediana (fit SOMENTE no treino — sem data leakage)
    print("\n[8] Imputação por mediana (fit no treino)")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURE_COLUMNS)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=FEATURE_COLUMNS)
    for col, median_val in zip(FEATURE_COLUMNS, imputer.statistics_):
        if X_train[col].isnull().any():
            print(f"  Mediana de '{col}': {median_val:.4f}")

    # 9 — Escalonamento MinMaxScaler (fit SOMENTE no treino — sem data leakage)
    print("\n[9] Escalonamento MinMaxScaler (fit no treino)")
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imp), columns=FEATURE_COLUMNS
    )
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=FEATURE_COLUMNS)

    # Verificação: zero valores ausentes
    assert X_train_scaled.isnull().sum().sum() == 0, "Nulos em X_train após scaling"
    assert X_test_scaled.isnull().sum().sum() == 0, "Nulos em X_test após scaling"
    print("  Verificação: nenhum valor ausente nos datasets finais")

    # 10 — Persistência
    y_train_out = y_train.reset_index(drop=True)
    y_test_out = y_test.reset_index(drop=True)

    X_train_scaled.to_csv(processed_dir / "X_train.csv", index=False)
    X_test_scaled.to_csv(processed_dir / "X_test.csv", index=False)
    y_train_out.to_csv(processed_dir / "y_train.csv", index=False)
    y_test_out.to_csv(processed_dir / "y_test.csv", index=False)
    joblib.dump(imputer, models_dir / "imputer.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")

    print("\n[10] Artefatos persistidos")
    print(f"  data/processed/X_train.csv  {X_train_scaled.shape}")
    print(f"  data/processed/X_test.csv   {X_test_scaled.shape}")
    print(f"  data/processed/y_train.csv  {y_train_out.shape}")
    print(f"  data/processed/y_test.csv   {y_test_out.shape}")
    print(f"  models/imputer.pkl")
    print(f"  models/scaler.pkl")
    print(f"\n  Features finais ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

    print("\nPré-processamento concluído com sucesso.")
    return X_train_scaled, X_test_scaled, y_train_out, y_test_out, imputer, scaler


if __name__ == "__main__":
    run_preprocessing()
