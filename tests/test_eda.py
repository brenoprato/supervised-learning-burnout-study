"""
Testes unitários e de integração para o módulo de Análise Exploratória de Dados (EDA).
Valida os critérios de aceitação da Feature 1 (Análise Exploratória).
Dataset: Remote Work Health Impact Survey 2025
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import (
    CATEGORICAL_COLS,
    DATA_RAW_PATH,
    NUMERIC_COLS,
    TARGET_COLUMN,
)
from eda import (
    load_data,
    plot_burnout_by_gender,
    plot_burnout_by_industry,
    plot_burnout_by_work_location,
    plot_burnout_distribution,
    plot_correlation_heatmap,
    report_missing_values,
)

# Colunas esperadas no novo dataset
EXPECTED_COLUMNS = [
    "Survey_Date",
    "Age",
    "Gender",
    "Region",
    "Industry",
    "Job_Role",
    "Work_Arrangement",
    "Hours_Per_Week",
    "Mental_Health_Status",
    "Burnout_Level",
    "Work_Life_Balance_Score",
    "Physical_Health_Issues",
    "Social_Isolation_Score",
    "Salary_Range",
]


# ── Checkpoint 1: Dataset carregado com dimensões corretas ────────────────────


def test_dataset_loads_successfully():
    df = load_data(DATA_RAW_PATH)
    assert not df.empty


def test_dataset_row_count():
    df = load_data(DATA_RAW_PATH)
    assert df.shape[0] >= 3000, f"Esperados ≥3.000 registros, obtidos {df.shape[0]}"


def test_dataset_column_count():
    df = load_data(DATA_RAW_PATH)
    assert df.shape[1] == 14, f"Esperadas 14 colunas, obtidas {df.shape[1]}"


# ── Checkpoint 2: Colunas e tipos corretos ────────────────────────────────────


def test_expected_columns_present():
    df = load_data(DATA_RAW_PATH)
    for col in EXPECTED_COLUMNS:
        assert col in df.columns, f"Coluna ausente: {col}"


def test_numeric_columns_are_numeric():
    df = load_data(DATA_RAW_PATH)
    for col in NUMERIC_COLS:
        assert pd.api.types.is_numeric_dtype(df[col]), (
            f"Coluna '{col}' deveria ser numérica, mas é {df[col].dtype}"
        )


def test_target_column_is_object():
    df = load_data(DATA_RAW_PATH)
    assert df[TARGET_COLUMN].dtype == object, (
        f"'{TARGET_COLUMN}' deveria ser object (categórico)"
    )


def test_target_values_are_valid():
    df = load_data(DATA_RAW_PATH)
    valid = {"Low", "Medium", "High"}
    actual = set(df[TARGET_COLUMN].dropna().unique())
    assert actual == valid, f"Valores inesperados em {TARGET_COLUMN}: {actual}"


# ── Checkpoint 3: Valores ausentes identificados ──────────────────────────────


def test_missing_values_exist():
    """Mental_Health_Status e Physical_Health_Issues têm NaN no dataset."""
    df = load_data(DATA_RAW_PATH)
    total_missing = df.isnull().sum().sum()
    assert total_missing > 0, "Esperados valores ausentes no dataset"


def test_missing_values_report_returns_dataframe():
    df = load_data(DATA_RAW_PATH)
    missing_df = report_missing_values(df)
    assert isinstance(missing_df, pd.DataFrame)
    assert "Ausentes" in missing_df.columns
    assert "Percentual (%)" in missing_df.columns


def test_missing_values_report_output(capsys):
    df = load_data(DATA_RAW_PATH)
    report_missing_values(df)
    captured = capsys.readouterr()
    assert "VALORES AUSENTES" in captured.out


def test_missing_in_expected_columns():
    """Apenas Mental_Health_Status e Physical_Health_Issues têm NaN."""
    df = load_data(DATA_RAW_PATH)
    missing = df.isnull().sum()
    cols_with_missing = set(missing[missing > 0].index)
    allowed = {"Mental_Health_Status", "Physical_Health_Issues"}
    unexpected = cols_with_missing - allowed
    assert not unexpected, f"Colunas inesperadas com NaN: {unexpected}"


# ── Critério: distribuição da variável alvo ─────────────────────────


def test_burnout_distribution_plot_generated(tmp_path):
    df = load_data(DATA_RAW_PATH)
    path = plot_burnout_distribution(df, tmp_path)
    assert path.exists(), "Gráfico de distribuição não foi gerado"
    assert path.name == "burnout_distribution.png"


# ── Critério: mapa de calor ─────────────────────────────────────────


def test_correlation_heatmap_generated(tmp_path):
    df = load_data(DATA_RAW_PATH)
    path = plot_correlation_heatmap(df, tmp_path)
    assert path.exists(), "Mapa de calor não foi gerado"
    assert path.name == "correlation_heatmap.png"


# ── Critério: análises por subgrupo ─────────────────────────────────


def test_work_arrangement_values():
    df = load_data(DATA_RAW_PATH)
    values = set(df["Work_Arrangement"].dropna().unique())
    assert values == {"Onsite", "Hybrid", "Remote"}, f"Valores inesperados: {values}"


def test_gender_values():
    df = load_data(DATA_RAW_PATH)
    values = set(df["Gender"].dropna().unique())
    assert values.issubset({"Male", "Female", "Non-binary", "Prefer not to say"}), (
        f"Valores inesperados: {values}"
    )


def test_burnout_by_work_location_generated(tmp_path):
    df = load_data(DATA_RAW_PATH)
    path = plot_burnout_by_work_location(df, tmp_path)
    assert path.exists()
    assert path.name == "burnout_by_work_location.png"


def test_burnout_by_gender_generated(tmp_path):
    df = load_data(DATA_RAW_PATH)
    path = plot_burnout_by_gender(df, tmp_path)
    assert path.exists()
    assert path.name == "burnout_by_gender.png"


def test_burnout_by_industry_generated(tmp_path):
    df = load_data(DATA_RAW_PATH)
    path = plot_burnout_by_industry(df, tmp_path)
    assert path.exists()
    assert path.name == "burnout_by_industry.png"


# ── Testes de borda ───────────────────────────────────────────────────────────


def test_load_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        load_data(Path("/nonexistent/path.csv"))


def test_missing_values_on_complete_dataframe(capsys):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    missing_df = report_missing_values(df)
    assert missing_df.empty
    captured = capsys.readouterr()
    assert "Nenhum valor ausente" in captured.out
