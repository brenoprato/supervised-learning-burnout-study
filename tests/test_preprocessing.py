"""
Testes unitários e de integração para o módulo de Pré-processamento.
Valida os critérios de aceitação da Feature 2 (Pré-processamento).
Dataset: Remote Work Health Impact Survey 2025
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import (
    DATA_RAW_PATH,
    ENCODING_MAP,
    FEATURE_COLUMNS,
    SEED,
    TARGET_COLUMN,
    TARGET_ENCODING,
)
from preprocessing import (
    _derive_physical_issue_flag,
    _drop_target_nulls,
    _encode_categoricals,
    _encode_target,
    _fill_mental_health_nulls,
    run_preprocessing,
)

# ── Checkpoint 1: imputação e ausência de nulos no output ────────────────────


def test_no_nulls_after_preprocessing(tmp_path):
    X_train, X_test, y_train, y_test, _, _ = run_preprocessing(
        processed_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
    )
    assert X_train.isnull().sum().sum() == 0, "Nulos em X_train após pré-processamento"
    assert X_test.isnull().sum().sum() == 0, "Nulos em X_test após pré-processamento"
    assert y_train.isnull().sum() == 0, "Nulos em y_train"
    assert y_test.isnull().sum() == 0, "Nulos em y_test"


# ── Checkpoint 2: target ausente removido ────────────────────────────────────


def test_drop_target_nulls_removes_rows():
    df = pd.DataFrame({TARGET_COLUMN: ["Low", None, "Medium", None, "High"]})
    result = _drop_target_nulls(df)
    assert len(result) == 3
    assert result[TARGET_COLUMN].isnull().sum() == 0


def test_drop_target_nulls_no_rows_removed_when_complete():
    df = pd.DataFrame({TARGET_COLUMN: ["Low", "Medium", "High"]})
    result = _drop_target_nulls(df)
    assert len(result) == 3


# ── Checkpoint 3: encoding do target ─────────────────────────────────────────


def test_encode_target_maps_correctly():
    df = pd.DataFrame({TARGET_COLUMN: ["Low", "Medium", "High", "Low"]})
    result = _encode_target(df)
    assert list(result[TARGET_COLUMN]) == [0, 1, 2, 0]


def test_target_values_after_preprocessing(tmp_path):
    _, _, y_train, y_test, _, _ = run_preprocessing(
        processed_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
    )
    valid = {0, 1, 2}
    assert set(y_train.unique()).issubset(valid), f"Valores inválidos em y_train"
    assert set(y_test.unique()).issubset(valid), f"Valores inválidos em y_test"


# ── Checkpoint 4: has_physical_issue derivado ────────────────────────────────


def test_physical_issue_flag_created():
    df = pd.DataFrame(
        {
            "Physical_Health_Issues": [
                "Back Pain",
                None,
                "Neck Pain; Eye Strain",
                None,
            ],
        }
    )
    result = _derive_physical_issue_flag(df)
    assert "has_physical_issue" in result.columns
    assert list(result["has_physical_issue"]) == [1, 0, 1, 0]


def test_physical_issue_flag_is_binary():
    df = pd.DataFrame(
        {
            "Physical_Health_Issues": ["Back Pain", None, "Eye Strain"],
        }
    )
    result = _derive_physical_issue_flag(df)
    assert set(result["has_physical_issue"].unique()).issubset({0, 1})


# ── Checkpoint 5: Mental_Health_Status NaN → "None" ──────────────────────────


def test_fill_mental_health_nulls():
    df = pd.DataFrame(
        {
            "Mental_Health_Status": ["Anxiety", None, "Burnout", None],
        }
    )
    result = _fill_mental_health_nulls(df)
    assert result["Mental_Health_Status"].isnull().sum() == 0
    assert "None" in result["Mental_Health_Status"].values


# ── Checkpoint 6: encoding das categóricas ───────────────────────────────────


def test_encoding_gender():
    df = pd.DataFrame({"Gender": ["Female", "Male", "Non-binary"]})
    result = _encode_categoricals(df)
    assert list(result["Gender"]) == [0, 1, 2]


def test_encoding_work_arrangement():
    df = pd.DataFrame({"Work_Arrangement": ["Onsite", "Hybrid", "Remote"]})
    result = _encode_categoricals(df)
    assert list(result["Work_Arrangement"]) == [0, 1, 2]


def test_encoding_salary_range():
    df = pd.DataFrame(
        {
            "Salary_Range": [
                "$40K-60K",
                "$60K-80K",
                "$80K-100K",
                "$100K-120K",
                "$120K+",
            ],
        }
    )
    result = _encode_categoricals(df)
    assert list(result["Salary_Range"]) == [0, 1, 2, 3, 4]


def test_encoded_columns_are_numeric(tmp_path):
    X_train, X_test, _, _, _, _ = run_preprocessing(
        processed_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
    )
    for col in FEATURE_COLUMNS:
        assert pd.api.types.is_numeric_dtype(X_train[col]), f"'{col}' não é numérica"
        assert pd.api.types.is_numeric_dtype(X_test[col]), f"'{col}' não é numérica"


# ── Checkpoint 7: sem data leakage no scaler ─────────────────────────────────


def test_no_data_leakage_scaler(tmp_path):
    """O scaler deve ser ajustado no treino; treino deve estar em [0, 1]."""
    X_train, X_test, _, _, _, scaler = run_preprocessing(
        processed_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
    )
    assert X_train.min().min() >= 0.0
    assert X_train.max().max() <= 1.0


def test_scaler_fit_only_on_train(tmp_path):
    """Scaler ajustado somente no treino deve ter tantos parâmetros quanto features."""
    _, _, _, _, _, scaler = run_preprocessing(
        processed_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
    )
    assert len(scaler.data_min_) == len(FEATURE_COLUMNS)
    assert len(scaler.data_max_) == len(FEATURE_COLUMNS)


# ── Checkpoint 8: divisão 80/20 ──────────────────────────────────────────────


def test_train_test_split_proportions(tmp_path):
    X_train, X_test, y_train, y_test, _, _ = run_preprocessing(
        processed_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
    )
    total = len(X_train) + len(X_test)
    train_pct = len(X_train) / total
    test_pct = len(X_test) / total
    assert abs(train_pct - 0.8) < 0.01, f"Treino esperado ~80%, obtido {train_pct:.2%}"
    assert abs(test_pct - 0.2) < 0.01, f"Teste esperado  ~20%, obtido {test_pct:.2%}"


def test_split_is_reproducible(tmp_path):
    X_train_a, _, _, _, _, _ = run_preprocessing(
        processed_dir=tmp_path / "p1",
        models_dir=tmp_path / "m1",
    )
    X_train_b, _, _, _, _, _ = run_preprocessing(
        processed_dir=tmp_path / "p2",
        models_dir=tmp_path / "m2",
    )
    pd.testing.assert_frame_equal(
        X_train_a.reset_index(drop=True),
        X_train_b.reset_index(drop=True),
    )


# ── Checkpoint 9: artefatos persistidos ──────────────────────────────────────


def test_processed_files_exist(tmp_path):
    run_preprocessing(
        processed_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
    )
    assert (tmp_path / "processed" / "X_train.csv").exists()
    assert (tmp_path / "processed" / "X_test.csv").exists()
    assert (tmp_path / "processed" / "y_train.csv").exists()
    assert (tmp_path / "processed" / "y_test.csv").exists()


def test_model_artifacts_exist(tmp_path):
    run_preprocessing(
        processed_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
    )
    assert (tmp_path / "models" / "imputer.pkl").exists()
    assert (tmp_path / "models" / "scaler.pkl").exists()


def test_feature_columns_match_spec(tmp_path):
    X_train, _, _, _, _, _ = run_preprocessing(
        processed_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
    )
    assert list(X_train.columns) == FEATURE_COLUMNS
