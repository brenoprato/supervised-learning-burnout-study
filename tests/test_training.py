"""
Testes unitários e de integração para o módulo de Treinamento dos Modelos.
Valida os critérios de aceitação da SPEC 2.3 e os checkpoints da Feature 3 do PLAN.
Dataset: Remote Work Health Impact Survey 2025

Estratégia: fixture com dataset sintético pequeno para testes unitários rápidos;
testes de integração carregam os modelos já salvos em models/ sem re-treinar.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import FEATURE_COLUMNS, MODELS_DIR, SEED, XGB_PARAM_GRID
from training import (
    train_linear_regression,
    train_random_forest,
    train_xgboost,
)

# ── Fixture: dataset sintético pequeno para testes rápidos ───────────────────


@pytest.fixture(scope="module")
def small_data():
    rng = np.random.default_rng(SEED)
    n = 300
    X = pd.DataFrame(rng.random((n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    # Target ordinal: 0, 1 ou 2 (Low, Medium, High)
    y = pd.Series(rng.integers(0, 3, n).astype(float), name="Burnout_Level")
    return X, y


# ── Checkpoint 1: Regressão Linear ───────────────────────────────────────────


def test_linear_regression_returns_model(small_data, tmp_path):
    X, y = small_data
    model = train_linear_regression(X, y, tmp_path)
    assert isinstance(model, LinearRegression)


def test_linear_regression_file_created(small_data, tmp_path):
    X, y = small_data
    train_linear_regression(X, y, tmp_path)
    assert (tmp_path / "linear_regression.pkl").exists()


def test_linear_regression_can_predict(small_data, tmp_path):
    X, y = small_data
    model = train_linear_regression(X, y, tmp_path)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert not np.any(np.isnan(preds))


def test_linear_regression_coef_count(small_data, tmp_path):
    X, y = small_data
    model = train_linear_regression(X, y, tmp_path)
    assert len(model.coef_) == len(FEATURE_COLUMNS)


# ── Checkpoint 2: Random Forest ──────────────────────────────────────────────


def test_random_forest_returns_model_and_importance(small_data, tmp_path):
    X, y = small_data
    model, importance = train_random_forest(X, y, tmp_path)
    assert isinstance(model, RandomForestRegressor)
    assert isinstance(importance, pd.Series)


def test_random_forest_file_created(small_data, tmp_path):
    X, y = small_data
    train_random_forest(X, y, tmp_path)
    assert (tmp_path / "random_forest.pkl").exists()


def test_random_forest_importance_sums_to_one(small_data, tmp_path):
    X, y = small_data
    _, importance = train_random_forest(X, y, tmp_path)
    assert abs(importance.sum() - 1.0) < 1e-6


def test_random_forest_importance_covers_all_features(small_data, tmp_path):
    X, y = small_data
    _, importance = train_random_forest(X, y, tmp_path)
    assert set(importance.index) == set(FEATURE_COLUMNS)


def test_random_forest_uses_fixed_seed(small_data, tmp_path):
    X, y = small_data
    d1, d2 = tmp_path / "a", tmp_path / "b"
    d1.mkdir()
    d2.mkdir()
    _, imp1 = train_random_forest(X, y, d1)
    _, imp2 = train_random_forest(X, y, d2)
    np.testing.assert_array_almost_equal(imp1.values, imp2.values)


# ── Checkpoint 3 & 4: XGBoost com GridSearchCV ───────────────────────────────


def test_xgboost_returns_model_and_params(small_data, tmp_path):
    X, y = small_data
    model, best_params = train_xgboost(X, y, tmp_path)
    assert isinstance(model, XGBRegressor)
    assert isinstance(best_params, dict)


def test_xgboost_file_created(small_data, tmp_path):
    X, y = small_data
    train_xgboost(X, y, tmp_path)
    assert (tmp_path / "xgboost_model.pkl").exists()


def test_xgboost_best_params_within_search_space(small_data, tmp_path):
    X, y = small_data
    _, best_params = train_xgboost(X, y, tmp_path)
    for param, values in XGB_PARAM_GRID.items():
        assert best_params[param] in values, (
            f"Hiperparâmetro '{param}' = {best_params[param]} "
            f"fora do espaço de busca {values}"
        )


def test_xgboost_can_predict(small_data, tmp_path):
    X, y = small_data
    model, _ = train_xgboost(X, y, tmp_path)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert not np.any(np.isnan(preds))


# ── Checkpoint 5: todos os 3 arquivos existem em models/ ─────────────────────


def test_all_three_model_files_exist():
    assert (MODELS_DIR / "linear_regression.pkl").exists(), (
        "linear_regression.pkl ausente"
    )
    assert (MODELS_DIR / "random_forest.pkl").exists(), "random_forest.pkl ausente"
    assert (MODELS_DIR / "xgboost_model.pkl").exists(), "xgboost_model.pkl ausente"


# ── Checkpoint 6: semente fixa — reprodutibilidade ───────────────────────────


def test_training_is_reproducible(small_data, tmp_path):
    X, y = small_data
    r1, r2 = tmp_path / "run1", tmp_path / "run2"
    r1.mkdir()
    r2.mkdir()
    m1 = train_linear_regression(X, y, r1)
    m2 = train_linear_regression(X, y, r2)
    np.testing.assert_array_almost_equal(m1.coef_, m2.coef_)


# ── Testes de integração: modelos salvos carregam e predizem corretamente ─────


def test_saved_linear_regression_loads():
    model = joblib.load(MODELS_DIR / "linear_regression.pkl")
    assert isinstance(model, LinearRegression)
    assert hasattr(model, "coef_")


def test_saved_random_forest_loads():
    model = joblib.load(MODELS_DIR / "random_forest.pkl")
    assert isinstance(model, RandomForestRegressor)
    assert hasattr(model, "feature_importances_")


def test_saved_xgboost_loads():
    model = joblib.load(MODELS_DIR / "xgboost_model.pkl")
    assert isinstance(model, XGBRegressor)


def test_saved_models_predict_on_test_data():
    X_test = pd.read_csv(
        Path(__file__).parent.parent / "data" / "processed" / "X_test.csv"
    )
    for name in ["linear_regression.pkl", "random_forest.pkl", "xgboost_model.pkl"]:
        model = joblib.load(MODELS_DIR / name)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test), f"{name}: tamanho errado nas predições"
        assert not np.any(np.isnan(preds)), f"{name}: predições com NaN"


def test_saved_models_predictions_in_valid_range():
    """Predições devem estar razoavelmente próximas de [0, 2] (Low=0, Med=1, High=2)."""
    X_test = pd.read_csv(
        Path(__file__).parent.parent / "data" / "processed" / "X_test.csv"
    )
    for name in ["linear_regression.pkl", "random_forest.pkl", "xgboost_model.pkl"]:
        model = joblib.load(MODELS_DIR / name)
        preds = model.predict(X_test)
        # Média deve estar entre 0 e 2
        assert 0.0 < preds.mean() < 2.0, (
            f"{name}: média das predições inesperada ({preds.mean():.3f})"
        )
