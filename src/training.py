"""
Objetivo: Treinar os três modelos de regressão ordinal em complexidade crescente.
Entradas:
  - data/processed/X_train.csv — features de treino (pré-processadas)
  - data/processed/y_train.csv — target de treino (0=Low, 1=Medium, 2=High)
Saídas:
  - models/linear_regression.pkl — Regressão Linear (baseline)
  - models/random_forest.pkl     — Random Forest Regressor
  - models/xgboost_model.pkl     — XGBoost Regressor (hiperparâmetros otimizados)

Nota: o target é ordinal (0/1/2). O pipeline de regressão é mantido como regressão (score normalizado em scoring.py).
O score final é normalizado para [0,1] dividindo a predição bruta por 2 em scoring.py.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_PROCESSED_DIR,
    FEATURE_COLUMNS,
    MODELS_DIR,
    SEED,
    XGB_PARAM_GRID,
)


def _load_train_data(processed_dir: Path):
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze()
    return X_train, y_train


def train_linear_regression(X_train, y_train, models_dir: Path):
    print("\n[1] Regressão Linear (baseline)")
    model = LinearRegression()
    model.fit(X_train, y_train)

    coef = pd.Series(model.coef_, index=FEATURE_COLUMNS).sort_values(
        key=abs, ascending=False
    )
    print("  Coeficientes (ordenados por magnitude):")
    for feat, val in coef.items():
        print(f"    {feat:<35} {val:+.4f}")

    path = models_dir / "linear_regression.pkl"
    joblib.dump(model, path)
    print(f"  Salvo: {path}")
    return model


def train_random_forest(X_train, y_train, models_dir: Path):
    print("\n[2] Random Forest Regressor")
    model = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)

    importance = pd.Series(
        model.feature_importances_, index=FEATURE_COLUMNS
    ).sort_values(ascending=False)
    print("  Importância das features:")
    for feat, val in importance.items():
        bar = "█" * int(val * 40)
        print(f"    {feat:<35} {val:.4f}  {bar}")

    path = models_dir / "random_forest.pkl"
    joblib.dump(model, path)
    print(f"  Salvo: {path}")
    return model, importance


def train_xgboost(X_train, y_train, models_dir: Path):
    print("\n[3] XGBoost Regressor (com GridSearchCV)")
    print(f"  Espaço de busca: {XGB_PARAM_GRID}")
    n_combos = 1
    for v in XGB_PARAM_GRID.values():
        n_combos *= len(v)
    print(f"  Combinações × folds: {n_combos} × 5 = {n_combos * 5} fits")

    base = XGBRegressor(
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )
    grid_search = GridSearchCV(
        base,
        XGB_PARAM_GRID,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    print(f"  Melhores hiperparâmetros: {best_params}")
    print(f"  Melhor MSE (CV):          {best_score:.6f}")

    model = grid_search.best_estimator_
    path = models_dir / "xgboost_model.pkl"
    joblib.dump(model, path)
    print(f"  Salvo: {path}")
    return model, best_params


def run_training(processed_dir=None, models_dir=None):
    if processed_dir is None:
        processed_dir = DATA_PROCESSED_DIR
    if models_dir is None:
        models_dir = MODELS_DIR

    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("TREINAMENTO DOS MODELOS — Remote Work Health Impact Survey 2025")
    print("=" * 60)

    X_train, y_train = _load_train_data(processed_dir)
    print(f"\nDados de treino carregados: {X_train.shape}")
    print(f"Distribuição do target — {y_train.value_counts().sort_index().to_dict()}")

    lr_model = train_linear_regression(X_train, y_train, models_dir)
    rf_model, rf_importance = train_random_forest(X_train, y_train, models_dir)
    xgb_model, xgb_best_params = train_xgboost(X_train, y_train, models_dir)

    print("\n" + "=" * 60)
    print("TREINAMENTO CONCLUÍDO")
    print("=" * 60)
    print("  models/linear_regression.pkl")
    print("  models/random_forest.pkl")
    print("  models/xgboost_model.pkl")

    return lr_model, rf_model, xgb_model, rf_importance, xgb_best_params


if __name__ == "__main__":
    run_training()
