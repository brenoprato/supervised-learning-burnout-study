"""
Objetivo: Avaliar e comparar os três modelos treinados sobre o conjunto de teste.
Entradas:
  - data/processed/X_train.csv, y_train.csv — usados para K-Fold CV
  - data/processed/X_test.csv,  y_test.csv  — conjunto de teste isolado
  - models/linear_regression.pkl, random_forest.pkl, xgboost_model.pkl
  - datasets/survey_2025.csv — para análise de equidade (variáveis sensíveis)
Saídas:
  - reports/figures/real_vs_pred_linear_regression.png
  - reports/figures/real_vs_pred_random_forest.png
  - reports/figures/real_vs_pred_xgboost.png
  - reports/figures/feature_importance_random_forest.png
  - reports/figures/feature_importance_xgboost.png
  - reports/figures/equity_by_gender.png
  - reports/figures/equity_by_work_location.png
  - reports/results/model_comparison.csv
  - reports/results/cv_results.csv
  - reports/results/conclusion.txt
"""

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_PATH,
    ENCODING_MAP,
    FEATURE_COLUMNS,
    MODELS_DIR,
    REPORTS_FIGURES_DIR,
    REPORTS_RESULTS_DIR,
    SEED,
    TARGET_COLUMN,
    TARGET_ENCODING,
)

MODEL_REGISTRY = {
    "Linear Regression": "linear_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost_model.pkl",
}


# ── I/O ───────────────────────────────────────────────────────────────────────


def _load_split(processed_dir: Path):
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def _load_models(models_dir: Path) -> dict:
    return {
        name: joblib.load(models_dir / fname) for name, fname in MODEL_REGISTRY.items()
    }


# ── Métricas ──────────────────────────────────────────────────────────────────


def compute_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE e R² no conjunto de teste."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_on_test(models: dict, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        preds = model.predict(X_test)
        m = compute_metrics(y_test, preds)
        rows.append({"Modelo": name, **m})
    return pd.DataFrame(rows).set_index("Modelo")


def cross_validate_models(models: dict, X_train, y_train, cv: int = 5) -> pd.DataFrame:
    kf = KFold(n_splits=cv, shuffle=True, random_state=SEED)
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }
    rows = []
    for name, model in models.items():
        cv_res = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring)
        rows.append(
            {
                "Modelo": name,
                "MAE_mean": -cv_res["test_MAE"].mean(),
                "MAE_std": cv_res["test_MAE"].std(),
                "RMSE_mean": -cv_res["test_RMSE"].mean(),
                "RMSE_std": cv_res["test_RMSE"].std(),
                "R2_mean": cv_res["test_R2"].mean(),
                "R2_std": cv_res["test_R2"].std(),
            }
        )
    return pd.DataFrame(rows).set_index("Modelo")


# ── Visualizações ──────────────────────────────────────────────────────────────


def plot_real_vs_predicted(models: dict, X_test, y_test, figures_dir: Path):
    """Scatter de valores reais vs preditos para cada modelo."""
    for name, model in models.items():
        preds = model.predict(X_test)
        fname = "real_vs_pred_" + name.lower().replace(" ", "_") + ".png"

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, preds, alpha=0.3, s=12, color="#4C72B0")
        lims = [
            min(float(y_test.min()), float(preds.min())) - 0.1,
            max(float(y_test.max()), float(preds.max())) + 0.1,
        ]
        ax.plot(lims, lims, "r--", linewidth=1.2, label="Predição perfeita")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Valor Real (0=Low, 1=Medium, 2=High)")
        ax.set_ylabel("Valor Predito")
        ax.set_title(f"Real vs Predito — {name}")
        ax.legend(fontsize=9)
        plt.tight_layout()
        path = figures_dir / fname
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Salvo: {path}")


def plot_feature_importance(models: dict, figures_dir: Path):
    """Importância de features para Random Forest e XGBoost."""
    targets = {
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
    }
    for name, slug in targets.items():
        if name not in models:
            continue
        model = models[name]
        importances = pd.Series(
            model.feature_importances_, index=FEATURE_COLUMNS
        ).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(9, 6))
        importances.plot(kind="barh", ax=ax, color="#4C72B0", edgecolor="white")
        ax.set_title(f"Importância das Features — {name}", fontsize=12)
        ax.set_xlabel("Importância relativa")
        plt.tight_layout()
        path = figures_dir / f"feature_importance_{slug}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Salvo: {path}")


def _reconstruct_col(series: pd.Series, label_map: dict) -> pd.Series:
    """
    Reconstrói a categoria original a partir de valores MinMaxScaler-escalados.
    O MinMaxScaler mapeia inteiros 0..N-1 para o intervalo [0,1];
    multiplicar por (N-1) e arredondar recupera o índice original.
    """
    n = len(label_map)  # número de categorias
    if n <= 1:
        return series.map(label_map)
    reconstructed = (series * (n - 1)).round().astype(int)
    return reconstructed.map(label_map)


def plot_equity_analysis(models: dict, X_test, y_test, figures_dir: Path):
    """Analisa MAE por subgrupo de gênero e regime de trabalho (RN-09)."""
    gender_map = {v: k for k, v in ENCODING_MAP["Gender"].items()}
    work_map = {v: k for k, v in ENCODING_MAP["Work_Arrangement"].items()}

    subgroups = {
        "Gender": ("Gender", gender_map),
        "Work_Arrangement": ("Work_Arrangement", work_map),
    }
    out_names = {
        "Gender": "equity_by_gender.png",
        "Work_Arrangement": "equity_by_work_location.png",
    }

    for col_key, (col, label_map) in subgroups.items():
        fig, axes = plt.subplots(
            1, len(models), figsize=(5 * len(models), 5), sharey=False
        )
        if len(models) == 1:
            axes = [axes]

        for ax, (name, model) in zip(axes, models.items()):
            preds = model.predict(X_test)
            df_eq = X_test.copy()
            df_eq["y_true"] = y_test.values
            df_eq["y_pred"] = preds
            # Reconstruir categoria original a partir do valor MinMaxScaler-escalado
            df_eq["group"] = _reconstruct_col(df_eq[col], label_map)

            mae_by_group = (
                df_eq.groupby("group")
                .apply(
                    lambda g: mean_absolute_error(g["y_true"], g["y_pred"]),
                    include_groups=False,
                )
                .rename("MAE")
                .sort_index()
            )

            colors = ["#4C72B0", "#C44E52", "#55A868", "#DD8452"]
            mae_by_group.plot(
                kind="bar",
                ax=ax,
                color=colors[: len(mae_by_group)],
                edgecolor="white",
                rot=20,
            )
            ax.set_title(name, fontsize=10)
            ax.set_xlabel(col)
            ax.set_ylabel("MAE")
            for bar in ax.patches:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{bar.get_height():.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        fig.suptitle(f"Análise de Equidade — MAE por {col}", y=1.02)
        plt.tight_layout()
        path = figures_dir / out_names[col_key]
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Salvo: {path}")


# ── Persistência ──────────────────────────────────────────────────────────────


def _print_table(title: str, df: pd.DataFrame):
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)
    print(df.round(6).to_string())


def save_results(
    metrics_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    results_dir: Path,
    best_model: str,
):
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.round(6).to_csv(results_dir / "model_comparison.csv")
    cv_df.round(6).to_csv(results_dir / "cv_results.csv")

    conclusion = (
        f"MODELO RECOMENDADO: {best_model}\n"
        f"Critério: maior R² e menor RMSE no conjunto de teste.\n\n"
        f"Métricas no conjunto de teste (target ordinal: 0=Low, 1=Medium, 2=High):\n"
        f"{metrics_df.round(6).to_string()}\n\n"
        f"Validação cruzada (K-Fold k=5):\n"
        f"{cv_df.round(6).to_string()}\n\n"
        f"Nota: dados de survey auto-reportados — ver docs/limitations.md.\n"
    )
    (results_dir / "conclusion.txt").write_text(conclusion, encoding="utf-8")
    print(f"\n  Salvo: {results_dir / 'model_comparison.csv'}")
    print(f"  Salvo: {results_dir / 'cv_results.csv'}")
    print(f"  Salvo: {results_dir / 'conclusion.txt'}")


# ── Orquestrador ──────────────────────────────────────────────────────────────


def run_evaluation(
    processed_dir=None,
    models_dir=None,
    figures_dir=None,
    results_dir=None,
):
    if processed_dir is None:
        processed_dir = DATA_PROCESSED_DIR
    if models_dir is None:
        models_dir = MODELS_DIR
    if figures_dir is None:
        figures_dir = REPORTS_FIGURES_DIR
    if results_dir is None:
        results_dir = REPORTS_RESULTS_DIR

    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)
    figures_dir = Path(figures_dir)
    results_dir = Path(results_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("AVALIAÇÃO — Remote Work Health Impact Survey 2025")
    print("=" * 60)

    X_train, X_test, y_train, y_test = _load_split(processed_dir)
    models = _load_models(models_dir)
    print(f"\nModelos carregados: {list(models.keys())}")
    print(f"Conjunto de teste : {X_test.shape[0]} registros")

    # 1 — Métricas no conjunto de teste
    print("\n[1] Calculando métricas no conjunto de teste")
    metrics_df = evaluate_on_test(models, X_test, y_test)
    _print_table("MÉTRICAS — CONJUNTO DE TESTE", metrics_df)

    # 2 — Validação cruzada K-Fold (k=5)
    print("\n[2] Validação cruzada K-Fold (k=5) no treino")
    cv_df = cross_validate_models(models, X_train, y_train, cv=5)
    _print_table("VALIDAÇÃO CRUZADA K-FOLD (k=5)", cv_df)

    # 3 — Gráficos real vs predito
    print("\n[3] Gerando gráficos real vs predito")
    plot_real_vs_predicted(models, X_test, y_test, figures_dir)

    # 4 — Importância de features
    print("\n[4] Gerando gráficos de importância de features")
    plot_feature_importance(models, figures_dir)

    # 5 — Análise de equidade
    print("\n[5] Análise de equidade por subgrupo")
    plot_equity_analysis(models, X_test, y_test, figures_dir)

    # 6 — Melhor modelo e persistência
    best_model = metrics_df["R2"].idxmax()
    print(f"\n[6] Melhor modelo por R²: {best_model}")
    save_results(metrics_df, cv_df, results_dir, best_model)

    print("\n" + "=" * 60)
    print(f"AVALIAÇÃO CONCLUÍDA — Modelo recomendado: {best_model}")
    print("=" * 60)

    return metrics_df, cv_df, best_model


if __name__ == "__main__":
    run_evaluation()
