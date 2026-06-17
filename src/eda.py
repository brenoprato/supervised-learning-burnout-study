"""
Objetivo: Análise Exploratória dos Dados (EDA) do dataset de predição de burnout.
Entradas: datasets/survey_2025.csv — Remote Work Health Impact Survey 2025.
Saídas:
  - reports/figures/burnout_distribution.png
  - reports/figures/correlation_heatmap.png
  - reports/figures/burnout_by_work_location.png
  - reports/figures/burnout_by_gender.png
  - reports/figures/burnout_by_industry.png
  - reports/results/eda_summary.txt
  - Relatório textual de missing values e estatísticas descritivas no stdout.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_RAW_PATH,
    REPORTS_FIGURES_DIR,
    REPORTS_RESULTS_DIR,
    TARGET_COLUMN,
    TARGET_ENCODING,
)

# Colunas numéricas para o heatmap de correlação
_NUMERIC_FOR_CORR = [
    "Age",
    "Hours_Per_Week",
    "Work_Life_Balance_Score",
    "Social_Isolation_Score",
]


def load_data(path=None) -> pd.DataFrame:
    if path is None:
        path = DATA_RAW_PATH
    df = pd.read_csv(path)
    return df


def inspect_data(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("INSPEÇÃO DO DATASET — Remote Work Health Impact Survey 2025")
    print("=" * 60)
    print(f"Dimensões: {df.shape[0]} registros × {df.shape[1]} colunas")
    print("\nColunas e tipos:")
    print(df.dtypes.to_string())
    print("\nPrimeiros registros:")
    print(df.head(3).to_string())
    print()


def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("VALORES AUSENTES")
    print("=" * 60)
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Ausentes": missing, "Percentual (%)": pct})
    missing_df = missing_df[missing_df["Ausentes"] > 0]
    if missing_df.empty:
        print("Nenhum valor ausente encontrado.")
    else:
        print(missing_df.to_string())
    print()
    return missing_df


def print_descriptive_stats(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("ESTATÍSTICAS DESCRITIVAS — VARIÁVEIS NUMÉRICAS")
    print("=" * 60)
    print(df.describe().round(3).to_string())
    print()


def _add_target_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna numérica do target para análises de correlação."""
    df = df.copy()
    df["Burnout_Numeric"] = df[TARGET_COLUMN].map(TARGET_ENCODING)
    return df


def plot_burnout_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    """Gráfico de barras da distribuição da variável alvo (Low/Medium/High)."""
    counts = df[TARGET_COLUMN].value_counts().reindex(["Low", "Medium", "High"])
    pcts = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=["#2ecc71", "#f39c12", "#e74c3c"],
        edgecolor="white",
        width=0.55,
    )
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 12,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title("Distribuição do Nível de Burnout", fontsize=13)
    ax.set_xlabel("Burnout_Level")
    ax.set_ylabel("Frequência")
    ax.set_ylim(0, counts.max() * 1.15)
    plt.tight_layout()

    path = output_dir / "burnout_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {path}")
    return path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """Heatmap de correlação de Pearson entre variáveis numéricas + alvo."""
    df_num = _add_target_numeric(df)
    cols = _NUMERIC_FOR_CORR + ["Burnout_Numeric"]
    corr = df_num[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax,
    )
    ax.set_title("Correlação de Pearson — Variáveis Numéricas", fontsize=12)
    plt.tight_layout()

    path = output_dir / "correlation_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {path}")
    return path


def plot_burnout_by_work_location(df: pd.DataFrame, output_dir: Path) -> Path:
    """Distribuição do Burnout por regime de trabalho (Onsite/Hybrid/Remote)."""
    order_arr = ["Onsite", "Hybrid", "Remote"]
    order_burn = ["Low", "Medium", "High"]

    cross = (
        df.groupby(["Work_Arrangement", TARGET_COLUMN])
        .size()
        .unstack(TARGET_COLUMN)
        .reindex(index=order_arr, columns=order_burn, fill_value=0)
    )
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    cross_pct.plot(
        kind="bar",
        stacked=True,
        color=["#2ecc71", "#f39c12", "#e74c3c"],
        edgecolor="white",
        ax=ax,
    )
    ax.set_title("Distribuição de Burnout por Regime de Trabalho", fontsize=12)
    ax.set_xlabel("Regime de Trabalho")
    ax.set_ylabel("Percentual (%)")
    ax.legend(title="Burnout Level", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()

    path = output_dir / "burnout_by_work_location.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {path}")
    return path


def plot_burnout_by_gender(df: pd.DataFrame, output_dir: Path) -> Path:
    """Distribuição percentual do Burnout por gênero."""
    order_burn = ["Low", "Medium", "High"]
    cross = (
        df.groupby(["Gender", TARGET_COLUMN])
        .size()
        .unstack(TARGET_COLUMN)
        .reindex(columns=order_burn, fill_value=0)
    )
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    cross_pct.plot(
        kind="bar",
        stacked=True,
        color=["#2ecc71", "#f39c12", "#e74c3c"],
        edgecolor="white",
        ax=ax,
    )
    ax.set_title("Distribuição de Burnout por Gênero", fontsize=12)
    ax.set_xlabel("Gênero")
    ax.set_ylabel("Percentual (%)")
    ax.legend(title="Burnout Level", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    plt.tight_layout()

    path = output_dir / "burnout_by_gender.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {path}")
    return path


def plot_burnout_by_industry(df: pd.DataFrame, output_dir: Path) -> Path:
    """Percentual de burnout Alto por setor/indústria (ranking)."""
    df_num = _add_target_numeric(df)
    high_pct = (
        df_num.groupby("Industry")["Burnout_Numeric"]
        .apply(lambda x: (x == 2).mean() * 100)
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    high_pct.plot(kind="barh", ax=ax, color="#e74c3c", edgecolor="white")
    ax.set_title("Percentual de Burnout Alto por Setor", fontsize=12)
    ax.set_xlabel("% com Burnout Alto")
    ax.set_ylabel("Setor")
    for bar in ax.patches:
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.1f}%",
            va="center",
            fontsize=9,
        )
    plt.tight_layout()

    path = output_dir / "burnout_by_industry.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {path}")
    return path


def print_subgroup_findings(df: pd.DataFrame) -> dict:
    """Imprime e retorna achados principais por subgrupo."""
    df_num = _add_target_numeric(df)
    findings = {}

    print("=" * 60)
    print("ACHADOS POR SUBGRUPO")
    print("=" * 60)

    # Distribuição do alvo
    dist = df[TARGET_COLUMN].value_counts(normalize=True).mul(100).round(1)
    findings["target_distribution"] = dist.to_dict()
    print("\nDistribuição do Burnout_Level:")
    for cat, pct in dist.items():
        print(f"  {cat:8s}: {pct:.1f}%")

    # Média numérica por regime de trabalho
    print("\nMédia de Burnout (0=Low,1=Med,2=High) por Work_Arrangement:")
    for arr, grp in df_num.groupby("Work_Arrangement"):
        m = grp["Burnout_Numeric"].mean()
        print(f"  {arr:10s}: {m:.3f}")
        findings[f"mean_burnout_{arr}"] = round(m, 3)

    # Correlações numéricas com o alvo
    corr_cols = _NUMERIC_FOR_CORR
    corrs = (
        df_num[corr_cols + ["Burnout_Numeric"]]
        .corr()["Burnout_Numeric"]
        .drop("Burnout_Numeric")
        .sort_values(key=abs, ascending=False)
    )
    print("\nCorrelações com Burnout_Numeric (Pearson):")
    for feat, val in corrs.items():
        print(f"  {feat:<30}: {val:+.4f}")
    findings["top_correlation"] = {corrs.index[0]: round(corrs.iloc[0], 4)}

    # Percentual de burnout alto por gênero
    print("\nPercentual de Burnout Alto por Gênero:")
    for gender, grp in df_num.groupby("Gender"):
        pct = (grp["Burnout_Numeric"] == 2).mean() * 100
        print(f"  {gender:20s}: {pct:.1f}%")

    print()
    return findings


def save_eda_summary(findings: dict, results_dir: Path) -> Path:
    """Salva resumo em texto dos achados da EDA."""
    results_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "=== EDA SUMMARY — Remote Work Health Impact Survey 2025 ===\n",
        "Dataset: https://www.kaggle.com/datasets/pratyushpuri/remote-work-health-impact-survey-2025\n",
        "\n--- Distribuição de Burnout_Level ---\n",
    ]
    dist = findings.get("target_distribution", {})
    for cat in ["Low", "Medium", "High"]:
        lines.append(f"  {cat}: {dist.get(cat, '?')}%\n")
    lines.append("\n--- Média Burnout por Work_Arrangement ---\n")
    for k, v in findings.items():
        if k.startswith("mean_burnout_"):
            arr = k.replace("mean_burnout_", "")
            lines.append(f"  {arr}: {v}\n")
    lines.append("\n--- Correlação mais forte com Burnout ---\n")
    for feat, val in findings.get("top_correlation", {}).items():
        lines.append(f"  {feat}: {val}\n")

    path = results_dir / "eda_summary.txt"
    path.write_text("".join(lines), encoding="utf-8")
    print(f"Salvo: {path}")
    return path


def run_eda(data_path=None, figures_dir=None, results_dir=None):
    if data_path is None:
        data_path = DATA_RAW_PATH
    if figures_dir is None:
        figures_dir = REPORTS_FIGURES_DIR
    if results_dir is None:
        results_dir = REPORTS_RESULTS_DIR

    figures_dir = Path(figures_dir)
    results_dir = Path(results_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ANÁLISE EXPLORATÓRIA DOS DADOS — BURNOUT PREDICTION")
    print("Dataset: Remote Work Health Impact Survey 2025")
    print("=" * 60 + "\n")

    df = load_data(data_path)
    inspect_data(df)
    report_missing_values(df)
    print_descriptive_stats(df)

    print("=" * 60)
    print("GERANDO VISUALIZAÇÕES")
    print("=" * 60)
    plot_burnout_distribution(df, figures_dir)
    plot_correlation_heatmap(df, figures_dir)
    plot_burnout_by_work_location(df, figures_dir)
    plot_burnout_by_gender(df, figures_dir)
    plot_burnout_by_industry(df, figures_dir)

    findings = print_subgroup_findings(df)
    save_eda_summary(findings, results_dir)

    print("\nEDA concluída com sucesso. Artefatos salvos em:", figures_dir)
    return df


if __name__ == "__main__":
    run_eda()
