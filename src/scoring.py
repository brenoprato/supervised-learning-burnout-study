"""
Objetivo: Calcular o score de risco de burnout de um único funcionário.
Dataset: Remote Work Health Impact Survey 2025
Fonte: https://www.kaggle.com/datasets/pratyushpuri/remote-work-health-impact-survey-2025

Entradas:
  - dict com atributos brutos do funcionário (campos do novo dataset — ver abaixo)
  - models/imputer.pkl, models/scaler.pkl, models/xgboost_model.pkl
Saídas:
  - score: float em [0, 1] (RN-01)  — clamp(raw_pred / 2, 0, 1)
  - classificação: "baixo" | "moderado" | "alto" (RN-02)

Normalização do score:
  O target é ordinal: Low=0, Medium=1, High=2.
  O modelo prediz valores aproximadamente em [0, 2].
  score = clamp(raw_prediction / 2, 0, 1) mapeia:
    Low  (≈ 0) → score ≈ 0.0 → "baixo"
    Medium (≈ 1) → score ≈ 0.5 → "moderado"
    High (≈ 2) → score ≈ 1.0 → "alto"

Uso independente (sem re-executar o pipeline):
    from src.scoring import predict_burnout
    score, risco = predict_burnout({
        "Age":                    35,
        "Gender":                 "Female",
        "Region":                 "Europe",
        "Industry":               "Technology",
        "Job_Role":               "Data Analyst",
        "Work_Arrangement":       "Remote",
        "Hours_Per_Week":         40,
        "Mental_Health_Status":   "None",   # ou None para ausente
        "Work_Life_Balance_Score": 4,
        "has_physical_issue":     0,
        "Social_Isolation_Score": 2,
        "Salary_Range":           "$80K-100K",
    })
"""

import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    ENCODING_MAP,
    FEATURE_COLUMNS,
    MODELS_DIR,
)

# Artefatos carregados uma única vez por processo (evita I/O repetido)
_cache: dict = {}


# ── Classificação de risco (RN-02) ────────────────────────────────────────────


def classify_risk(score: float) -> str:
    """
    Mapeia score contínuo para categoria de risco conforme RN-02:
      [0.0, 0.3]  → "baixo"    — nenhuma ação imediata
      (0.3, 0.6)  → "moderado" — atenção recomendada
      [0.6, 1.0]  → "alto"     — intervenção necessária
    """
    if score <= 0.3:
        return "baixo"
    elif score < 0.6:
        return "moderado"
    else:
        return "alto"


# ── Pré-processamento do registro individual ──────────────────────────────────


def _preprocess_employee(employee: dict) -> pd.DataFrame:
    """
    Aplica as mesmas transformações do pipeline de preprocessing.py
    a um único registro. Retorna DataFrame de 1 linha com FEATURE_COLUMNS.

    Transformações aplicadas:
      - has_physical_issue: usa o valor direto (0 ou 1) se fornecido;
        None → 0 (sem problema físico reportado)
      - Mental_Health_Status: None ou ausente → "None" antes do encoding
      - Todas as categóricas: mapeadas via ENCODING_MAP (valor desconhecido → NaN)
    """
    row = {}

    # Campos numéricos diretos
    for col in [
        "Age",
        "Hours_Per_Week",
        "Work_Life_Balance_Score",
        "Social_Isolation_Score",
    ]:
        row[col] = employee.get(col, np.nan)

    # Campo derivado: has_physical_issue
    # O usuário pode passar True/False, 1/0 ou omitir
    raw_phys = employee.get("has_physical_issue", None)
    if raw_phys is None:
        row["has_physical_issue"] = 0
    else:
        row["has_physical_issue"] = int(bool(raw_phys))

    # Campos categóricos: encoding via ENCODING_MAP
    for col in [
        "Gender",
        "Region",
        "Industry",
        "Job_Role",
        "Work_Arrangement",
        "Salary_Range",
    ]:
        raw_val = employee.get(col)
        mapping = ENCODING_MAP.get(col, {})
        row[col] = mapping.get(raw_val, np.nan)

    # Mental_Health_Status: None string ou Python None → "None" (sem diagnóstico)
    mhs_raw = employee.get("Mental_Health_Status")
    if mhs_raw is None or (isinstance(mhs_raw, float) and np.isnan(mhs_raw)):
        mhs_raw = "None"
    row["Mental_Health_Status"] = ENCODING_MAP["Mental_Health_Status"].get(
        mhs_raw, np.nan
    )

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def _get_artifacts(models_dir: Path) -> dict:
    """Carrega e armazena em cache os três artefatos necessários."""
    key = str(models_dir)
    if key not in _cache:
        _cache[key] = {
            "imputer": joblib.load(models_dir / "imputer.pkl"),
            "scaler": joblib.load(models_dir / "scaler.pkl"),
            "model": joblib.load(models_dir / "xgboost_model.pkl"),
        }
    return _cache[key]


# ── Função principal de scoring ───────────────────────────────────────────────


def predict_burnout(
    employee: dict,
    models_dir: Optional[Path] = None,
) -> tuple:
    """
    Recebe os atributos brutos de um funcionário e retorna o score de risco
    de burnout normalizado e a classificação correspondente.

    Parâmetros
    ----------
    employee : dict
        Atributos do funcionário. Campos esperados:
          "Age"                    — int, 22–65
          "Gender"                 — "Female" | "Male" | "Non-binary" | "Prefer not to say"
          "Region"                 — "Africa" | "Asia" | "Europe" | "North America"
                                     | "Oceania" | "South America"
          "Industry"               — ex: "Technology", "Healthcare", "Finance" ...
          "Job_Role"               — ex: "Software Engineer", "Data Analyst" ...
          "Work_Arrangement"       — "Onsite" | "Hybrid" | "Remote"
          "Hours_Per_Week"         — int, 35–65
          "Mental_Health_Status"   — "None" | "Anxiety" | "Depression" | "Burnout"
                                     | "PTSD" | "ADHD" | "Stress Disorder" | None
          "Work_Life_Balance_Score" — int, 1–5  (5 = melhor equilíbrio)
          "has_physical_issue"     — 0 | 1  (1 = tem problemas físicos)
          "Social_Isolation_Score" — int, 1–5  (5 = mais isolado)
          "Salary_Range"           — "$40K-60K" | "$60K-80K" | "$80K-100K"
                                     | "$100K-120K" | "$120K+"

    models_dir : Path, opcional
        Diretório com os artefatos. Padrão: config.MODELS_DIR.

    Retorna
    -------
    (score, classificação) : (float, str)
        score          — valor contínuo em [0, 1], onde 0 = sem risco, 1 = máximo risco
        classificação  — "baixo" (≤0.3) | "moderado" (0.3–0.6) | "alto" (>0.6)
    """
    if models_dir is None:
        models_dir = MODELS_DIR

    artifacts = _get_artifacts(Path(models_dir))
    imputer = artifacts["imputer"]
    scaler = artifacts["scaler"]
    model = artifacts["model"]

    X = _preprocess_employee(employee)
    X_imp = pd.DataFrame(imputer.transform(X), columns=FEATURE_COLUMNS)
    X_scaled = pd.DataFrame(scaler.transform(X_imp), columns=FEATURE_COLUMNS)

    # Predição bruta: aproximadamente em [0, 2] (Low=0, Medium=1, High=2)
    raw_pred = float(model.predict(X_scaled)[0])
    # Normaliza para [0, 1]
    score = float(np.clip(raw_pred / 2.0, 0.0, 1.0))
    risk = classify_risk(score)

    return score, risk


# ── Execução via CLI — demonstra os 3 casos de referência ─────────────────────

if __name__ == "__main__":
    cases = [
        {
            "label": "Baixo risco — Remote, bom equilíbrio, sem problemas de saúde",
            "employee": {
                "Age": 30,
                "Gender": "Female",
                "Region": "Europe",
                "Industry": "Technology",
                "Job_Role": "Data Analyst",
                "Work_Arrangement": "Remote",
                "Hours_Per_Week": 38,
                "Mental_Health_Status": "None",
                "Work_Life_Balance_Score": 5,
                "has_physical_issue": 0,
                "Social_Isolation_Score": 1,
                "Salary_Range": "$80K-100K",
            },
        },
        {
            "label": "Risco moderado — Hybrid, equilíbrio médio, ansiedade",
            "employee": {
                "Age": 40,
                "Gender": "Male",
                "Region": "North America",
                "Industry": "Finance",
                "Job_Role": "Project Manager",
                "Work_Arrangement": "Hybrid",
                "Hours_Per_Week": 50,
                "Mental_Health_Status": "Anxiety",
                "Work_Life_Balance_Score": 3,
                "has_physical_issue": 0,
                "Social_Isolation_Score": 3,
                "Salary_Range": "$60K-80K",
            },
        },
        {
            "label": "Alto risco — Onsite, muitas horas, Burnout diagnosticado",
            "employee": {
                "Age": 52,
                "Gender": "Male",
                "Region": "Asia",
                "Industry": "Customer Service",
                "Job_Role": "IT Support",
                "Work_Arrangement": "Onsite",
                "Hours_Per_Week": 65,
                "Mental_Health_Status": "Burnout",
                "Work_Life_Balance_Score": 1,
                "has_physical_issue": 1,
                "Social_Isolation_Score": 5,
                "Salary_Range": "$40K-60K",
            },
        },
    ]

    print("\n" + "=" * 60)
    print("FUNÇÃO DE SCORING — CASOS DE DEMONSTRAÇÃO")
    print("Dataset: Remote Work Health Impact Survey 2025")
    print("=" * 60)

    for case in cases:
        score, risk = predict_burnout(case["employee"])
        print(f"\n{case['label']}")
        print(f"  Score : {score:.4f}")
        print(f"  Risco : {risk}")
