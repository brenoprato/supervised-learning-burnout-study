"""
Objetivo: Configurações globais compartilhadas por todos os scripts do pipeline.
Entradas: Nenhuma (módulo de constantes).
Saídas:   Constantes e caminhos importáveis pelos demais módulos.

Dataset: Remote Work Health Impact Survey 2025
Fonte: https://www.kaggle.com/datasets/pratyushpuri/remote-work-health-impact-survey-2025
Autor: Pratyush Puri — coletado em junho de 2025
Decisão de modelagem: Burnout_Level é categórica ordinal (Low/Medium/High).
  Convertida para numérica (Low=0, Medium=1, High=2) e tratada como
  regressão ordinal. Score final = clamp(pred / 2, 0, 1).
"""

from pathlib import Path

# ── Reprodutibilidade ─────────────────────────────────────────────────────────
SEED = 42

# ── Caminhos ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "datasets" / "survey_2025.csv"
DATA_RAW_PATH = DATASET_PATH  # alias

DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR = DATA_PROCESSED_DIR  # alias

DATA_PROCESSED_X_TRAIN = DATA_PROCESSED_DIR / "X_train.csv"
DATA_PROCESSED_X_TEST = DATA_PROCESSED_DIR / "X_test.csv"
DATA_PROCESSED_Y_TRAIN = DATA_PROCESSED_DIR / "y_train.csv"
DATA_PROCESSED_Y_TEST = DATA_PROCESSED_DIR / "y_test.csv"

MODELS_DIR = BASE_DIR / "models"
REPORTS_FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR = REPORTS_FIGURES_DIR  # alias
REPORTS_RESULTS_DIR = BASE_DIR / "reports" / "results"
RESULTS_DIR = REPORTS_RESULTS_DIR  # alias

# ── Dataset ───────────────────────────────────────────────────────────────────
TARGET_COLUMN = "Burnout_Level"  # ordinal: Low=0, Medium=1, High=2
ID_COLUMN = None  # sem identificador único
DATE_COLUMN = None  # Survey_Date descartada (sem data de admissão)

# Colunas removidas no pré-processamento:
#   Survey_Date        — todos os registros são de junho/2025 (sem valor preditivo)
#   Physical_Health_Issues — texto multi-label; substituído por has_physical_issue
DROP_COLS = ["Survey_Date", "Physical_Health_Issues"]

# Features numéricas diretas (sem transformação necessária além de impute/scale)
NUMERIC_COLS = [
    "Age",
    "Hours_Per_Week",
    "Work_Life_Balance_Score",
    "Social_Isolation_Score",
]

# Features categóricas a encodar via ENCODING_MAP
CATEGORICAL_COLS = [
    "Gender",
    "Region",
    "Industry",
    "Job_Role",
    "Work_Arrangement",
    "Mental_Health_Status",  # NaN preenchido com "None" antes do encoding
    "Salary_Range",
]

# ── Encoding das variáveis categóricas ────────────────────────────────────────
# Variáveis com ordem natural: encoding ordinal.
# Variáveis nominais: label encoding por ordem alfabética.
ENCODING_MAP = {
    # Nominal
    "Gender": {
        "Female": 0,
        "Male": 1,
        "Non-binary": 2,
        "Prefer not to say": 3,
    },
    # Nominal — por ordem alfabética das regiões
    "Region": {
        "Africa": 0,
        "Asia": 1,
        "Europe": 2,
        "North America": 3,
        "Oceania": 4,
        "South America": 5,
    },
    # Nominal — por ordem alfabética dos setores
    "Industry": {
        "Customer Service": 0,
        "Education": 1,
        "Finance": 2,
        "Healthcare": 3,
        "Manufacturing": 4,
        "Marketing": 5,
        "Professional Services": 6,
        "Retail": 7,
        "Technology": 8,
    },
    # Nominal — por ordem alfabética dos cargos
    "Job_Role": {
        "Account Manager": 0,
        "Business Analyst": 1,
        "Customer Service Manager": 2,
        "Data Analyst": 3,
        "Data Scientist": 4,
        "DevOps Engineer": 5,
        "Digital Marketing Specialist": 6,
        "Executive Assistant": 7,
        "HR Manager": 8,
        "IT Support": 9,
        "Operations Manager": 10,
        "Product Manager": 11,
        "Project Manager": 12,
        "Quality Assurance": 13,
        "Research Scientist": 14,
        "Sales Representative": 15,
        "Social Media Manager": 16,
        "Software Engineer": 17,
        "Technical Writer": 18,
        "UX Designer": 19,
    },
    # Ordinal: crescente em flexibilidade de local de trabalho
    "Work_Arrangement": {"Onsite": 0, "Hybrid": 1, "Remote": 2},
    # Nominal (condições de saúde mental sem hierarquia clara)
    "Mental_Health_Status": {
        "None": 0,
        "ADHD": 1,
        "Anxiety": 2,
        "Burnout": 3,
        "Depression": 4,
        "PTSD": 5,
        "Stress Disorder": 6,
    },
    # Ordinal: faixas salariais em ordem crescente de renda
    "Salary_Range": {
        "$40K-60K": 0,
        "$60K-80K": 1,
        "$80K-100K": 2,
        "$100K-120K": 3,
        "$120K+": 4,
    },
}

# Alias para compatibilidade interna
CATEGORICAL_ENCODING = ENCODING_MAP

# ── Conversão da variável alvo ordinal para numérica ─────────────────────────
TARGET_ENCODING = {"Low": 0, "Medium": 1, "High": 2}

# ── Ordem final das features após pré-processamento ──────────────────────────
# 12 features: 4 numéricas + 7 categóricas encodadas + 1 derivada (has_physical_issue)
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Region",
    "Industry",
    "Job_Role",
    "Work_Arrangement",
    "Hours_Per_Week",
    "Mental_Health_Status",
    "Work_Life_Balance_Score",
    "has_physical_issue",
    "Social_Isolation_Score",
    "Salary_Range",
]

# ── Split ─────────────────────────────────────────────────────────────────────
TEST_SIZE = 0.20

# ── XGBoost GridSearch ────────────────────────────────────────────────────────
XGB_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.10, 0.20],
}
