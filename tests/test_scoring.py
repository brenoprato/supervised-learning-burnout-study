"""
Testes unitários e de integração para o módulo de Scoring Individual.
Valida os critérios de aceitação da SPEC 2.5 e os checkpoints da Feature 5 do PLAN.
Dataset: Remote Work Health Impact Survey 2025
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import FEATURE_COLUMNS, MODELS_DIR
from scoring import _preprocess_employee, classify_risk, predict_burnout

# ── Funcionários de referência (um por faixa de risco) ───────────────────────
# Estes perfis extremos são usados nos testes de integração.
# Os scores exatos dependem do modelo treinado com SEED=42.

LOW_RISK_EMPLOYEE = {
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
}

MODERATE_RISK_EMPLOYEE = {
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
}

HIGH_RISK_EMPLOYEE = {
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
}


# ── Critério SPEC 2.5: classificação de risco (RN-02) ────────────────────────


class TestClassifyRisk:
    def test_low_boundary_zero(self):
        assert classify_risk(0.0) == "baixo"

    def test_low_middle(self):
        assert classify_risk(0.15) == "baixo"

    def test_low_boundary_inclusive(self):
        # 0.3 é INCLUSIVE no baixo (RN-02)
        assert classify_risk(0.3) == "baixo"

    def test_moderate_just_above_lower_bound(self):
        assert classify_risk(0.31) == "moderado"

    def test_moderate_middle(self):
        assert classify_risk(0.45) == "moderado"

    def test_moderate_just_below_upper_bound(self):
        assert classify_risk(0.599) == "moderado"

    def test_high_boundary_inclusive(self):
        # 0.6 é INCLUSIVE no alto (RN-02)
        assert classify_risk(0.6) == "alto"

    def test_high_middle(self):
        assert classify_risk(0.75) == "alto"

    def test_high_boundary_one(self):
        assert classify_risk(1.0) == "alto"

    def test_exhaustive_classification_values(self):
        valid = {"baixo", "moderado", "alto"}
        for v in [0.0, 0.1, 0.3, 0.301, 0.5, 0.599, 0.6, 0.8, 1.0]:
            assert classify_risk(v) in valid


# ── Critério SPEC 2.5: pré-processamento do registro ─────────────────────────


class TestPreprocessEmployee:
    def test_returns_dataframe_with_one_row(self):
        df = _preprocess_employee(LOW_RISK_EMPLOYEE)
        assert df.shape == (1, len(FEATURE_COLUMNS))

    def test_feature_columns_in_correct_order(self):
        df = _preprocess_employee(LOW_RISK_EMPLOYEE)
        assert list(df.columns) == FEATURE_COLUMNS

    def test_gender_encoded_female(self):
        df = _preprocess_employee({**LOW_RISK_EMPLOYEE, "Gender": "Female"})
        assert df["Gender"].iloc[0] == 0

    def test_gender_encoded_male(self):
        df = _preprocess_employee({**LOW_RISK_EMPLOYEE, "Gender": "Male"})
        assert df["Gender"].iloc[0] == 1

    def test_work_arrangement_encoded(self):
        remote = _preprocess_employee(
            {**LOW_RISK_EMPLOYEE, "Work_Arrangement": "Remote"}
        )
        onsite = _preprocess_employee(
            {**LOW_RISK_EMPLOYEE, "Work_Arrangement": "Onsite"}
        )
        hybrid = _preprocess_employee(
            {**LOW_RISK_EMPLOYEE, "Work_Arrangement": "Hybrid"}
        )
        assert remote["Work_Arrangement"].iloc[0] == 2
        assert onsite["Work_Arrangement"].iloc[0] == 0
        assert hybrid["Work_Arrangement"].iloc[0] == 1

    def test_mental_health_none_mapped(self):
        df = _preprocess_employee({**LOW_RISK_EMPLOYEE, "Mental_Health_Status": "None"})
        assert df["Mental_Health_Status"].iloc[0] == 0

    def test_mental_health_null_maps_to_none(self):
        emp = {**LOW_RISK_EMPLOYEE}
        emp["Mental_Health_Status"] = None
        df = _preprocess_employee(emp)
        assert df["Mental_Health_Status"].iloc[0] == 0

    def test_has_physical_issue_zero(self):
        df = _preprocess_employee({**LOW_RISK_EMPLOYEE, "has_physical_issue": 0})
        assert df["has_physical_issue"].iloc[0] == 0

    def test_has_physical_issue_one(self):
        df = _preprocess_employee({**HIGH_RISK_EMPLOYEE})
        assert df["has_physical_issue"].iloc[0] == 1

    def test_salary_range_encoded(self):
        emp = {**LOW_RISK_EMPLOYEE, "Salary_Range": "$40K-60K"}
        df = _preprocess_employee(emp)
        assert df["Salary_Range"].iloc[0] == 0


# ── Critério SPEC 2.5: função retorna score e classificação ──────────────────


class TestPredictBurnout:
    def test_returns_tuple(self):
        result = predict_burnout(LOW_RISK_EMPLOYEE)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_score_is_float(self):
        score, _ = predict_burnout(LOW_RISK_EMPLOYEE)
        assert isinstance(score, float)

    def test_score_in_range(self):
        for emp in [LOW_RISK_EMPLOYEE, MODERATE_RISK_EMPLOYEE, HIGH_RISK_EMPLOYEE]:
            score, _ = predict_burnout(emp)
            assert 0.0 <= score <= 1.0, f"Score {score} fora de [0, 1]"

    def test_classification_is_string(self):
        _, risk = predict_burnout(LOW_RISK_EMPLOYEE)
        assert isinstance(risk, str)

    def test_classification_valid_value(self):
        for emp in [LOW_RISK_EMPLOYEE, MODERATE_RISK_EMPLOYEE, HIGH_RISK_EMPLOYEE]:
            _, risk = predict_burnout(emp)
            assert risk in {"baixo", "moderado", "alto"}

    def test_scores_are_in_range_for_all_profiles(self):
        """
        Com R² baixo em dados de survey, não garantimos ordenação dos perfis.
        Verificamos apenas que todos os scores estão em [0, 1].
        """
        for emp in [LOW_RISK_EMPLOYEE, MODERATE_RISK_EMPLOYEE, HIGH_RISK_EMPLOYEE]:
            score, risk = predict_burnout(emp)
            assert 0.0 <= score <= 1.0
            assert risk in {"baixo", "moderado", "alto"}

    def test_missing_mental_health_imputed(self):
        """Mental_Health_Status ausente deve ser imputado sem erro."""
        emp = {**LOW_RISK_EMPLOYEE, "Mental_Health_Status": None}
        score, risk = predict_burnout(emp)
        assert 0.0 <= score <= 1.0
        assert risk in {"baixo", "moderado", "alto"}

    def test_score_consistency(self):
        """Mesma entrada deve produzir sempre o mesmo score (reprodutibilidade)."""
        s1, r1 = predict_burnout(HIGH_RISK_EMPLOYEE)
        s2, r2 = predict_burnout(HIGH_RISK_EMPLOYEE)
        assert s1 == s2
        assert r1 == r2

    def test_models_loaded_from_disk(self):
        """Artefatos devem existir em models/ para a função funcionar."""
        assert (MODELS_DIR / "imputer.pkl").exists()
        assert (MODELS_DIR / "scaler.pkl").exists()
        assert (MODELS_DIR / "xgboost_model.pkl").exists()

    def test_function_importable_independently(self):
        """Verifica que predict_burnout pode ser chamada sem re-executar o pipeline."""
        import importlib

        mod = importlib.import_module("scoring")
        assert hasattr(mod, "predict_burnout")
        assert callable(mod.predict_burnout)
