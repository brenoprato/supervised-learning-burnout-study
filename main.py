"""
Orquestrador do pipeline completo de predição de burnout.
Executa as cinco etapas em sequência ou permite rodar cada uma isoladamente.

Uso:
    python main.py                       # pipeline completo
    python main.py --step eda
    python main.py --step preprocessing
    python main.py --step training
    python main.py --step evaluation
    python main.py --step scoring
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def _demo_scoring():
    """Demonstração da função de scoring com perfis do novo dataset."""
    from scoring import predict_burnout

    print("\n" + "=" * 60)
    print("DEMONSTRAÇÃO — FUNÇÃO DE SCORING INDIVIDUAL")
    print("Dataset: Remote Work Health Impact Survey 2025")
    print("=" * 60)
    cases = [
        (
            "Perfil Remote/alto WLB",
            {
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
        ),
        (
            "Perfil Hybrid/moderado",
            {
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
        ),
        (
            "Perfil Onsite/alto burnout",
            {
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
        ),
    ]
    for label, emp in cases:
        score, risk = predict_burnout(emp)
        print(f"  {label:<28}  score={score:.4f}  risco={risk}")


def run_all():
    from eda import run_eda
    from evaluation import run_evaluation
    from preprocessing import run_preprocessing
    from training import run_training

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETO — PREDIÇÃO DE BURNOUT")
    print("=" * 60)

    run_eda()
    run_preprocessing()
    run_training()
    run_evaluation()
    _demo_scoring()

    print("\nPipeline concluído com sucesso.")


STEPS = {
    "eda": lambda: __import__("eda").run_eda(),
    "preprocessing": lambda: __import__("preprocessing").run_preprocessing(),
    "training": lambda: __import__("training").run_training(),
    "evaluation": lambda: __import__("evaluation").run_evaluation(),
    "scoring": _demo_scoring,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de predição de burnout")
    parser.add_argument(
        "--step",
        choices=list(STEPS.keys()),
        help="Executar somente uma etapa do pipeline",
    )
    args = parser.parse_args()
    STEPS[args.step]() if args.step else run_all()
