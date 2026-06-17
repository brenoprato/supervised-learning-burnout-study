#!/usr/bin/env python3
"""
prever.py — Interface interativa de terminal para predição de risco de burnout.
Dataset: Remote Work Health Impact Survey 2025
Fonte: https://www.kaggle.com/datasets/pratyushpuri/remote-work-health-impact-survey-2025
Autor do dataset: Pratyush Puri (Kaggle, junho de 2025)

Uso: python prever.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from scoring import predict_burnout

# ── Helpers de entrada ────────────────────────────────────────────────────────


def _barra(score: float, width: int = 20) -> str:
    """Barra de progresso proporcional ao score (20 caracteres)."""
    filled = round(score * width)
    return "#" * filled + " " * (width - filled)


def _input_opcao(prompt: str, opcoes: dict) -> str:
    """Exibe menu numerado e retorna o valor correspondente."""
    while True:
        print(prompt)
        for k, v in opcoes.items():
            print(f"  [{k}] {v}")
        escolha = input("  Escolha: ").strip()
        if escolha in opcoes:
            return opcoes[escolha]
        print(f"  ✗ Opção inválida. Digite um número entre 1 e {len(opcoes)}.\n")


def _input_inteiro(prompt: str, minimo: int, maximo: int) -> int:
    """Solicita inteiro dentro de intervalo válido."""
    while True:
        try:
            valor = int(input(f"{prompt} ({minimo}–{maximo}): ").strip())
            if minimo <= valor <= maximo:
                return valor
            print(f"  ✗ Valor fora do intervalo [{minimo}, {maximo}].\n")
        except ValueError:
            print("  ✗ Digite um número inteiro válido.\n")


# ── Interface principal ───────────────────────────────────────────────────────


def main():
    print("\n" + "=" * 56)
    print("  Sistema de Predição de Risco de Burnout")
    print("  Dataset: Remote Work Health Impact Survey 2025")
    print("  Modelo: XGBoost (melhor desempenho)")
    print("=" * 56 + "\n")

    nome = (
        input("Nome do funcionário (apenas para exibição): ").strip() or "Funcionário"
    )

    # ── Gênero ──────────────────────────────────────────────────────────────
    genero = _input_opcao(
        "\nGênero:",
        {"1": "Female", "2": "Male", "3": "Non-binary", "4": "Prefer not to say"},
    )

    # ── Idade ───────────────────────────────────────────────────────────────
    idade = _input_inteiro("\nIdade", 18, 70)

    # ── Região ──────────────────────────────────────────────────────────────
    regiao = _input_opcao(
        "\nRegião geográfica:",
        {
            "1": "Africa",
            "2": "Asia",
            "3": "Europe",
            "4": "North America",
            "5": "Oceania",
            "6": "South America",
        },
    )

    # ── Setor/Indústria ──────────────────────────────────────────────────────
    setor = _input_opcao(
        "\nSetor / Indústria:",
        {
            "1": "Customer Service",
            "2": "Education",
            "3": "Finance",
            "4": "Healthcare",
            "5": "Manufacturing",
            "6": "Marketing",
            "7": "Professional Services",
            "8": "Retail",
            "9": "Technology",
        },
    )

    # ── Cargo ────────────────────────────────────────────────────────────────
    cargo = _input_opcao(
        "\nCargo:",
        {
            "1": "Account Manager",
            "2": "Business Analyst",
            "3": "Customer Service Manager",
            "4": "Data Analyst",
            "5": "Data Scientist",
            "6": "DevOps Engineer",
            "7": "Digital Marketing Specialist",
            "8": "Executive Assistant",
            "9": "HR Manager",
            "10": "IT Support",
            "11": "Operations Manager",
            "12": "Product Manager",
            "13": "Project Manager",
            "14": "Quality Assurance",
            "15": "Research Scientist",
            "16": "Sales Representative",
            "17": "Social Media Manager",
            "18": "Software Engineer",
            "19": "Technical Writer",
            "20": "UX Designer",
        },
    )

    # ── Regime de trabalho ───────────────────────────────────────────────────
    regime = _input_opcao(
        "\nRegime de trabalho:",
        {"1": "Onsite", "2": "Hybrid", "3": "Remote"},
    )

    # ── Horas semanais ───────────────────────────────────────────────────────
    horas = _input_inteiro("\nHoras trabalhadas por semana", 35, 65)

    # ── Status de saúde mental ───────────────────────────────────────────────
    saude_mental = _input_opcao(
        "\nCondição de saúde mental (se houver diagnóstico):",
        {
            "1": "None",
            "2": "ADHD",
            "3": "Anxiety",
            "4": "Burnout",
            "5": "Depression",
            "6": "PTSD",
            "7": "Stress Disorder",
        },
    )

    # ── Equilíbrio vida-trabalho ─────────────────────────────────────────────
    wlb = _input_inteiro(
        "\nEquilíbrio vida-trabalho\n  (1 = muito ruim, 5 = excelente)",
        1,
        5,
    )

    # ── Problemas físicos ────────────────────────────────────────────────────
    tem_problema_fisico = _input_opcao(
        "\nFuncionário relata problemas físicos (dores, lesões)?",
        {"1": "Sim", "2": "Não"},
    )
    has_physical = 1 if tem_problema_fisico == "Sim" else 0

    # ── Isolamento social ────────────────────────────────────────────────────
    isolamento = _input_inteiro(
        "\nNível de isolamento social percebido\n"
        "  (1 = muito conectado, 5 = muito isolado)",
        1,
        5,
    )

    # ── Faixa salarial ───────────────────────────────────────────────────────
    salario = _input_opcao(
        "\nFaixa salarial anual (USD):",
        {
            "1": "$40K-60K",
            "2": "$60K-80K",
            "3": "$80K-100K",
            "4": "$100K-120K",
            "5": "$120K+",
        },
    )

    # ── Predição ─────────────────────────────────────────────────────────────
    employee = {
        "Age": idade,
        "Gender": genero,
        "Region": regiao,
        "Industry": setor,
        "Job_Role": cargo,
        "Work_Arrangement": regime,
        "Hours_Per_Week": horas,
        "Mental_Health_Status": saude_mental,
        "Work_Life_Balance_Score": wlb,
        "has_physical_issue": has_physical,
        "Social_Isolation_Score": isolamento,
        "Salary_Range": salario,
    }

    score, risco = predict_burnout(employee)

    msg = {
        "baixo": "Nenhuma ação imediata necessária.",
        "moderado": "Atenção recomendada. Acompanhar de perto.",
        "alto": "Intervenção necessária. Acionar equipe de saúde ocupacional.",
    }

    print("\n" + "=" * 50)
    print(f"   RESULTADO — {nome.upper()}")
    print("=" * 50)
    print(f"  Score    : {score:.4f}  [{_barra(score)}]")
    print(f"  Risco    : {risco.upper()}")
    print(f"  Orientação: {msg[risco]}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()


# ── Casos de referência (valores obtidos após treinamento com SEED=42) ─────────
# Caso 1 — Baixo risco:
#   Age=30, Gender=Female, Region=Europe, Industry=Technology,
#   Job_Role=Data Analyst, Work_Arrangement=Remote, Hours_Per_Week=38,
#   Mental_Health_Status=None, Work_Life_Balance_Score=5,
#   has_physical_issue=0, Social_Isolation_Score=1, Salary_Range=$80K-100K
#   → score ≈ 0.10–0.25, risco = baixo
#
# Caso 2 — Risco moderado:
#   Age=40, Gender=Male, Region=North America, Industry=Finance,
#   Job_Role=Project Manager, Work_Arrangement=Hybrid, Hours_Per_Week=50,
#   Mental_Health_Status=Anxiety, Work_Life_Balance_Score=3,
#   has_physical_issue=0, Social_Isolation_Score=3, Salary_Range=$60K-80K
#   → score ≈ 0.35–0.55, risco = moderado
#
# Caso 3 — Alto risco:
#   Age=52, Gender=Male, Region=Asia, Industry=Customer Service,
#   Job_Role=IT Support, Work_Arrangement=Onsite, Hours_Per_Week=65,
#   Mental_Health_Status=Burnout, Work_Life_Balance_Score=1,
#   has_physical_issue=1, Social_Isolation_Score=5, Salary_Range=$40K-60K
#   → score ≈ 0.65–0.90, risco = alto
