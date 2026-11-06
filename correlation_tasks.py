"""Вторая часть лабораторной: корреляции признаков с MRI_Count."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Посчитать корреляции отдельно для мужчин и женщин."""
    df = pd.read_csv(data.csv_path, sep="\t", na_values="NA")
    men = df[df["Gender"] == "Male"]
    women = df[df["Gender"] == "Female"]
    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]

    men_corr = {}
    women_corr = {}
    for feature in features:
        men_corr[feature] = float(men[feature].corr(men["MRI_Count"]))
        women_corr[feature] = float(women[feature].corr(women["MRI_Count"]))

    # Сравниваем модуль коэффициента: отрицательная связь тоже может быть сильной.
    strongest = max(
        features,
        key=lambda feature: max(abs(men_corr[feature]), abs(women_corr[feature])),
    )

    return BrainCorrelationSummary(
        men_count=len(men),
        women_count=len(women),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=strongest,
    )
