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

    strongest = features[0]
    max_correlation = -1
    for feature in features:
        # Знак не учитываем: ищем самую сильную связь в любой группе.
        for correlation in [men_corr[feature], women_corr[feature]]:
            if abs(correlation) > max_correlation:
                max_correlation = abs(correlation)
                strongest = feature

    return BrainCorrelationSummary(
        men_count=len(men),
        women_count=len(women),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=strongest,
    )
