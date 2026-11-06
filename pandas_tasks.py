"""Первая часть лабораторной: анализ пассажиров Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Загрузить таблицу и посчитать основные показатели."""
    df = pd.read_csv(data.csv_path)

    missing = df.isna().sum()
    mean_age = df.groupby("Pclass")["Age"].mean()
    survival_rate = df.groupby("Pclass")["Survived"].mean()

    return TitanicSummary(
        row_count=len(df),
        missing_by_column={column: int(count) for column, count in missing.items()},
        adults_over_30_count=int((df["Age"] > 30).sum()),
        mean_age_by_pclass={int(pclass): float(age) for pclass, age in mean_age.items()},
        survival_rate_by_pclass={
            int(pclass): float(rate) for pclass, rate in survival_rate.items()
        },
        highest_fares=df["Fare"].nlargest(5).tolist(),
    )
