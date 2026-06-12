"""
FAPE — Student Performance EDA
Phase 4 — Exploratory Data Analysis
Education Domain

EDA on Student Performance dataset — 1,044 records across two subjects.
Math (395 students) and Portuguese (649 students) from two Portuguese schools.
Understanding demographic distributions, family background effects,
and educational fairness patterns before baseline model training.

Source: Cortez & Silva (2008) — UCI ML Repository ID: 320
Sensitive attributes: sex (F=0, M=1), age
Target: G3 final grade binarized at median (above median = 1)
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from student_loader import load_student_performance

SEX_LABELS = {0: "Female", 1: "Male"}


def run_eda():
    print("FAPE Phase 4 — Student Performance EDA")
    print("=" * 50)

    datasets = load_student_performance()

    for subject, data in datasets.items():
        df = pd.concat([data["X"], data["y"]], axis=1)
        df.columns = list(data["X"].columns) + ["label"]
        meta = data["metadata"]

        print(f"\n" + "="*50)
        print(f"Dataset: {meta['name']} ({subject})")
        print(f"Records: {meta['n_samples']} | Features: {meta['n_features']}")
        print(f"Median grade: {meta['median_grade']} | Positive rate: {meta['positive_rate']:.1%}")

        print(f"\n--- Label Distribution ---")
        label_counts = df["label"].value_counts()
        label_pct = df["label"].value_counts(normalize=True)
        print(f"  Below median (0): {label_counts[0]:,} ({label_pct[0]:.1%})")
        print(f"  Above median (1): {label_counts[1]:,} ({label_pct[1]:.1%})")

        print(f"\n--- Sex Distribution ---")
        for code, label in SEX_LABELS.items():
            subset = df[df["sex"] == code]
            if len(subset) == 0:
                continue
            rate = subset["label"].mean()
            print(f"  {label:<10} n={len(subset):,} ({len(subset)/len(df):.1%}) | above median: {rate:.1%}")

        print(f"\n--- Age Distribution ---")
        print(f"  Mean: {df['age'].mean():.1f} | Min: {df['age'].min()} | Max: {df['age'].max()}")
        age_perf = df.groupby("age")["label"].mean()
        for age, rate in age_perf.items():
            count = (df["age"] == age).sum()
            print(f"  Age {age}: n={count} | above median: {rate:.1%}")

        print(f"\n--- Parental Education ---")
        print(f"  Mother education (Medu) mean: {df['Medu'].mean():.2f}")
        print(f"  Father education (Fedu) mean: {df['Fedu'].mean():.2f}")
        medu_corr = df["Medu"].corr(df["label"])
        fedu_corr = df["Fedu"].corr(df["label"])
        print(f"  Medu-label correlation: {medu_corr:.3f}")
        print(f"  Fedu-label correlation: {fedu_corr:.3f}")

        print(f"\n--- Study Time ---")
        study_perf = df.groupby("studytime")["label"].mean()
        for st, rate in study_perf.items():
            count = (df["studytime"] == st).sum()
            print(f"  Study time {st}: n={count} | above median: {rate:.1%}")

        print(f"\n--- Past Failures ---")
        fail_perf = df.groupby("failures")["label"].mean()
        for f, rate in fail_perf.items():
            count = (df["failures"] == f).sum()
            print(f"  Failures {f}: n={count} | above median: {rate:.1%}")
        fail_corr = df["failures"].corr(df["label"])
        print(f"  Failures-label correlation: {fail_corr:.3f}")

        print(f"\n--- Absences ---")
        print(f"  Mean absences: {df['absences'].mean():.2f}")
        abs_corr = df["absences"].corr(df["label"])
        print(f"  Absences-label correlation: {abs_corr:.3f}")

        print(f"\n--- Alcohol Consumption ---")
        dalc_corr = df["Dalc"].corr(df["label"])
        walc_corr = df["Walc"].corr(df["label"])
        print(f"  Workday alcohol (Dalc) mean: {df['Dalc'].mean():.2f} | correlation: {dalc_corr:.3f}")
        print(f"  Weekend alcohol (Walc) mean: {df['Walc'].mean():.2f} | correlation: {walc_corr:.3f}")

        print(f"\n--- School Support ---")
        for col in ["schoolsup", "famsup", "paid", "higher", "internet"]:
            rate_0 = df[df[col] == 0]["label"].mean()
            rate_1 = df[df[col] == 1]["label"].mean()
            print(f"  {col:<12} no={rate_0:.1%} yes={rate_1:.1%} | diff={abs(rate_1-rate_0):.1%}")
        print(f"  Note: schoolsup paradox — students WITH support perform worse.")
        print(f"  This is a selection effect: struggling students receive more support,")
        print(f"  not evidence that support causes poor performance.")

        print(f"\n--- Intersectional (Sex x Failures) ---")
        pivot = df.groupby(["sex", "failures"])["label"].mean().unstack(fill_value=float("nan"))
        pivot.index = [SEX_LABELS.get(i, i) for i in pivot.index]
        print(pivot.round(3).to_string())

        print(f"\n--- Feature Correlations with Label ---")
        numeric_cols = ["age", "Medu", "Fedu", "traveltime", "studytime",
                       "failures", "famrel", "freetime", "goout",
                       "Dalc", "Walc", "health", "absences"]
        correlations = df[numeric_cols + ["label"]].corr()["label"].drop("label").sort_values(ascending=False)
        for feat, val in correlations.items():
            print(f"  {feat:<15} {val:.3f}")

        print(f"\n--- Missing Values ---")
        nulls = df.isnull().sum()
        if nulls.sum() == 0:
            print("  No missing values")
        else:
            print(nulls[nulls > 0])

        print(f"\n--- Key Fairness Observations ---")
        female_rate = df[df["sex"] == 0]["label"].mean()
        male_rate = df[df["sex"] == 1]["label"].mean()
        print(f"  Female above median rate: {female_rate:.1%}")
        print(f"  Male above median rate:   {male_rate:.1%}")
        print(f"  Sex gap:                  {abs(female_rate - male_rate):.1%}")
        print(f"  Zero failures above median: {df[df['failures']==0]['label'].mean():.1%}")
        print(f"  1+ failures above median:   {df[df['failures']>0]['label'].mean():.1%}")
        print(f"  Note: Ages 20-22 have very small samples (n<=6) — interpret with caution")
        if subject == 'portuguese':
            print(f"  Note: Sex gap reverses vs math — females outperform males in Portuguese")


    print(f"\n--- Subject Comparison (Math vs Portuguese) ---")
    math_df = None
    por_df = None
    for subject, data in datasets.items():
        df = pd.concat([data["X"], data["y"]], axis=1)
        df.columns = list(data["X"].columns) + ["label"]
        if subject == "math":
            math_df = df
        else:
            por_df = df

    print(f"  Sex gap — Math:       Male {math_df[math_df['sex']==1]['label'].mean():.1%} vs Female {math_df[math_df['sex']==0]['label'].mean():.1%} (Male higher)")
    print(f"  Sex gap — Portuguese: Female {por_df[por_df['sex']==0]['label'].mean():.1%} vs Male {por_df[por_df['sex']==1]['label'].mean():.1%} (Female higher)")
    print(f"  Key finding: Sex performance gap reverses between subjects")
    print(f"  Failures impact — Math:       0 failures {math_df[math_df['failures']==0]['label'].mean():.1%} vs 1+ failures {math_df[math_df['failures']>0]['label'].mean():.1%}")
    print(f"  Failures impact — Portuguese: 0 failures {por_df[por_df['failures']==0]['label'].mean():.1%} vs 1+ failures {por_df[por_df['failures']>0]['label'].mean():.1%}")

    print(f"\n--- Student Performance EDA complete ---")
    print(f"  Total records: 1,044 (math: 395 + portuguese: 649)")
    print(f"  Ready for Stage 1 preprocessing and baseline modeling")

    return datasets


if __name__ == "__main__":
    datasets = run_eda()
