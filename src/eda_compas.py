"""
FAPE — COMPAS EDA
Phase 4 — Exploratory Data Analysis
Criminal Justice Domain

EDA on COMPAS recidivism dataset — 6,172 records from Broward County Florida (2013-2014).
Understanding demographic distributions, COMPAS algorithmic bias patterns,
false positive/negative rate disparities, and proxy variable bias before
baseline model training.

Source: ProPublica — Angwin et al. (2016)
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_compas


def run_eda():
    print("FAPE Phase 4 — COMPAS EDA")
    print("=" * 50)

    df = load_compas()

    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")

    print(f"\n--- Label Distribution ---")
    label_counts = df["label"].value_counts()
    label_pct = df["label"].value_counts(normalize=True)
    print(f"  No recidivism (0): {label_counts[0]:,} ({label_pct[0]:.1%})")
    print(f"  Recidivism (1):    {label_counts[1]:,} ({label_pct[1]:.1%})")

    print(f"\n--- Race Distribution ---")
    race_counts = df["race"].value_counts()
    for race, count in race_counts.items():
        pct = count / len(df)
        recid_rate = df[df["race"] == race]["label"].mean()
        print(f"  {race:<30} n={count:,} ({pct:.1%}) | recidivism rate: {recid_rate:.1%}")

    print(f"\n--- Sex Distribution ---")
    sex_counts = df["sex"].value_counts()
    for sex, count in sex_counts.items():
        pct = count / len(df)
        recid_rate = df[df["sex"] == sex]["label"].mean()
        print(f"  {sex:<10} n={count:,} ({pct:.1%}) | recidivism rate: {recid_rate:.1%}")

    print(f"\n--- Age Distribution ---")
    print(f"  Mean: {df['age'].mean():.1f}")
    print(f"  Median: {df['age'].median():.1f}")
    print(f"  Min: {df['age'].min()} | Max: {df['age'].max()}")
    age_recid = df.groupby(pd.cut(df["age"], bins=[18,25,35,45,55,100]), observed=False)["label"].mean()
    print(f"  Recidivism by age group:")
    for age_grp, rate in age_recid.items():
        print(f"    {str(age_grp):<15} {rate:.1%}")

    print(f"\n--- COMPAS Decile Score Distribution ---")
    print(df["decile_score"].describe().round(2))
    print(f"\n  Decile score mean by race:")
    race_score = df.groupby("race")["decile_score"].mean().sort_values(ascending=False)
    for race, score in race_score.items():
        print(f"    {race:<30} mean score: {score:.2f}")

    print(f"\n--- Decile Score vs Actual Recidivism ---")
    score_recid = df.groupby("decile_score")["label"].mean()
    print(f"  Calibration — recidivism rate per decile score:")
    for score, rate in score_recid.items():
        print(f"    Score {score}: {rate:.1%}")

    print(f"\n--- False Positive and Negative Rates by Race ---")
    print(f"  False positive: did not reoffend (label=0) but scored high risk (decile>=7)")
    print(f"  False negative: did reoffend (label=1) but scored low risk (decile<=4)")
    for race in df["race"].value_counts().index:
        race_df = df[df["race"] == race]
        if len(race_df) < 10:
            continue
        no_recid = race_df[race_df["label"] == 0]
        did_recid = race_df[race_df["label"] == 1]
        fpr = (no_recid["decile_score"] >= 7).mean() if len(no_recid) > 0 else 0
        fnr = (did_recid["decile_score"] <= 4).mean() if len(did_recid) > 0 else 0
        print(f"  {race:<30} FPR: {fpr:.1%} | FNR: {fnr:.1%}")

    print(f"\n--- Prior Convictions by Race ---")
    race_priors = df.groupby("race")["priors_count"].mean().sort_values(ascending=False)
    for race, mean_priors in race_priors.items():
        print(f"  {race:<30} mean priors: {mean_priors:.2f}")

    print(f"\n--- Prior Convictions ---")
    print(f"  Mean: {df['priors_count'].mean():.2f}")
    print(f"  Median: {df['priors_count'].median():.0f}")
    print(f"  Max: {df['priors_count'].max()}")
    prior_recid_corr = df["priors_count"].corr(df["label"])
    print(f"  Priors-recidivism correlation: {prior_recid_corr:.3f}")

    print(f"\n--- Charge Degree vs Recidivism ---")
    charge_stats = df.groupby("c_charge_degree").agg(
        count=("label", "count"),
        recid_rate=("label", "mean")
    )
    for degree, row in charge_stats.iterrows():
        label = "Felony" if degree == "F" else "Misdemeanor"
        print(f"  {label} ({degree}): n={int(row['count']):,} | recidivism rate: {row['recid_rate']:.1%}")

    print(f"\n--- Intersectional Analysis (Race x Sex) ---")
    pivot = df.groupby(["race","sex"])["label"].mean().unstack()
    print(pivot.to_string(float_format=lambda x: f"{x:.1%}"))

    print(f"\n--- Feature Correlations with Label ---")
    numeric_cols = ["age", "priors_count", "decile_score", "days_b_screening_arrest"]
    correlations = df[numeric_cols + ["label"]].corr()["label"].drop("label").sort_values(ascending=False)
    for feat, val in correlations.items():
        print(f"  {feat:<30} {val:.3f}")

    print(f"\n--- Missing Values ---")
    nulls = df.isnull().sum()
    if nulls.sum() == 0:
        print("  No missing values")
    else:
        print(nulls[nulls > 0])

    print(f"\n--- Key Fairness Observations ---")
    aa_rate = df[df["race"] == "African-American"]["label"].mean()
    ca_rate = df[df["race"] == "Caucasian"]["label"].mean()
    aa_score = df[df["race"] == "African-American"]["decile_score"].mean()
    ca_score = df[df["race"] == "Caucasian"]["decile_score"].mean()
    aa_no_recid = df[(df["race"]=="African-American") & (df["label"]==0)]
    ca_no_recid = df[(df["race"]=="Caucasian") & (df["label"]==0)]
    aa_fpr = (aa_no_recid["decile_score"] >= 7).mean()
    ca_fpr = (ca_no_recid["decile_score"] >= 7).mean()
    print(f"  African-American recidivism rate: {aa_rate:.1%}")
    print(f"  Caucasian recidivism rate:        {ca_rate:.1%}")
    print(f"  Race gap:                         {abs(aa_rate - ca_rate):.1%}")
    print(f"  African-American mean COMPAS score: {aa_score:.2f}")
    print(f"  Caucasian mean COMPAS score:        {ca_score:.2f}")
    print(f"  Score gap:                          {abs(aa_score - ca_score):.2f}")
    print(f"  African-American false positive rate: {aa_fpr:.1%}")
    print(f"  Caucasian false positive rate:        {ca_fpr:.1%}")
    print(f"  FPR gap:                              {abs(aa_fpr - ca_fpr):.1%}")
    print(f"  Note: Native American (n=11) and Asian (n=31) results unreliable due to small sample size")

    print(f"\n--- COMPAS EDA complete ---")
    print(f"  Total records: {len(df):,}")
    print(f"  Ready for Stage 1 preprocessing and baseline modeling")

    return df


if __name__ == "__main__":
    df = run_eda()
