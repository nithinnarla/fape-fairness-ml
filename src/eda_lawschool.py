"""
FAPE, Law School Admissions EDA
Phase 4, Exploratory Data Analysis
Education/Legal Domain

EDA on Law School Admissions dataset, 18,692 records.
Wightman (1998) LSAC National Longitudinal Bar Passage Study,
adapted by LeQuy et al. via FairGround corpus.

Sensitive attributes: racetxt (0=minority, 1=White), male (0=Female, 1=Male)
Target: pass_bar, bar passage (binary)
Note: 90.2% pass rate, highly imbalanced dataset.
Note: law_school_tensorflow unavailable (HTTP 403), using law_school_lequy.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from lawschool_loader import load_law_school

RACE_LABELS = {0: "Minority", 1: "White"}
SEX_LABELS = {0.0: "Female", 1.0: "Male"}


def run_eda():
    print("FAPE Phase 4, Law School Admissions EDA")
    print("=" * 50)

    dataset = load_law_school()
    data = dataset["law_school"]
    df = pd.concat([data["X"], data["y"]], axis=1)
    df.columns = list(data["X"].columns) + ["label"]
    meta = data["metadata"]

    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")

    print(f"\n--- Label Distribution ---")
    label_counts = df["label"].value_counts().sort_index()
    label_pct = df["label"].value_counts(normalize=True).sort_index()
    print(f"  Failed bar (0): {label_counts[0]:,} ({label_pct[0]:.1%})")
    print(f"  Passed bar (1): {label_counts[1]:,} ({label_pct[1]:.1%})")
    print(f"  Note: 90.2% pass rate, severe class imbalance")

    print(f"\n--- Race Distribution ---")
    for code, label in RACE_LABELS.items():
        subset = df[df["racetxt"] == code]
        rate = subset["label"].mean()
        pct = len(subset) / len(df)
        print(f"  {label:<12} n={len(subset):,} ({pct:.1%}) | pass rate: {rate:.1%}")

    print(f"\n--- Sex Distribution ---")
    for code, label in SEX_LABELS.items():
        subset = df[df["male"] == code]
        if len(subset) == 0:
            continue
        rate = subset["label"].mean()
        pct = len(subset) / len(df)
        print(f"  {label:<10} n={len(subset):,} ({pct:.1%}) | pass rate: {rate:.1%}")

    print(f"\n--- LSAT Score Distribution ---")
    print(f"  Mean: {df['lsat'].mean():.2f}")
    print(f"  Median: {df['lsat'].median():.2f}")
    print(f"  Min: {df['lsat'].min():.2f} | Max: {df['lsat'].max():.2f}")
    lsat_corr = df["lsat"].corr(df["label"])
    print(f"  LSAT-pass correlation: {lsat_corr:.3f}")
    for code, label in RACE_LABELS.items():
        mean_lsat = df[df["racetxt"] == code]["lsat"].mean()
        print(f"  {label:<12} mean LSAT: {mean_lsat:.2f}")

    print(f"\n--- Undergraduate GPA (UGPA) ---")
    print(f"  Mean: {df['ugpa'].mean():.3f}")
    print(f"  Median: {df['ugpa'].median():.3f}")
    ugpa_corr = df["ugpa"].corr(df["label"])
    print(f"  UGPA-pass correlation: {ugpa_corr:.3f}")
    for code, label in RACE_LABELS.items():
        mean_ugpa = df[df["racetxt"] == code]["ugpa"].mean()
        print(f"  {label:<12} mean UGPA: {mean_ugpa:.3f}")

    print(f"\n--- School Tier vs Pass Rate ---")
    tier_stats = df.groupby("tier").agg(
        count=("label", "count"),
        pass_rate=("label", "mean")
    ).reset_index()
    for _, row in tier_stats.iterrows():
        print(f"  Tier {int(row['tier'])}: n={int(row['count']):,} | pass rate: {row['pass_rate']:.1%}")

    print(f"\n--- Family Income vs Pass Rate ---")
    print(f"  Mean family income: {df['fam_inc'].mean():.2f}")
    fam_corr = df["fam_inc"].corr(df["label"])
    print(f"  Family income-pass correlation: {fam_corr:.3f}")

    print(f"\n--- Intersectional (Race x Sex) ---")
    pivot = df.groupby(["racetxt", "male"])["label"].mean().unstack()
    pivot.index = [RACE_LABELS.get(i, str(i)) for i in pivot.index]
    pivot.columns = [SEX_LABELS.get(c, str(c)) for c in pivot.columns]
    print(pivot.round(3).to_string())

    print(f"\n--- Feature Correlations with Label ---")
    numeric_cols = ["lsat", "ugpa", "zfygpa", "zgpa", "decile1b",
                   "decile3", "fulltime", "fam_inc", "male", "racetxt", "tier"]
    correlations = df[numeric_cols + ["label"]].corr()["label"].drop("label").sort_values(ascending=False)
    for feat, val in correlations.items():
        print(f"  {feat:<15} {val:.3f}")

    print(f"\n--- Missing Values ---")
    nulls = df.isnull().sum()
    if nulls.sum() == 0:
        print("  No missing values")
    else:
        print(nulls[nulls > 0])


    print(f"\n--- LSAT vs UGPA Correlation by Race ---")
    for race in sorted(df['racetxt'].dropna().unique()):
        subset = df[df['racetxt']==race]
        if len(subset) < 50: continue
        corr = subset['lsat'].corr(subset['ugpa'])
        print(f"  race={int(race):<5} n={len(subset):,} mean_lsat={subset['lsat'].mean():.2f} mean_ugpa={subset['ugpa'].mean():.2f} corr={corr:.3f}")
    print(f"  Note: Racial group separation in LSAT-UGPA space, key fairness finding")

    print(f"\n--- Key Fairness Observations ---")
    white_pass = df[df["racetxt"] == 1]["label"].mean()
    minority_pass = df[df["racetxt"] == 0]["label"].mean()
    male_pass = df[df["male"] == 1.0]["label"].mean()
    female_pass = df[df["male"] == 0.0]["label"].mean()
    white_lsat = df[df["racetxt"] == 1]["lsat"].mean()
    minority_lsat = df[df["racetxt"] == 0]["lsat"].mean()
    print(f"  White pass rate:         {white_pass:.1%}")
    print(f"  Minority pass rate:      {minority_pass:.1%}")
    print(f"  Race gap:                {abs(white_pass - minority_pass):.1%}")
    print(f"  Male pass rate:          {male_pass:.1%}")
    print(f"  Female pass rate:        {female_pass:.1%}")
    print(f"  Sex gap:                 {abs(male_pass - female_pass):.1%}")
    print(f"  White mean LSAT:         {white_lsat:.2f}")
    print(f"  Minority mean LSAT:      {minority_lsat:.2f}")
    print(f"  LSAT gap:                {abs(white_lsat - minority_lsat):.2f}")
    print(f"  Note: Minority group n=1,201 (6.4%), small sample relative to White n=17,491")
    print(f"  Note: 90.2% overall pass rate means fairness metrics will be dominated")
    print(f"  by the majority class, FAPE Stage 2 will use equalized odds not accuracy")

    print(f"\n--- Law School EDA complete ---")
    print(f"  Total records: {len(df):,}")
    print(f"  Ready for Stage 1 preprocessing and baseline modeling")

    return df


if __name__ == "__main__":
    df = run_eda()
