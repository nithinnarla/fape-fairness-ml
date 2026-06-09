"""
FAPE — Folktables ACS EDA
Phase 4 — Exploratory Data Analysis
Socioeconomic Domain

EDA on US Census ACS 2021 data — 1,589,032 records across
all 50 states. Understanding income inequality patterns,
demographic distributions, and fairness-relevant signals
before baseline model training.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_folktables_acs

# Census race codes
RACE_LABELS = {
    1: "White",
    2: "Black/African-American",
    3: "American Indian",
    4: "Alaska Native",
    5: "American Indian + Alaska Native",
    6: "Asian",
    7: "Native Hawaiian/Pacific Islander",
    8: "Other",
    9: "Two or more races"
}

SEX_LABELS = {1: "Male", 2: "Female"}


def run_eda():
    print("FAPE Phase 4 — Folktables ACS EDA")
    print("=" * 50)

    df = load_folktables_acs()

    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")

    print(f"\n--- Label Distribution ---")
    label_counts = df["label"].value_counts()
    label_pct = df["label"].value_counts(normalize=True)
    print(f"Income <= $50k (0): {label_counts[0]:,} ({label_pct[0]:.1%})")
    print(f"Income >  $50k (1): {label_counts[1]:,} ({label_pct[1]:.1%})")

    print(f"\n--- Race Distribution (RAC1P) ---")
    race_counts = df["RAC1P"].value_counts().sort_index()
    for code, count in race_counts.items():
        label = RACE_LABELS.get(code, f"Code {code}")
        pct = count / len(df)
        income_rate = df[df["RAC1P"] == code]["label"].mean()
        print(f"  {label:<35} n={count:,} ({pct:.1%}) | income >$50k: {income_rate:.1%}")

    print(f"\n--- Sex Distribution (SEX) ---")
    sex_counts = df["SEX"].value_counts().sort_index()
    for code, count in sex_counts.items():
        label = SEX_LABELS.get(code, f"Code {code}")
        pct = count / len(df)
        income_rate = df[df["SEX"] == code]["label"].mean()
        print(f"  {label:<10} n={count:,} ({pct:.1%}) | income >$50k: {income_rate:.1%}")

    print(f"\n--- Age Distribution (AGEP) ---")
    print(f"  Mean: {df['AGEP'].mean():.1f}")
    print(f"  Median: {df['AGEP'].median():.1f}")
    print(f"  Min: {df['AGEP'].min()} | Max: {df['AGEP'].max()}")

    print(f"\n--- Education Distribution (SCHL) ---")
    print(f"  Mean: {df['SCHL'].mean():.1f}")
    print(f"  Median: {df['SCHL'].median():.1f}")

    print(f"\n--- Hours Worked Per Week (WKHP) ---")
    print(f"  Mean: {df['WKHP'].mean():.1f}")
    print(f"  Median: {df['WKHP'].median():.1f}")

    print(f"\n--- Disability Status (DIS) ---")
    dis_counts = df["DIS"].value_counts()
    for code, count in dis_counts.items():
        pct = count / len(df)
        income_rate = df[df["DIS"] == code]["label"].mean()
        label = "With disability" if code == 1 else "Without disability"
        print(f"  {label:<20} n={count:,} ({pct:.1%}) | income >$50k: {income_rate:.1%}")

    print(f"\n--- Missing Values ---")
    nulls = df.isnull().sum()
    if nulls.sum() == 0:
        print("  No missing values")
    else:
        print(nulls[nulls > 0])

    print(f"\n--- Key Fairness Observations ---")
    white_rate = df[df["RAC1P"] == 1]["label"].mean()
    black_rate = df[df["RAC1P"] == 2]["label"].mean()
    male_rate = df[df["SEX"] == 1]["label"].mean()
    female_rate = df[df["SEX"] == 2]["label"].mean()
    print(f"  White income >$50k rate:          {white_rate:.1%}")
    print(f"  Black/African-American rate:       {black_rate:.1%}")
    print(f"  Race gap (White - Black):          {abs(white_rate - black_rate):.1%}")
    print(f"  Male income >$50k rate:            {male_rate:.1%}")
    print(f"  Female income >$50k rate:          {female_rate:.1%}")
    print(f"  Sex gap (Male - Female):           {abs(male_rate - female_rate):.1%}")

    print(f"\n--- Folktables ACS EDA complete ---")
    print(f"  Total records: {len(df):,} across all 50 states")
    print(f"  Ready for Stage 1 preprocessing and baseline modeling")

    return df


if __name__ == "__main__":
    df = run_eda()
