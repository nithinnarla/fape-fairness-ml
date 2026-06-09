"""
FAPE — COMPAS EDA
Phase 4 — Exploratory Data Analysis
Criminal Justice Domain

First EDA script — understanding demographic distributions,
label balance, and fairness-relevant patterns in COMPAS
before any model training begins.
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
    label_counts = df['label'].value_counts()
    label_pct = df['label'].value_counts(normalize=True)
    print(f"No recidivism (0): {label_counts[0]:,} ({label_pct[0]:.1%})")
    print(f"Recidivism (1):    {label_counts[1]:,} ({label_pct[1]:.1%})")

    print(f"\n--- Race Distribution ---")
    race_counts = df['race'].value_counts()
    for race, count in race_counts.items():
        pct = count / len(df)
        recid_rate = df[df['race'] == race]['label'].mean()
        print(f"  {race:<30} n={count:,} ({pct:.1%}) | recidivism rate: {recid_rate:.1%}")

    print(f"\n--- Sex Distribution ---")
    sex_counts = df['sex'].value_counts()
    for sex, count in sex_counts.items():
        pct = count / len(df)
        recid_rate = df[df['sex'] == sex]['label'].mean()
        print(f"  {sex:<10} n={count:,} ({pct:.1%}) | recidivism rate: {recid_rate:.1%}")

    print(f"\n--- Age Distribution ---")
    print(f"  Mean: {df['age'].mean():.1f}")
    print(f"  Median: {df['age'].median():.1f}")
    print(f"  Min: {df['age'].min()} | Max: {df['age'].max()}")

    print(f"\n--- COMPAS Decile Score Distribution ---")
    print(df['decile_score'].describe().round(2))

    print(f"\n--- Prior Convictions ---")
    print(f"  Mean: {df['priors_count'].mean():.2f}")
    print(f"  Median: {df['priors_count'].median():.0f}")
    print(f"  Max: {df['priors_count'].max()}")

    print(f"\n--- Missing Values ---")
    nulls = df.isnull().sum()
    if nulls.sum() == 0:
        print("  No missing values")
    else:
        print(nulls[nulls > 0])

    print(f"\n--- Key Fairness Observations ---")
    african_american_rate = df[df['race'] == 'African-American']['label'].mean()
    caucasian_rate = df[df['race'] == 'Caucasian']['label'].mean()
    print(f"  African-American recidivism rate: {african_american_rate:.1%}")
    print(f"  Caucasian recidivism rate: {caucasian_rate:.1%}")
    print(f"  Rate difference: {abs(african_american_rate - caucasian_rate):.1%}")
    print(f"  Note: Raw rate difference does not imply algorithmic bias.")

    print(f"\n--- COMPAS EDA complete ---")
    print(f"  Total records: {len(df):,}")
    print(f"  Ready for Stage 1 preprocessing and baseline modeling")

    return df


if __name__ == "__main__":
    df = run_eda()
