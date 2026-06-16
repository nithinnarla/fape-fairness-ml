"""
FAPE — SBA Agricultural Loans EDA
Phase 4 — Exploratory Data Analysis
Agriculture/Financial Domain

EDA on SBA 7(a) Agricultural Loans — 15,845 records from FY1991-2024.
NAICS-11 filtered subset of SBA FOIA dataset.
No direct race/ethnicity demographics — SBA redacts per privacy policy.
Geographic and loan-size proxies used per ECOA fair lending standard.

Source: U.S. Small Business Administration FOIA 7(a) FY1991-2024
Sensitive attributes: borrstate (geographic proxy), businesstype
Target: loan_default_binary (1=Charged Off, 0=Paid In Full)
Note: 5.2% default rate — severe class imbalance.
Note: FAPE uses equalized odds not accuracy for this domain.
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sba_agricultural_loader import load_sba_agricultural

BTYPE_LABELS = {-1:'Unknown', 0:'Corporation', 1:'Individual', 2:'Partnership'}
STATE_MAP = {0: 'AK', 1: 'AL', 2: 'AR', 3: 'AZ', 4: 'CA', 5: 'CO', 6: 'CT', 7: 'DC', 8: 'DE', 9: 'FL', 10: 'GA', 11: 'GU', 12: 'HI', 13: 'IA', 14: 'ID', 15: 'IL', 16: 'IN', 17: 'KS', 18: 'KY', 19: 'LA', 20: 'MA', 21: 'MD', 22: 'ME', 23: 'MI', 24: 'MN', 25: 'MO', 26: 'MP', 27: 'MS', 28: 'MT', 29: 'NC', 30: 'ND', 31: 'NE', 32: 'NH', 33: 'NJ', 34: 'NM', 35: 'NV', 36: 'NY', 37: 'OH', 38: 'OK', 39: 'OR', 40: 'PA', 41: 'PR', 42: 'RI', 43: 'SC', 44: 'SD', 45: 'TN', 46: 'TX', 47: 'UT', 48: 'VA', 49: 'VI', 50: 'VT', 51: 'WA', 52: 'WI', 53: 'WV', 54: 'WY'}
NAICS_MAP = {16: 'Broilers and Other Meat Type Chicken Production', 60: 'Logging', 102: 'Support Activities for Animal Production', 20: 'Chicken Egg Production', 95: 'Soil Preparation, Planting, and Cultivating', 17: 'Broilers (trailing space variant — same as 16)', 93: 'Shellfish Fishing', 11: 'Beef Cattle Ranching and Farming', 6: 'All Other Miscellaneous Crop Farming', 103: 'Support Activities for Forestry', 41: 'Finfish Fishing', 64: 'Nursery and Tree Production', 55: 'Hog and Pig Farming', 79: 'Other Poultry Production', 58: 'Horses and Other Equine Production', 31: 'Dairy Cattle and Milk Production', 2: 'All Other Animal Production'}

# borrstate encoding — alphabetical order of states appearing in SBA data
# MS dominates due to poultry farming concentration in Mississippi


def run_eda():
    print("FAPE Phase 4 — SBA Agricultural Loans EDA")
    print("=" * 50)

    dataset = load_sba_agricultural()
    data = dataset["sba_agricultural"]
    df = pd.concat([data["X"], data["y"]], axis=1)
    df.columns = list(data["X"].columns) + ["label"]
    meta = data["metadata"]

    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")

    print(f"\n--- Label Distribution ---")
    label_counts = df["label"].value_counts().sort_index()
    label_pct = df["label"].value_counts(normalize=True).sort_index()
    print(f"  Paid In Full (0): {label_counts[0]:,} ({label_pct[0]:.1%})")
    print(f"  Charged Off (1):  {label_counts[1]:,} ({label_pct[1]:.1%})")
    print(f"  Note: 5.2% default rate — severe class imbalance")
    print(f"  Note: FAPE Stage 2 uses equalized odds not accuracy")

    print(f"\n--- Business Type vs Default Rate ---")
    btype_stats = df.groupby("businesstype").agg(
        count=("label","count"),
        default_rate=("label","mean")
    ).reset_index()
    for _, row in btype_stats.iterrows():
        name = BTYPE_LABELS.get(int(row["businesstype"]), str(int(row["businesstype"])))
        print(f"  {name:<15} n={int(row['count']):,} | default rate: {row['default_rate']:.1%}")
    print(f"  Note: -1 = unknown businesstype (n=34 records)")

    print(f"\n--- Geographic Distribution (Top 15 States) ---")
    state_counts = df.groupby("borrstate").agg(
        count=("label","count"),
        default_rate=("label","mean")
    ).reset_index().sort_values("count", ascending=False).head(15)
    for _, row in state_counts.iterrows():
        state_name = STATE_MAP.get(int(row["borrstate"]), str(int(row["borrstate"])))
        print(f"  {state_name:<5} n={int(row['count']):,} | default rate: {row['default_rate']:.1%}")

    print(f"\n--- Loan Approval Amount Distribution ---")
    print(f"  Mean: ${df['grossapproval'].mean():,.0f}")
    print(f"  Median: ${df['grossapproval'].median():,.0f}")
    print(f"  Min: ${df['grossapproval'].min():,.0f} | Max: ${df['grossapproval'].max():,.0f}")
    amt_corr = df["grossapproval"].corr(df["label"])
    print(f"  Loan amount-default correlation: {amt_corr:.3f}")
    for label, group in df.groupby("label"):
        print(f"  {'Default' if label==1 else 'Paid'} mean loan: ${group['grossapproval'].mean():,.0f}")

    print(f"\n--- Loan Term vs Default Rate ---")
    term_stats = df.groupby("terminmonths").agg(
        count=("label","count"),
        default_rate=("label","mean")
    ).reset_index().sort_values("count", ascending=False).head(10)
    for _, row in term_stats.iterrows():
        print(f"  {int(row['terminmonths'])} months: n={int(row['count']):,} | default rate: {row['default_rate']:.1%}")
    print(f"  Note: Standard terms show low default rates — non-standard/legacy terms drive overall 5.2% rate")

    print(f"\n--- Approval Year vs Default Rate ---")
    df["decade"] = (df["approvalfy"] // 10 * 10).astype(int)
    decade_stats = df.groupby("decade").agg(
        count=("label","count"),
        default_rate=("label","mean")
    ).reset_index()
    for _, row in decade_stats.iterrows():
        print(f"  {int(row['decade'])}s: n={int(row['count']):,} | default rate: {row['default_rate']:.1%}")
    print(f"  Note: 2000s decade highest default rate (10.9%) — financial crisis effect")


    print(f"\n--- NAICS Agricultural Subcategory vs Default Rate ---")
    naics_stats = df.groupby("naicsdescription").agg(
        count=("label","count"),
        default_rate=("label","mean")
    ).reset_index().sort_values("count", ascending=False).head(10)
    for _, row in naics_stats.iterrows():
        naics_name = NAICS_MAP.get(int(row["naicsdescription"]), f"NAICS_{int(row['naicsdescription'])}")
        print(f"  {naics_name[:40]:<40} n={int(row['count']):,} | default rate: {row['default_rate']:.1%}")

    print(f"\n--- Jobs Supported Distribution ---")
    jobs_valid = df[df["jobssupported"] >= 0]["jobssupported"]
    print(f"  Mean jobs: {jobs_valid.mean():.1f}")
    print(f"  Median jobs: {jobs_valid.median():.0f}")
    jobs_corr = df["jobssupported"].corr(df["label"])
    print(f"  Jobs-default correlation: {jobs_corr:.3f}")


    print(f"\n--- Decade x Business Type Interaction ---")
    df["btype_name"] = df["businesstype"].map({-1:"Unknown", 0:"Corporation", 1:"Individual", 2:"Partnership"})
    df["decade"] = (df["approvalfy"] // 10 * 10).astype(int)
    pivot = df.groupby(["decade","btype_name"])["label"].mean().unstack()
    print(f"  Default rate by decade x business type:")
    print(pivot.round(3).to_string())
    print(f"  Note: Individual farmers hit hardest in 2000s crisis (11.8% vs Corporation 10.8%)")
    print(f"  Note: Partnership most protected in crisis era (8.6%)")

    print(f"\n--- Feature Correlations with Label ---")
    numeric_cols = ["grossapproval","terminmonths","approvalfy",
                   "borrstate","businesstype","naicsdescription",
                   "projectstate","jobssupported"]
    correlations = df[numeric_cols+["label"]].corr()["label"].drop("label").sort_values(ascending=False)
    for feat, val in correlations.items():
        print(f"  {feat:<20} {val:.3f}")

    print(f"\n--- Missing Values ---")
    nulls = df.isnull().sum()
    if nulls.sum() == 0:
        print("  No missing values")
    else:
        print(nulls[nulls > 0])

    print(f"\n--- Key Fairness Observations ---")
    corp_rate = df[df["businesstype"]==0]["label"].mean()
    ind_rate = df[df["businesstype"]==1]["label"].mean()
    part_rate = df[df["businesstype"]==2]["label"].mean()
    state_default = df.groupby("borrstate")["label"].mean()
    print(f"  Corporation default rate:  {corp_rate:.1%}")
    print(f"  Individual default rate:   {ind_rate:.1%}")
    print(f"  Partnership default rate:  {part_rate:.1%}")
    print(f"  Geographic spread: {state_default.min():.1%} to {state_default.max():.1%}")
    print(f"  Note: No race/ethnicity data — SBA privacy policy")
    print(f"  Note: MS dominates dataset (n=3,670) — poultry farming concentration")
    print(f"  Note: LA 14.4% default rate — highest geographic rate, likely Hurricane Katrina impact on 2000s loans")
    print(f"  Note: 5.2% default rate requires careful threshold selection in Stage 2")

    print(f"\n--- SBA Agricultural EDA complete ---")
    print(f"  Total records: {len(df):,}")
    print(f"  Ready for Stage 1 preprocessing and baseline modeling")

    return df


if __name__ == "__main__":
    df = run_eda()
