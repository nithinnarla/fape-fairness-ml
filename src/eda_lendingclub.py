"""
FAPE — Lending Club EDA
Phase 4 — Exploratory Data Analysis
Financial Services Domain

EDA on Lending Club loan data — 500,000 sample from 1,348,099 records
(2007-2018 Q4). No direct race/gender data available — ECOA compliance
means financial institutions cannot collect protected characteristics.
Socioeconomic proxies used per fairness literature standard.

Source: Kaggle — wordsforthewise/lending-club
Citation: Lending Club 2007-2018 Q4
Sensitive attributes: annual_inc_band, emp_length, home_ownership, addr_state
Target: loan_default_binary (1=default, 0=fully paid)
Note: Follows Kozodoi et al. (2022) preprocessing for financial fairness.
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from lending_club_loader import load_lending_club

GRADE_LABELS = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F", 6:"G"}
PURPOSE_LABELS = {0:"car", 1:"credit_card", 2:"debt_consolidation",
                  3:"educational", 4:"home_improvement", 5:"house",
                  6:"major_purchase", 7:"medical", 8:"moving",
                  9:"other", 10:"renewable_energy", 11:"small_business",
                  12:"vacation", 13:"wedding"}
HOME_LABELS = {0:"ANY", 1:"MORTGAGE", 2:"NONE", 3:"OTHER", 4:"OWN", 5:"RENT"}
STATE_MAP = {0:'AK',1:'AL',2:'AR',3:'AZ',4:'CA',5:'CO',6:'CT',7:'DC',8:'DE',
9:'FL',10:'GA',11:'HI',12:'ID',13:'IL',14:'IN',15:'KS',16:'KY',17:'LA',
18:'MA',19:'MD',20:'ME',21:'MI',22:'MN',23:'MO',24:'MS',25:'MT',26:'NC',
27:'ND',28:'NE',29:'NH',30:'NJ',31:'NM',32:'NV',33:'NY',34:'OH',35:'OK',
36:'OR',37:'PA',38:'RI',39:'SC',40:'SD',41:'TN',42:'TX',43:'UT',44:'VA',
45:'VT',46:'WA',47:'WI',48:'WV',49:'WY'}
INC_LABELS = {-1:"Unknown", 0:"Low", 1:"Lower-Mid", 2:"Upper-Mid", 3:"High"}


def run_eda():
    print("FAPE Phase 4 — Lending Club EDA")
    print("=" * 50)

    dataset = load_lending_club(sample_size=500000)
    data = dataset["lending_club"]
    df = pd.concat([data["X"], data["y"]], axis=1)
    df.columns = list(data["X"].columns) + ["label"]
    meta = data["metadata"]

    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")

    print(f"\n--- Label Distribution ---")
    label_counts = df["label"].value_counts().sort_index()
    label_pct = df["label"].value_counts(normalize=True).sort_index()
    print(f"  Fully Paid (0):  {label_counts[0]:,} ({label_pct[0]:.1%})")
    print(f"  Default (1):     {label_counts[1]:,} ({label_pct[1]:.1%})")
    print(f"  Note: 20% default rate — moderately imbalanced")

    print(f"\n--- Loan Grade vs Default Rate ---")
    grade_stats = df.groupby("grade").agg(
        count=("label","count"),
        default_rate=("label","mean")
    ).reset_index()
    for _, row in grade_stats.iterrows():
        grade_name = GRADE_LABELS.get(int(row["grade"]), str(int(row["grade"])))
        print(f"  Grade {grade_name}: n={int(row['count']):,} | default rate: {row['default_rate']:.1%}")
    print(f"  Note: Grade is strongest predictor — monotonic A→G default rate increase")

    print(f"\n--- Home Ownership vs Default Rate ---")
    home_stats = df.groupby("home_ownership").agg(
        count=("label","count"),
        default_rate=("label","mean")
    ).reset_index()
    for _, row in home_stats.iterrows():
        home_name = HOME_LABELS.get(int(row["home_ownership"]), str(int(row["home_ownership"])))
        print(f"  {home_name:<12} n={int(row['count']):,} | default rate: {row['default_rate']:.1%}")

    print(f"  Note: ANY (n~123), NONE (n~21), OTHER (n~54) — small samples, interpret with caution")
    print(f"\n--- Employment Length vs Default Rate ---")
    emp_stats = df.groupby("emp_length").agg(
        count=("label","count"),
        default_rate=("label","mean")
    ).reset_index().dropna()
    for _, row in emp_stats.iterrows():
        print(f"  {int(row['emp_length']):>2} years: n={int(row['count']):,} | default rate: {row['default_rate']:.1%}")

    print(f"  Note: < 1 year emp_length dropped as NaN during numeric extraction — see loader")
    print(f"\n--- Income Band vs Default Rate ---")
    inc_stats = df.groupby("annual_inc_band").agg(
        count=("label","count"),
        default_rate=("label","mean")
    ).reset_index()
    for _, row in inc_stats.iterrows():
        inc_name = INC_LABELS.get(int(row["annual_inc_band"]), str(int(row["annual_inc_band"])))
        print(f"  {inc_name:<12} n={int(row['count']):,} | default rate: {row['default_rate']:.1%}")
    inc_corr = df["annual_inc_band"].corr(df["label"])
    print(f"  Income band-default correlation: {inc_corr:.3f}")

    print(f"\n--- Loan Amount Distribution ---")
    print(f"  Mean: ${df['loan_amnt'].mean():,.0f}")
    print(f"  Median: ${df['loan_amnt'].median():,.0f}")
    print(f"  Min: ${df['loan_amnt'].min():,.0f} | Max: ${df['loan_amnt'].max():,.0f}")
    amt_corr = df["loan_amnt"].corr(df["label"])
    print(f"  Loan amount-default correlation: {amt_corr:.3f}")

    print(f"\n--- Interest Rate vs Default Rate ---")
    print(f"  Mean: {df['int_rate'].mean():.2f}%")
    int_corr = df["int_rate"].corr(df["label"])
    print(f"  Interest rate-default correlation: {int_corr:.3f}")
    for label, group in df.groupby("label"):
        print(f"  {'Default' if label==1 else 'Paid'} mean int rate: {group['int_rate'].mean():.2f}%")

    print(f"\n--- DTI vs Default Rate ---")
    print(f"  Mean DTI: {df['dti'].mean():.2f}")
    dti_corr = df["dti"].corr(df["label"])
    print(f"  DTI-default correlation: {dti_corr:.3f}")

    print(f"\n--- Loan Purpose vs Default Rate ---")
    purpose_stats = df.groupby("purpose").agg(
        count=("label","count"),
        default_rate=("label","mean")
    ).reset_index().sort_values("count", ascending=False).head(10)
    for _, row in purpose_stats.iterrows():
        purpose_name = PURPOSE_LABELS.get(int(row["purpose"]), str(int(row["purpose"])))
        print(f"  {purpose_name:<22} n={int(row['count']):,} | default rate: {row['default_rate']:.1%}")

    print(f"\n--- FICO Score vs Default Rate ---")
    print(f"  FICO low mean: {df['fico_range_low'].mean():.1f}")
    print(f"  FICO high mean: {df['fico_range_high'].mean():.1f}")
    fico_corr = df["fico_range_low"].corr(df["label"])
    print(f"  FICO low-default correlation: {fico_corr:.3f}")
    for label, group in df.groupby("label"):
        print(f"  {'Default' if label==1 else 'Paid'} mean FICO: {group['fico_range_low'].mean():.1f}")



    print(f"\n--- Geographic Proxy (State) vs Default Rate ---")
    state_stats = df.groupby("addr_state").agg(
        count=("label","count"),
        default_rate=("label","mean")
    ).reset_index()
    state_stats = state_stats[state_stats["count"] >= 100]
    top5_high = state_stats.nlargest(5, "default_rate")
    top5_low = state_stats.nsmallest(5, "default_rate")
    print(f"  Total states with n>=100: {len(state_stats)}")
    print(f"  Default rate range: {state_stats['default_rate'].min():.1%} to {state_stats['default_rate'].max():.1%}")
    print(f"  Geographic spread: {state_stats['default_rate'].max()-state_stats['default_rate'].min():.1%}")
    print(f"  Top 5 highest default states:")
    for _, row in top5_high.iterrows():
        print(f"    {STATE_MAP.get(int(row['addr_state']),str(int(row['addr_state'])))}: {row['default_rate']:.1%} (n={int(row['count']):,})")
    print(f"  Top 5 lowest default states:")
    for _, row in top5_low.iterrows():
        print(f"    {STATE_MAP.get(int(row['addr_state']),str(int(row['addr_state'])))}: {row['default_rate']:.1%} (n={int(row['count']):,})")
    print(f"  Note: addr_state encodes geographic proxy for redlining patterns")

    print(f"\n--- Feature Correlations with Label ---")
    numeric_cols = ["loan_amnt","int_rate","installment","emp_length",
                   "annual_inc_band","dti","delinq_2yrs","fico_range_low",
                   "inq_last_6mths","open_acc","pub_rec","revol_bal",
                   "revol_util","total_acc","home_ownership","grade"]
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
    low_inc = df[df["annual_inc_band"]==0]["label"].mean()
    high_inc = df[df["annual_inc_band"]==3]["label"].mean()
    rent_rate = df[df["home_ownership"]==5]["label"].mean()
    mortgage_rate = df[df["home_ownership"]==1]["label"].mean()
    print(f"  Low income default rate:      {low_inc:.1%}")
    print(f"  High income default rate:     {high_inc:.1%}")
    print(f"  Income gap:                   {abs(low_inc-high_inc):.1%}")
    print(f"  Renters default rate:         {rent_rate:.1%}")
    print(f"  Mortgage holders default:     {mortgage_rate:.1%}")
    print(f"  Housing gap:                  {abs(rent_rate-mortgage_rate):.1%}")
    print(f"  Note: No direct race/gender data — ECOA compliance")
    print(f"  Note: Socioeconomic proxies may encode racial disparities")
    print(f"  Note: Geographic proxy (addr_state) captures redlining patterns")

    print(f"\n--- Lending Club EDA complete ---")
    print(f"  Sample records: {len(df):,} (from 1,348,099 total)")
    print(f"  Ready for Stage 1 preprocessing and baseline modeling")

    return df


if __name__ == "__main__":
    df = run_eda()
