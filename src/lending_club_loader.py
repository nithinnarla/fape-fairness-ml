"""
Lending Club Loan Data Loader, FAPE Phase 4
Source: Kaggle, wordsforthewise/lending-club
Period: 2007-2018 Q4
Source: 2.26M accepted loan applications (full Kaggle dataset), FAPE verified subset: 1,348,099 records after filtering

Lending Club does not collect race or gender data, a documented
limitation of financial services ML fairness research. FAPE uses
socioeconomic proxies consistent with the fairness literature:
- addr_state: geographic proxy for demographic composition
- annual_inc: income as socioeconomic sensitive attribute
- emp_length: employment stability proxy
- home_ownership: housing status proxy

Target: loan_status binarized, Fully Paid (0) vs Charged Off/Default (1)
This follows the standard approach in Kozodoi et al. (2022) and
Verma & Rubin (2018) for financial services fairness evaluation.

Why include despite no direct demographics:
Documented geographic and income-based disparities in loan approval
and default prediction make this a canonical financial fairness dataset.
ECOA (Equal Credit Opportunity Act) prohibits discrimination on
protected characteristics, proxy-based auditing is the field standard.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = (
    'data/raw/lending_club/accepted_2007_to_2018q4.csv/'
    'accepted_2007_to_2018Q4.csv'
)

# Target loan statuses, binary classification
# 1 = default/loss, 0 = fully paid
DEFAULT_STATUSES = {'Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off'}
PAID_STATUSES = {'Fully Paid', 'Does not meet the credit policy. Status:Fully Paid'}

# Socioeconomic proxy sensitive attributes
SENSITIVE_ATTRS = ['annual_inc_band', 'emp_length', 'home_ownership', 'addr_state']

# Core feature columns, exclude direct loan outcome leakage
FEATURE_COLS = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade',
    'sub_grade', 'emp_length', 'home_ownership', 'annual_inc',
    'verification_status', 'purpose', 'addr_state', 'dti',
    'delinq_2yrs', 'fico_range_low', 'fico_range_high',
    'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal',
    'revol_util', 'total_acc'
]


def load_lending_club(
    data_path: str = DATA_PATH,
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> dict:
    """
    Load Lending Club loan data for FAPE financial domain evaluation.

    Applies standard preprocessing from the fairness literature:
    - Filter to binary outcome loans only (Fully Paid vs Charged Off)
    - Binarize loan_status as target variable
    - Engineer annual_inc_band as income-based sensitive attribute
    - Encode categorical features for ML pipeline

    Why sample_size parameter:
    At 2.26M records, full dataset training is computationally expensive.
    Default loads all records, Phase 4 EDA uses 500K sample for speed.
    Final evaluation uses full dataset for reported results.

    Args:
        data_path: Path to Lending Club CSV file
        sample_size: Optional row limit for EDA and development
        random_state: Reproducibility seed

    Returns:
        Dictionary with X, y, metadata, sensitive attributes
    """
    print(f"Loading Lending Club data from {data_path}...")

    # Load with dtype optimization for memory efficiency
    df = pd.read_csv(
        data_path,
        low_memory=False,
        usecols=FEATURE_COLS + ['loan_status']
    )

    print(f"  Raw records: {len(df):,}")

    # Filter to binary outcome only
    df = df[df['loan_status'].isin(DEFAULT_STATUSES | PAID_STATUSES)]
    print(f"  After binary outcome filter: {len(df):,}")

    # Binarize target, 1 = default, 0 = fully paid
    y = df['loan_status'].apply(
        lambda x: 1 if x in DEFAULT_STATUSES else 0
    )

    # Engineer income band as sensitive attribute
    df['annual_inc_band'] = pd.qcut(
        df['annual_inc'].clip(upper=500000),
        q=4,
        labels=['low', 'lower_mid', 'upper_mid', 'high']
    )

    # Clean numeric columns
    df['int_rate'] = pd.to_numeric(
        df['int_rate'].astype(str).str.replace('%', ''), errors='coerce'
    )
    df['revol_util'] = pd.to_numeric(
        df['revol_util'].astype(str).str.replace('%', ''), errors='coerce'
    )
    df['term'] = pd.to_numeric(
        df['term'].astype(str).str.extract(r'(\d+)')[0], errors='coerce'
    )
    df['emp_length'] = pd.to_numeric(
        df['emp_length'].astype(str).str.extract(r'(\d+)')[0], errors='coerce'
    )

    # Encode categorical features
    cat_cols = ['grade', 'sub_grade', 'home_ownership',
                'verification_status', 'purpose',
                'addr_state', 'annual_inc_band']
    for col in cat_cols:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes

    # Build feature matrix
    X = df[FEATURE_COLS + ['annual_inc_band']].copy()
    X = X.fillna(X.median(numeric_only=True))

    # Optional sample for development
    if sample_size and len(X) > sample_size:
        idx = X.sample(n=sample_size, random_state=random_state).index
        X = X.loc[idx]
        y = y.loc[idx]
        print(f"  Sampled to: {len(X):,} records")

    positive_rate = float(y.mean())

    metadata = {
        'name': 'lending_club',
        'source': 'Kaggle, wordsforthewise/lending-club',
        'citation': 'Lending Club 2007-2018 Q4',
        'n_samples': len(X),
        'n_features': len(X.columns),
        'sensitive_attrs': SENSITIVE_ATTRS,
        'target': 'loan_default_binary',
        'positive_rate': positive_rate,
        'domain': 'Financial Services',
        'note': (
            'No direct race/gender data available, socioeconomic proxies '
            'used per ECOA fairness auditing standard. Follows Kozodoi et al. '
            '(2022) preprocessing for financial fairness evaluation.'
        )
    }

    print(f"  OK: lending_club: {len(X):,} rows | "
          f"{len(X.columns)} features | "
          f"default rate: {positive_rate:.3f}")

    return {
        'lending_club': {
            'X': X,
            'y': y,
            'metadata': metadata,
            'sensitive_attrs': SENSITIVE_ATTRS
        }
    }


if __name__ == '__main__':
    print("Loading Lending Club for FAPE financial domain evaluation...")
    print("=" * 60)

    dataset = load_lending_club()
    meta = dataset['lending_club']['metadata']

    print(f"\nLending Club Summary:")
    print(f"  Records:        {meta['n_samples']:,}")
    print(f"  Features:       {meta['n_features']}")
    print(f"  Default rate:   {meta['positive_rate']:.3f}")
    print(f"  Sensitive:      {meta['sensitive_attrs']}")
    print(f"  Domain:         {meta['domain']}")
    print(f"\nNote: {meta['note']}")
