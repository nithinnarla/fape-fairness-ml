"""
SBA 7(a) Agricultural Loans Loader — FAPE Phase 4
Source: U.S. Small Business Administration FOIA Dataset
Period: FY1991-2024 (4 files)
Agricultural filter: NAICS code prefix 11 (Agriculture, Forestry, Fishing)

Individual-level farm business loan records with binary outcome.
Used as FAPE's US-specific agricultural fairness evaluation dataset.

Why SBA 7(a) for agricultural fairness:
USDA individual-level farm data is protected under CIPSEA and
requires a formal research agreement. SBA 7(a) FOIA data is
publicly available and covers agricultural business loans
(NAICS-11) with loan approval, terms, and repayment outcomes.
No direct race/ethnicity demographics — socioeconomic and
geographic proxies used per ECOA fair lending standard.

Sensitive attributes: borrstate (geographic proxy), businesstype,
                     grossapproval (loan size proxy for wealth)
Target: loan_default_binary — Charged Off (1) vs Paid In Full (0)
Domain: Agriculture/Financial
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data/raw/agricultural/sba_loans'

FILES = {
    'FY1991-1999': 'sba_7a_fy1991_fy1999.csv',
    'FY2000-2009': 'sba_7a_fy2000_fy2009.csv',
    'FY2010-2019': 'sba_7a_fy2010_fy2019.csv',
    'FY2020-Present': 'sba_7a_fy2020_present.csv'
}

DEFAULT_STATUSES = {'CHGOFF'}
PAID_STATUSES = {'P I F'}


def load_sba_agricultural(data_dir: str = DATA_DIR) -> dict:
    """
    Load SBA 7(a) agricultural loans across FY1991-2024.

    Filters to NAICS-11 (Agriculture, Forestry, Fishing, Hunting)
    across all four decade files. Applies binary outcome labeling
    consistent with Lending Club loader — Charged Off = 1 (default),
    Paid In Full = 0 (repaid).

    Note on demographics:
    SBA FOIA data contains no race/ethnicity/sex fields — redacted
    per SBA privacy policy. Geographic (state) and loan-size proxies
    are used as sensitive attributes, consistent with ECOA fair
    lending audit methodology used by CFPB.

    Args:
        data_dir: Directory containing SBA CSV files

    Returns:
        Dictionary with X, y, metadata, sensitive attributes
    """
    data_path = Path(data_dir)
    print("Loading SBA 7(a) agricultural loans FY1991-2024...")

    dfs = []
    for period, filename in FILES.items():
        filepath = data_path / filename
        df = pd.read_csv(filepath, low_memory=False)
        ag_df = df[df['naicscode'].astype(str).str.startswith('11', na=False)]
        print(f"  {period}: {len(df):,} total | {len(ag_df):,} agricultural NAICS-11")
        dfs.append(ag_df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Combined agricultural records: {len(combined):,}")

    # Filter to binary outcome only
    binary_df = combined[
        combined['loanstatus'].isin(DEFAULT_STATUSES | PAID_STATUSES)
    ].copy()
    print(f"  After binary outcome filter: {len(binary_df):,}")

    # Binary target
    y = binary_df['loanstatus'].apply(
        lambda x: 1 if x in DEFAULT_STATUSES else 0
    )

    # Feature engineering
    binary_df['grossapproval'] = pd.to_numeric(
        binary_df['grossapproval'], errors='coerce'
    )
    binary_df['terminmonths'] = pd.to_numeric(
        binary_df['terminmonths'], errors='coerce'
    )
    binary_df['approvalfy'] = pd.to_numeric(
        binary_df['approvalfy'], errors='coerce'
    )

    # Encode categoricals
    for col in ['borrstate', 'businesstype', 'naicsdescription',
                'projectstate', 'deliverymethod']:
        if col in binary_df.columns:
            binary_df[col] = pd.Categorical(binary_df[col]).codes

    feature_cols = [
        'grossapproval', 'terminmonths', 'approvalfy',
        'borrstate', 'businesstype', 'naicsdescription',
        'projectstate', 'jobssupported'
    ]
    feature_cols = [c for c in feature_cols if c in binary_df.columns]
    X = binary_df[feature_cols].copy().fillna(-1)

    sensitive_attrs = ['borrstate', 'businesstype']
    positive_rate = float(y.mean())

    print(f"\n  Loan status distribution:")
    print(f"    Paid In Full: {(y==0).sum():,}")
    print(f"    Charged Off:  {(y==1).sum():,}")

    print(f"\n  Top agricultural NAICS categories:")
    print(combined['naicsdescription'].value_counts().head(5).to_string())

    print(f"\n  ✓ sba_agricultural: {len(X):,} records | "
          f"{len(feature_cols)} features | "
          f"default rate: {positive_rate:.3f}")

    metadata = {
        'name': 'sba_7a_agricultural',
        'source': 'SBA FOIA 7(a) dataset — NAICS-11 filter',
        'citation': 'U.S. Small Business Administration FOIA 7(a) FY1991-2024',
        'n_samples': len(X),
        'n_features': len(feature_cols),
        'sensitive_attrs': sensitive_attrs,
        'target': 'loan_default_binary',
        'positive_rate': positive_rate,
        'domain': 'Agriculture/Financial',
        'note': (
            'No direct race/ethnicity demographics available — '
            'SBA redacts borrower demographics per privacy policy. '
            'Geographic and loan-size proxies used per ECOA standard. '
            '21,926 total NAICS-11 records; binary outcome subset used.'
        )
    }

    return {
        'sba_agricultural': {
            'X': X,
            'y': y,
            'metadata': metadata,
            'sensitive_attrs': sensitive_attrs
        }
    }


if __name__ == '__main__':
    print("Loading SBA 7(a) agricultural loans for FAPE...")
    print("=" * 60)

    dataset = load_sba_agricultural()
    meta = dataset['sba_agricultural']['metadata']

    print(f"\nSBA Agricultural Summary:")
    print(f"  Records:      {meta['n_samples']:,}")
    print(f"  Features:     {meta['n_features']}")
    print(f"  Default rate: {meta['positive_rate']:.3f}")
    print(f"  Sensitive:    {meta['sensitive_attrs']}")
    print(f"  Domain:       {meta['domain']}")
