"""
LSMS-ISA Nigeria GHS-Panel Wave 4 Loader — FAPE Phase 4
Source: World Bank Living Standards Measurement Study (LSMS-ISA)
Survey: Nigeria General Household Survey Panel 2018-2019, Wave 4
Reference: Azzarri et al. (2025) Scientific Data, Nature

Individual-level agricultural household survey with demographic
attributes and farm outcome variables. Used as FAPE's primary
individual-level agricultural fairness evaluation dataset.

Why LSMS-ISA for agricultural fairness:
No publicly available US individual-level agricultural dataset
with demographic attributes exists — USDA NASS and ARMS are
aggregate only due to CIPSEA confidentiality protections.
LSMS-ISA provides the only large-scale publicly downloadable
individual-level agricultural dataset with sex, age, and
education demographics suitable for ML fairness evaluation.

Sensitive attributes: sex (s1q2), education (s1q7)
Target: food security / consumption adequacy (binary)
        derived from totcons_final.csv household consumption
Domain: Agriculture (Nigeria 2018-2019)
Records: 30,337 individuals across 4,980 households
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data/raw/agricultural/lsms/NGA_2018_GHSP-W4_v03_M_CSV'


def load_lsms_nigeria(data_dir: str = DATA_DIR) -> dict:
    """
    Load Nigeria GHS-Panel Wave 4 for FAPE agricultural fairness evaluation.

    Merges household roster (demographics) with consumption outcome
    to create an individual-level dataset suitable for fairness
    metric computation. Binary target: household above/below
    median per-capita consumption — proxy for food security.

    Why median consumption as target:
    Direct farm outcome variables (crop yields, sales) are at
    plot level and require complex aggregation. Per-capita
    consumption is the standard welfare outcome in agricultural
    development economics and directly reflects farm productivity
    and household food security — the core outcome AIFARMS and
    agricultural AI advisory systems affect.

    Args:
        data_dir: Path to NGA_2018_GHSP-W4_v03_M_CSV directory

    Returns:
        Dictionary with X, y, metadata, sensitive attributes
    """
    data_path = Path(data_dir)
    print(f"Loading LSMS-ISA Nigeria GHS-Panel Wave 4...")

    # Load household roster — individual demographics
    roster = pd.read_csv(data_path / 'sect1_harvestw4.csv', low_memory=False)
    print(f"  Household roster: {len(roster):,} individuals")

    # Load consumption outcome — household welfare
    consumption = pd.read_csv(data_path / 'totcons_final.csv', low_memory=False)
    print(f"  Consumption data: {len(consumption):,} households")

    # Clean roster demographics
    roster_clean = roster[[
        'hhid', 'indiv', 'zone', 'state', 'sector',
        's1q2',   # sex: 1=male, 2=female
        's1q4',   # age
        's1q7',   # education level
        's1q27',  # occupation
        's1q19',  # marital status
    ]].copy()

    roster_clean.columns = [
        'hhid', 'individual_id', 'zone', 'state', 'sector',
        'sex', 'age', 'education', 'occupation', 'marital_status'
    ]

    # Merge with household consumption
    df = roster_clean.merge(
        consumption[['hhid', 'wt_wave4', 'hhsize', 'food_own1']],
        on='hhid',
        how='inner'
    )

    print(f"  After merge: {len(df):,} individuals with consumption data")

    # Compute per-capita consumption proxy
    df['food_pc'] = df['food_own1'] / df['hhsize']

    # Binary target — above median per-capita food consumption = 1
    median_food = df['food_pc'].median()
    y = (df['food_pc'] > median_food).astype(int)

    # Feature columns
    feature_cols = ['zone', 'state', 'sector', 'age',
                   'education', 'occupation', 'marital_status', 'hhsize']
    X = df[feature_cols].copy()

    # Encode categoricals
    for col in X.select_dtypes(include='object').columns:
        X[col] = pd.Categorical(X[col]).codes
    X = X.fillna(-1)

    # Sensitive attributes
    sensitive_attrs = ['sex', 'education']

    positive_rate = float(y.mean())

    # Demographic breakdown
    print(f"\n  Demographic breakdown:")
    sex_map = {1: 'Male', 2: 'Female'}
    for sex_val, sex_name in sex_map.items():
        count = (df['sex'] == sex_val).sum()
        print(f"    {sex_name}: {count:,} ({count/len(df)*100:.1f}%)")

    edu_counts = df['education'].value_counts().head(5)
    print(f"  Education levels (top 5): {edu_counts.to_dict()}")

    print(f"\n  ✓ lsms_nigeria_wave4: {len(X):,} individuals | "
          f"{len(feature_cols)} features | "
          f"positive rate: {positive_rate:.3f} | "
          f"sensitive: {sensitive_attrs}")

    metadata = {
        'name': 'lsms_nigeria_ghsp_wave4',
        'source': 'World Bank LSMS-ISA',
        'citation': 'NGA GHS-Panel Wave 4 2018-2019; Azzarri et al. (2025) Scientific Data',
        'n_samples': len(X),
        'n_features': len(feature_cols),
        'sensitive_attrs': sensitive_attrs,
        'target': 'above_median_per_capita_food_consumption',
        'positive_rate': positive_rate,
        'domain': 'Agriculture (Nigeria)',
        'note': (
            'Individual-level agricultural household survey. '
            'No publicly available US individual-level agricultural '
            'fairness dataset exists due to CIPSEA confidentiality — '
            'LSMS-ISA provides the only large-scale alternative. '
            'Target: household food security proxy via per-capita consumption.'
        )
    }

    return {
        'lsms_nigeria': {
            'X': X,
            'y': y,
            'df': df,
            'metadata': metadata,
            'sensitive_attrs': sensitive_attrs
        }
    }


if __name__ == '__main__':
    print("Loading LSMS-ISA Nigeria for FAPE agricultural domain...")
    print("=" * 60)

    dataset = load_lsms_nigeria()
    meta = dataset['lsms_nigeria']['metadata']

    print(f"\nLSMS-ISA Summary:")
    print(f"  Records:    {meta['n_samples']:,}")
    print(f"  Features:   {meta['n_features']}")
    print(f"  Sensitive:  {meta['sensitive_attrs']}")
    print(f"  Target:     {meta['target']}")
    print(f"  Positive rate: {meta['positive_rate']:.3f}")
    print(f"  Citation:   {meta['citation']}")
