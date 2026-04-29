"""
Student Performance Dataset Loader — FAPE Phase 4
Cortez & Silva (2008) — UCI ML Repository
Dataset ID: 320

Two student performance datasets:
- student-mat.csv: Math course (395 students)
- student-por.csv: Portuguese language course (649 students)

Used in FAPE as the small-scale education domain dataset.
Tests whether the fairness framework holds at 649 records —
most fairness papers never evaluate at this scale.

Sensitive attributes: sex, age
Target: G3 (final grade) binarized at median
"""

import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import zipfile
import io


def load_student_performance(data_dir: str = 'data') -> dict:
    """
    Load Student Performance dataset from UCI ML Repository.

    Downloads both math and Portuguese variants. Binarizes
    the G3 final grade at the median — students above median
    are labeled 1 (passing), below median labeled 0.

    Why binarize at median rather than pass/fail threshold:
    The raw grade scale (0-20) has different distributions
    across the two subjects. Median binarization ensures
    comparable positive rates across datasets — critical
    for cross-domain fairness evaluation in FAPE Stage 3.

    Args:
        data_dir: Directory to save downloaded data

    Returns:
        Dictionary with 'math' and 'portuguese' dataset entries
    """
    Path(data_dir).mkdir(exist_ok=True)

    url = 'https://archive.ics.uci.edu/static/public/320/student+performance.zip'

    print("Downloading Student Performance dataset from UCI ML Repository...")

    try:
        with urllib.request.urlopen(url) as response:
            zip_data = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as outer:
            inner_data = outer.read('student.zip')
            with zipfile.ZipFile(io.BytesIO(inner_data)) as inner:
                with inner.open('student-mat.csv') as f:
                    df_math = pd.read_csv(f, sep=';')
                with inner.open('student-por.csv') as f:
                    df_por = pd.read_csv(f, sep=';')

        print(f"  Downloaded: student-mat ({len(df_math)} rows), "
              f"student-por ({len(df_por)} rows)")

    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Attempting alternate approach...")
        df_math, df_por = _download_alternate(data_dir)

    datasets = {}

    for name, df in [('math', df_math), ('portuguese', df_por)]:
        # Binarize G3 at median — above median = 1, at or below = 0
        median_grade = df['G3'].median()
        y = (df['G3'] > median_grade).astype(int)

        # Feature columns — exclude all three grade columns
        # G1 and G2 are intermediate grades that directly predict G3
        # Including them would make the task trivial
        feature_cols = [c for c in df.columns
                       if c not in ['G1', 'G2', 'G3']]
        X = df[feature_cols].copy()

        # Encode categorical features
        cat_cols = X.select_dtypes(include='object').columns
        for col in cat_cols:
            X[col] = pd.Categorical(X[col]).codes

        sensitive_attrs = ['sex', 'age']
        positive_rate = float(y.mean())

        datasets[name] = {
            'X': X,
            'y': y,
            'metadata': {
                'name': f'student_{name}',
                'n_samples': len(X),
                'n_features': len(feature_cols),
                'sensitive_attrs': sensitive_attrs,
                'target': 'G3_binarized_at_median',
                'positive_rate': positive_rate,
                'median_grade': median_grade,
                'note': 'G1/G2 excluded to prevent target leakage'
            },
            'sensitive_attrs': sensitive_attrs
        }

        print(f"  ✓ student_{name}: {len(X):,} rows | "
              f"{len(feature_cols)} features | "
              f"positive rate: {positive_rate:.3f} | "
              f"median grade: {median_grade}")

    return datasets


def _download_alternate(data_dir: str) -> tuple:
    """
    Alternate download via direct file URLs if zip fails.
    """
    base = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00320/'
    df_math = pd.read_csv(base + 'student-mat.csv', sep=';')
    df_por = pd.read_csv(base + 'student-por.csv', sep=';')
    return df_math, df_por


if __name__ == '__main__':
    print("Loading Student Performance datasets for FAPE education domain...")
    print("=" * 60)

    datasets = load_student_performance()

    print(f"\nStudent Performance Summary:")
    for name, content in datasets.items():
        meta = content['metadata']
        print(f"  {meta['name']}: {meta['n_samples']} rows | "
              f"{meta['n_features']} features | "
              f"positive rate: {meta['positive_rate']:.3f}")

    print(f"\nTotal records: "
          f"{sum(d['metadata']['n_samples'] for d in datasets.values())}")
    print("Note: student_language (649 rows) is FAPE's small-scale "
          "education benchmark")
