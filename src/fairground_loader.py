"""
FairGround Corpus Loader, FAPE Phase 4
Fabris et al. (2025), Bias Begins with Data: The FairGround Corpus
arXiv: 2510.22363
Package: fairml-datasets (github.com/reliable-ai/fairground)

Loads fairness-annotated tabular datasets with standardized
preprocessing for FAPE's cross-domain generalization evaluation.
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


def load_fairground_corpus(
    include_large: bool = False,
    max_datasets: Optional[int] = None
) -> dict:
    """
    Load FairGround Corpus datasets via fairml_datasets package.

    Uses dataset.load() and dataset.to_pandas() to retrieve each
    of the 38 fairness-annotated tabular datasets. Sensitive
    attribute metadata is pre-validated by FairGround, eliminates
    the arbitrary preprocessing choices Fabris et al. (2025)
    identified as the core reproducibility problem in fair ML.

    Args:
        include_large: Whether to include large datasets (>100K rows)
        max_datasets: Cap number of datasets (None = load all)

    Returns:
        Dictionary of {dataset_id: {X, y, sensitive_attrs, metadata}}
    """
    from fairml_datasets import Datasets

    datasets = Datasets(include_large_datasets=include_large)
    print(f"FairGround corpus: {len(datasets)} datasets available")

    corpus = {}
    failed = []
    total_records = 0
    count = 0

    for dataset in datasets:
        if max_datasets and count >= max_datasets:
            break

        dataset_id = getattr(dataset, 'dataset_id', f'dataset_{count}')

        try:
            # Load raw data
            dataset.load()

            # Convert to pandas
            df = dataset.to_pandas()

            # Get target column
            target_col = dataset.get_target_column()

            # Get sensitive columns
            sensitive_cols = dataset.sensitive_columns \
                if hasattr(dataset, 'sensitive_columns') else []

            # Split features and target
            feature_cols = [c for c in df.columns if c != target_col]
            X = df[feature_cols]
            y = df[target_col]

            n_samples = len(df)
            n_features = len(feature_cols)
            positive_rate = float(y.mean()) if y.dtype in [int, float, bool] \
                else None

            metadata = {
                'id': dataset_id,
                'name': getattr(dataset, 'name', dataset_id),
                'n_samples': n_samples,
                'n_features': n_features,
                'sensitive_columns': sensitive_cols,
                'target_column': target_col,
                'positive_rate': positive_rate,
            }

            corpus[dataset_id] = {
                'X': X,
                'y': y,
                'df': df,
                'metadata': metadata,
                'sensitive_cols': sensitive_cols
            }

            total_records += n_samples
            count += 1
            print(f"  OK: {dataset_id}: {n_samples:,} rows | "
                  f"{n_features} features | "
                  f"sensitive: {sensitive_cols}")

        except Exception as e:
            failed.append((dataset_id, str(e)))
            print(f"  FAIL: {dataset_id}: {str(e)[:80]}")
            count += 1

    print(f"\nFairGround load complete:")
    print(f"  Loaded:  {len(corpus)} datasets")
    print(f"  Failed:  {len(failed)} datasets")
    print(f"  Records: {total_records:,} total")

    if failed:
        print(f"\nFailed datasets:")
        for name, err in failed:
            print(f"  {name}: {err[:80]}")

    return corpus


def get_fairground_summary(corpus: dict) -> pd.DataFrame:
    """
    Build summary DataFrame of loaded FairGround datasets.

    Maps each dataset to size, features, and sensitive attributes
    for systematic cross-domain fairness evaluation in FAPE Stage 3.
    """
    rows = []
    for name, content in corpus.items():
        meta = content['metadata']
        rows.append({
            'dataset': meta.get('name', name),
            'dataset_id': name,
            'n_samples': meta['n_samples'],
            'n_features': meta['n_features'],
            'sensitive_columns': ', '.join(
                str(c) for c in meta['sensitive_columns']
            ),
            'positive_rate': round(meta['positive_rate'], 3)
            if meta['positive_rate'] is not None else None
        })

    return pd.DataFrame(rows).sort_values('n_samples', ascending=False)


if __name__ == '__main__':
    print("Loading FairGround Corpus for FAPE cross-domain evaluation...")
    print("=" * 60)

    corpus = load_fairground_corpus(include_large=False)

    if corpus:
        summary = get_fairground_summary(corpus)
        print("\nFairGround Corpus Summary:")
        print(summary.to_string(index=False))

        summary.to_csv('data/fairground_summary.csv', index=False)
        print("\nSummary saved to data/fairground_summary.csv")
        print(f"Total records: {summary['n_samples'].sum():,}")
        print(f"Total datasets: {len(summary)}")
