"""
FairGround Corpus Loader — FAPE Phase 4
Fabris et al. (2025) — Bias Begins with Data: The FairGround Corpus
arXiv: 2510.22363

Loads 44 fairness-annotated tabular datasets with standardized
preprocessing. Used as multi-domain validation corpus for FAPE's
cross-domain generalization evaluation.
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


def install_fairground():
    """Install fairground package if not present."""
    import subprocess
    import sys
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install',
        'fairground', '--break-system-packages', '-q'
    ])


def load_fairground_corpus(
    max_datasets: Optional[int] = None,
    target_domains: Optional[list] = None
) -> dict:
    """
    Load FairGround Corpus datasets with fairness metadata.

    The FairGround package provides standardized loading for 44
    tabular fairness datasets. We load all available datasets
    that align with FAPE's tabular classification scope —
    criminal justice, healthcare, education, financial services,
    and socioeconomic domains.

    Why FairGround over individual dataset loading:
    - Standardized preprocessing removes arbitrary data cleaning choices
    - Fairness-relevant metadata (sensitive attributes, protected groups)
      is pre-annotated and validated
    - Addresses reproducibility gap identified in Fabris et al. (2025) —
      ad hoc dataset selection is a known source of fairness research bias

    Args:
        max_datasets: Cap number of datasets loaded (None = load all)
        target_domains: Filter to specific domains if needed

    Returns:
        Dictionary of {dataset_name: {data, metadata, sensitive_attrs}}
    """
    try:
        from fairground import load_dataset, list_datasets
    except ImportError:
        print("Installing fairground package...")
        install_fairground()
        from fairground import load_dataset, list_datasets

    available = list_datasets()
    print(f"FairGround corpus: {len(available)} datasets available")

    if max_datasets:
        available = available[:max_datasets]

    corpus = {}
    failed = []
    total_records = 0

    for dataset_name in available:
        try:
            dataset = load_dataset(dataset_name)

            # Extract core components
            X = dataset.X if hasattr(dataset, 'X') else dataset.data
            y = dataset.y if hasattr(dataset, 'y') else dataset.target

            # Get sensitive attributes from FairGround metadata
            sensitive_attrs = []
            if hasattr(dataset, 'sensitive_attribute'):
                sensitive_attrs = [dataset.sensitive_attribute]
            elif hasattr(dataset, 'sensitive_attributes'):
                sensitive_attrs = dataset.sensitive_attributes

            # Build metadata record
            metadata = {
                'name': dataset_name,
                'n_samples': len(X),
                'n_features': X.shape[1] if hasattr(X, 'shape') else len(X.columns),
                'sensitive_attributes': sensitive_attrs,
                'domain': getattr(dataset, 'domain', 'unspecified'),
                'positive_rate': float(np.mean(y)) if y is not None else None,
            }

            corpus[dataset_name] = {
                'X': X,
                'y': y,
                'metadata': metadata,
                'sensitive_attrs': sensitive_attrs
            }

            total_records += metadata['n_samples']
            print(f"  ✓ {dataset_name}: {metadata['n_samples']:,} records, "
                  f"sensitive attrs: {sensitive_attrs}")

        except Exception as e:
            failed.append((dataset_name, str(e)))
            print(f"  ✗ {dataset_name}: {str(e)[:60]}")

    print(f"\nFairGround load complete:")
    print(f"  Loaded:  {len(corpus)} datasets")
    print(f"  Failed:  {len(failed)} datasets")
    print(f"  Records: {total_records:,} total")

    if failed:
        print(f"\nFailed datasets (will investigate individually):")
        for name, err in failed:
            print(f"  {name}: {err[:80]}")

    return corpus


def get_fairground_summary(corpus: dict) -> pd.DataFrame:
    """
    Build summary DataFrame of loaded FairGround datasets.

    Provides the metadata table needed for FAPE's cross-domain
    evaluation — maps each dataset to its domain, size, and
    sensitive attributes for systematic fairness evaluation.
    """
    rows = []
    for name, content in corpus.items():
        meta = content['metadata']
        rows.append({
            'dataset': name,
            'domain': meta.get('domain', 'unspecified'),
            'n_samples': meta['n_samples'],
            'n_features': meta['n_features'],
            'sensitive_attributes': ', '.join(meta['sensitive_attributes']),
            'positive_rate': round(meta['positive_rate'], 3)
            if meta['positive_rate'] else None
        })

    summary = pd.DataFrame(rows).sort_values('n_samples', ascending=False)
    return summary


def filter_by_domain(corpus: dict, domains: list) -> dict:
    """
    Filter FairGround corpus to target domains.

    FAPE evaluates across criminal justice, healthcare, education,
    and financial services. This filter isolates datasets from
    those domains for domain-specific fairness analysis.
    """
    filtered = {
        name: content
        for name, content in corpus.items()
        if content['metadata'].get('domain', '').lower() in
        [d.lower() for d in domains]
    }
    print(f"Filtered to {len(filtered)} datasets in domains: {domains}")
    return filtered


if __name__ == '__main__':
    print("Loading FairGround Corpus for FAPE cross-domain evaluation...")
    print("=" * 60)

    corpus = load_fairground_corpus()

    if corpus:
        summary = get_fairground_summary(corpus)
        print("\nFairGround Corpus Summary:")
        print(summary.to_string(index=False))

        # Save summary for documentation
        summary.to_csv('data/fairground_summary.csv', index=False)
        print("\nSummary saved to data/fairground_summary.csv")

        # Domain breakdown
        print("\nDomain distribution:")
        domain_counts = summary.groupby('domain').agg(
            datasets=('dataset', 'count'),
            total_records=('n_samples', 'sum')
        ).sort_values('total_records', ascending=False)
        print(domain_counts.to_string())
