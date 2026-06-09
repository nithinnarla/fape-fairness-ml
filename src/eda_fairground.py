"""
FAPE — FairGround Corpus EDA
Phase 4 — Exploratory Data Analysis
Multi-Domain Benchmark

EDA on FairGround corpus — 37 fairness-annotated datasets
totaling 1,964,010 records across criminal justice, financial,
healthcare, education, and socioeconomic domains.

FairGround addresses the benchmark monoculture problem identified
by Fabris et al. (2025) — pre-validated sensitive attribute
metadata eliminates arbitrary preprocessing choices.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from fairground_loader import load_fairground_corpus


def run_eda():
    print("FAPE Phase 4 — FairGround Corpus EDA")
    print("=" * 50)

    corpus = load_fairground_corpus()

    print(f"\n--- Corpus Overview ---")
    print(f"  Datasets loaded: {len(corpus)}")
    total_records = sum(corpus[k]['metadata']['n_samples'] for k in corpus)
    print(f"  Total records:   {total_records:,}")

    print(f"\n--- Dataset Summary ---")
    print(f"  {'Dataset':<40} {'Records':>10} {'Features':>10} Sensitive Attrs")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*30}")

    sizes = []
    for dataset_id, data in corpus.items():
        meta = data['metadata']
        n = meta['n_samples']
        f = meta['n_features']
        sens = meta['sensitive_columns']
        sizes.append((dataset_id, n, f, sens))
        print(f"  {dataset_id:<40} {n:>10,} {f:>10} {sens}")

    print(f"\n--- Size Distribution ---")
    sizes_sorted = sorted(sizes, key=lambda x: x[1], reverse=True)
    print(f"  Largest:  {sizes_sorted[0][0]} — {sizes_sorted[0][1]:,} records")
    print(f"  Smallest: {sizes_sorted[-1][0]} — {sizes_sorted[-1][1]:,} records")
    print(f"  Median:   {sorted([s[1] for s in sizes])[len(sizes)//2]:,} records")

    print(f"\n--- Sensitive Attribute Coverage ---")
    all_attrs = []
    for _, data in corpus.items():
        all_attrs.extend(data['metadata']['sensitive_columns'])
    attr_counts = pd.Series(all_attrs).value_counts()
    print(f"  Most common sensitive attributes:")
    for attr, count in attr_counts.head(10).items():
        print(f"    {attr:<30} appears in {count} datasets")

    print(f"\n--- Domain Coverage ---")
    criminal = [k for k in corpus if any(x in k for x in ['compas', 'stop_question', 'chicago', 'ricci'])]
    financial = [k for k in corpus if any(x in k for x in ['credit', 'bank'])]
    healthcare = [k for k in corpus if any(x in k for x in ['meps', 'heart', 'arrhythmia'])]
    education = [k for k in corpus if any(x in k for x in ['student', 'law_school', 'nursery'])]
    socioeconomic = [k for k in corpus if any(x in k for x in ['adult', 'folktables', 'dutch', 'communities'])]
    other = [k for k in corpus if k not in criminal + financial + healthcare + education + socioeconomic]

    print(f"  Criminal justice:  {len(criminal)} datasets")
    for d in criminal:
        print(f"    - {d}")
    print(f"  Financial:         {len(financial)} datasets")
    for d in financial:
        print(f"    - {d}")
    print(f"  Healthcare:        {len(healthcare)} datasets")
    for d in healthcare:
        print(f"    - {d}")
    print(f"  Education:         {len(education)} datasets")
    for d in education:
        print(f"    - {d}")
    print(f"  Socioeconomic:     {len(socioeconomic)} datasets")
    for d in socioeconomic:
        print(f"    - {d}")
    print(f"  Other/synthetic:   {len(other)} datasets")
    for d in other:
        print(f"    - {d}")

    print(f"\n--- Label Balance Across Datasets ---")
    balanced = []
    imbalanced = []
    no_rate = []
    for dataset_id, data in corpus.items():
        rate = data['metadata']['positive_rate']
        if rate is None:
            no_rate.append(dataset_id)
        elif 0.3 <= rate <= 0.7:
            balanced.append((dataset_id, rate))
        else:
            imbalanced.append((dataset_id, rate))

    print(f"  Balanced (30-70%):  {len(balanced)} datasets")
    print(f"  Imbalanced:         {len(imbalanced)} datasets")
    print(f"  No rate computed:   {len(no_rate)} datasets")

    print(f"\n--- FairGround EDA complete ---")
    print(f"  Total datasets: {len(corpus)} loaded (1 failed — law_school_tensorflow HTTP 403)")
    print(f"  Total records:  {total_records:,}")
    print(f"  Ready for cross-domain fairness baseline modeling")

    return corpus


if __name__ == "__main__":
    corpus = run_eda()
