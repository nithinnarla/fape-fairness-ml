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


def get_domain(name):
    if any(x in name for x in ["compas","stop_question","chicago","ricci"]):
        return "Criminal Justice"
    elif any(x in name for x in ["credit","bank"]):
        return "Financial"
    elif any(x in name for x in ["meps","heart","arrhythmia"]):
        return "Healthcare"
    elif any(x in name for x in ["student","law_school","nursery"]):
        return "Education"
    elif any(x in name for x in ["adult","folktables","dutch","communities"]):
        return "Socioeconomic"
    else:
        return "Other/Synthetic"


def get_balance(rate):
    if rate is None or pd.isna(rate):
        return "No rate computed"
    elif rate < 0 or rate > 1:
        return "Non-binary target"
    elif 0.3 <= rate <= 0.7:
        return "Balanced (30-70%)"
    else:
        return "Imbalanced"


def run_eda():
    print("FAPE Phase 4 — FairGround Corpus EDA")
    print("=" * 50)

    corpus = load_fairground_corpus()

    print(f"\n--- Corpus Overview ---")
    print(f"  Datasets loaded: {len(corpus)}")
    total_records = sum(corpus[k]["metadata"]["n_samples"] for k in corpus)
    print(f"  Total records:   {total_records:,}")
    print(f"  Failed:          1 (law_school_tensorflow HTTP 403)")

    print(f"\n--- Dataset Summary ---")
    print(f"  {'Dataset':<40} {'Records':>10} {'Features':>10} Sensitive Attrs")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*30}")
    sizes = []
    for dataset_id, data in corpus.items():
        meta = data["metadata"]
        n = meta["n_samples"]
        f = meta["n_features"]
        sens = meta["sensitive_columns"]
        sizes.append((dataset_id, n, f, sens))
        print(f"  {dataset_id:<40} {n:>10,} {f:>10} {sens}")

    print(f"\n--- Size Distribution ---")
    sizes_sorted = sorted(sizes, key=lambda x: x[1], reverse=True)
    print(f"  Largest:  {sizes_sorted[0][0]} — {sizes_sorted[0][1]:,} records")
    print(f"  Smallest: {sizes_sorted[-1][0]} — {sizes_sorted[-1][1]:,} records")
    print(f"  Median:   {sorted([s[1] for s in sizes])[len(sizes)//2]:,} records")

    print(f"\n--- Domain Coverage ---")
    domain_list = ["Criminal Justice","Financial","Healthcare","Education","Socioeconomic","Other/Synthetic"]
    for domain in domain_list:
        domain_keys = [k for k in corpus if get_domain(k) == domain]
        total = sum(corpus[k]["metadata"]["n_samples"] for k in domain_keys)
        print(f"  {domain:<20} {len(domain_keys)} datasets | {total:,} records")
        for k in domain_keys:
            print(f"    - {k}")

    print(f"\n--- Sensitive Attributes by Domain ---")
    for domain in domain_list:
        domain_keys = [k for k in corpus if get_domain(k) == domain]
        attrs = []
        for k in domain_keys:
            attrs.extend(corpus[k]["metadata"]["sensitive_columns"])
        attr_counts = pd.Series(attrs).value_counts()
        print(f"  {domain}: {list(attr_counts.index)}")

    print(f"\n--- Sensitive Attribute Coverage ---")
    all_attrs = []
    for _, data in corpus.items():
        all_attrs.extend(data["metadata"]["sensitive_columns"])
    attr_counts = pd.Series(all_attrs).value_counts()
    print(f"  Most common sensitive attributes:")
    for attr, count in attr_counts.head(10).items():
        print(f"    {attr:<30} appears in {count} datasets")

    print(f"\n--- Label Balance Across Datasets ---")
    balanced = []
    imbalanced = []
    no_rate = []
    non_binary = []
    for dataset_id, data in corpus.items():
        rate = data["metadata"]["positive_rate"]
        balance = get_balance(rate)
        if balance == "Balanced (30-70%)":
            balanced.append(dataset_id)
        elif balance == "Imbalanced":
            imbalanced.append(dataset_id)
        elif balance == "Non-binary target":
            non_binary.append(dataset_id)
        else:
            no_rate.append(dataset_id)
    print(f"  Balanced (30-70%):  {len(balanced)} datasets")
    print(f"  Imbalanced:         {len(imbalanced)} datasets")
    print(f"  Non-binary target:  {len(non_binary)} datasets")
    print(f"  No rate computed:   {len(no_rate)} datasets")

    print(f"\n--- Feature Count Distribution ---")
    n_features_list = [corpus[k]["metadata"]["n_features"] for k in corpus]
    n_features_series = pd.Series(n_features_list)
    print(f"  Min: {n_features_series.min()} | Max: {n_features_series.max()}")
    print(f"  Median: {n_features_series.median():.0f}")
    outliers = [(k, corpus[k]["metadata"]["n_features"]) for k in corpus if corpus[k]["metadata"]["n_features"] > 200]
    print(f"  Datasets with >200 features: {len(outliers)}")
    for name, nf in outliers:
        print(f"    {name}: {nf}")


    print(f"\n--- Records per Domain by Label Balance ---")
    domain_balance = {}
    for domain in domain_list:
        domain_keys = [k for k in corpus if get_domain(k) == domain]
        balance_counts = {}
        for k in domain_keys:
            rate = corpus[k]["metadata"]["positive_rate"]
            balance = get_balance(rate)
            n = corpus[k]["metadata"]["n_samples"]
            balance_counts[balance] = balance_counts.get(balance, 0) + n
        domain_balance[domain] = balance_counts
        print(f"  {domain}:")
        for b, n in sorted(balance_counts.items()):
            print(f"    {b:<25} {n:,} records")

    print(f"\n--- Top 10 Largest Datasets ---")
    top10 = sorted([(k, corpus[k]["metadata"]["n_samples"]) for k in corpus], key=lambda x: x[1], reverse=True)[:10]
    for name, n in top10:
        print(f"  {name:<45} {n:,}")

    print(f"\n--- FairGround EDA complete ---")
    print(f"  Total datasets: {len(corpus)} loaded (1 failed — law_school_tensorflow HTTP 403)")
    print(f"  Total records:  {total_records:,}")
    print(f"  Ready for cross-domain fairness baseline modeling")

    return corpus


if __name__ == "__main__":
    corpus = run_eda()
