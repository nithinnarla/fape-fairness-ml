"""
Law School Admissions Loader, FAPE Phase 4
Wightman (1998), LSAC National Longitudinal Bar Passage Study

Source: FairGround Corpus (law_school_lequy)
LeQuy et al. adaptation of the Wightman LSAC dataset.
18,692 law school applicants across race and gender groups.

Why use the LeQuy version over the TensorFlow version:
The TensorFlow Datasets hosted version (law_school_tensorflow)
returns HTTP 403, Google Storage access blocked. The LeQuy
version is hosted on GitHub and loads cleanly. Both trace back
to the same Wightman (1998) LSAC source data.

Sensitive attributes: racetxt (race), male (sex)
Target: pass_bar (bar passage, binary)
Domain: Education/Legal
"""

import pandas as pd
import numpy as np


def load_law_school() -> dict:
    """
    Load Law School Admissions dataset via FairGround corpus.

    Uses the law_school_lequy variant, the LeQuy et al. adaptation
    of Wightman's LSAC dataset. Documented racial and gender disparities
    in bar passage rates make this a well-understood fairness benchmark
    for FAPE's education/legal domain evaluation.

    Why this matters for FAPE:
    The law school dataset captures a different fairness challenge than
    COMPAS. COMPAS bias operates through risk score assignment.
    Law school bias operates through selection and credentialing.
    the fairness intervention must hold across both mechanisms
    to justify FAPE's cross-domain generalization claim.

    Returns:
        Dictionary with dataset, metadata, and sensitive attribute info
    """
    from fairml_datasets import Datasets

    print("Loading Law School Admissions dataset via FairGround...")

    datasets = Datasets()
    law_dataset = None

    for d in datasets:
        if d.dataset_id == 'law_school_lequy':
            law_dataset = d
            break

    if law_dataset is None:
        raise RuntimeError("law_school_lequy not found in FairGround corpus")

    law_dataset.load()
    df = law_dataset.to_pandas()

    target_col = law_dataset.get_target_column()
    sensitive_cols = law_dataset.sensitive_columns

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Encode categorical features
    cat_cols = X.select_dtypes(include='object').columns
    for col in cat_cols:
        X[col] = pd.Categorical(X[col]).codes

    # pass_bar stored as object dtype, convert to numeric
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)
    positive_rate = float(y.mean())

    metadata = {
        'name': 'law_school_admissions',
        'source': 'FairGround law_school_lequy',
        'citation': 'Wightman (1998), LSAC National Longitudinal Bar Passage Study',
        'n_samples': len(X),
        'n_features': len(feature_cols),
        'sensitive_cols': sensitive_cols,
        'target': target_col,
        'positive_rate': positive_rate,
        'domain': 'Education/Legal',
        'note': 'LeQuy et al. adaptation of Wightman LSAC data, '
                'law_school_tensorflow unavailable (HTTP 403)'
    }

    print(f"  OK: law_school_admissions: {len(X):,} rows | "
          f"{len(feature_cols)} features | "
          f"sensitive: {sensitive_cols} | "
          f"positive rate: {positive_rate:.3f}")

    return {
        'law_school': {
            'X': X,
            'y': y,
            'metadata': metadata,
            'sensitive_cols': sensitive_cols
        }
    }


if __name__ == '__main__':
    print("Loading Law School Admissions for FAPE education/legal domain...")
    print("=" * 60)

    dataset = load_law_school()
    meta = dataset['law_school']['metadata']

    print(f"\nLaw School Summary:")
    print(f"  Records:    {meta['n_samples']:,}")
    print(f"  Features:   {meta['n_features']}")
    print(f"  Sensitive:  {meta['sensitive_cols']}")
    print(f"  Positive rate: {meta['positive_rate']:.3f}")
    print(f"  Citation:   {meta['citation']}")
