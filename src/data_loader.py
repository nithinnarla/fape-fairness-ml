import pandas as pd
import numpy as np
from pathlib import Path


def load_compas(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load and preprocess the COMPAS recidivism dataset.
    
    Source: ProPublica - Broward County Florida (2013-2014)
    Records: 7,000+ criminal defendants
    Target: Two-year recidivism outcome
    """
    filepath = Path(data_dir) / "compas-scores-two-years.csv"
    
    df = pd.read_csv(filepath)
    
    # Filter to relevant population — same criteria ProPublica used
    # Removes records with missing data or irrelevant charge categories
    df = df[
        (df["days_b_screening_arrest"] <= 30) &
        (df["days_b_screening_arrest"] >= -30) &
        (df["is_recid"] != -1) &
        (df["c_charge_degree"] != "O") &
        (df["score_text"] != "N/A")
    ]
    
    # Select features relevant to fairness analysis
    features = [
        "age", "c_charge_degree", "race", "sex",
        "priors_count", "days_b_screening_arrest",
        "decile_score", "score_text", "is_recid",
        "two_year_recid"
    ]
    df = df[features].copy()
    
    # Rename target for clarity
    df = df.rename(columns={"two_year_recid": "label"})
    
    # Basic null check
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"Warning: null values found:\n{null_counts[null_counts > 0]}")
    
    print(f"COMPAS loaded: {len(df):,} records, "
          f"{df['race'].nunique()} racial groups, "
          f"recidivism rate: {df['label'].mean():.1%}")
    
    return df


if __name__ == "__main__":
    df = load_compas()
    print(df.head())
    print(df["race"].value_counts())
