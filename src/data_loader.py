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


def load_folktables_acs() -> pd.DataFrame:
    """
    Load the Folktables ACS dataset for income fairness analysis.

    Source: US Census American Community Survey (2021)
    Records: 3,000,000+ socioeconomic records across all 50 states
    Target: Annual income > $50,000
    Replaces legacy Adult Income dataset (Ding et al., 2021)

    Features selected for fairness analysis:
        AGEP     - Age
        SEX      - Sex
        RAC1P    - Race
        SCHL     - Educational attainment
        MAR      - Marital status
        WKHP     - Hours worked per week
        COW      - Class of worker
        DIS      - Disability status
        POVPIP   - Income to poverty ratio
        NATIVITY - Native or foreign born
    """
    from folktables import ACSDataSource

    print("Loading Folktables ACS data — first run downloads per state, please wait...")

    data_source = ACSDataSource(
        survey_year="2021",
        horizon="1-Year",
        survey="person"
    )

    all_states = [
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
    ]

    acs_features = [
        "AGEP", "SEX", "RAC1P", "SCHL", "MAR",
        "WKHP", "COW", "DIS", "POVPIP", "NATIVITY"
    ]

    chunks = []
    for state in all_states:
        try:
            state_data = data_source.get_data(states=[state], download=True)

            # Filter to working-age adults with valid income records
            # Process per state to keep memory footprint manageable
            state_data = state_data[
                (state_data["AGEP"] >= 16) &
                (state_data["AGEP"] <= 90) &
                (state_data["PINCP"].notna())
            ].copy()

            # Binary income label: annual income > $50,000
            state_data["label"] = (state_data["PINCP"] > 50000).astype(int)

            state_data = state_data[acs_features + ["label"]].dropna()
            chunks.append(state_data)
            print(f"  {state}: {len(state_data):,} records")

        except Exception as e:
            print(f"  {state}: skipped ({e})")

    df = pd.concat(chunks, ignore_index=True)

    print(f"Folktables ACS loaded: {len(df):,} records, "
          f"income >$50k rate: {df['label'].mean():.1%}")

    return df


if __name__ == "__main__":
    print("=== COMPAS ===")
    compas = load_compas()
    print(compas.head())
    print(compas["race"].value_counts())

    print("\n=== Folktables ACS ===")
    acs = load_folktables_acs()
    print(acs.head())
    print(acs.shape)
