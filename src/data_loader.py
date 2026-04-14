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


def load_folktables_acs(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load the Folktables ACS Income task dataset.
    
    Source: US Census American Community Survey (2021-2023)
    Records: 3,000,000+ socioeconomic records
    Target: Annual income > $50,000
    Replaces legacy Adult Income dataset (Ding et al., 2021)
    """
    from folktables import ACSDataSource, ACSIncome

    # Download ACS data for all 50 states, 2021 survey year
    # Data downloads automatically to local cache on first run
    print("Loading Folktables ACS data — first run downloads ~500MB, please wait...")

    data_source = ACSDataSource(
        survey_year="2021",
        horizon="1-Year",
        survey="person"
    )

    # Load all US states for maximum coverage
    # 10 states selected for geographic + demographic diversity
    # Covers urban/rural, high/low income, racially diverse populations
    # Load all 50 states one by one and concatenate
    # Avoids memory crash from loading all states simultaneously
    all_states = ["AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
                  "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
                  "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
                  "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
                  "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"]

    chunks = []
    for state in all_states:
        try:
            state_data = data_source.get_data(states=[state], download=True)

            # Filter and process each state before concatenating
            # Keeps memory footprint small — never load full 3M+ raw at once
            state_data = state_data[
                (state_data["AGEP"] >= 16) &
                (state_data["AGEP"] <= 90) &
                (state_data["PINCP"].notna())
            ].copy()

            state_data["label"] = (state_data["PINCP"] > 50000).astype(int)

            acs_features = ["AGEP","SEX","RAC1P","SCHL","MAR",
                           "WKHP","COW","DIS","POVPIP","NATIVITY","label"]

            state_data = state_data[acs_features].dropna()
            chunks.append(state_data)
            print(f"  {state}: {len(state_data):,} records")

        except Exception as e:
            print(f"  {state}: skipped ({e})")

    acs_data = pd.concat(chunks, ignore_index=True)
    print(f"Total ACS records after filtering: {len(acs_data):,}")

    # Select fairness-relevant features available in 2021 ACS format
    # AGEP=age, SEX=sex, RAC1P=race, SCHL=education, MAR=marital status
    # WKHP=hours worked, ESR=employment status, PINCP=total income
    # POVPIP=poverty ratio, COW=class of worker, DIS=disability status
    acs_features = [
        "AGEP", "SEX", "RAC1P", "SCHL", "MAR",
        "WKHP", "COW", "DIS", "POVPIP", "NATIVITY"
    ]

    df = acs_data.copy()

    print(f"Folktables ACS loaded: {len(df):,} records, "
          f"income >$50k rate: {df['label'].mean():.1%}")

    return df


if __name__ == "__main__":
    df = load_folktables_acs()
    print(df.head())
    print(df.shape)
