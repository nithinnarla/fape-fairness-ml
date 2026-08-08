
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/raw/agricultural/usda_nass_producers_2022.csv"

RACE_GROUPS = [
    "AMERICAN INDIAN OR ALASKA NATIVE",
    "ASIAN",
    "BLACK OR AFRICAN AMERICAN",
    "HISPANIC",
    "WHITE",
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER",
    "MORE THAN ONE RACE"
]

def load_usda_nass(data_path=DATA_PATH):
    print(f"Loading USDA NASS Census 2022...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"  Total records: {len(df):,}")

    race_mask = df["short_desc"].str.contains(
        "AMERICAN INDIAN|ASIAN|BLACK|HISPANIC|WHITE|PACIFIC ISLANDER|MORE THAN ONE RACE",
        case=False, na=False
    )
    race_df = df[race_mask].copy()
    print(f"  Race-related records: {len(race_df):,}")

    race_df["value_clean"] = pd.to_numeric(
        race_df["Value"].astype(str)
        .str.replace(",", "")
        .str.replace("(D)", "NaN")
        .str.replace("(Z)", "0"),
        errors="coerce"
    )

    outcomes = []
    for race in RACE_GROUPS:
        race_records = race_df[
            race_df["short_desc"].str.contains(race, case=False, na=False)
        ]
        if len(race_records) == 0:
            continue

        n_producers = race_records[
            race_records["short_desc"].str.contains("NUMBER OF PRODUCERS", case=False, na=False) &
            ~race_records["short_desc"].str.contains(
                "DAY|DAYS|DECISION|OCCUPATION|HIRED|ESTATE|LAND|LIVE|MARKET", case=False, na=False
            )
        ]["value_clean"].sum()

        n_operations = race_records[
            race_records["short_desc"].str.contains("NUMBER OF OPERATIONS", case=False, na=False)
        ]["value_clean"].sum()

        acres = race_records[
            race_records["short_desc"].str.contains("ACRES OPERATED", case=False, na=False)
        ]["value_clean"].sum()

        outcomes.append({
            "race_group": race,
            "n_producers": n_producers,
            "n_operations": n_operations,
            "acres_operated": acres,
        })

    summary_df = pd.DataFrame(outcomes)

    print(f"\n  USDA NASS 2022, Producer Demographics Summary:")
    print(f"  {'Race Group':<45} {'Producers':>12} {'Operations':>12}")
    print(f"  {'-'*72}")
    for _, row in summary_df.iterrows():
        print(f"  {row['race_group']:<45} "
              f"{row['n_producers']:>12,.0f} "
              f"{row['n_operations']:>12,.0f}")

    print(f"\n  Total race groups loaded: {len(summary_df)}")
    return {"usda_nass": {"raw": df, "race_summary": summary_df, "metadata": {"n_records": len(df), "n_race_records": len(race_df)}}}

if __name__ == "__main__":
    dataset = load_usda_nass()
