
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

    # BUG FIX 2026-08-17: the previous version summed value_clean across every
    # domaincat_desc breakdown (area operated, economic class, farm sales, NAICS,
    # organization, producers-per-farm, tenure, typology - 8 overlapping dimensions
    # plus the single true total), inflating every figure by roughly 8x on top of
    # cross-contaminating race groups via loose substring matching (e.g. "WHITE"
    # matched "HISPANIC, WHITE" rows too). The correct row for a clean national
    # total is domaincat_desc == "NOT SPECIFIED", used with an exact short_desc
    # match per race group, not a substring match.
    outcomes = []
    for race in RACE_GROUPS:
        producer_desc = f"PRODUCERS, {race} - NUMBER OF PRODUCERS"
        operations_desc = f"PRODUCERS, {race} - NUMBER OF OPERATIONS"
        acres_desc = f"PRODUCERS, {race} - ACRES OPERATED"

        n_producers_rows = race_df[
            (race_df["short_desc"] == producer_desc) &
            (race_df["domaincat_desc"] == "NOT SPECIFIED")
        ]["value_clean"]
        n_producers = n_producers_rows.iloc[0] if len(n_producers_rows) > 0 else np.nan

        n_operations_rows = race_df[
            (race_df["short_desc"] == operations_desc) &
            (race_df["domaincat_desc"] == "NOT SPECIFIED")
        ]["value_clean"]
        n_operations = n_operations_rows.iloc[0] if len(n_operations_rows) > 0 else np.nan

        acres_rows = race_df[
            (race_df["short_desc"] == acres_desc) &
            (race_df["domaincat_desc"] == "NOT SPECIFIED")
        ]["value_clean"]
        acres = acres_rows.iloc[0] if len(acres_rows) > 0 else np.nan

        if pd.isna(n_producers) and pd.isna(n_operations) and pd.isna(acres):
            continue

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
