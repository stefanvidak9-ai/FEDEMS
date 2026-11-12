
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "Dataset" / "daily"
OUTPUT_DIR = PROJECT / "docs" / "data_reports"

WINDOW = 7
TARGET = "HtSpace_kWh_day"

FEATURES = [
    "HtSpace_kWh_day",
    "Tout_mean_C",
    "SolGlob_kWh_m2_day",
    "WindSpd_mean_m_s",
]

files = sorted(DATA_DIR.glob("building_*_daily.csv"))
assert len(files) == 17, f"Expected 17 files, found {len(files)}"

client_dates = {}

for path in files:
    bid = int(path.stem.split("_")[1])

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    required = ["date"] + FEATURES
    missing = set(required) - set(df.columns)
    assert not missing, f"{bid}: missing columns {missing}"

    assert df["date"].notna().all()
    assert not df["date"].duplicated().any()

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_dates = set()

    for i in range(WINDOW, len(df)):
        history = df.iloc[i-WINDOW:i]
        target = df.iloc[i]

        dates = df.iloc[i-WINDOW:i+1]["date"]

        consecutive = (
            dates.diff().iloc[1:] == pd.Timedelta(days=1)
        ).all()

        valid_history = history[FEATURES].notna().all().all()
        valid_target = pd.notna(target[TARGET])

        if consecutive and valid_history and valid_target:
            valid_dates.add(target["date"].strftime("%Y-%m-%d"))

    client_dates[bid] = valid_dates

common_dates = set.intersection(*client_dates.values())

rows = [
    {
        "building_id": bid,
        "individual_samples": len(dates),
        "common_federated_samples": len(common_dates),
    }
    for bid, dates in sorted(client_dates.items())
]

summary = pd.DataFrame(rows)

assert summary["building_id"].nunique() == 17
assert len(summary) == 17

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

summary.to_csv(
    OUTPUT_DIR / "step19_federated_sample_availability.csv",
    index=False,
)

pd.DataFrame({
    "date": sorted(common_dates)
}).to_csv(
    OUTPUT_DIR / "step19_common_federated_target_dates.csv",
    index=False,
)

print("\nSTEP 19 — REBUILT SAMPLE AVAILABILITY")
print(summary.to_string(index=False))

print("\nUnique clients:", summary["building_id"].nunique())
print("Total individual samples:", summary["individual_samples"].sum())
print("Common federated samples:", len(common_dates))
print("\nSaved both CSV files.")
