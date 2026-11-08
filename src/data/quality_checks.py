from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAILY = ROOT / "Dataset/daily"
RAW = ROOT / "Dataset"
DOC = ROOT / "docs/data_reports"

FEATURES = [
    "HtSpace_kWh_day",
    "Tout_mean_C",
    "SolGlob_kWh_m2_day",
    "WindSpd_mean_m_s",
]

samples = pd.read_csv(
    DOC / "step19_federated_sample_availability.csv"
).set_index("building_id")

quality_rows = []
summary_rows = []

for bid in range(6479, 6496):
    df = pd.read_csv(
        DAILY / f"building_{bid}_daily.csv",
        parse_dates=["date"]
    )

    assert len(df) == 1461
    assert not df["date"].duplicated().any()

    df = df.set_index("date").sort_index()

    expected_dates = pd.date_range(
        df.index.min(), df.index.max(), freq="D"
    )
    assert df.index.equals(expected_dates)

    target = df["HtSpace_kWh_day"]
    valid_dates = target.dropna().index

    runs = (
        valid_dates.to_series()
        .diff()
        .ne(pd.Timedelta(days=1))
        .cumsum()
    )

    longest = int(
        valid_dates.to_series().groupby(runs).size().max()
    )

    metadata = {}
    with open(
        RAW / f"building_{bid}.txt",
        encoding="utf-8-sig"
    ) as f:
        for line in f:
            if ";" in line:
                key, value = line.strip().split(";", 1)
                metadata[key.strip().lower()] = value.strip()

    area = float(metadata["floor_area"])
    sample_count = int(samples.loc[bid, "individual_samples"])

    quality_rows.append({
        "building_id": bid,
        "daily_records": len(df),
        "date_start": df.index.min().date(),
        "date_end": df.index.max().date(),
        "duplicate_dates": 0,
        "valid_heating_days": int(target.notna().sum()),
        "missing_heating_days": int(target.isna().sum()),
        "valid_temperature_days":
            int(df["Tout_mean_C"].notna().sum()),
        "valid_solar_days":
            int(df["SolGlob_kWh_m2_day"].notna().sum()),
        "valid_wind_days":
            int(df["WindSpd_mean_m_s"].notna().sum()),
        "complete_heating_feature_days":
            int(df[FEATURES].notna().all(axis=1).sum()),
        "longest_heating_run_days": longest,
        "negative_heating_days": int((target.dropna() < 0).sum()),
        "negative_solar_days": int(
            (df["SolGlob_kWh_m2_day"].dropna() < 0).sum()
        ),
        "mean_heating_kWh_day": target.mean(),
        "std_heating_kWh_day": target.std(),
        "min_heating_kWh_day": target.min(),
        "max_heating_kWh_day": target.max(),
        "usable_cnn_samples": sample_count,
    })

    summary_rows.append({
        "building_id": bid,
        "location": metadata.get("location"),
        "year_of_construction":
            metadata.get("year_of_construction"),
        "floor_area_m2": area,
        "number_of_units": metadata.get("number_of_units"),
        "number_of_buildings":
            metadata.get("number_of_buildings"),
        "sh_heat_source": metadata.get("sh_heat_source"),
        "dhw_heat_source": metadata.get("dhw_heat_source"),
        "valid_heating_days": int(target.notna().sum()),
        "longest_heating_run_days": longest,
        "mean_heating_kWh_day": target.mean(),
        "mean_heating_kWh_m2_day": target.mean() / area,
        "usable_cnn_samples": sample_count,
    })

quality = pd.DataFrame(quality_rows)
summary = pd.DataFrame(summary_rows)

# Validate before saving
for report in (quality, summary):
    assert len(report) == 17
    assert report["building_id"].nunique() == 17
    assert report["building_id"].is_unique
    assert report["usable_cnn_samples"].sum() == 5423

quality.to_csv(DOC / "daily_data_quality.csv", index=False)
summary.to_csv(DOC / "preprocessing_summary.csv", index=False)

print("\nSTEP 19 — QUALITY REPORTS REBUILT")
print("Quality rows:", len(quality))
print("Summary rows:", len(summary))
print("Unique clients:", quality["building_id"].nunique())
print("Total CNN samples:", quality["usable_cnn_samples"].sum())

print("\nPer-client results:")
print(
    quality[
        ["building_id", "valid_heating_days", "usable_cnn_samples"]
    ].to_string(index=False)
)

print("\nBoth reports saved successfully.")
