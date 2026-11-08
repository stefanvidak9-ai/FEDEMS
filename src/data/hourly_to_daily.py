
import csv
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "daily"

BUILDING_IDS = list(range(6479, 6496))

REQUIRED_COLUMNS = [
    "HtSpace",
    "Tout",
    "SolGlob",
    "WindSpd",
]


# --------------------------------------------------
# 1. Locate the time-series header
# --------------------------------------------------

def find_header_and_separator(file_path):

    with open(file_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    header_index = None

    for i, line in enumerate(lines):
        normalized = line.lower()

        if "timestamp" in normalized and "htspace" in normalized:
            header_index = i
            break

    if header_index is None:
        raise ValueError("Could not find timestamp/HtSpace header")

    try:
        delimiter = csv.Sniffer().sniff(
            lines[header_index],
            delimiters=",;\t"
        ).delimiter
    except csv.Error:
        delimiter = ","

    return header_index, delimiter


# --------------------------------------------------
# 2. Read hourly data
# --------------------------------------------------

def load_hourly_data(file_path):

    header_index, delimiter = find_header_and_separator(file_path)

    df = pd.read_csv(
        file_path,
        skiprows=header_index,
        sep=delimiter,
        encoding="utf-8-sig",
        engine="python"
    )

    df.columns = df.columns.astype(str).str.strip()

    column_lookup = {
        col.lower(): col for col in df.columns
    }

    required = ["timestamp"] + [
        col.lower() for col in REQUIRED_COLUMNS
    ]

    missing = [
        col for col in required
        if col not in column_lookup
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    rename_map = {
        column_lookup["timestamp"]: "Timestamp"
    }

    for col in REQUIRED_COLUMNS:
        rename_map[column_lookup[col.lower()]] = col

    df = df.rename(columns=rename_map)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    # Preserve local calendar dates from source timestamps.
    df["Date"] = df["Timestamp"].dt.strftime("%Y-%m-%d")

    df = df.dropna(subset=["Date"]).copy()

    return df


# --------------------------------------------------
# 3. Aggregate hourly data into daily data
# --------------------------------------------------

def aggregate_daily(df, building_id):

    results = []

    for date, group in df.groupby("Date", sort=True):

        unique_hours = group["Timestamp"].nunique()

        hourly_complete = (
            len(group) == 24
            and unique_hours == 24
            and set(group["Timestamp"].dt.hour) == set(range(24))
        )

        row = {
            "date": date,
            "building_id": building_id,
            "n_rows": len(group),
            "n_unique_hours": unique_hours,
            "hourly_complete": hourly_complete,
        }

        # Space-heating energy
        heating = group["HtSpace"].dropna()

        heating_valid = (
            len(heating) == 24
            and hourly_complete
            and (heating >= 0).all()
        )

        row["HtSpace_valid_hours"] = len(heating)

        row["HtSpace_kWh_day"] = (
            heating.sum() / 1000
            if heating_valid else np.nan
        )

        # Weather variables
        for col, output_name in [
            ("Tout", "Tout_mean_C"),
            ("SolGlob", "SolGlob_kWh_m2_day"),
            ("WindSpd", "WindSpd_mean_m_s"),
        ]:

            values = group[col].dropna()

            row[f"{col}_valid_hours"] = len(values)

            if col == "SolGlob":

                negative_count = int((values < 0).sum())

                row["SolGlob_negative_hours"] = negative_count

                # Documented preprocessing decision:
                # clip negative irradiance readings to zero.
                values = values.clip(lower=0)

            valid = (
                len(values) == 24
                and hourly_complete
            )

            if col == "WindSpd":
                valid = valid and (values >= 0).all()

            if valid:

                if col == "Tout":
                    row[output_name] = values.mean()

                elif col == "SolGlob":
                    row[output_name] = values.sum() / 1000

                elif col == "WindSpd":
                    row[output_name] = values.mean()

            else:
                row[output_name] = np.nan

        results.append(row)

    return pd.DataFrame(results)


# --------------------------------------------------
# 4. Calculate preliminary client summary
# --------------------------------------------------

def longest_heating_run(daily_df):

    valid = daily_df[
        daily_df["HtSpace_kWh_day"].notna()
    ].copy()

    if valid.empty:
        return 0

    dates = pd.to_datetime(valid["date"])

    gaps = dates.diff().dt.days

    groups = gaps.ne(1).cumsum()

    return int(
        valid.groupby(groups).size().max()
    )


# --------------------------------------------------
# 5. Process all candidate buildings
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_daily = []
    summaries = []
    errors = []

    for building_id in BUILDING_IDS:

        file_path = INPUT_DIR / f"building_{building_id}.txt"

        print("\n" + "=" * 55)
        print(f"Processing Building {building_id}")

        if not file_path.exists():
            print("ERROR: Original file not found")
            errors.append({
                "building_id": building_id,
                "error": "Original file not found"
            })
            continue

        try:

            hourly_df = load_hourly_data(file_path)

            daily_df = aggregate_daily(
                hourly_df, building_id
            )

            output_file = (
                OUTPUT_DIR /
                f"building_{building_id}_daily.csv"
            )

            daily_df.to_csv(output_file, index=False)

            all_daily.append(daily_df)

            summary = {
                "building_id": building_id,
                "hourly_records": len(hourly_df),
                "daily_records": len(daily_df),
                "complete_heating_days":
                    daily_df["HtSpace_kWh_day"].notna().sum(),
                "complete_temperature_days":
                    daily_df["Tout_mean_C"].notna().sum(),
                "complete_solar_days":
                    daily_df["SolGlob_kWh_m2_day"].notna().sum(),
                "complete_wind_days":
                    daily_df["WindSpd_mean_m_s"].notna().sum(),
                "negative_solar_readings_corrected":
                    daily_df["SolGlob_negative_hours"].sum(),
                "longest_heating_run":
                    longest_heating_run(daily_df),
            }

            summaries.append(summary)

            print("Hourly records:", len(hourly_df))
            print("Daily records:", len(daily_df))
            print(
                "Complete heating days:",
                summary["complete_heating_days"]
            )
            print(
                "Longest heating run:",
                summary["longest_heating_run"]
            )
            print("Saved:", output_file)

        except Exception as e:

            print("ERROR:", str(e))

            errors.append({
                "building_id": building_id,
                "error": str(e)
            })

    # Save combined daily dataset
    if all_daily:

        combined = pd.concat(
            all_daily,
            ignore_index=True
        )

        combined.to_csv(
            OUTPUT_DIR / "all_candidates_daily.csv",
            index=False
        )

    # Save preliminary processing summary
    if summaries:

        pd.DataFrame(summaries).to_csv(
            PROJECT_DIR / "docs/data_reports" /
            "step18_processing_summary.csv",
            index=False
        )

    # Save processing errors, if any
    if errors:

        pd.DataFrame(errors).to_csv(
            PROJECT_DIR / "docs/data_reports" /
            "step18_processing_errors.csv",
            index=False
        )

    print("\n" + "=" * 55)
    print("BATCH PROCESSING FINISHED")
    print("Successfully processed:", len(summaries))
    print("Errors:", len(errors))
    print("Candidate buildings:", len(BUILDING_IDS))


if __name__ == "__main__":
    main()
