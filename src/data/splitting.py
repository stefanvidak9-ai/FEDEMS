"""Recreate the shared, chronological train/val/test target-date split."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/data_reports/step19_common_federated_target_dates.csv"
DEST = ROOT / "docs/data_reports/step21_temporal_split.csv"


def build_split():
    df = pd.read_csv(SOURCE, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if df["date"].isna().any() or not df["date"].is_unique:
        raise ValueError("Invalid or duplicate target dates")
    df["split"] = "test"
    df.loc[df["date"] <= "2022-08-31", "split"] = "train"
    df.loc[df["date"].between("2022-09-01", "2022-10-31"), "split"] = "val"
    sizes = df["split"].value_counts().to_dict()
    if sizes != {"train": 203, "val": 59, "test": 45}:
        raise ValueError(f"Unexpected temporal split counts: {sizes}")
    return df


def main():
    df = build_split()
    DEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEST, index=False)
    print(f"Wrote {len(df)} target dates to {DEST}")


if __name__ == "__main__":
    main()
