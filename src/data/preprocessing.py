"""Regenerate 17 CNN-ready client splits with training-only global z-score scaling.

Run from the repository root: python -m src.data.preprocessing
Only use eligible common target dates documented in step21_temporal_split.csv.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.data.feature_engineering import FEATURES, TARGET, WINDOW_DAYS
from src.data.create_windows import build_windows

ROOT = Path(__file__).resolve().parents[2]
DAILY = ROOT / "Dataset/daily"
PROCESSED = ROOT / "Dataset/processed"
SPLITS = ROOT / "docs/data_reports/step21_temporal_split.csv"


def main():
    split_df = pd.read_csv(SPLITS, parse_dates=["date"])
    split_df = split_df.sort_values("date").reset_index(drop=True)
    assert len(split_df) == 307 and split_df["date"].is_unique
    split_names = split_df["split"].to_numpy()
    assert {key: int((split_names == key).sum()) for key in ("train", "val", "test")} == {
        "train": 203, "val": 59, "test": 45
    }
    dates = split_df["date"]
    clients = {}
    for bid in range(6479, 6496):
        csv_path = DAILY / f"building_{bid}_daily.csv"
        daily = pd.read_csv(csv_path)
        clients[bid] = build_windows(daily, dates)
    train_x = np.concatenate([x[split_names == "train"] for x, y in clients.values()])
    train_y = np.concatenate([y[split_names == "train"] for x, y in clients.values()])
    feature_mean = train_x.mean(axis=(0, 2))
    feature_std = train_x.std(axis=(0, 2))
    target_mean = train_y.mean()
    target_std = train_y.std()
    if not (feature_std > 0).all() or not target_std > 0:
        raise ValueError("Zero training standard deviation")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    scaler = {
        "features": FEATURES, "target": TARGET,
        "window_days": WINDOW_DAYS, "forecast_horizon_days": 1,
        "normalization": "global training-only z-score",
        "feature_mean": feature_mean.tolist(), "feature_std": feature_std.tolist(),
        "target_mean": float(target_mean), "target_std": float(target_std),
        "training_target_count": int(len(train_y)),
    }
    with (PROCESSED / "global_scaler.json").open("w") as f:
        json.dump(scaler, f, indent=2)
    for bid, (x, y) in clients.items():
        x_normalized = ((x - feature_mean[None, :, None]) / feature_std[None, :, None]).astype(np.float32)
        y_normalized = ((y - target_mean) / target_std).astype(np.float32)
        out = PROCESSED / f"building_{bid}"
        out.mkdir(parents=True, exist_ok=True)
        for split, filename in (("train", "train.npz"), ("val", "validation.npz"), ("test", "test.npz")):
            mask = split_names == split
            np.savez_compressed(out / filename, X=x_normalized[mask], y=y_normalized[mask])
    print(f"Regenerated {len(clients)} clients in {PROCESSED}")


if __name__ == "__main__":
    main()
