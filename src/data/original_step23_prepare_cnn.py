from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data/daily"
PREPARED = ROOT / "data/prepared"
SPLIT_FILE = ROOT / "documentation/step21_temporal_split.csv"

FEATURES = [
    "HtSpace_kWh_day",
    "Tout_mean_C",
    "SolGlob_kWh_m2_day",
    "WindSpd_mean_m_s",
]
TARGET = "HtSpace_kWh_day"
WINDOW = 7

PREPARED.mkdir(parents=True, exist_ok=True)

split_df = pd.read_csv(SPLIT_FILE, parse_dates=["date"])
split_df = split_df.sort_values("date").reset_index(drop=True)

assert len(split_df) == 307
assert split_df["date"].is_unique
assert set(split_df["split"]) == {"train", "val", "test"}

expected = {"train": 203, "val": 59, "test": 45}
for split, count in expected.items():
    assert (split_df["split"] == split).sum() == count

assert (
    split_df.loc[split_df["split"] == "train", "date"].max()
    < split_df.loc[split_df["split"] == "val", "date"].min()
)
assert (
    split_df.loc[split_df["split"] == "val", "date"].max()
    < split_df.loc[split_df["split"] == "test", "date"].min()
)

# Collect each client's eligible seven-day windows.
clients = {}
feature_sum = np.zeros(len(FEATURES), dtype=np.float64)
feature_sum_sq = np.zeros(len(FEATURES), dtype=np.float64)
feature_count = 0

target_sum = 0.0
target_sum_sq = 0.0
target_count = 0

splits = split_df["split"].to_numpy()

for bid in range(6479, 6496):
    file = DAILY / f"building_{bid}_daily.csv"

    df = pd.read_csv(file, parse_dates=["date"])
    assert df["date"].is_unique
    df = df.set_index("date").sort_index()

    X, y = [], []

    for target_date in split_df["date"]:
        history_dates = pd.date_range(
            end=target_date - pd.Timedelta(days=1),
            periods=WINDOW,
            freq="D",
        )

        history = df.reindex(history_dates)[FEATURES]
        target = df.loc[target_date, TARGET]

        values = history.to_numpy(dtype=np.float64)
        target = float(target)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Client {bid}: invalid history for {target_date}"
            )
        if not np.isfinite(target):
            raise ValueError(
                f"Client {bid}: invalid target for {target_date}"
            )

        X.append(values.T)  # (4 features, 7 days)
        y.append(target)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    assert X.shape == (307, 4, 7)
    assert y.shape == (307,)

    clients[bid] = (X, y)

    # Training-only statistics for global normalization.
    train_X = X[splits == "train"]
    train_y = y[splits == "train"]

    feature_sum += train_X.sum(axis=(0, 2))
    feature_sum_sq += np.square(train_X).sum(axis=(0, 2))
    feature_count += train_X.shape[0] * WINDOW

    target_sum += train_y.sum()
    target_sum_sq += np.square(train_y).sum()
    target_count += len(train_y)

# Calculate shared normalization parameters.
feature_mean = feature_sum / feature_count
feature_std = np.sqrt(
    np.maximum(
        feature_sum_sq / feature_count - feature_mean**2,
        0.0,
    )
)

target_mean = target_sum / target_count
target_std = np.sqrt(
    max(target_sum_sq / target_count - target_mean**2, 0.0)
)

assert (feature_std > 0).all()
assert target_std > 0

scaler = {
    "features": FEATURES,
    "target": TARGET,
    "window_days": WINDOW,
    "forecast_horizon_days": 1,
    "normalization": "global training-only z-score",
    "feature_mean": feature_mean.tolist(),
    "feature_std": feature_std.tolist(),
    "target_mean": float(target_mean),
    "target_std": float(target_std),
    "training_target_count": target_count,
}

with open(PREPARED / "global_scaler.json", "w") as f:
    json.dump(scaler, f, indent=2)

# Normalize and save each client's arrays.
for bid, (X, y) in clients.items():
    X_normalized = (
        X - feature_mean[None, :, None]
    ) / feature_std[None, :, None]

    y_normalized = (y - target_mean) / target_std

    output = {}
    for split in ("train", "val", "test"):
        mask = splits == split

        output[f"X_{split}"] = X_normalized[mask].astype(
            np.float32
        )
        output[f"y_{split}"] = y_normalized[mask].astype(
            np.float32
        )

    np.savez_compressed(
        PREPARED / f"client_{bid}.npz",
        **output,
    )

print("\nSTEP 23 — CNN DATA PREPARATION")

print("\nGlobal normalization statistics:")
for i, name in enumerate(FEATURES):
    print(
        f"{name}: mean={feature_mean[i]:.4f}, "
        f"std={feature_std[i]:.4f}"
    )

print(
    f"\nTarget: mean={target_mean:.4f}, "
    f"std={target_std:.4f}"
)

print("\nClient sample shapes:")
for bid in (6479, 6491, 6495):
    data = np.load(PREPARED / f"client_{bid}.npz")
    print(f"\nClient {bid}:")
    for split in ("train", "val", "test"):
        print(
            f"  {split}: X={data[f'X_{split}'].shape}, "
            f"y={data[f'y_{split}'].shape}"
        )

print("\nCreated 17 client datasets and global_scaler.json")
