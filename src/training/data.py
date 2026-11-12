"""Load and validate the finalized Phase 5 window datasets."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from src.data.feature_engineering import FEATURES, TARGET, WINDOW_DAYS

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Scaler:
    features: tuple[str, ...]
    target: str
    window_days: int
    target_mean: float
    target_std: float
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.target_std + self.target_mean


@dataclass(frozen=True)
class WindowSplit:
    X: np.ndarray
    y: np.ndarray
    dates: tuple[str, ...]


@dataclass(frozen=True)
class BuildingData:
    building_id: int
    splits: dict[str, WindowSplit]


def load_scaler(processed_dir: str | Path) -> Scaler:
    """Read the Phase 5 scaler and verify its one-day, seven-day contract."""
    path = Path(processed_dir) / "global_scaler.json"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read Phase 5 scaler at {path}: {error}") from error

    try:
        features = tuple(str(value) for value in metadata["features"])
        target = str(metadata["target"])
        window_days = int(metadata["window_days"])
        horizon = int(metadata["forecast_horizon_days"])
        feature_mean = tuple(float(value) for value in metadata["feature_mean"])
        feature_std = tuple(float(value) for value in metadata["feature_std"])
        target_mean = float(metadata["target_mean"])
        target_std = float(metadata["target_std"])
        normalization = metadata["normalization"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid Phase 5 scaler metadata in {path}: {error}") from error

    if features != tuple(FEATURES):
        raise ValueError(
            f"Scaler feature order must match Phase 4/5 FEATURES: {tuple(FEATURES)}"
        )
    if target != TARGET or target not in features:
        raise ValueError(f"Scaler target must match the Phase 4/5 target {TARGET!r}")
    if window_days != WINDOW_DAYS or horizon != 1:
        raise ValueError("Expected the finalized seven-day lookback and one-day forecast horizon")
    if normalization != "global training-only z-score":
        raise ValueError(f"Unsupported Phase 5 normalization: {normalization!r}")
    if len(feature_mean) != len(features) or len(feature_std) != len(features):
        raise ValueError("Scaler statistics do not match the feature count")
    statistics = (*feature_mean, *feature_std, target_mean, target_std)
    if not all(math.isfinite(value) for value in statistics):
        raise ValueError("Scaler statistics must all be finite")
    if any(value <= 0 for value in feature_std) or target_std <= 0:
        raise ValueError("Scaler standard deviations must be positive")

    return Scaler(
        features=features,
        target=target,
        window_days=window_days,
        target_mean=target_mean,
        target_std=target_std,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def load_split_dates(split_file: str | Path) -> dict[str, tuple[str, ...]]:
    """Load authoritative chronological target dates from the Phase 5 split."""
    path = Path(split_file)
    dates: dict[str, list[str]] = {split: [] for split in SPLITS}
    seen: set[str] = set()
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or not {"date", "split"}.issubset(reader.fieldnames):
                raise ValueError(f"Split file {path} must contain date and split columns")
            for row_number, row in enumerate(reader, start=2):
                try:
                    target_date = date.fromisoformat(row["date"].strip()).isoformat()
                except (AttributeError, ValueError) as error:
                    raise ValueError(f"Invalid target date on row {row_number} of {path}") from error
                if target_date in seen:
                    raise ValueError(f"Duplicate target date {target_date} in {path}")
                seen.add(target_date)
                split = row["split"].strip().lower()
                normalized_split = "validation" if split == "val" else split
                if normalized_split not in dates:
                    raise ValueError(
                        f"Invalid split {split!r} on row {row_number} of {path}"
                    )
                dates[normalized_split].append(target_date)
    except OSError as error:
        raise ValueError(f"Cannot read Phase 5 split file {path}: {error}") from error

    if any(not values for values in dates.values()):
        raise ValueError(f"Phase 5 split file {path} must contain train, val, and test dates")
    for split_dates in dates.values():
        if split_dates != sorted(split_dates):
            raise ValueError(f"Target dates in {path} must be chronological within each split")
    if max(dates["train"]) >= min(dates["validation"]):
        raise ValueError("Training dates must precede validation dates")
    if max(dates["validation"]) >= min(dates["test"]):
        raise ValueError("Validation dates must precede test dates")
    return {split: tuple(values) for split, values in dates.items()}


def _read_window_file(path: Path, feature_count: int, window_days: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            if not {"X", "y"}.issubset(data.files):
                raise ValueError(f"Window file {path} must contain X and y arrays")
            X = np.asarray(data["X"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.float32).reshape(-1)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot load Phase 5 window file {path}: {error}") from error

    if X.ndim != 3 or X.shape[1:] != (feature_count, window_days):
        raise ValueError(
            f"{path} has input shape {X.shape}; expected [samples, {feature_count}, {window_days}]"
        )
    if X.shape[0] == 0 or y.size == 0:
        raise ValueError(f"{path} contains no usable samples")
    if X.shape[0] != y.size:
        raise ValueError(f"{path} has {X.shape[0]} inputs but {y.size} targets")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"{path} contains non-finite model inputs or targets")
    return X, y


def load_buildings(
    processed_dir: str | Path,
    split_file: str | Path,
    building_ids: list[int] | tuple[int, ...],
) -> tuple[Scaler, dict[int, BuildingData]]:
    """Load Phase 5 NPZs without changing their split or scaling policy."""
    if not building_ids:
        raise ValueError("At least one building ID must be selected")
    if len(set(building_ids)) != len(building_ids):
        raise ValueError("Building IDs must be unique")

    scaler = load_scaler(processed_dir)
    dates_by_split = load_split_dates(split_file)
    building_data: dict[int, BuildingData] = {}
    for building_id in building_ids:
        if building_id <= 0:
            raise ValueError(f"Invalid building ID: {building_id}")
        directory = Path(processed_dir) / f"building_{building_id}"
        splits: dict[str, WindowSplit] = {}
        for split in SPLITS:
            filename = "validation.npz" if split == "validation" else f"{split}.npz"
            path = directory / filename
            X, y = _read_window_file(path, len(scaler.features), scaler.window_days)
            split_dates = dates_by_split[split]
            if len(split_dates) != y.size:
                raise ValueError(
                    f"Building {building_id} {split} has {y.size} samples but the "
                    f"official split has {len(split_dates)} target dates"
                )
            # Phase 5 NPZs omit dates; preprocessing writes rows in official split order.
            splits[split] = WindowSplit(X=X, y=y, dates=split_dates)
        building_data[building_id] = BuildingData(building_id=building_id, splits=splits)
    return scaler, building_data
