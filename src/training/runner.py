"""Step 32–34 experiment orchestration and artifact writing."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.baselines.evaluate import evaluate_forecasts
from src.evaluation import regression_metrics
from src.training.data import BuildingData, Scaler, WindowSplit, load_buildings
from src.training.engine import TrainingConfig, predict_kwh, train_model

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILDINGS = (6479, 6480, 6481, 6482)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    _write_csv(
        path,
        ("epoch", "train_loss_normalized_mse", "validation_mae_kwh", "validation_rmse_kwh"),
        history,
    )


def _prediction_rows(
    building_id: int,
    split_name: str,
    split: WindowSplit,
    predicted_kwh: np.ndarray,
    scaler: Scaler,
) -> list[dict[str, Any]]:
    actual_kwh = scaler.inverse_target(split.y)
    if len(split.dates) != len(actual_kwh) or len(actual_kwh) != len(predicted_kwh):
        raise ValueError(f"Prediction and target alignment failed for building {building_id}")
    return [
        {
            "building_id": building_id,
            "split": split_name,
            "target_date": target_date,
            "actual_kwh": float(actual),
            "predicted_kwh": float(predicted),
        }
        for target_date, actual, predicted in zip(
            split.dates, actual_kwh, predicted_kwh, strict=True
        )
    ]


def _evaluate_split(
    model: Any,
    building_id: int,
    split_name: str,
    split: WindowSplit,
    scaler: Scaler,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    predicted_kwh = predict_kwh(model, split.X, scaler.target_mean, scaler.target_std)
    rows = _prediction_rows(building_id, split_name, split, predicted_kwh, scaler)
    metrics = regression_metrics(
        [row["actual_kwh"] for row in rows],
        [row["predicted_kwh"] for row in rows],
    )
    return metrics, rows


def _baseline_rows(
    buildings: dict[int, BuildingData],
    scaler: Scaler,
) -> list[dict[str, Any]]:
    target_index = scaler.features.index(scaler.target)
    rows: list[dict[str, Any]] = []
    for building_id, building in buildings.items():
        for split_name in ("validation", "test"):
            split = building.splits[split_name]
            actual_kwh = scaler.inverse_target(split.y)
            inputs_kwh = (
                split.X[:, target_index, :].astype(np.float64)
                * scaler.feature_std[target_index]
                + scaler.feature_mean[target_index]
            )
            for target_date, actual, history in zip(
                split.dates, actual_kwh, inputs_kwh, strict=True
            ):
                rows.append(
                    {
                        "building_id": str(building_id),
                        "target_date": target_date,
                        "actual_kwh": float(actual),
                        "lag1_kwh": float(history[-1]),
                        "lag7_kwh": float(history[0]),
                        "split": split_name,
                    }
                )
    return rows


def _comparison_rows(
    prediction_rows: list[dict[str, Any]],
    baseline_input_rows: list[dict[str, Any]],
    cnn_name: str = "cnn",
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for result in evaluate_forecasts(baseline_input_rows):
        results.append(
            {
                "model": result["baseline"],
                "split": result["split"],
                "scope": result["scope"],
                "building_id": result["building_id"],
                "n": result["n"],
                "mae_kwh": result["mae_kwh"],
                "rmse_kwh": result["rmse_kwh"],
            }
        )
    for split_name in ("validation", "test"):
        split_rows = [row for row in prediction_rows if row["split"] == split_name]
        building_ids = sorted({row["building_id"] for row in split_rows})
        for building_id in building_ids:
            building_rows = [row for row in split_rows if row["building_id"] == building_id]
            metrics = regression_metrics(
                [row["actual_kwh"] for row in building_rows],
                [row["predicted_kwh"] for row in building_rows],
            )
            results.append(
                {
                    "model": cnn_name,
                    "split": split_name,
                    "scope": "building",
                    "building_id": building_id,
                    **metrics,
                }
            )
        pooled = regression_metrics(
            [row["actual_kwh"] for row in split_rows],
            [row["predicted_kwh"] for row in split_rows],
        )
        results.append(
            {
                "model": cnn_name,
                "split": split_name,
                "scope": "aggregate",
                "building_id": "",
                **pooled,
            }
        )
    return results


def _fit(
    train_X: np.ndarray,
    train_y: np.ndarray,
    validation_X: np.ndarray,
    validation_y: np.ndarray,
    scaler: Scaler,
    config: TrainingConfig,
    checkpoint_path: Path,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    return train_model(
        train_X,
        train_y,
        validation_X,
        validation_y,
        target_mean=scaler.target_mean,
        target_std=scaler.target_std,
        feature_names=scaler.features,
        target_name=scaler.target,
        window_days=scaler.window_days,
        config=config,
        checkpoint_path=checkpoint_path,
        run_metadata=run_metadata,
    )


def _evaluate_building(
    model: Any,
    building: BuildingData,
    scaler: Scaler,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_split: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for split_name in ("validation", "test"):
        metrics, rows = _evaluate_split(
            model,
            building.building_id,
            split_name,
            building.splits[split_name],
            scaler,
        )
        per_split[split_name] = metrics
        all_rows.extend(rows)
    return per_split, all_rows


def _aggregate_metrics(per_building: dict[int, dict[str, Any]], split_name: str) -> dict[str, Any]:
    metrics_by_building = [values[split_name] for values in per_building.values()]
    counts = [int(item["n"]) for item in metrics_by_building]
    total = sum(counts)
    if total == 0:
        raise ValueError(f"No {split_name} observations are available for aggregate metrics")
    mae = sum(float(item["mae_kwh"]) * item["n"] for item in metrics_by_building) / total
    squared_error = sum(float(item["rmse_kwh"]) ** 2 * item["n"] for item in metrics_by_building)
    rmse = math.sqrt(squared_error / total)
    return {"n": total, "mae_kwh": mae, "rmse_kwh": rmse}


def _save_experiment(
    run_dir: Path,
    summary: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    history_by_building: dict[int | str, list[dict[str, Any]]],
) -> None:
    _write_json(run_dir / "metrics.json", summary)
    _write_csv(
        run_dir / "predictions.csv",
        ("building_id", "split", "target_date", "actual_kwh", "predicted_kwh"),
        prediction_rows,
    )
    _write_csv(
        run_dir / "comparison.csv",
        ("model", "split", "scope", "building_id", "n", "mae_kwh", "rmse_kwh"),
        comparison,
    )
    for building_id, history in history_by_building.items():
        history_name = (
            "history_centralized.csv"
            if building_id == "centralized"
            else f"history_building_{building_id}.csv"
        )
        _write_history(run_dir / history_name, history)


def _model_config(scaler: Scaler) -> dict[str, Any]:
    return {
        "architecture": "DailyHeatingCNN (Step 31)",
        "num_features": len(scaler.features),
        "feature_names": list(scaler.features),
        "input_shape_per_sample": [len(scaler.features), scaler.window_days],
        "conv1_channels": 16,
        "conv2_channels": 32,
        "kernel_size": 3,
    }


def _data_fingerprint(
    processed_dir: Path,
    split_file: Path,
    building_ids: list[int] | tuple[int, ...],
) -> dict[str, str]:
    """Fingerprint the exact Phase 5 artifacts used by an experiment."""
    paths = {
        "global_scaler.json": Path(processed_dir) / "global_scaler.json",
        "target_split.csv": Path(split_file),
    }
    for building_id in sorted(building_ids):
        directory = Path(processed_dir) / f"building_{building_id}"
        paths.update(
            {
                f"building_{building_id}/{filename}": directory / filename
                for filename in ("train.npz", "validation.npz", "test.npz")
            }
        )
    fingerprints: dict[str, str] = {}
    for name, path in paths.items():
        try:
            fingerprints[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError(f"Cannot fingerprint Phase 5 artifact {path}: {error}") from error
    return fingerprints


def _require_matching_step33(
    central_summary: dict[str, Any],
    building_ids: list[int],
    config: TrainingConfig,
    scaler: Scaler,
    data_fingerprint: dict[str, str],
) -> None:
    if set(central_summary.get("building_ids", [])) != set(building_ids):
        raise ValueError(
            "Step 34 building IDs must match Step 33 exactly for a valid comparison"
        )
    expected_config = vars(config)
    central_config = central_summary.get("training_config", {})
    if any(central_config.get(key) != value for key, value in expected_config.items()):
        raise ValueError(
            "Step 34 training settings must match Step 33 (epochs, batch size, "
            "learning rate, patience, and seed) for a valid comparison"
        )
    if central_summary.get("model_config") != _model_config(scaler):
        raise ValueError("Step 33 and Step 34 must use the same CNN and feature contract")
    if central_summary.get("phase5_artifact_sha256") != data_fingerprint:
        raise ValueError(
            "Step 33 and Step 34 must use identical Phase 5 scaler, split, and "
            "selected-building window files"
        )


def run_step32(
    building_id: int,
    *,
    processed_dir: Path,
    split_file: Path,
    output_dir: Path,
    config: TrainingConfig,
) -> dict[str, Any]:
    scaler, buildings = load_buildings(processed_dir, split_file, [building_id])
    building = buildings[building_id]
    run_dir = output_dir / f"step32_building_{building_id}"
    checkpoint_path = run_dir / "best_model.pt"
    fit = _fit(
        building.splits["train"].X,
        building.splits["train"].y,
        building.splits["validation"].X,
        building.splits["validation"].y,
        scaler,
        config,
        checkpoint_path,
        {"step": "step32", "building_id": building_id},
    )
    per_building, prediction_rows = _evaluate_building(fit["model"], building, scaler)
    summary = {
        "step": "step32",
        "building_ids": [building_id],
        "sample_counts": {
            split: int(len(building.splits[split].y)) for split in ("train", "validation", "test")
        },
        "training_config": {
            **vars(config),
            "optimizer": "Adam",
            "loss": "MSE on Phase 5 normalized targets",
            "checkpoint_selection": "lowest validation RMSE in kWh/day",
        },
        "model_config": _model_config(scaler),
        "phase5_artifact_sha256": _data_fingerprint(
            processed_dir, split_file, [building_id]
        ),
        "duration_seconds": fit["duration_seconds"],
        "best_epoch": fit["best_epoch"],
        "best_validation_mae_kwh": fit["best_validation_mae_kwh"],
        "best_validation_rmse_kwh": fit["best_validation_rmse_kwh"],
        "per_building": {
            str(building_id): {
                "training_samples": int(len(building.splits["train"].y)),
                "validation_samples": int(len(building.splits["validation"].y)),
                "test_samples": int(len(building.splits["test"].y)),
                **per_building,
                "checkpoint": str(checkpoint_path),
            }
        },
        "aggregate": {
            split: per_building[split] for split in ("validation", "test")
        },
    }
    comparison = _comparison_rows(
        prediction_rows,
        _baseline_rows(buildings, scaler),
        cnn_name="step32_cnn",
    )
    _save_experiment(
        run_dir,
        summary,
        prediction_rows,
        comparison,
        {building_id: fit["history"]},
    )
    return summary


def run_step33(
    building_ids: list[int],
    *,
    processed_dir: Path,
    split_file: Path,
    output_dir: Path,
    config: TrainingConfig,
) -> dict[str, Any]:
    scaler, buildings = load_buildings(processed_dir, split_file, building_ids)
    train_X = np.concatenate([item.splits["train"].X for item in buildings.values()])
    train_y = np.concatenate([item.splits["train"].y for item in buildings.values()])
    validation_X = np.concatenate([item.splits["validation"].X for item in buildings.values()])
    validation_y = np.concatenate([item.splits["validation"].y for item in buildings.values()])
    run_dir = output_dir / "step33"
    checkpoint_path = run_dir / "best_model.pt"
    fit = _fit(
        train_X,
        train_y,
        validation_X,
        validation_y,
        scaler,
        config,
        checkpoint_path,
        {"step": "step33", "building_ids": building_ids},
    )

    per_building: dict[int, dict[str, Any]] = {}
    prediction_rows: list[dict[str, Any]] = []
    for building_id, building in buildings.items():
        per_split, rows = _evaluate_building(fit["model"], building, scaler)
        per_building[building_id] = {
            "training_samples": int(len(building.splits["train"].y)),
            "validation_samples": int(len(building.splits["validation"].y)),
            "test_samples": int(len(building.splits["test"].y)),
            **per_split,
            "checkpoint": str(checkpoint_path),
        }
        prediction_rows.extend(rows)
    summary = {
        "step": "step33",
        "building_ids": building_ids,
        "training_samples": int(len(train_y)),
        "validation_samples": int(len(validation_y)),
        "training_config": {
            **vars(config),
            "optimizer": "Adam",
            "loss": "MSE on Phase 5 normalized targets",
            "checkpoint_selection": "lowest pooled validation RMSE in kWh/day",
        },
        "model_config": _model_config(scaler),
        "phase5_artifact_sha256": _data_fingerprint(
            processed_dir, split_file, building_ids
        ),
        "duration_seconds": fit["duration_seconds"],
        "best_epoch": fit["best_epoch"],
        "best_validation_mae_kwh": fit["best_validation_mae_kwh"],
        "best_validation_rmse_kwh": fit["best_validation_rmse_kwh"],
        "per_building": {str(key): value for key, value in per_building.items()},
        "aggregate": {
            split: _aggregate_metrics(per_building, split)
            for split in ("validation", "test")
        },
        "checkpoint": str(checkpoint_path),
    }
    comparison = _comparison_rows(
        prediction_rows,
        _baseline_rows(buildings, scaler),
        cnn_name="centralized_cnn",
    )
    _save_experiment(
        run_dir,
        summary,
        prediction_rows,
        comparison,
        {"centralized": fit["history"]},
    )
    return summary


def run_step34(
    building_ids: list[int],
    *,
    processed_dir: Path,
    split_file: Path,
    output_dir: Path,
    config: TrainingConfig,
) -> dict[str, Any]:
    central_metrics_path = output_dir / "step33" / "metrics.json"
    try:
        central_summary = json.loads(central_metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Run Step 33 first; expected centralized metrics at {central_metrics_path}"
        ) from error
    scaler, buildings = load_buildings(processed_dir, split_file, building_ids)
    data_fingerprint = _data_fingerprint(processed_dir, split_file, building_ids)
    _require_matching_step33(
        central_summary,
        building_ids,
        config,
        scaler,
        data_fingerprint,
    )
    run_dir = output_dir / "step34"
    per_building: dict[int, dict[str, Any]] = {}
    prediction_rows: list[dict[str, Any]] = []
    history_by_building: dict[int, list[dict[str, Any]]] = {}
    total_duration = 0.0

    for building_id, building in buildings.items():
        checkpoint_path = run_dir / f"building_{building_id}" / "best_model.pt"
        fit = _fit(
            building.splits["train"].X,
            building.splits["train"].y,
            building.splits["validation"].X,
            building.splits["validation"].y,
            scaler,
            config,
            checkpoint_path,
            {"step": "step34", "building_id": building_id},
        )
        total_duration += fit["duration_seconds"]
        history_by_building[building_id] = fit["history"]
        per_split, rows = _evaluate_building(fit["model"], building, scaler)
        per_building[building_id] = {
            "training_samples": int(len(building.splits["train"].y)),
            "validation_samples": int(len(building.splits["validation"].y)),
            "test_samples": int(len(building.splits["test"].y)),
            "duration_seconds": fit["duration_seconds"],
            "best_epoch": fit["best_epoch"],
            "best_validation_mae_kwh": fit["best_validation_mae_kwh"],
            "best_validation_rmse_kwh": fit["best_validation_rmse_kwh"],
            **per_split,
            "training_config": vars(config),
            "checkpoint": str(checkpoint_path),
        }
        prediction_rows.extend(rows)

    summary = {
        "step": "step34",
        "building_ids": building_ids,
        "training_config": {
            **vars(config),
            "optimizer": "Adam",
            "loss": "MSE on Phase 5 normalized targets",
            "checkpoint_selection": "lowest client-specific validation RMSE in kWh/day",
            "parameter_exchange": False,
        },
        "model_config": _model_config(scaler),
        "phase5_artifact_sha256": data_fingerprint,
        "duration_seconds": total_duration,
        "per_building": {str(key): value for key, value in per_building.items()},
        "aggregate": {
            split: _aggregate_metrics(per_building, split)
            for split in ("validation", "test")
        },
    }
    comparison = _comparison_rows(
        prediction_rows,
        _baseline_rows(buildings, scaler),
        cnn_name="local_cnn",
    )
    for split_name in ("validation", "test"):
        central_metrics = central_summary["aggregate"][split_name]
        comparison.append(
            {
                "model": "centralized_cnn",
                "split": split_name,
                "scope": "aggregate",
                "building_id": "",
                **central_metrics,
            }
        )
        for building_id in building_ids:
            central_building = central_summary["per_building"][str(building_id)][split_name]
            comparison.append(
                {
                    "model": "centralized_cnn",
                    "split": split_name,
                    "scope": "building",
                    "building_id": building_id,
                    **central_building,
                }
            )
        comparison.append(
            {
                "model": "local_cnn",
                "split": split_name,
                "scope": "aggregate",
                "building_id": "",
                **summary["aggregate"][split_name],
            }
        )
    _save_experiment(run_dir, summary, prediction_rows, comparison, history_by_building)
    return summary
