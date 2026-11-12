import csv
import json

import numpy as np
import pytest
import torch

from src.evaluation import regression_metrics
from src.training.data import load_buildings
from src.training.engine import TrainingConfig, create_model, load_checkpoint, train_model
from src.training.runner import run_step34


def _write_phase5_fixture(root, building_ids=(6479,)):
    processed = root / "processed"
    processed.mkdir()
    metadata = {
        "features": [
            "HtSpace_kWh_day",
            "Tout_mean_C",
            "SolGlob_kWh_m2_day",
            "WindSpd_mean_m_s",
        ],
        "target": "HtSpace_kWh_day",
        "window_days": 7,
        "forecast_horizon_days": 1,
        "normalization": "global training-only z-score",
        "feature_mean": [100.0, 0.0, 0.0, 0.0],
        "feature_std": [20.0, 1.0, 1.0, 1.0],
        "target_mean": 100.0,
        "target_std": 20.0,
    }
    (processed / "global_scaler.json").write_text(json.dumps(metadata), encoding="utf-8")
    split_file = root / "split.csv"
    with split_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["date", "split"])
        writer.writerows(
            [
                ["2022-01-01", "train"],
                ["2022-01-02", "train"],
                ["2022-01-03", "val"],
                ["2022-01-04", "test"],
            ]
        )
    for building_id in building_ids:
        building_dir = processed / f"building_{building_id}"
        building_dir.mkdir()
        for filename, count in (("train.npz", 2), ("validation.npz", 1), ("test.npz", 1)):
            X = np.zeros((count, 4, 7), dtype=np.float32)
            y = np.zeros(count, dtype=np.float32)
            np.savez_compressed(building_dir / filename, X=X, y=y)
    return processed, split_file


def test_regression_metrics_pool_squared_errors_in_original_units():
    metrics = regression_metrics([0.0, 2.0], [1.0, 4.0])

    assert metrics["n"] == 2
    assert metrics["mae_kwh"] == pytest.approx(1.5)
    assert metrics["rmse_kwh"] == pytest.approx(np.sqrt(2.5))


def test_phase5_loader_preserves_split_dates_and_expected_tensor_shape(tmp_path):
    processed, split_file = _write_phase5_fixture(tmp_path)

    scaler, buildings = load_buildings(processed, split_file, [6479])

    assert scaler.features == (
        "HtSpace_kWh_day",
        "Tout_mean_C",
        "SolGlob_kWh_m2_day",
        "WindSpd_mean_m_s",
    )
    assert buildings[6479].splits["train"].X.shape == (2, 4, 7)
    assert buildings[6479].splits["validation"].dates == ("2022-01-03",)
    assert buildings[6479].splits["test"].dates == ("2022-01-04",)


def test_phase5_loader_rejects_misaligned_sample_counts(tmp_path):
    processed, split_file = _write_phase5_fixture(tmp_path)
    np.savez_compressed(
        processed / "building_6479" / "test.npz",
        X=np.zeros((2, 4, 7), dtype=np.float32),
        y=np.zeros(2, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="official split"):
        load_buildings(processed, split_file, [6479])


def test_training_decreases_loss_and_checkpoint_reload_matches(tmp_path):
    torch.manual_seed(4)
    rng = np.random.default_rng(4)
    X = rng.normal(size=(64, 2, 7)).astype(np.float32)
    y = (0.5 * X[:, 0, -1] - 0.2 * X[:, 1, 0]).astype(np.float32)
    checkpoint_path = tmp_path / "best_model.pt"

    result = train_model(
        X,
        y,
        X[:16],
        y[:16],
        target_mean=100.0,
        target_std=20.0,
        feature_names=("HtSpace_kWh_day", "Tout_mean_C"),
        target_name="HtSpace_kWh_day",
        window_days=7,
        config=TrainingConfig(epochs=5, batch_size=16, learning_rate=0.01, patience=5, seed=3),
        checkpoint_path=checkpoint_path,
    )
    restored, checkpoint = load_checkpoint(checkpoint_path)
    inputs = torch.from_numpy(X[:4])

    with torch.inference_mode():
        expected = result["model"](inputs)
        actual = restored(inputs)
    torch.testing.assert_close(actual, expected)
    assert checkpoint["best_epoch"] >= 1
    assert all(np.isfinite(row["train_loss_normalized_mse"]) for row in result["history"])
    assert result["history"][-1]["train_loss_normalized_mse"] < result["history"][0]["train_loss_normalized_mse"]


def test_model_factory_uses_step31_cnn_with_configured_feature_count():
    model = create_model(4)

    assert model.num_features == 4
    assert model(torch.zeros((3, 4, 7), dtype=torch.float32)).shape == (3, 1)


def test_step34_rejects_training_settings_that_differ_from_step33(tmp_path):
    processed, split_file = _write_phase5_fixture(tmp_path)
    output_dir = tmp_path / "results"
    central_dir = output_dir / "step33"
    central_dir.mkdir(parents=True)
    central_config = vars(TrainingConfig())
    central_config["seed"] = 13
    (central_dir / "metrics.json").write_text(
        json.dumps({"building_ids": [6479], "training_config": central_config}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="training settings must match Step 33"):
        run_step34(
            [6479],
            processed_dir=processed,
            split_file=split_file,
            output_dir=output_dir,
            config=TrainingConfig(),
        )


def test_step34_rejects_different_phase5_artifacts_from_step33(tmp_path):
    processed, split_file = _write_phase5_fixture(tmp_path)
    output_dir = tmp_path / "results"
    central_dir = output_dir / "step33"
    central_dir.mkdir(parents=True)
    metadata = json.loads((processed / "global_scaler.json").read_text(encoding="utf-8"))
    central_model_config = {
        "architecture": "DailyHeatingCNN (Step 31)",
        "num_features": 4,
        "feature_names": metadata["features"],
        "input_shape_per_sample": [4, 7],
        "conv1_channels": 16,
        "conv2_channels": 32,
        "kernel_size": 3,
    }
    (central_dir / "metrics.json").write_text(
        json.dumps(
            {
                "building_ids": [6479],
                "training_config": vars(TrainingConfig()),
                "model_config": central_model_config,
                "phase5_artifact_sha256": {"global_scaler.json": "different-data"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identical Phase 5 scaler, split"):
        run_step34(
            [6479],
            processed_dir=processed,
            split_file=split_file,
            output_dir=output_dir,
            config=TrainingConfig(),
        )


def test_phase5_loader_rejects_feature_order_that_differs_from_phase4(tmp_path):
    processed, _ = _write_phase5_fixture(tmp_path)
    scaler_path = processed / "global_scaler.json"
    metadata = json.loads(scaler_path.read_text(encoding="utf-8"))
    metadata["features"][0], metadata["features"][1] = (
        metadata["features"][1],
        metadata["features"][0],
    )
    scaler_path.write_text(json.dumps(metadata), encoding="utf-8")

    from src.training.data import load_scaler

    with pytest.raises(ValueError, match="feature order must match Phase 4/5"):
        load_scaler(processed)
