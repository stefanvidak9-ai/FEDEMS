"""Deterministic CPU training and checkpoint utilities for the Step 31 CNN."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import regression_metrics
from src.models import DailyHeatingCNN


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    patience: int = 15
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and the CPU DataLoader."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def create_model(num_features: int) -> DailyHeatingCNN:
    """Instantiate the established Step 31 architecture on CPU."""
    return DailyHeatingCNN(num_features=num_features).cpu()


def _validated_tensors(X: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = np.asarray(X, dtype=np.float32)
    targets = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    if inputs.ndim != 3 or inputs.shape[0] == 0 or inputs.shape[1] == 0 or inputs.shape[2] == 0:
        raise ValueError("Inputs must have shape [non-empty samples, features, time]")
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("Input and target sample counts must match")
    if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise ValueError("Inputs and targets must be finite")
    return torch.from_numpy(inputs.copy()), torch.from_numpy(targets.copy())


def _predict_normalized(model: DailyHeatingCNN, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    inputs, _ = _validated_tensors(X, np.zeros(len(X), dtype=np.float32))
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in inputs.split(batch_size):
            predictions.append(model(batch).squeeze(1).cpu().numpy())
    return np.concatenate(predictions).astype(np.float64, copy=False)


def predict_kwh(
    model: DailyHeatingCNN,
    X: np.ndarray,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    """Return one-day-ahead predictions in original kWh/day units."""
    normalized = _predict_normalized(model, X)
    predictions = normalized * target_std + target_mean
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions")
    return predictions


def load_checkpoint(
    checkpoint_path: str | Path,
    model: DailyHeatingCNN | None = None,
) -> tuple[DailyHeatingCNN, dict[str, Any]]:
    """Load a saved CNN and its scale/configuration metadata on CPU."""
    path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Cannot load checkpoint {path}: {error}") from error
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported or invalid checkpoint format at {path}")

    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict) or not isinstance(model_config.get("num_features"), int):
        raise ValueError(f"Checkpoint {path} is missing model configuration")
    if model is None:
        model = create_model(model_config["num_features"])
    if model.num_features != model_config["num_features"]:
        raise ValueError("Provided model feature count does not match checkpoint")
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise ValueError(f"Checkpoint weights are incompatible with the model: {error}") from error
    return model.cpu(), checkpoint


def train_model(
    train_X: np.ndarray,
    train_y: np.ndarray,
    validation_X: np.ndarray,
    validation_y: np.ndarray,
    *,
    target_mean: float,
    target_std: float,
    feature_names: tuple[str, ...],
    target_name: str,
    window_days: int,
    config: TrainingConfig,
    checkpoint_path: str | Path,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train with normalized MSE and select the best checkpoint by val RMSE."""
    train_inputs, train_targets = _validated_tensors(train_X, train_y)
    validation_inputs, validation_targets = _validated_tensors(validation_X, validation_y)
    if train_inputs.shape[1:] != validation_inputs.shape[1:]:
        raise ValueError("Training and validation input shapes must match")
    if len(feature_names) != train_inputs.shape[1] or window_days != train_inputs.shape[2]:
        raise ValueError("Feature metadata does not match the input tensor shape")
    if not np.isfinite(target_mean) or not np.isfinite(target_std) or target_std <= 0:
        raise ValueError("Target scaler must have finite mean and positive standard deviation")

    set_random_seed(config.seed)
    model = create_model(train_inputs.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(train_inputs, train_targets),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    started_at = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        observed = 0
        for batch_inputs, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            if any(
                parameter.grad is None or not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.requires_grad
            ):
                raise ValueError(f"Invalid gradient at epoch {epoch}")
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_targets)
            observed += len(batch_targets)

        validation_predictions = _predict_normalized(
            model, validation_inputs.numpy()
        ) * target_std + target_mean
        validation_actual = validation_targets.squeeze(1).numpy() * target_std + target_mean
        metrics = regression_metrics(validation_actual, validation_predictions)
        epoch_loss = total_loss / observed
        if not np.isfinite(epoch_loss):
            raise ValueError(f"Non-finite training loss at epoch {epoch}")
        history.append(
            {
                "epoch": epoch,
                "train_loss_normalized_mse": epoch_loss,
                "validation_mae_kwh": metrics["mae_kwh"],
                "validation_rmse_kwh": metrics["rmse_kwh"],
            }
        )

        if metrics["rmse_kwh"] < best_rmse:
            best_rmse = metrics["rmse_kwh"]
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = {
                "format_version": 1,
                "model_config": {"num_features": len(feature_names)},
                "model_state_dict": model.state_dict(),
                "feature_names": list(feature_names),
                "target_name": target_name,
                "window_days": window_days,
                "target_mean": float(target_mean),
                "target_std": float(target_std),
                "training_config": asdict(config),
                "best_epoch": best_epoch,
                "best_validation_mae_kwh": metrics["mae_kwh"],
                "best_validation_rmse_kwh": metrics["rmse_kwh"],
                "run_metadata": run_metadata or {},
            }
            torch.save(checkpoint, path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    duration_seconds = time.perf_counter() - started_at
    best_model, best_checkpoint = load_checkpoint(path)
    return {
        "model": best_model,
        "checkpoint": best_checkpoint,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_mae_kwh": float(best_checkpoint["best_validation_mae_kwh"]),
        "best_validation_rmse_kwh": float(best_checkpoint["best_validation_rmse_kwh"]),
        "duration_seconds": duration_seconds,
        "checkpoint_path": str(path),
    }
