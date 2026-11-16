"""Check CPU-only CNN inference, model-state transfer, and optional Flower."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys

import numpy as np
import torch

from src.federated.parameters import load_model_parameters, model_parameters
from src.models import DailyHeatingCNN
from src.training.engine import load_checkpoint


def verify_environment(checkpoint_path: str | None = None) -> dict[str, object]:
    checkpoint_metadata = None
    if checkpoint_path:
        model, checkpoint_metadata = load_checkpoint(checkpoint_path)
        if model.num_features != 4:
            raise ValueError("Phase 6 checkpoint must contain the four-feature CNN")
    else:
        model = DailyHeatingCNN(num_features=4).cpu()
    model.eval()
    clone = DailyHeatingCNN(num_features=4).cpu().eval()
    parameters = model_parameters(model)
    load_model_parameters(clone, parameters)
    sample = torch.zeros((2, 4, 7), dtype=torch.float32)
    with torch.inference_mode():
        expected = model(sample)
        actual = clone(sample)
    torch.testing.assert_close(actual, expected)
    flower_version: str | None
    try:
        flower_version = importlib.metadata.version("flwr")
    except importlib.metadata.PackageNotFoundError:
        flower_version = None
    report: dict[str, object] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "architecture": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "flower": flower_version,
        "device": "cpu",
        "input_shape": list(sample.shape),
        "output_shape": list(actual.shape),
        "model_parameter_arrays": len(parameters),
        "model_serialization_round_trip": "passed",
        "flower_status": "installed" if flower_version else "not-installed",
        "checkpoint_loaded": checkpoint_path is not None,
    }
    if checkpoint_path:
        report["checkpoint_path"] = checkpoint_path
        report["checkpoint_best_epoch"] = checkpoint_metadata["best_epoch"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        help="Optional trusted Step 32–34 best_model.pt to load before inference",
    )
    args = parser.parse_args()
    report = verify_environment(args.checkpoint)
    print(json.dumps(report, indent=2))
    return 0 if report["flower"] else 1


if __name__ == "__main__":
    sys.exit(main())
