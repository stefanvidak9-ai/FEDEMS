"""Verify FEDEMS CNN inference and parameter round-trip on a CPU-only Pi."""

from __future__ import annotations

import json
import platform

import torch

from src.federated.parameters import load_model_parameters, model_parameters
from src.models import DailyHeatingCNN


def main() -> int:
    model = DailyHeatingCNN(num_features=4).cpu().eval()
    clone = DailyHeatingCNN(num_features=4).cpu().eval()
    parameters = model_parameters(model)
    load_model_parameters(clone, parameters)
    sample = torch.zeros((2, 4, 7), dtype=torch.float32, device="cpu")
    with torch.inference_mode():
        expected = model(sample)
        actual = clone(sample)
    torch.testing.assert_close(actual, expected)
    if actual.shape != (2, 1) or not torch.isfinite(actual).all():
        raise RuntimeError("CNN produced an invalid output")

    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "architecture": platform.machine(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": "cpu",
                "input_shape": list(sample.shape),
                "output_shape": list(actual.shape),
                "parameter_arrays": len(parameters),
                "parameter_round_trip": "passed",
                "inference": "passed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
