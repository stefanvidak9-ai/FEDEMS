import math

import pytest
import torch
from torch import nn

from src.models import DailyHeatingCNN


def test_model_initializes_with_configurable_feature_count():
    model = DailyHeatingCNN(num_features=10)

    assert isinstance(model, nn.Module)
    assert model.num_features == 10
    assert model.feature_extractor[0].in_channels == 10


def test_output_dimensions_for_example_batch():
    model = DailyHeatingCNN(num_features=10)
    predictions = model(torch.randn(32, 10, 7, dtype=torch.float32))

    assert predictions.shape == (32, 1)


def test_single_sample_inference():
    model = DailyHeatingCNN(num_features=10)
    predictions = model(torch.randn(1, 10, 7, dtype=torch.float32))

    assert predictions.shape == (1, 1)


@pytest.mark.parametrize("num_features", [1, 4, 17])
def test_alternative_feature_counts(num_features):
    model = DailyHeatingCNN(num_features=num_features)
    predictions = model(torch.randn(3, num_features, 7, dtype=torch.float32))

    assert predictions.shape == (3, 1)
    assert torch.isfinite(predictions).all()


def test_forward_pass_is_finite_and_runs_on_cpu():
    model = DailyHeatingCNN(num_features=10).cpu().eval()
    inputs = torch.randn(8, 10, 7, dtype=torch.float32, device="cpu")

    with torch.no_grad():
        predictions = model(inputs)

    assert predictions.shape == (8, 1)
    assert predictions.device.type == "cpu"
    assert torch.isfinite(predictions).all()


def test_backward_pass_produces_finite_parameter_gradients():
    model = DailyHeatingCNN(num_features=6)
    inputs = torch.randn(8, 6, 7, dtype=torch.float32)
    synthetic_targets = torch.randn(8, 1, dtype=torch.float32)

    loss = nn.functional.mse_loss(model(inputs), synthetic_targets)
    loss.backward()

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable_parameters
    assert all(parameter.grad is not None for parameter in trainable_parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable_parameters)


def test_parameter_count_and_float32_storage_estimate():
    model = DailyHeatingCNN(num_features=10)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    assert parameter_count == 2097
    assert parameter_count * torch.tensor([], dtype=torch.float32).element_size() == 8388
    assert math.isclose(parameter_count * 4 / 1024, 8.19140625)


def test_state_dict_save_and_load_preserves_predictions(tmp_path):
    model = DailyHeatingCNN(num_features=10).eval()
    inputs = torch.randn(5, 10, 7, dtype=torch.float32)
    with torch.no_grad():
        expected = model(inputs)

    checkpoint_path = tmp_path / "daily_heating_cnn_state.pt"
    torch.save(model.state_dict(), checkpoint_path)
    restored = DailyHeatingCNN(num_features=10).eval()
    restored.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))

    with torch.no_grad():
        actual = restored(inputs)
    torch.testing.assert_close(actual, expected)


def test_input_shape_feature_count_and_dtype_are_validated():
    model = DailyHeatingCNN(num_features=3)

    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(3, 7))
    with pytest.raises(ValueError, match="input features"):
        model(torch.randn(2, 4, 7))
    with pytest.raises(TypeError, match="torch.float32"):
        model(torch.randn(2, 3, 7, dtype=torch.float64))