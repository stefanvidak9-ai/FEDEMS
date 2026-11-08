"""Compact 1D-CNN architecture for next-day heating-energy regression."""

from __future__ import annotations

import torch
from torch import nn


class DailyHeatingCNN(nn.Module):
    """Predict one target-day heating value from channel-first daily history.

    Args:
        num_features: Number of input features/channels per day.
        conv1_channels: Output channels in the first temporal convolution.
        conv2_channels: Output channels in the second temporal convolution.
        kernel_size: Odd temporal convolution width; padding preserves length.

    Input tensors have shape ``[batch_size, num_features, sequence_length]``
    and dtype ``torch.float32``. The output has shape ``[batch_size, 1]``.
    Adaptive pooling allows the sequence length to vary; FEDEMS initial use is
    seven days.
    """

    def __init__(
        self,
        num_features: int,
        conv1_channels: int = 16,
        conv2_channels: int = 32,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if num_features < 1:
            raise ValueError("num_features must be a positive integer")
        if conv1_channels < 1 or conv2_channels < 1:
            raise ValueError("convolution channel counts must be positive integers")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.num_features = num_features
        padding = kernel_size // 2
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(num_features, conv1_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(conv1_channels, conv2_channels, kernel_size, padding=padding),
            nn.ReLU(),
        )
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.regression_head = nn.Linear(conv2_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one linear regression prediction for each input sample."""
        if inputs.ndim != 3:
            raise ValueError(
                "inputs must have shape [batch_size, num_features, sequence_length]"
            )
        if inputs.shape[1] != self.num_features:
            raise ValueError(
                f"expected {self.num_features} input features, got {inputs.shape[1]}"
            )
        if inputs.shape[0] < 1 or inputs.shape[2] < 1:
            raise ValueError("batch size and sequence length must be positive")
        if inputs.dtype != torch.float32:
            raise TypeError("inputs must have dtype torch.float32")

        features = self.feature_extractor(inputs)
        pooled = self.global_average_pool(features).squeeze(-1)
        return self.regression_head(pooled)