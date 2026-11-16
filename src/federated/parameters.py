"""Safe model-parameter serialization and weighted aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ClientUpdate:
    client_id: str
    num_examples: int
    parameters: tuple[np.ndarray, ...]
    metrics: dict[str, float]


@dataclass(frozen=True)
class AggregatedUpdate:
    parameters: tuple[np.ndarray, ...]
    num_examples: int
    metrics: dict[str, float]


def model_parameters(model: nn.Module) -> tuple[np.ndarray, ...]:
    """Return a detached, copied NumPy representation of a model state."""
    return tuple(
        tensor.detach().cpu().numpy().copy()
        for tensor in model.state_dict().values()
    )


def load_model_parameters(
    model: nn.Module,
    parameters: Sequence[np.ndarray],
) -> None:
    """Load NumPy parameters after validating the complete model-state shape."""
    state = model.state_dict()
    if len(parameters) != len(state):
        raise ValueError(
            f"Expected {len(state)} parameter arrays, received {len(parameters)}"
        )

    updated: dict[str, torch.Tensor] = {}
    for (name, tensor), value in zip(state.items(), parameters, strict=True):
        array = np.asarray(value)
        if array.shape != tuple(tensor.shape):
            raise ValueError(
                f"Parameter {name!r} has shape {array.shape}; "
                f"expected {tuple(tensor.shape)}"
            )
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
            array.dtype, np.complexfloating
        ):
            raise TypeError(f"Parameter {name!r} must have a numeric dtype")
        if not np.isfinite(array).all():
            raise ValueError(f"Parameter {name!r} contains non-finite values")
        converted = torch.as_tensor(
            np.ascontiguousarray(array),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        if converted.is_floating_point() or converted.is_complex():
            if not torch.isfinite(converted).all():
                raise ValueError(
                    f"Parameter {name!r} is not finite in the model's {tensor.dtype} dtype"
                )
        elif tensor.dtype == torch.bool:
            if not np.isin(array, (0, 1)).all():
                raise ValueError(f"Parameter {name!r} cannot be represented as bool")
        else:
            destination = tensor.detach().cpu().numpy().dtype
            bounds = np.iinfo(destination)
            if np.any(array < bounds.min) or np.any(array > bounds.max):
                raise ValueError(
                    f"Parameter {name!r} is outside the model's {tensor.dtype} range"
                )
            if np.issubdtype(array.dtype, np.floating) and not np.equal(
                array, np.trunc(array)
            ).all():
                raise ValueError(
                    f"Parameter {name!r} contains fractional values for integer state"
                )
        updated[name] = converted

    model.load_state_dict(updated, strict=True)


def _validated_update_arrays(
    update: ClientUpdate,
    expected_shapes: tuple[tuple[int, ...], ...] | None,
) -> tuple[tuple[np.ndarray, ...], tuple[tuple[int, ...], ...]]:
    if not isinstance(update.client_id, str) or not update.client_id.strip():
        raise ValueError("Client IDs must not be empty")
    if (
        isinstance(update.num_examples, bool)
        or not isinstance(update.num_examples, (int, np.integer))
        or update.num_examples < 1
    ):
        raise ValueError("Client example counts must be positive integers")
    if not update.parameters:
        raise ValueError(f"Client {update.client_id!r} returned no parameters")

    arrays = tuple(np.asarray(value) for value in update.parameters)
    shapes = tuple(array.shape for array in arrays)
    if expected_shapes is not None and shapes != expected_shapes:
        raise ValueError(
            f"Client {update.client_id!r} returned parameter shapes {shapes}; "
            f"expected {expected_shapes}"
        )
    for array in arrays:
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
            array.dtype, np.complexfloating
        ):
            raise TypeError("Client parameters must have numeric dtypes")
        if not np.isfinite(array).all():
            raise ValueError(
                f"Client {update.client_id!r} returned non-finite parameters"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            converted = array.astype(np.float64)
        if not np.isfinite(converted).all():
            raise ValueError(
                f"Client {update.client_id!r} parameters cannot be represented "
                "as float64 for aggregation"
            )
    for name, value in update.metrics.items():
        if (
            not name.strip()
            or isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
        ):
            raise ValueError(
                f"Client {update.client_id!r} returned an invalid metric"
            )
    return arrays, shapes


def aggregate_client_updates(
    updates: Sequence[ClientUpdate],
    *,
    min_clients: int = 2,
) -> AggregatedUpdate:
    """Compute an example-weighted average, requiring the configured quorum."""
    if min_clients < 1:
        raise ValueError("min_clients must be positive")
    if len(updates) < min_clients:
        raise ValueError(
            f"Expected at least {min_clients} client updates, received {len(updates)}"
        )
    if len({update.client_id for update in updates}) != len(updates):
        raise ValueError("Client update IDs must be unique")

    validated: list[tuple[ClientUpdate, tuple[np.ndarray, ...]]] = []
    expected_shapes: tuple[tuple[int, ...], ...] | None = None
    expected_dtypes: tuple[np.dtype, ...] | None = None
    metric_names: set[str] | None = None
    for update in updates:
        arrays, shapes = _validated_update_arrays(update, expected_shapes)
        expected_shapes = shapes
        dtypes = tuple(array.dtype for array in arrays)
        if expected_dtypes is None:
            expected_dtypes = dtypes
        elif dtypes != expected_dtypes:
            raise ValueError("All clients must return matching parameter dtypes")
        names = set(update.metrics)
        if metric_names is None:
            metric_names = names
        elif names != metric_names:
            raise ValueError("All clients must return the same metric names")
        validated.append((update, arrays))

    total_examples = sum(update.num_examples for update, _ in validated)
    aggregated: list[np.ndarray] = []
    for index in range(len(validated[0][1])):
        with np.errstate(over="ignore", invalid="ignore"):
            averaged = np.average(
                np.stack(
                    [
                        arrays[index].astype(np.float64)
                        for _, arrays in validated
                    ]
                ),
                axis=0,
                weights=[update.num_examples for update, _ in validated],
            )
            converted = averaged.astype(validated[0][1][index].dtype, copy=False)
        if not np.isfinite(converted).all():
            raise ValueError("Aggregated parameters are not finite in the client dtype")
        aggregated.append(converted)
    aggregated_parameters = tuple(aggregated)
    metrics: dict[str, float] = {}
    for name in sorted(metric_names or ()):
        with np.errstate(over="ignore", invalid="ignore"):
            metric = float(
                np.average(
                [update.metrics[name] for update, _ in validated],
                weights=[update.num_examples for update, _ in validated],
                )
            )
        if not np.isfinite(metric):
            raise ValueError(f"Aggregated metric {name!r} is not finite")
        metrics[name] = metric
    return AggregatedUpdate(
        parameters=aggregated_parameters,
        num_examples=total_examples,
        metrics=metrics,
    )
