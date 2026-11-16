"""Reusable federated-learning infrastructure for FEDEMS."""

from src.federated.parameters import (
    ClientUpdate,
    AggregatedUpdate,
    aggregate_client_updates,
    load_model_parameters,
    model_parameters,
)

__all__ = [
    "AggregatedUpdate",
    "ClientUpdate",
    "aggregate_client_updates",
    "load_model_parameters",
    "model_parameters",
]
