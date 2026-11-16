import csv
import json
import subprocess
from datetime import date, timedelta

import numpy as np
import pytest
import torch

from deployment.inventory import InventoryHost, inspect_host, load_allowlist, write_inventory
from src.federated.allocation import validate_allocation
from src.federated.parameters import (
    ClientUpdate,
    aggregate_client_updates,
    load_model_parameters,
    model_parameters,
)
from src.models import DailyHeatingCNN


def _write_phase5_dataset(root, building_ids):
    processed = root / "processed"
    processed.mkdir()
    features = [
        "HtSpace_kWh_day",
        "Tout_mean_C",
        "SolGlob_kWh_m2_day",
        "WindSpd_mean_m_s",
    ]
    metadata = {
        "features": features,
        "target": "HtSpace_kWh_day",
        "window_days": 7,
        "forecast_horizon_days": 1,
        "normalization": "global training-only z-score",
        "feature_mean": [0.0] * 4,
        "feature_std": [1.0] * 4,
        "target_mean": 0.0,
        "target_std": 1.0,
    }
    (processed / "global_scaler.json").write_text(json.dumps(metadata), encoding="utf-8")
    split_file = root / "split.csv"
    with split_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", "split"])
        start = date(2022, 1, 1)
        for index in range(307):
            split = "train" if index < 203 else "val" if index < 262 else "test"
            writer.writerow([str(start + timedelta(days=index)), split])
    for building_id in building_ids:
        building_dir = processed / f"building_{building_id}"
        building_dir.mkdir()
        for name, count in (("train", 203), ("validation", 59), ("test", 45)):
            np.savez_compressed(
                building_dir / f"{name}.npz",
                X=np.zeros((count, 4, 7), dtype=np.float32),
                y=np.zeros(count, dtype=np.float32),
            )
    return processed, split_file


def _mapping_for_buildings(building_ids):
    return {
        "schema_version": 1,
        "profiles": {
            str(len(building_ids)): [
                {
                    "logical_client_id": f"client-{index + 1:02d}",
                    "physical_pi_id": f"node-{52 + index}",
                    "host_status": "candidate-unverified",
                    "building_id": building_id,
                    "data_path": f"Dataset/processed/building_{building_id}",
                    "expected_samples": {
                        "train": 203,
                        "validation": 59,
                        "test": 45,
                    },
                }
                for index, building_id in enumerate(building_ids)
            ]
        },
    }


def test_model_parameters_round_trip():
    model = DailyHeatingCNN(num_features=4)
    original = model_parameters(model)
    replacement = tuple(array + 0.25 for array in original)

    load_model_parameters(model, replacement)

    for actual, expected in zip(model_parameters(model), replacement, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_model_parameter_loading_rejects_bad_count_shape_and_non_finite_values():
    model = DailyHeatingCNN(num_features=4)
    parameters = model_parameters(model)

    with pytest.raises(ValueError, match="Expected"):
        load_model_parameters(model, parameters[:-1])
    wrong_shape = list(parameters)
    wrong_shape[0] = np.zeros((1,), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        load_model_parameters(model, wrong_shape)
    non_finite = list(parameters)
    non_finite[0] = np.full_like(parameters[0], np.nan)
    with pytest.raises(ValueError, match="non-finite"):
        load_model_parameters(model, non_finite)
    overflow = list(parameters)
    overflow[0] = np.full(parameters[0].shape, 1e300, dtype=np.float64)
    with pytest.raises(ValueError, match="not finite in the model"):
        load_model_parameters(model, overflow)


def test_weighted_aggregation_is_predictable_and_aggregates_metrics():
    result = aggregate_client_updates(
        [
            ClientUpdate("client-a", 1, (np.array([2.0], dtype=np.float32),), {"loss": 4.0}),
            ClientUpdate("client-b", 3, (np.array([6.0], dtype=np.float32),), {"loss": 8.0}),
        ]
    )

    np.testing.assert_allclose(result.parameters[0], [5.0])
    assert result.num_examples == 4
    assert result.metrics["loss"] == pytest.approx(7.0)


def test_aggregation_requires_two_unique_compatible_clients():
    one = ClientUpdate("client-a", 1, (np.array([1.0]),), {"loss": 1.0})
    with pytest.raises(ValueError, match="at least 2"):
        aggregate_client_updates([one])
    with pytest.raises(ValueError, match="unique"):
        aggregate_client_updates([one, one])
    with pytest.raises(ValueError, match="shapes"):
        aggregate_client_updates(
            [
                one,
                ClientUpdate("client-b", 1, (np.array([1.0, 2.0]),), {"loss": 1.0}),
            ]
        )
    with pytest.raises(ValueError, match="same metric names"):
        aggregate_client_updates(
            [
                one,
                ClientUpdate("client-b", 1, (np.array([1.0]),), {"accuracy": 1.0}),
            ]
        )


def test_aggregation_rejects_invalid_counts_and_nan_metrics():
    with pytest.raises(ValueError, match="positive integers"):
        aggregate_client_updates(
            [
                ClientUpdate("client-a", 0, (np.array([1.0]),), {"loss": 1.0}),
                ClientUpdate("client-b", 1, (np.array([1.0]),), {"loss": 1.0}),
            ]
        )
    with pytest.raises(ValueError, match="matching parameter dtypes"):
        aggregate_client_updates(
            [
                ClientUpdate("client-a", 1, (np.array([1.0], dtype=np.float32),), {"loss": 1.0}),
                ClientUpdate("client-b", 1, (np.array([1.0], dtype=np.float64),), {"loss": 1.0}),
            ]
        )
    with pytest.raises(ValueError, match="invalid metric"):
        aggregate_client_updates(
            [
                ClientUpdate("client-a", 1, (np.array([1.0]),), {"loss": float("nan")}),
                ClientUpdate("client-b", 1, (np.array([1.0]),), {"loss": 1.0}),
            ]
        )


def test_host_allowlist_rejects_secrets_and_duplicate_aliases(tmp_path):
    config = tmp_path / "hosts.json"
    config.write_text(json.dumps({"hosts": [{"device_id": "pi", "ssh_alias": "pi", "role": "client"}]}))
    assert load_allowlist(config) == [InventoryHost("pi", "pi", "client")]

    config.write_text(
        json.dumps(
            {
                "hosts": [
                    {
                        "device_id": "pi",
                        "ssh_alias": "pi",
                        "role": "client",
                        "password": "must-not-be-accepted",
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="unsupported fields"):
        load_allowlist(config)

    config.write_text(
        json.dumps(
            {
                "hosts": [
                    {"device_id": "pi-a", "ssh_alias": "pi", "role": "client"},
                    {"device_id": "pi-b", "ssh_alias": "pi", "role": "client"},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="SSH aliases must be unique"):
        load_allowlist(config)


def test_inventory_unreachable_host_is_reported_without_raising(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fail)

    record = inspect_host(InventoryHost("pi-1", "pi-1", "client"), timeout=0.1)

    assert record["status"] == "unreachable"
    assert record["inventory"] is None


def test_inventory_parses_successful_hardware_response(monkeypatch):
    inventory = {
        "model": "Raspberry Pi 5 Model B",
        "architecture": "aarch64",
        "physical_cores": 4,
        "logical_cores": 4,
        "memory_total_bytes": 8_000_000_000,
        "software_available": {"torch": True, "flwr": False},
    }

    def succeed(*args, **kwargs):
        assert "StrictHostKeyChecking=yes" in args[0]
        assert "pi-1" in args[0]
        return subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=json.dumps(inventory), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", succeed)

    record = inspect_host(InventoryHost("pi-1", "pi-1", "client"))

    assert record["status"] == "inspected"
    assert record["inventory"] == inventory


def test_sanitized_inventory_omits_hostnames_and_ip_addresses(tmp_path):
    record = {
        "device_id": "pi-1",
        "ssh_alias": "private-alias",
        "role": "client",
        "status": "inspected",
        "error": None,
        "inventory": {
            "hostname": "private-host",
            "network_interfaces": [{"name": "eth0", "addresses": ["192.0.2.1"]}],
        },
    }
    unreachable = {
        "device_id": "pi-2",
        "ssh_alias": "private-alias-2",
        "role": "client",
        "status": "unreachable",
        "error": "private host 192.0.2.2 did not accept SSH",
        "inventory": None,
    }
    json_path = tmp_path / "inventory.json"
    csv_path = tmp_path / "inventory.csv"

    write_inventory([record, unreachable], json_path, csv_path, sanitize=True)

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["inspected_count"] == 1
    assert saved["hosts"][0]["ssh_alias"] is None
    assert saved["hosts"][0]["inventory"]["hostname"] is None
    assert "addresses" not in saved["hosts"][0]["inventory"]["network_interfaces"][0]
    assert saved["hosts"][1]["ssh_alias"] is None
    assert "192.0.2.2" not in saved["hosts"][1]["error"]


def test_mapping_validation_checks_synthetic_phase5_data_and_keeps_host_status(tmp_path):
    processed, split_file = _write_phase5_dataset(tmp_path, [6479, 6480])
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps(_mapping_for_buildings([6479, 6480])))

    result = validate_allocation(
        mapping,
        profile=2,
        processed_dir=processed,
        split_file=split_file,
    )

    assert result["client_count"] == 2
    assert result["unique_buildings"] == 2
    assert result["all_hosts_inventory_verified"] is False
    assert all(client["data_validation"] == "passed" for client in result["clients"])
    assert result["clients"][0]["actual_samples"] == {
        "train": 203,
        "validation": 59,
        "test": 45,
    }


def test_mapping_validation_rejects_reused_physical_pis_and_buildings(tmp_path):
    processed, split_file = _write_phase5_dataset(tmp_path, [6479, 6480])
    document = _mapping_for_buildings([6479, 6480])
    document["profiles"]["2"][1]["physical_pi_id"] = "node-52"
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="physical_pi_id values must be unique"):
        validate_allocation(
            mapping,
            profile=2,
            processed_dir=processed,
            split_file=split_file,
        )

    document["profiles"]["2"][1]["physical_pi_id"] = "node-53"
    document["profiles"]["2"][1]["building_id"] = 6479
    document["profiles"]["2"][1]["data_path"] = "Dataset/processed/building_6479"
    mapping.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="building_id values must be unique"):
        validate_allocation(
            mapping,
            profile=2,
            processed_dir=processed,
            split_file=split_file,
        )
