"""Validate logical Flower clients against the finalized Phase 5 datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.training.data import load_buildings

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = ROOT / "configs" / "raspberry-pi-clients.json"
DEFAULT_PROCESSED = ROOT / "Dataset" / "processed"
DEFAULT_SPLIT = ROOT / "docs" / "data_reports" / "step21_temporal_split.csv"
SAMPLE_COUNTS = {"train": 203, "validation": 59, "test": 45}


def validate_allocation(
    mapping_path: str | Path = DEFAULT_MAPPING,
    *,
    profile: int,
    processed_dir: str | Path = DEFAULT_PROCESSED,
    split_file: str | Path = DEFAULT_SPLIT,
) -> dict[str, Any]:
    """Return verified data counts while retaining hardware-verification status."""
    if profile not in {2, 4, 8, 17}:
        raise ValueError("Profiles must contain 2, 4, 8, or 17 clients")
    path = Path(mapping_path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read client mapping {path}: {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Client mapping must use schema_version 1")
    profiles = document.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("Client mapping must contain a profiles object")
    assignments = profiles.get(str(profile))
    if not isinstance(assignments, list) or len(assignments) != profile:
        raise ValueError(f"Profile {profile} must contain exactly {profile} clients")

    required_fields = {
        "logical_client_id",
        "physical_pi_id",
        "host_status",
        "building_id",
        "data_path",
        "expected_samples",
    }
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict) or set(assignment) != required_fields:
            raise ValueError(f"Profile {profile} assignment {index} has an invalid schema")
    for assignment in assignments:
        for field in ("logical_client_id", "physical_pi_id", "host_status"):
            if not isinstance(assignment[field], str) or not assignment[field].strip():
                raise ValueError(f"{field} must be a non-empty string")
        if assignment["host_status"] not in {
            "candidate-unverified",
            "inventory-verified",
        }:
            raise ValueError(
                f"Unsupported host status {assignment['host_status']!r}"
            )
        if not isinstance(assignment["building_id"], int) or isinstance(
            assignment["building_id"], bool
        ):
            raise ValueError("Building IDs must be integers")
        expected_path = f"Dataset/processed/building_{assignment['building_id']}"
        if assignment["data_path"] != expected_path:
            raise ValueError(
                f"Building {assignment['building_id']} must use {expected_path}"
            )
        if assignment["expected_samples"] != SAMPLE_COUNTS:
            raise ValueError(
                f"Building {assignment['building_id']} has unexpected sample metadata"
            )
    for field in ("logical_client_id", "physical_pi_id", "building_id"):
        values = [assignment[field] for assignment in assignments]
        if len(set(values)) != len(values):
            raise ValueError(f"Profile {profile} {field} values must be unique")

    building_ids = [assignment["building_id"] for assignment in assignments]
    _, buildings = load_buildings(processed_dir, split_file, building_ids)
    validated: list[dict[str, Any]] = []
    for assignment in assignments:
        actual_counts = {
            name: int(split.y.size)
            for name, split in buildings[assignment["building_id"]].splits.items()
        }
        if actual_counts != assignment["expected_samples"]:
            raise ValueError(
                f"Building {assignment['building_id']} sample counts {actual_counts} "
                f"do not match the mapping"
            )
        validated.append(
            {
                **assignment,
                "actual_samples": actual_counts,
                "data_validation": "passed",
            }
        )
    return {
        "profile": profile,
        "client_count": len(validated),
        "unique_buildings": len({item["building_id"] for item in validated}),
        "all_hosts_inventory_verified": all(
            item["host_status"] == "inventory-verified" for item in validated
        ),
        "clients": validated,
    }
