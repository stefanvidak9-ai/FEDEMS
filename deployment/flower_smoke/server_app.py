"""Flower ServerApp that verifies two synthetic updates and their aggregation."""

from __future__ import annotations

import json
import math
import time

import numpy as np
from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.serverapp import Grid, ServerApp

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    started = time.perf_counter()
    node_ids = sorted(grid.get_node_ids())
    if len(node_ids) != 2:
        raise RuntimeError(f"Expected exactly two connected clients; found {len(node_ids)}")
    synthetic_clients = (
        ("smoke-client-1", 1, 2.0, 4.0),
        ("smoke-client-2", 3, 6.0, 8.0),
    )
    requests: list[Message] = []
    request_bytes = 0
    for node_id, (client_id, examples, update_value, loss) in zip(
        node_ids, synthetic_clients, strict=True
    ):
        content = RecordDict(
            {
                "probe": ConfigRecord(
                    {
                        "client_id": client_id,
                        "parameters": [0.0, 0.0],
                        "shape": [2],
                        "num_examples": examples,
                        "update_value": update_value,
                        "loss": loss,
                    }
                )
            }
        )
        request_bytes += len(
            json.dumps(dict(content["probe"]), sort_keys=True).encode("utf-8")
        )
        requests.append(
            grid.create_message(
                content=content,
                message_type="query.parameter_probe",
                dst_node_id=node_id,
                group_id="phase7-smoke-round-1",
            )
        )

    replies = list(grid.send_and_receive(requests, timeout=30))
    if len(replies) != 2:
        raise RuntimeError(f"Expected exactly two client replies; received {len(replies)}")
    expected_ids = {item[0] for item in synthetic_clients}
    updates: list[tuple[int, np.ndarray, float]] = []
    reply_bytes = 0
    client_elapsed_ms: list[float] = []
    for reply in replies:
        record = reply.content["probe"]
        client_id = str(record["client_id"])
        if client_id not in expected_ids:
            raise RuntimeError(f"Unexpected client response identity: {client_id}")
        shape = tuple(int(value) for value in record["shape"])
        flat = np.asarray(record["parameters"], dtype=np.float64)
        if math.prod(shape) != flat.size or shape != (2,):
            raise RuntimeError("Client returned an invalid parameter tensor shape")
        values = flat.reshape(shape)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Client {client_id} returned non-finite parameters")
        examples = int(record["num_examples"])
        loss = float(record["loss"])
        if examples < 1 or not np.isfinite(loss):
            raise RuntimeError(f"Client {client_id} returned invalid metrics")
        expected = next(item for item in synthetic_clients if item[0] == client_id)
        if examples != expected[1] or not np.array_equal(
            values, np.full((2,), expected[2], dtype=np.float64)
        ):
            raise RuntimeError(f"Client {client_id} response did not match its probe")
        updates.append((examples, values, loss))
        client_elapsed_ms.append(float(record["elapsed_ms"]))
        reply_bytes += len(json.dumps(dict(record), sort_keys=True).encode("utf-8"))

    total_examples = sum(examples for examples, _, _ in updates)
    aggregated = np.average(
        np.stack([values for _, values, _ in updates]),
        axis=0,
        weights=[examples for examples, _, _ in updates],
    )
    metrics = {
        "loss": float(
            np.average(
                [loss for _, _, loss in updates],
                weights=[examples for examples, _, _ in updates],
            )
        )
    }
    expected_parameters = np.full((2,), 5.0, dtype=np.float64)
    if total_examples != 4 or not np.array_equal(aggregated, expected_parameters):
        raise RuntimeError(
            f"Aggregation mismatch: examples={total_examples}, parameters={aggregated}"
        )
    if metrics["loss"] != 7.0:
        raise RuntimeError(f"Metric aggregation mismatch: {metrics}")

    print(
        json.dumps(
            {
                "status": "passed",
                "round": 1,
                "participants": sorted(expected_ids),
                "flower_node_ids": {
                    client_id: node_id
                    for node_id, (client_id, _, _, _) in zip(node_ids, synthetic_clients, strict=True)
                },
                "parameter_count": int(aggregated.size),
                "aggregated_parameters": aggregated.tolist(),
                "weighted_metrics": metrics,
                "total_examples": total_examples,
                "request_json_bytes": request_bytes,
                "reply_json_bytes": reply_bytes,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "client_elapsed_ms": client_elapsed_ms,
                "training_performed": False,
                "real_data_accessed": False,
            },
            sort_keys=True,
        )
    )
