"""Synthetic Flower ClientApp used only for the Phase 7 communication probe."""

from __future__ import annotations

import math
import time

import numpy as np
from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp

app = ClientApp()


@app.query("parameter_probe")
def parameter_probe(message: Message, context: Context) -> Message:
    started = time.perf_counter()
    probe = message.content["probe"]
    flat = np.asarray(probe["parameters"], dtype=np.float32)
    shape = tuple(int(value) for value in probe["shape"])
    if math.prod(shape) != flat.size:
        raise ValueError("Received parameter shape does not match payload size")
    parameters = flat.reshape(shape)
    num_examples = int(probe["num_examples"])
    update_value = float(probe["update_value"])
    loss = float(probe["loss"])
    if num_examples < 1 or not np.isfinite(parameters).all():
        raise ValueError("Received invalid synthetic model payload")
    updated = parameters + update_value
    response = ConfigRecord(
        {
            "client_id": str(probe["client_id"]),
            "parameters": updated.reshape(-1).astype(float).tolist(),
            "shape": list(updated.shape),
            "num_examples": num_examples,
            "loss": loss,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
    )
    return Message(
        content=RecordDict({"probe": response}),
        reply_to=message,
    )
