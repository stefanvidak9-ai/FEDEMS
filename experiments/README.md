# Experiments

Phase 6 experiment settings are supplied as command-line options to
`python -m src.training step32`, `step33`, or `step34`; the defaults and
overrides are documented in [the Phase 6 training guide](../src/training/README.md).
Each run records its model/training configuration and SHA-256 fingerprints of
the Phase 5 scaler, target-date split, and selected building windows in
`metrics.json`.

Generated checkpoints, histories, predictions, and metrics belong in
`results/phase6/`, not in this directory or Git. Step 34 checks that its
centralized comparison used the same selected buildings, exact Phase 5 input
artifacts, CNN contract, and training settings as Step 33.
