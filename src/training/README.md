# Phase 6 Training

`src.training` implements Steps 32–34 using the Step 31 `DailyHeatingCNN` and
the finalized Phase 5 windows under `Dataset/processed/`. It validates the
seven-day, next-day contract, feature count/order, finite values, sample counts,
and chronological target-date alignment. It does not rebuild or resplit data.

Run commands from the repository root:

```powershell
python -m src.training step32 --building-id 6479
python -m src.training step33 --buildings 6479 6480 6481 6482
python -m src.training step34 --buildings 6479 6480 6481 6482
```

Step 34 requires a Step 33 run with the same building IDs, Phase 5 artifact
hashes, and training settings. All commands accept `--data-dir`,
`--split-file`, `--output-dir`, `--epochs`, `--batch-size`, `--learning-rate`,
`--patience`, and `--seed`. Defaults are CPU training, 100 epochs, batch size
32, learning rate 0.001, patience 15, and seed 42.

The shared trainer uses Adam and MSE on Phase 5 normalized targets, selects
checkpoints by validation RMSE in kWh/day, and evaluates validation/test
predictions after inverse scaling. Checkpoints include the Step 31 model
state, feature and target metadata, training settings, and best epoch.
Generated checkpoints, histories, predictions, metrics, and baseline
comparisons are written under `results/phase6/` and are not tracked by Git.

Run the training tests and full suite with:

```powershell
python -m pytest tests/test_training.py
python -m pytest
```

Federated orchestration and Flower/FedAvg are not implemented here.
