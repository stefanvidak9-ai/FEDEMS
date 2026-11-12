# Phase 6: Baselines and CNN Experiments (Steps 30–34)

Phase 6 uses Person A's finalized Phase 4 daily aggregation and Phase 5
chronological splits/windows. Its task is to predict next-day
`HtSpace_kWh_day` (kWh/day) from the preceding seven complete days. The
feature order is `HtSpace_kWh_day`, `Tout_mean_C`, `SolGlob_kWh_m2_day`, and
`WindSpd_mean_m_s`. Phase 5's global training-only z-score is reused as-is;
the training code validates these artifacts and does not change preprocessing,
missing-data handling, or splits.

## Implemented Steps

- [Step 30: persistence and weekly seasonal-naive baselines](step30_baselines.md)
- [Step 31: reusable daily 1D-CNN architecture](step31_cnn.md)
- [Step 32: single-building CNN training](step32_cnn_local_training.md)
- [Step 33: centralized CNN training](step33_centralized_cnn.md)
- [Step 34: independent local CNN training](step34_local_cnn.md)

Steps 32–34 share the Step 31 `DailyHeatingCNN`, a deterministic CPU training
loop, validation-based checkpoint selection, inverse-scaled MAE/RMSE, and the
Step 30 baseline evaluator. Aggregate RMSE is pooled from squared errors.
Step 34 checks that its centralized comparison uses the same clients, model,
training settings, and exact Phase 5 artifacts as Step 33. Generated outputs
are written to the Git-ignored `results/phase6/` directory.

## Run the Experiments

Run from the repository root; the defaults are 100 epochs, batch size 32,
learning rate 0.001, patience 15, and seed 42. Data and output paths are
configurable; see [the training guide](../../src/training/README.md).

```powershell
python -m src.training step32 --building-id 6479
python -m src.training step33 --buildings 6479 6480 6481 6482
python -m src.training step34 --buildings 6479 6480 6481 6482
```

Run Step 33 before Step 34, using the same building IDs and training options.
These steps do not implement Flower/FedAvg or measure Raspberry Pi runtime;
those remain later work.
