# Step 32: Train and Verify the CNN Locally

Step 32 trains the existing `DailyHeatingCNN` from Step 31 on one COFACTOR
heating central. It consumes the finalized Phase 5 `train.npz`,
`validation.npz`, and `test.npz` windows and their
`global_scaler.json`; it does not regenerate or repartition data.
Phase 5 NPZs store arrays without date labels, so the loader aligns each
sample position with the official chronological Step 21 target-date list and
rejects any sample-count mismatch.

The inputs retain Phase 5's global, training-only z-score and feature order
(`HtSpace_kWh_day`, `Tout_mean_C`, `SolGlob_kWh_m2_day`,
`WindSpd_mean_m_s`). Each input has shape `[4, 7]`: four features for the
seven complete days before the target day. The target is next-day
`HtSpace_kWh_day` in kWh/day. MSE is optimized on normalized targets, while
validation and test MAE/RMSE are reported after inverse transformation in
kWh/day. The best checkpoint is selected by validation RMSE; test data is
used only for final evaluation.

From the repository root:

```powershell
python -m src.training step32 --building-id 6479 --epochs 100 --batch-size 32 --learning-rate 0.001 --patience 15 --seed 42
```

`--data-dir`, `--split-file`, and `--output-dir` override the default
Phase 5 inputs and `results/phase6` output directory. The run writes
`metrics.json`, `history_building_<id>.csv`, `predictions.csv`,
`comparison.csv`, and `best_model.pt`. Checkpoints include the model
state, feature/target scaling metadata, seed, configuration, and selected
validation epoch; `src.training.load_checkpoint` reconstructs a fresh
Step 31 model from the saved checkpoint.
