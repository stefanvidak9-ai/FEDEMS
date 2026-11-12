# Step 33: Centralized CNN Training

Step 33 trains one Step 31 `DailyHeatingCNN` on concatenated Phase 5 training
windows for a configurable set of buildings (default IDs: 6479–6482).
Validation and test observations remain in their original per-building
partitions. The model is selected by pooled validation RMSE and evaluated
separately on each building, then pooled across all rows; aggregate RMSE is
computed from pooled squared errors, not by averaging building RMSE values.

All clients use the same seven-day, four-feature input contract and the
existing Phase 5 global training-only scaler. Metrics are inverse-scaled and
reported in kWh/day. For same-building comparisons, the existing Step 30
persistence and weekly seasonal-naive methods are scored from the lag1 and
lag7 values in the same Phase 5 windows.

Run Step 33 from the repository root:

```powershell
python -m src.training step33 --buildings 6479 6480 6481 6482 --epochs 100 --batch-size 32 --learning-rate 0.001 --patience 15 --seed 42
```

The result directory is `results/phase6/step33/` by default. It contains the
best checkpoint, per-epoch history, date-aligned validation/test predictions,
per-building and pooled `metrics.json`, and `comparison.csv`. Options
`--data-dir`, `--split-file`, and `--output-dir` are configurable.
