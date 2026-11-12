# Phase 6 Evaluation

`src.evaluation.regression_metrics` provides shared finite-input MAE and RMSE
for original-scale kWh/day predictions. It rejects empty, mismatched, or
non-finite arrays and reports the scored sample count. Centralized aggregate
RMSE pools squared errors across observations; it is not the mean of
per-building RMSE values.

Steps 32–34 write date-aligned validation/test predictions and per-building
and aggregate metrics to `results/phase6/`. They reuse Step 30's
`src.baselines.evaluate.evaluate_forecasts` for persistence and weekly
seasonal-naive forecasts, preserving that module's API and eligibility rules.
The Step 30 standalone CSV evaluator remains available with
`python -m src.baselines.evaluate <daily-csv>`.
