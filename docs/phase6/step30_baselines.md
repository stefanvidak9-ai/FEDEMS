# Step 30: Simple Space-Heating Baselines

This step provides two next-day forecasts for residential space-heating energy
(`HtSpace`, kWh/day): persistence (`lag1_kwh`) and weekly seasonal naive
(`lag7_kwh`). The forecasting functions receive only their lag value, never
the target-day actual. `src.baselines.evaluate` reports per-building and
building-pooled MAE and RMSE for validation and test rows; R² is included only
when it is defined.

## Input Data

The evaluator accepts a comma-separated daily CSV with these columns:

| Column | Meaning |
| --- | --- |
| `building_id` | Stable building/client identifier |
| `target_date` | Forecast target calendar date, as `YYYY-MM-DD` |
| `actual_kwh` | Observed target-day space-heating energy in kWh/day |
| `lag1_kwh` | Actual energy from the same building on target date minus one day |
| `lag7_kwh` | Actual energy from the same building on target date minus seven days |
| `split` | `validation` or `test` |

Blank values are missing. Zero is a valid demand value. Both baselines are
scored on the same rows: the target and both lag values must be present. This
keeps their building/date samples directly comparable; the output includes
each scored sample count. Duplicate building/target-date rows are rejected.
Aggregate metrics pool scored rows across buildings (micro-average).

When starting from daily observations with `building_id`, `target_date`, and
`actual_kwh`, `derive_lags` creates exact calendar-day lags. It keeps buildings
separate and uses only observations dated before each target. An adapter for
the later processed NPZ data can map confirmed arrays into these fields; no NPZ
keys, normalization, or aggregation rules are assumed here. The hourly source
files are not read or transformed by these baselines.

## Install, Test, and Run

```powershell
python -m pip install -r requirements.txt
python -m pytest
python -m src.baselines.evaluate tests/data/synthetic_example.csv
```

The last command uses fabricated values in `tests/data/synthetic_example.csv`.
The CLI labels this output **SYNTHETIC EXAMPLE ONLY - NOT RESEARCH RESULTS**.
Do not use these metrics as COFACTOR findings.

## Comparing with the CNN

The future CNN adapter and this evaluator must use the same building/client
IDs, target dates, validation/test split, target units, and missing-target
policy. Score models on matching rows and report scored sample counts. The
baselines do not implement a CNN, Flower, or federated training.