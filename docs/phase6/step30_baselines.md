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
separate and uses only observations dated before each target. The Step 32–34
runner adapts the finalized Phase 5 seven-day windows to the same evaluator:
the first and last historical target-feature values are lag 7 and lag 1,
respectively, and the target and lag values are inverse-scaled to kWh/day. The
baseline code does not read or transform hourly source files.

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

The Step 32–34 runs compare CNNs and baselines on the same building/client IDs,
target dates, validation/test split, and kWh/day target values. Since valid
Phase 5 windows have finite inputs and targets, all rows are eligible for both
lag baselines and the comparison records their scored counts. The baselines do
not implement Flower or federated training.