# Daily Feature Specification

## Project
Federated Learning on EMS Hardware

## Dataset
COFACTOR-Residential Dataset 1

## Prediction Task
Daily residential space-heating energy forecasting.

## Prediction Target
Original column: HtSpace
Target unit: kWh/day

Formula:
E_daily = sum(HtSpace_hourly) / 1000

Only days with 24 valid hourly measurements are accepted.

## Daily Input Features

| Original Column | Daily Operation | Unit |
|---|---|---|
| HtSpace | Sum / 1000 | kWh/day |
| Tout | Daily arithmetic mean | °C |
| SolGlob | Integrate hourly irradiance | kWh/m²/day |
| WindSpd | Daily arithmetic mean | m/s |

## Timestamp
Output format: YYYY-MM-DD
Source timezone: CET (UTC+01:00), as documented in README.

## Missing Data Rules
- Do not replace missing heating values with zero.
- Do not calculate incomplete daily heating totals.
- Preserve missing dates.
- Do not connect discontinuous observations.

## Forecasting Configuration
Input window: 7 consecutive days.
Prediction horizon: 1 day.
Model: 1D CNN.
Federated algorithm: FedAvg.

## Client Definition
One eligible heating central = one FL client.
