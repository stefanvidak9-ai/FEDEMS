# Step 18 — Building 6479 Aggregation Report

## 1. Source Data
- Original file: building_6479.txt
- Original resolution: Hourly
- Date range: 2019-01-01 to 2022-12-31
- Total daily records: 1461

## 2. Daily Aggregation

| Variable | Daily operation | Unit |
|---|---|---|
| HtSpace | Sum of 24 hourly readings / 1000 | kWh/day |
| Tout | Daily mean | °C |
| SolGlob | Sum of hourly irradiance / 1000 | kWh/m²/day |
| WindSpd | Daily mean | m/s |

## 3. Daily Data Coverage

| Variable | Valid days | Missing days |
|---|---:|---:|
| HtSpace | 335 | 1126 |
| Tout | 1421 | 40 |
| SolGlob | 1417 | 44 |
| WindSpd | 1421 | 40 |

## 4. Solar Radiation Correction

- Negative hourly readings: 1405
- Correction: Negative readings clipped to zero.
- Complete solar days: 1417
- Minimum daily solar radiation: 0.006222 kWh/m²/day
- Mean daily solar radiation: 2.720240 kWh/m²/day
- Maximum daily solar radiation: 8.348911 kWh/m²/day

## 5. Heating Continuity

| Start date | End date | Consecutive days |
|---|---|---:|
| 2022-01-20 | 2022-10-29 | 283 |
| 2022-10-31 | 2022-12-21 | 52 |

Total complete heating days: 335

## 6. CNN Sample Preparation Assessment

- Historical input window: 7 consecutive days
- Prediction horizon: 1 day
- Potential samples based on heating continuity: 321
- Samples with complete heating and all four input features: 321

Calculation:

(283 - 7) + (52 - 7) = 321

These are potential samples before chronological train/validation/test splitting.

## 7. Preliminary Conclusion

Building 6479 has sufficient continuous heating observations
for further investigation in the forecasting experiment.

The building is retained provisionally.

Final eligibility will be determined after comparing all selected
heating centrals in Step 19.

## 8. Processing Files

- Source: data/raw/building_6479.txt
- Daily output: data/daily/building_6479_daily.csv
- Processing script: preprocessing/hourly_to_daily.py

## 9. Status

Step 18 — Building 6479 initial aggregation and sample assessment:
Completed provisionally.

Step 19 — Full daily data quality assessment:
Pending.
