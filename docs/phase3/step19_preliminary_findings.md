# Step 19 — Preliminary Cross-Building Analysis

## Dataset
- Candidate clients: 17
- Building IDs: 6479–6495
- Daily records per client: 1461

## Heating data
- Complete heating days: 321–336 per client
- Longest continuous heating runs: 269–284 days
- Common dates with all 17 heating targets: 321

## Energy consumption
- Lowest mean: Building 6479, 500.896 kWh/day
- Highest mean: Building 6495, 823.229 kWh/day
- Highest/lowest mean ratio: 1.644

## Correlation
- Daily heating-level correlation:
  minimum 0.9867, maximum 0.9992, mean 0.9958
- Daily heating-change correlation:
  approximately 0.852–0.990

No exact duplicate heating series were detected.

## Preliminary interpretation
The candidate clients show different average energy
consumption levels but highly synchronized daily
heating patterns.

Possible explanations include shared weather conditions
and similar heating-demand responses. Building metadata
must be examined before attributing these differences
to building characteristics.

## Client selection
All 17 clients are retained provisionally.
No client is excluded based on prediction performance
or correlation alone.

## Remaining work
- Calculate valid CNN samples for each client.
- Calculate common valid federated samples.
- Examine building metadata.
- Complete daily data quality assessment.

## CNN sample availability

Forecasting configuration:
- Historical input window: 7 consecutive days
- Prediction horizon: 1 day
- Historical features: HtSpace, Tout, SolGlob, WindSpd

Individual usable samples:
- Minimum: 307 (Building 6491)
- Maximum: 322 (Building 6493)
- Common valid prediction dates across all 17 clients: 307

All 17 candidate clients are retained provisionally.

These are total eligible samples before chronological
train/validation/test assignment, not training samples alone.

## Preliminary conclusion

The 17 clients provide sufficient data to proceed with
a small-scale daily forecasting experiment, subject to
the subsequent chronological split and evaluation checks.
