# Step 19 — Building Metadata Analysis

## Selected clients
17 heating centrals:
6479–6495

## Common characteristics
- Location: Oslo, Norway
- Construction year: 1966
- Building category: Apt
- Number of buildings: 1 per heating central
- Space heating: DH (district heating)
- Domestic hot water: DH (district heating)

## Building differences
- Floor area: 2,304–5,000 m2
- Apartments: 24–50

## Interpretation
The 17 clients have similar recorded building
characteristics and identical heating technologies.

Their daily heating demand levels differ, while their
temporal patterns are highly correlated.

Building size is one possible factor contributing to
differences in absolute heating consumption.

## Target decision
Primary target:
HtSpace_kWh_day

Additional analysis:
HtSpace_kWh_day / Floor_area

Absolute daily heating energy remains the primary
prediction target because the project investigates
forecasting total heating demand at each heating central.

Area-normalized demand will be used for supplementary
cross-building comparisons.

## Client selection
All 17 clients remain provisionally selected.
