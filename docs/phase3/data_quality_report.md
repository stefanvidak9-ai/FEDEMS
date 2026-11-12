# Heating-data quality summary

All 17 selected clients have 1,461 daily calendar rows (2019-01-01 through 2022-12-31). Valid space-heating days range from 321 to 336; other periods are missing target data. Independent window validation produced 5,423 individual seven-day next-day examples and 307 forecast target dates usable by every client. Valid heating target dates are concentrated in 2022. All four selected inputs are complete when the heating target is valid.

See `docs/data_reports/daily_data_quality.csv`, `preprocessing_summary.csv`, `step19_federated_sample_availability.csv` and `step19_common_federated_target_dates.csv`. These were rebuilt independently after a bug in the old Step 19 reporting loop.
