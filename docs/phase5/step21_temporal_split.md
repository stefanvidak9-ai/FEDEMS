# Chronological train/validation/test split

The target-date index is in `docs/data_reports/step21_temporal_split.csv`. 307 common dates are split chronologically: train 203 (2022-02-10–2022-08-31), validation 59 (2022-09-01–2022-10-31), test 45 (2022-11-01–2022-12-21). One date gap occurs, but only windows with seven consecutive observed history days are retained. Splitting is by target date, not random. Adjacent-window history can overlap; validation and test targets do not enter training.
