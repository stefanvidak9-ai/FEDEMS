# Hourly-to-daily aggregation (Person A)

Daily data for each of 17 clients is provided in `Dataset/daily/`. Features: heating sum (kWh/day), average outdoor temperature (°C), summed global solar irradiation (kWh/m²/day), average wind speed (m/s). Incomplete days are not considered valid. See `feature_specification.md` and `step18_building_6479_report.md`. Original aggregation implementation is in `src/data/hourly_to_daily.py`; inspect its adapted file paths before regenerating daily data.
