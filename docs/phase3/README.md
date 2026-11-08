# Data quality (Person A)

Validate each client's daily time axis, heating coverage and seven-day next-day window eligibility. The independent audit found 321–336 valid daily heating observations per client, 5,423 individual CNN windows and 307 target dates eligible for every client. Supporting verified CSVs are in `docs/data_reports/`. Run `python -m src.data.quality_checks` and `python -m src.data.validate_windows` from the repository root. The former Step 19 script had a reporting-loop indentation bug; the corrected rebuild scripts are used here.
