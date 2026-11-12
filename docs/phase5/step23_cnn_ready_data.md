# CNN-ready building datasets

For each heating-centre client, `Dataset/processed/building_<id>/` contains `train.npz`, `validation.npz`, `test.npz` with arrays named `X` and `y`. `X` is float32 in PyTorch Conv1d order `(N,4,7)`; `y` is float32 `(N,)`, a normalized next-day space-heating target. Dataset sizes are 203, 59, and 45 samples per client. Feature order: historical space heating, outdoor temperature, global solar irradiation and wind speed. `Dataset/processed/global_scaler.json` contains training-only global z-score statistics for unscaling output back to kWh/day.

The chosen shared target dates align all 17 clients, although FedAvg does not require date alignment. The complete per-client eligible sample inventory is also available in `docs/data_reports/`.
