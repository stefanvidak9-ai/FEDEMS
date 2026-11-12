# Step 34: Independent Local CNN Training

Step 34 trains a separate Step 31 `DailyHeatingCNN` for each selected
building (default IDs: 6479–6482). Every model sees only its own Phase 5
training windows; no weights, gradients, or samples are exchanged. Its own
validation partition selects its checkpoint. Per-building validation and test
metrics, training duration, sample counts, seed, configuration, and
checkpoint path are recorded.

The feature order, seven-day lookback, target, global training-only scaling,
CPU execution, optimizer, loss, batch size, epoch limit, and stopping
criterion are the same as the centralized configuration when matching
options are used. Step 34 verifies that its building IDs, configuration, CNN
contract, and SHA-256 fingerprints of the scaler, split, and window files
match Step 33 before comparing results. Metrics are inverse-scaled to kWh/day.
The comparison also includes Step 30 baselines. Test observations are not
used to select checkpoints.

Run from the repository root:

```powershell
python -m src.training step34 --buildings 6479 6480 6481 6482 --epochs 100 --batch-size 32 --learning-rate 0.001 --patience 15 --seed 42
```

Outputs are written to `results/phase6/step34/`, with an independent
`building_<id>/best_model.pt` per building, along with histories,
date-aligned predictions, metrics, and `comparison.csv`. Change the selected
IDs or training parameters through command-line options; no Python source
change is needed.
