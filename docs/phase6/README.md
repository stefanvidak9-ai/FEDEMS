# Phase 6: CNN Architecture and Baselines (Steps 30–34)

Step 30 is implemented in `src/baselines/`; see
[step30_baselines.md](step30_baselines.md). Steps 31–34 are future work. The
following is a suggested organization, not a replacement for the team's
official step definitions:

- Define the model-facing daily input/target contract with the preprocessing
  owner. Record units, ordering, missing-value policy, and agreed train,
  validation, and test dates before adding a data adapter.
- Add the 1D-CNN architecture under `src/models/` with an explicit input/output
  shape contract and focused model tests.
- Add only the training workflows assigned to this phase under `src/training/`;
  keep centralized and single-client/local training distinct and reproducible.
- Compare CNN forecasts against Step 30 on identical client IDs, forecast dates,
  splits, units, and eligible target rows using shared metrics in
  `src/evaluation/`.

Do not infer NPZ keys or normalization from filenames. Confirm these with the
preprocessing owner and document the adapter contract before implementation.