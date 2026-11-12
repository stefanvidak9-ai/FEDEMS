# FEDEMS

Federated Learning on EMS Edge Hardware (FEDEMS) studies residential energy
forecasting on edge devices, with a path from reproducible centralized models
to federated learning on Raspberry Pis.

## Project Scope

The current dataset is COFACTOR Residential Dataset 1. Phase 4 provides
processed daily observations; Phase 5 provides chronological splits and
seven-day windows. Phase 6 predicts next-day total space-heating energy
(`HtSpace_kWh_day`, kWh/day) from the preceding seven complete days.

The project is organized into research phases:

- Phases 2–5: dataset preparation and preprocessing, owned by a teammate.
- Phase 6, Steps 30–34: forecasting baselines and CNN work, owned here.
- Phase 7, Steps 35–40: Raspberry Pi infrastructure, owned here.
- Phase 8: future Flower/FedAvg federated learning.

Steps 30–34 are implemented: Step 30 provides persistence and weekly
seasonal-naive baselines; Step 31 defines the compact PyTorch 1D-CNN; and
Steps 32–34 train and evaluate single-building, centralized, and independent
local CNN experiments. See the [Phase 6 guide](docs/phase6/README.md) for the
confirmed feature/data contract and run commands. Flower/FedAvg is future
work; Phase 7 covers Raspberry Pi infrastructure and runtime experiments.

## Repository Layout

- `Dataset/`: supplied COFACTOR-Residential Dataset 1 files; source data is
  kept unchanged.
- `src/baselines/`: Step 30 lag-only forecasting and evaluation.
- `src/models/`: reusable model architectures, including the Step 31 1D-CNN.
- `src/training/`: Steps 32–34 CNN training and experiment entry points.
- `src/federated/`: future Flower/FedAvg integration (Phase 8).
- `src/evaluation/`: shared original-scale regression metrics.
- `deployment/`: Raspberry Pi setup and deployment procedures (Phase 7).
- `configs/`: versioned, non-secret experiment configuration.
- `experiments/`: Phase 6 experiment reproducibility guidance.
- `results/`: generated metrics and artifacts; generated files are ignored.
- `tests/`: automated data, baseline, model, and training tests; synthetic
  fixtures are tests only, not research results.
- `docs/`: project, phase, and step documentation.

See [docs/README.md](docs/README.md) for documentation navigation and phase
planning.

## Development

Use a project virtual environment, then install dependencies and run the test
suite:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
```

To run the Step 30 synthetic example:

```powershell
python -m src.baselines.evaluate tests/data/synthetic_example.csv
```

The example is synthetic and is explicitly labeled as not research results.
Research evaluations must use agreed daily data, split definitions, units, and
the same client IDs and forecast dates for every compared model. The Phase 6
runner uses the finalized Phase 5 windows and records artifact fingerprints.

## Collaboration and Git Remotes

Work is developed in the personal GitHub repository. Completed, reviewed work
is transferred to the official TUM GitLab repository according to the team's
collaboration process. Do not commit credentials, private configuration, or
generated training artifacts. Dataset handling and licensing requirements
must be followed when transferring or publishing research outputs.