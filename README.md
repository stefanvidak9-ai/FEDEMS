# FEDEMS

Federated Learning on EMS Edge Hardware (FEDEMS) studies residential energy
forecasting on edge devices, with a path from reproducible centralized models
to federated learning on Raspberry Pis.

## Project Scope

The current dataset is COFACTOR-Residential Dataset 1. Its source measurements
are hourly; a separate team member is preparing daily data. The forecasting
target is next-day total space-heating energy (`HtSpace`, kWh/day), using the
preceding seven complete days as input history.

The project is organized into research phases:

- Phases 2–5: dataset preparation and preprocessing, owned by a teammate.
- Phase 6, Steps 30–34: forecasting baselines and CNN work, owned here.
- Phase 7, Steps 35–40: Raspberry Pi infrastructure, owned here.
- Phase 8: future Flower/FedAvg federated learning.

Steps 30 and 31 are implemented: Step 30 provides persistence and weekly
seasonal-naive baselines; Step 31 adds a compact configurable PyTorch 1D-CNN
architecture. See [Step 30 documentation](docs/phase6/step30_baselines.md) and
[Step 31 documentation](docs/phase6/step31_cnn.md). CNN training,
federated-learning, and Raspberry Pi deployment are not implemented yet.

## Repository Layout

- `Dataset/`: supplied COFACTOR-Residential Dataset 1 files; source data is
  kept unchanged.
- `src/baselines/`: Step 30 lag-only forecasting and evaluation.
- `src/models/`: reusable model architectures, including the Step 31 1D-CNN.
- `src/training/`: future centralized and local model-training workflows.
- `src/federated/`: future Flower/FedAvg integration (Phase 8).
- `src/evaluation/`: shared evaluation and monitoring utilities.
- `deployment/`: Raspberry Pi setup and deployment procedures (Phase 7).
- `configs/`: versioned, non-secret experiment configuration.
- `experiments/`: reproducible experiment definitions and metadata.
- `results/`: generated metrics and artifacts; generated files are ignored.
- `tests/`: automated tests, including synthetic Step 30 and Step 31 checks.
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
the same client IDs and forecast dates for every compared model.

## Collaboration and Git Remotes

Work is developed in the personal GitHub repository. Completed, reviewed work
is transferred to the official TUM GitLab repository according to the team's
collaboration process. Do not commit credentials, private configuration, or
generated training artifacts. Dataset handling and licensing requirements
must be followed when transferring or publishing research outputs.