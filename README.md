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

Step 30 is implemented. Its persistence and weekly seasonal-naive baselines,
CSV contract, metrics, and commands are documented in
[docs/phase6/step30_baselines.md](docs/phase6/step30_baselines.md). No CNN,
training, federated-learning, or Raspberry Pi deployment implementation is
included yet.

## Repository Layout

- `Dataset/`: supplied COFACTOR-Residential Dataset 1 files; source data is
  kept unchanged.
- `src/baselines/`: Step 30 lag-only forecasting and evaluation.
- `src/models/`: future model architectures, including the 1D-CNN.
- `src/training/`: future centralized and local model-training workflows.
- `src/federated/`: future Flower/FedAvg integration (Phase 8).
- `src/evaluation/`: shared evaluation and monitoring utilities.
- `deployment/`: Raspberry Pi setup and deployment procedures (Phase 7).
- `configs/`: versioned, non-secret experiment configuration.
- `experiments/`: reproducible experiment definitions and metadata.
- `results/`: generated metrics and artifacts; generated files are ignored.
- `tests/`: automated tests, including synthetic Step 30 examples.
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