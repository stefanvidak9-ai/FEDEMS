# Federated Infrastructure

Phase 7 adds safe model-state serialization and weighted aggregation in
`parameters.py`, plus Phase 5 client/building allocation validation in
`allocation.py`. These helpers do not train a federated CNN.

The communication-only Flower ServerApp/ClientApp and same-machine deployment
instructions live in `deployment/flower_smoke/`. The app uses synthetic
parameters and must not be run with the insecure local profile on a shared
network. Physical deployment requires the TLS and SuperNode identity
authentication described in [Phase 7](../../docs/phase7/README.md).

Phase 8 owns the future FedAvg CNN strategy and client-side data training.