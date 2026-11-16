# Phase 7 Deployment Utilities

- `inventory.py`: read-only hardware inspection of an explicit OpenSSH alias
  allowlist; defaults to a local, ignored config file and ignored results.
- `validate_mapping.py`: checks 2/4/8/17-client mapping profiles against the
  finalized Phase 5 split and NPZ windows.
- `verify_environment.py`: verifies CPU-only CNN inference, model-state
  serialization, and optional Flower availability.
- `verify_pi_cnn.py`: minimal CPU inference and model-parameter round-trip
  check intended to be copied to an ARM64 Pi with the required FEDEMS source
  modules.
- `flower_smoke/`: minimal Flower 1.43.0 ServerApp/ClientApp communication test;
  it uses synthetic parameters and performs no training.
- `requirements-flower.txt`: optional Flower dependency, separate from the
  Phase 6 base requirements.

See the [Phase 7 runbook](../docs/phase7/README.md) for install, inventory,
network-security, local smoke-test, and physical-deployment procedures. Host
allowlists, SSH keys, passwords, TLS private keys, and generated inventory
results must remain local and uncommitted. No system configuration, firewall,
service, or TUM network settings are changed by these utilities.