# Phase 7: Raspberry Pi Infrastructure (Steps 35–40)

Phase 7 is future work; no device setup or deployment scripts are implemented.
Use `deployment/` for the eventual reproducible runbooks and automation. A
practical plan for the official step definitions is to cover:

- Device inventory, hardware identity, and supported operating-system image.
- Secure first-boot, network, user, and SSH configuration without committing
  keys or credentials.
- A tested Python/runtime and dependency installation process appropriate to
  the Raspberry Pi hardware.
- Repeatable project installation, configuration, and data/model placement.
- Device health, resource, and connectivity checks, with logs kept outside Git.
- A clean-device validation procedure and recovery/update instructions.

Confirm the exact Step 35–40 mapping with the project plan before turning this
outline into scripts. Keep training outputs and device-specific secrets out of
the repository.