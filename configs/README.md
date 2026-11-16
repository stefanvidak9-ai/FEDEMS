# Configurations

- `data.yaml`: existing Phase 4/5 data contract.
- `raspberry-pi-clients.json`: candidate 2/4/8/17-client allocations using
  distinct existing building IDs and Phase 5 sample counts. Pi IDs are
  explicitly marked unverified until inventory confirms them.
- `raspberry-pi-hosts.example.json`: empty example for the read-only inventory
  allowlist. Copy it to the ignored `raspberry-pi-hosts.local.json` and enter
  only verified SSH aliases; keep hostnames, usernames, addresses, SSH keys,
  and credentials in local OpenSSH configuration.

Never commit passwords, SSH or TLS private keys, tokens, or live machine
configuration.