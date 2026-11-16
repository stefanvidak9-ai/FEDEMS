"""Read-only inventory of explicitly allowlisted hosts over OpenSSH."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "raspberry-pi-hosts.local.json"
DEFAULT_JSON = ROOT / "results" / "phase7" / "hardware-inventory.json"
DEFAULT_CSV = ROOT / "results" / "phase7" / "hardware-inventory.csv"
_ALIAS = re.compile(r"[A-Za-z0-9_.-]+\Z")

_REMOTE_INVENTORY = r"""
import importlib.util, json, os, platform, shutil, socket, subprocess

def read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as stream:
            return stream.read().strip()
    except OSError:
        return None

def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None

def meminfo():
    values = {}
    for line in (read("/proc/meminfo") or "").splitlines():
        key, _, remainder = line.partition(":")
        parts = remainder.split()
        if parts and parts[0].isdigit():
            values[key] = int(parts[0]) * 1024
    return values

cpuinfo = read("/proc/cpuinfo") or ""
model = read("/proc/device-tree/model")
if model:
    model = model.replace("\x00", "").strip()
cpu_model = None
for line in cpuinfo.splitlines():
    key, separator, value = line.partition(":")
    if separator and key.strip().lower() in {"model name", "hardware", "cpu model"}:
        cpu_model = value.strip()
        break
try:
    cpu_layout = json.loads(command(["lscpu", "-J"]) or "{}")["lscpu"]
    cpu_fields = {item["field"].rstrip(":"): item["data"] for item in cpu_layout}
except (KeyError, json.JSONDecodeError, TypeError):
    cpu_fields = {}
try:
    topology = command(["lscpu", "-p=CORE,SOCKET"])
    core_socket_pairs = {
        tuple(line.split(","))
        for line in (topology or "").splitlines()
        if line and not line.startswith("#")
    }
    physical = (
        len(core_socket_pairs)
        if core_socket_pairs
        else int(cpu_fields["Core(s) per socket"]) * int(cpu_fields["Socket(s)"])
    )
except (KeyError, TypeError, ValueError):
    physical = None
logical = os.cpu_count() or int(cpu_fields.get("CPU(s)", 0)) or None
memory = meminfo()
block_devices = command(["lsblk", "-b", "-J", "-o", "NAME,TYPE,SIZE,ROTA,TRAN,MOUNTPOINT,MODEL"])
try:
    storage = json.loads(block_devices)["blockdevices"] if block_devices else []
except (KeyError, json.JSONDecodeError):
    storage = []
try:
    addresses = json.loads(command(["ip", "-j", "address"]) or "[]")
except json.JSONDecodeError:
    addresses = []
interfaces = []
for link in addresses:
    name = link.get("ifname")
    speed = read("/sys/class/net/" + str(name) + "/speed") if name else None
    interfaces.append({
        "name": name,
        "state": link.get("operstate"),
        "addresses": [item.get("local") for item in link.get("addr_info", []) if item.get("local")],
        "speed_mbps": int(speed) if speed and speed.isdigit() else None,
    })
temperature = None
raw_temp = read("/sys/class/thermal/thermal_zone0/temp")
if raw_temp and raw_temp.isdigit():
    temperature = int(raw_temp) / 1000.0
if temperature is None:
    output = command(["vcgencmd", "measure_temp"])
    if output and output.startswith("temp="):
        try:
            temperature = float(output[5:].split("'")[0])
        except ValueError:
            pass
try:
    disk = shutil.disk_usage("/")
    disk_info = {"total_bytes": disk.total, "free_bytes": disk.free}
except OSError:
    disk_info = None
os_release = {}
for line in (read("/etc/os-release") or "").splitlines():
    key, separator, value = line.partition("=")
    if separator and key in {"PRETTY_NAME", "ID", "VERSION_ID"}:
        os_release[key] = value.strip().strip('"')
software = {}
for module in ("numpy", "pandas", "torch", "flwr"):
    spec = importlib.util.find_spec(module)
    software[module] = spec is not None
print(json.dumps({
    "hostname": socket.gethostname(),
    "model": model,
    "architecture": platform.machine(),
    "cpu_model": cpu_model or platform.processor() or None,
    "physical_cores": physical,
    "logical_cores": logical,
    "memory_total_bytes": memory.get("MemTotal"),
    "memory_available_bytes": memory.get("MemAvailable"),
    "storage_devices": storage,
    "root_disk": disk_info,
    "operating_system": os_release,
    "kernel": platform.release(),
    "python": platform.python_version(),
    "network_interfaces": interfaces,
    "cpu_temperature_c": temperature,
    "software_available": software,
    "write_permissions": {
        "home": os.access(os.path.expanduser("~"), os.W_OK),
        "temporary_directory": os.access("/tmp", os.W_OK),
    },
}, separators=(",", ":"), sort_keys=True))
"""


@dataclass(frozen=True)
class InventoryHost:
    device_id: str
    ssh_alias: str
    role: str
    port: int | None = None
    jump_host_alias: str | None = None


def load_allowlist(path: str | Path) -> list[InventoryHost]:
    """Load a secret-free allowlist; SSH credentials stay in OpenSSH config."""
    config_path = Path(path)
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read inventory config {config_path}: {error}") from error
    if not isinstance(document, dict) or set(document) != {"hosts"}:
        raise ValueError("Inventory config must contain only a 'hosts' array")
    if not isinstance(document["hosts"], list):
        raise ValueError("Inventory config 'hosts' must be an array")

    hosts: list[InventoryHost] = []
    allowed_fields = {"device_id", "ssh_alias", "role", "port", "jump_host_alias"}
    for index, item in enumerate(document["hosts"]):
        if not isinstance(item, dict):
            raise ValueError(f"Inventory host {index} must be an object")
        extra = set(item) - allowed_fields
        if extra:
            raise ValueError(
                f"Inventory host {index} has unsupported fields: {sorted(extra)}"
            )
        device_id = item.get("device_id")
        ssh_alias = item.get("ssh_alias")
        role = item.get("role")
        if not all(isinstance(value, str) and value.strip() for value in (device_id, role)):
            raise ValueError(f"Inventory host {index} requires device_id and role")
        if not isinstance(ssh_alias, str) or not _ALIAS.fullmatch(ssh_alias):
            raise ValueError(
                f"Inventory host {index} requires an SSH alias from OpenSSH config"
            )
        port = item.get("port")
        if port is not None and (
            isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535
        ):
            raise ValueError(f"Inventory host {index} has an invalid SSH port")
        jump = item.get("jump_host_alias")
        if jump is not None and (not isinstance(jump, str) or not _ALIAS.fullmatch(jump)):
            raise ValueError(f"Inventory host {index} has an invalid jump-host alias")
        hosts.append(
            InventoryHost(
                device_id=device_id.strip(),
                ssh_alias=ssh_alias,
                role=role.strip(),
                port=port,
                jump_host_alias=jump,
            )
        )

    if len({host.device_id for host in hosts}) != len(hosts):
        raise ValueError("Inventory device IDs must be unique")
    if len({host.ssh_alias for host in hosts}) != len(hosts):
        raise ValueError("Inventory SSH aliases must be unique")
    return hosts


def _sanitize_inventory(record: dict[str, Any]) -> dict[str, Any]:
    inventory = record.get("inventory")
    safe_record = {
        **record,
        "ssh_alias": None,
        "error": (
            "connection or inventory failed; see private local report for details"
            if record.get("error")
            else None
        ),
    }
    if not isinstance(inventory, dict):
        return safe_record
    inventory = dict(inventory)
    inventory["hostname"] = None
    inventory["network_interfaces"] = [
        {key: value for key, value in interface.items() if key != "addresses"}
        for interface in inventory.get("network_interfaces", [])
    ]
    return {
        **safe_record,
        "inventory": inventory,
    }


def inspect_host(host: InventoryHost, *, timeout: float = 20.0) -> dict[str, Any]:
    """Run read-only Python hardware inspection through strict OpenSSH."""
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=8",
    ]
    if host.port is not None:
        command.extend(["-p", str(host.port)])
    if host.jump_host_alias is not None:
        command.extend(["-J", host.jump_host_alias])
    command.extend([host.ssh_alias, "python3 -"])
    try:
        completed = subprocess.run(
            command,
            input=_REMOTE_INVENTORY,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "device_id": host.device_id,
            "ssh_alias": host.ssh_alias,
            "role": host.role,
            "status": "unreachable",
            "error": "SSH inventory command timed out",
            "inventory": None,
        }
    if completed.returncode != 0:
        return {
            "device_id": host.device_id,
            "ssh_alias": host.ssh_alias,
            "role": host.role,
            "status": "unreachable",
            "error": completed.stderr.strip() or f"ssh exited {completed.returncode}",
            "inventory": None,
        }
    try:
        inventory = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        return {
            "device_id": host.device_id,
            "ssh_alias": host.ssh_alias,
            "role": host.role,
            "status": "inventory_error",
            "error": f"Remote inventory returned invalid JSON: {error}",
            "inventory": None,
        }
    return {
        "device_id": host.device_id,
        "ssh_alias": host.ssh_alias,
        "role": host.role,
        "status": "inspected",
        "error": None,
        "inventory": inventory,
    }


def write_inventory(
    records: list[dict[str, Any]],
    json_path: str | Path,
    csv_path: str | Path,
    *,
    sanitize: bool = False,
) -> None:
    output = [_sanitize_inventory(record) for record in records] if sanitize else records
    json_file = Path(json_path)
    csv_file = Path(csv_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.write_text(
        json.dumps({"inspected_count": sum(r["status"] == "inspected" for r in output),
                    "hosts": output}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fields = ("device_id", "ssh_alias", "role", "status", "error", "inventory_json")
    with csv_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in output:
            writer.writerow(
                {
                    **{key: record.get(key) for key in fields if key != "inventory_json"},
                    "inventory_json": json.dumps(record.get("inventory"), sort_keys=True),
                }
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        hosts = load_allowlist(args.config)
    except ValueError as error:
        parser.error(str(error))
    records = [inspect_host(host, timeout=args.timeout) for host in hosts]
    write_inventory(records, args.json_out, args.csv_out, sanitize=args.sanitize)
    inspected = sum(record["status"] == "inspected" for record in records)
    print(
        f"Inspected {inspected}/{len(hosts)} explicitly allowlisted devices; "
        f"JSON: {args.json_out}; CSV: {args.csv_out}"
    )
    if not hosts:
        print("No devices were inspected because the allowlist is empty.")
    return int(
        not hosts or any(record["status"] != "inspected" for record in records)
    )


if __name__ == "__main__":
    sys.exit(main())
