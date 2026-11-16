"""Run and clean up a loopback-only two-SuperNode Flower smoke test."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "deployment" / "flower_smoke"
LOG_DIR = ROOT / "results" / "phase7" / "logs"
FLOWER_HOME = ROOT / "results" / "phase7" / "flower-home"
READY_TIMEOUT_SECONDS = 30.0
RUN_TIMEOUT_SECONDS = 180.0


def _executable(name: str) -> str:
    suffix = ".exe" if os.name == "nt" else ""
    executable = Path(sys.executable).parent / f"{name}{suffix}"
    if not executable.is_file():
        raise FileNotFoundError(
            f"Missing {name}; install deployment/requirements-flower.txt "
            "in the active Python environment"
        )
    return str(executable)


def _ensure_ports_free(ports: tuple[int, ...]) -> None:
    for port in ports:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                raise RuntimeError(
                    f"Loopback port {port} is already in use; refusing to attach"
                )
        except OSError:
            pass


def _wait_for_port(
    port: int,
    process: subprocess.Popen[bytes],
    log_path: Path,
) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Flower process exited with {process.returncode}; inspect {log_path}"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"Flower service did not open loopback port {port}")


def _stop_processes(
    processes: list[subprocess.Popen[bytes]],
    logs: list[TextIO],
) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    for stream in logs:
        stream.close()


def run_smoke_test() -> dict[str, object]:
    _ensure_ports_free((8000, 9094, 9095))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FLOWER_HOME.mkdir(parents=True, exist_ok=True)
    (FLOWER_HOME / "config.toml").write_text(
        '[superlink]\ndefault = "local-deployment"\n\n'
        '[superlink.local-deployment]\n'
        'address = "127.0.0.1:8000"\n'
        "insecure = true\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PATH"] = (
        str(Path(sys.executable).parent)
        + os.pathsep
        + environment.get("PATH", "")
    )
    environment["FLWR_HOME"] = str(FLOWER_HOME    )
    processes: list[subprocess.Popen[bytes]] = []
    logs: list[TextIO] = []
    service_specs = (
        (
            "superlink",
            [_executable("flower-superlink"), "--insecure",
             "--disable-runtime-dependency-installation"],
            8000,
        ),
        (
            "supernode-1",
            [_executable("flower-supernode"), "--insecure", "--superlink",
             "127.0.0.1:8000", "--host", "127.0.0.1", "--port", "9094"],
            9094,
        ),
        (
            "supernode-2",
            [_executable("flower-supernode"), "--insecure", "--superlink",
             "127.0.0.1:8000", "--host", "127.0.0.1", "--port", "9095"],
            9095,
        ),
    )
    try:
        for name, command, port in service_specs:
            log_path = LOG_DIR / f"{name}.log"
            stream = log_path.open("w", encoding="utf-8")
            logs.append(stream)
            process = subprocess.Popen(
                command,
                cwd=APP_DIR,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            processes.append(process)
            _wait_for_port(port, process, log_path)

        run = subprocess.run(
            [_executable("flwr"), "run", ".", "--stream"],
            cwd=APP_DIR,
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_SECONDS,
            check=False,
            env=environment,
        )
        run_log = LOG_DIR / "flower-run.log"
        run_output = run.stdout + run.stderr
        run_log.write_text(run_output, encoding="utf-8")
        if run.returncode != 0:
            raise RuntimeError(
                f"Flower run failed with exit code {run.returncode}; inspect {run_log}"
            )

        report = None
        for line in run_output.splitlines():
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and candidate.get("status") == "passed":
                report = candidate
        if report is None:
            raise RuntimeError(
                f"Flower run exited successfully without a passing report; inspect {run_log}"
            )
        if (
            report.get("participants") != ["smoke-client-1", "smoke-client-2"]
            or report.get("aggregated_parameters") != [5.0, 5.0]
            or report.get("total_examples") != 4
            or report.get("weighted_metrics") != {"loss": 7.0}
            or report.get("training_performed") is not False
            or report.get("real_data_accessed") is not False
        ):
            raise RuntimeError(f"Flower smoke result failed its assertions: {report}")
        return report
    finally:
        _stop_processes(processes, logs)


def main() -> int:
    report = run_smoke_test()
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Flower process logs: {LOG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
