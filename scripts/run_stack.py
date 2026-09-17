#!/usr/bin/env python
"""ONE COMMAND: start the application and all of its dependencies.

    python scripts/run_stack.py

Starts, supervises and shuts down together:

  * the infer-lab FastAPI server          (always)
  * the Node.js live dashboard            (if node is installed)
  * Prometheus                            (if a binary is found -- see --prometheus)
  * Grafana                               (if a binary is found -- see --grafana)

Everything runs as a plain local process. No Docker, no containers, no
administrator rights. Missing optional components are reported and skipped;
the core server always comes up.

Ctrl+C shuts the whole stack down in reverse order.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OBS = ROOT / "observability"
LOGS = ROOT / "artifacts" / "logs"
sys.path.insert(0, str(SRC))

GREEN, RED, YELLOW, BLUE, DIM, RESET = (
    ("\033[32m", "\033[31m", "\033[33m", "\033[34m", "\033[2m", "\033[0m")
    if sys.stdout.isatty() and os.name != "nt" else ("", "", "", "", "", "")
)


def info(msg: str) -> None:
    print(f"{BLUE}[stack]{RESET} {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"{GREEN}[ ok ]{RESET} {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"{YELLOW}[skip]{RESET} {msg}", flush=True)


def err(msg: str) -> None:
    print(f"{RED}[fail]{RESET} {msg}", flush=True)


# --------------------------------------------------------------------------- helpers
def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def wait_for_http(url: str, timeout: float = 45.0, interval: float = 0.3) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(interval)
    return False


def find_binary(name: str, extra_hints: list[Path]) -> Path | None:
    found = shutil.which(name)
    if found:
        return Path(found)
    for hint in extra_hints:
        for candidate in (hint, hint / name, hint / f"{name}.exe"):
            if candidate.is_file():
                return candidate
        if hint.is_dir():
            matches = sorted(hint.rglob(f"{name}.exe")) or sorted(hint.rglob(name))
            for match in matches:
                if match.is_file():
                    return match
    return None


# --------------------------------------------------------------------------- process
@dataclass
class Service:
    name: str
    cmd: list[str]
    cwd: Path
    health_url: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    process: subprocess.Popen | None = None
    log_path: Path | None = None
    optional: bool = True

    def start(self) -> bool:
        LOGS.mkdir(parents=True, exist_ok=True)
        self.log_path = LOGS / f"{self.name}.log"
        handle = self.log_path.open("w", encoding="utf-8")
        environment = {**os.environ, **self.env}
        try:
            self.process = subprocess.Popen(
                self.cmd, cwd=self.cwd, stdout=handle, stderr=subprocess.STDOUT,
                env=environment,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if os.name == "nt" else 0,
            )
        except (OSError, FileNotFoundError) as exc:
            err(f"{self.name}: could not start ({exc})")
            return False

        if self.health_url:
            if not wait_for_http(self.health_url):
                err(f"{self.name}: did not become healthy -- see {self.log_path}")
                self.stop()
                return False
        else:
            time.sleep(1.0)
            if self.process.poll() is not None:
                err(f"{self.name}: exited immediately -- see {self.log_path}")
                return False
        return True

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            if os.name == "nt":
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.terminate()
            self.process.wait(timeout=10)
        except (subprocess.TimeoutExpired, OSError):
            self.process.kill()

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None


# --------------------------------------------------------------------------- main
def build_services(args: argparse.Namespace) -> list[Service]:
    services: list[Service] = []

    # ---- 1. the inference server (mandatory)
    services.append(Service(
        name="infer-lab-server",
        cmd=[sys.executable, "-m", "uvicorn", "infer_lab.server.api:app",
             "--host", args.host, "--port", str(args.port), "--log-level", "warning"],
        cwd=ROOT,
        health_url=f"http://127.0.0.1:{args.port}/health",
        env={"PYTHONPATH": str(SRC)},
        optional=False,
    ))

    # ---- 2. Node dashboard
    if not args.no_dashboard and shutil.which("node"):
        services.append(Service(
            name="dashboard",
            cmd=["node", str(ROOT / "clients" / "node" / "dashboard.js")],
            cwd=ROOT / "clients" / "node",
            health_url=f"http://127.0.0.1:{args.dashboard_port}/",
            env={"INFER_LAB_URL": f"http://127.0.0.1:{args.port}",
                 "DASHBOARD_PORT": str(args.dashboard_port)},
        ))
    elif not args.no_dashboard:
        warn("dashboard: node not found on PATH")

    # ---- 3. Prometheus
    if not args.no_prometheus:
        hints = [Path(args.prometheus)] if args.prometheus else []
        hints += [ROOT / "vendor" / "prometheus", Path.home() / "prometheus"]
        binary = find_binary("prometheus", hints)
        if binary:
            services.append(Service(
                name="prometheus",
                cmd=[str(binary), f"--config.file={OBS / 'prometheus.yml'}",
                     f"--storage.tsdb.path={ROOT / 'artifacts' / 'prometheus-data'}",
                     f"--web.listen-address=127.0.0.1:{args.prometheus_port}"],
                cwd=OBS,
                health_url=f"http://127.0.0.1:{args.prometheus_port}/-/ready",
            ))
        else:
            warn("prometheus: binary not found "
                 "(unzip it to ./vendor/prometheus/ or pass --prometheus PATH)")

    # ---- 4. Grafana
    if not args.no_grafana:
        hints = [Path(args.grafana)] if args.grafana else []
        hints += [ROOT / "vendor" / "grafana", Path("C:/Program Files/GrafanaLabs")]
        binary = find_binary("grafana-server", hints) or find_binary("grafana", hints)
        if binary:
            services.append(Service(
                name="grafana",
                cmd=[str(binary), "server", "--homepath", str(binary.parent.parent)]
                if binary.name.startswith("grafana") and "server" not in binary.name
                else [str(binary)],
                cwd=binary.parent,
                health_url=f"http://127.0.0.1:{args.grafana_port}/api/health",
            ))
        else:
            warn("grafana: binary not found (install it or pass --grafana PATH)")

    return services


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start infer-lab and its dependencies")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dashboard-port", type=int, default=3000)
    parser.add_argument("--prometheus-port", type=int, default=9090)
    parser.add_argument("--grafana-port", type=int, default=3001)
    parser.add_argument("--prometheus", help="path to the prometheus binary")
    parser.add_argument("--grafana", help="path to the grafana binary")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--no-prometheus", action="store_true")
    parser.add_argument("--no-grafana", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="start, run one request, print URLs, then shut down")
    args = parser.parse_args(argv)

    if not port_is_free(args.port):
        err(f"port {args.port} is already in use -- pass --port to pick another")
        return 1

    info("starting infer-lab stack (no containers, all local processes)")
    services = build_services(args)
    started: list[Service] = []

    shutdown_done = threading.Event()

    def shutdown() -> None:
        # Idempotent: both the --smoke path and the finally block call this.
        if shutdown_done.is_set():
            return
        shutdown_done.set()
        info("shutting down (reverse order)")
        for service in reversed(started):
            service.stop()
            ok(f"{service.name} stopped")

    try:
        for service in services:
            info(f"starting {service.name} ...")
            if service.start():
                started.append(service)
                ok(f"{service.name} up  {DIM}(log: {service.log_path}){RESET}")
            elif not service.optional:
                err("mandatory service failed to start")
                shutdown()
                return 1

        base = f"http://127.0.0.1:{args.port}"
        print(f"\n{GREEN}stack is up{RESET}")
        print(f"  API            {base}")
        print(f"  interactive    {base}/docs")
        print(f"  metrics        {base}/metrics")
        print(f"  engine stats   {base}/stats")
        for service in started:
            if service.name == "dashboard":
                print(f"  dashboard      http://127.0.0.1:{args.dashboard_port}")
            if service.name == "prometheus":
                print(f"  prometheus     http://127.0.0.1:{args.prometheus_port}")
            if service.name == "grafana":
                print(f"  grafana        http://127.0.0.1:{args.grafana_port} "
                      f"(import {OBS / 'grafana_dashboard.json'})")

        print(f"\n  try it:\n    curl -X POST {base}/generate "
              f'-H "Content-Type: application/json" '
              f'-d "{{\\"prompt\\":\\"hello\\",\\"max_tokens\\":16}}"')

        if args.smoke:
            import json
            request = urllib.request.Request(
                f"{base}/generate",
                data=json.dumps({"prompt": "smoke test", "max_tokens": 8}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read())
            ok(f"smoke request returned {body['output_tokens']} tokens")
            shutdown()
            return 0

        print(f"\n{DIM}Ctrl+C to stop the whole stack{RESET}\n")
        stop_event = threading.Event()

        def handle_signal(*_: object) -> None:
            stop_event.set()

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        while not stop_event.is_set():
            for service in started:
                if not service.alive and not service.optional:
                    err(f"{service.name} died -- see {service.log_path}")
                    stop_event.set()
            stop_event.wait(1.0)

    except KeyboardInterrupt:
        pass
    finally:
        shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
