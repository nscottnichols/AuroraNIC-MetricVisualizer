import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "examples" / "legacy_fixture"


def free_port():
    with socket.socket() as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return server_socket.getsockname()[1]


def request_json(url, timeout=1):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_entrypoint_serves_fixture_and_shuts_down_cleanly():
    with tempfile.TemporaryDirectory(prefix="nic-entrypoint-") as temporary_directory:
        fixture_root = Path(temporary_directory) / "fixture"
        shutil.copytree(FIXTURE_ROOT, fixture_root)
        port = free_port()
        command = [
            sys.executable,
            str(REPOSITORY_ROOT / "src" / "main.py"),
            "--base_dir",
            str(fixture_root),
            "--num_interfaces",
            "2",
            "--metric_names_file",
            str(fixture_root / "metric_names.txt"),
            "--benchmark",
            "oneccl",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 15
            last_error = None
            while time.monotonic() < deadline:
                try:
                    health_status, health = request_json(
                        f"http://127.0.0.1:{port}/healthz"
                    )
                    readiness_status, readiness = request_json(
                        f"http://127.0.0.1:{port}/readyz"
                    )
                    if (
                        health_status == 200
                        and readiness_status == 200
                        and readiness["status"] == "ready"
                    ):
                        break
                except (OSError, urllib.error.URLError) as exc:
                    last_error = exc
                time.sleep(0.1)
            else:
                raise AssertionError(f"entry point did not become ready: {last_error}")

            assert health == {"status": "ok"}
            assert (fixture_root / "cached_data.json").is_file()
            assert (fixture_root / "cached_bench_oneccl.json").is_file()

            process.terminate()
            process.wait(timeout=10)
            assert process.returncode == 0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()
