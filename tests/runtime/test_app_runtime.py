import json
import os
import socket
import subprocess
import sys
import textwrap
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
RUNTIME_ROOT = Path(__file__).resolve().parent


def load_runtime_modules():
    sys.path.insert(0, str(SOURCE_ROOT))
    sys.path.insert(0, str(RUNTIME_ROOT))
    try:
        import dash
        import flask
        import numpy
        import plotly
    except ImportError as exc:
        raise unittest.SkipTest(f"runtime dependencies unavailable: {exc}") from exc


load_runtime_modules()

from app import create_app, resolve_runtime_config, run_app
from synthetic_fixture import create_minimal_app_data


class RuntimeConfigTest(unittest.TestCase):
    def test_defaults_to_loopback(self):
        config = resolve_runtime_config()
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 8050)

    def test_reads_environment_override(self):
        with mock.patch.dict(
            os.environ, {"AURORA_APP_HOST": "localhost", "AURORA_APP_PORT": "9090"}
        ):
            config = resolve_runtime_config()
        self.assertEqual(config.host, "localhost")
        self.assertEqual(config.port, 9090)

    def test_rejects_invalid_port(self):
        with self.assertRaisesRegex(ValueError, "Invalid Dash server port"):
            resolve_runtime_config(port="not-a-port")


class AppHealthReadinessTest(unittest.TestCase):
    def create_client(self, **fixture_overrides):
        zero_values = fixture_overrides.pop("zero_values", False)
        fixture = create_minimal_app_data(zero_values=zero_values)
        fixture.update(fixture_overrides)
        app = create_app(**fixture)
        return app, app.server.test_client()

    def test_health_and_readiness_succeed(self):
        _, client = self.create_client()
        health_response = client.get("/healthz")
        readiness_response = client.get("/readyz")
        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(readiness_response.status_code, 200)
        self.assertEqual(health_response.get_json(), {"status": "ok"})
        self.assertEqual(readiness_response.get_json()["status"], "ready")

    def test_zero_values_are_ready(self):
        _, client = self.create_client(zero_values=True)
        response = client.get("/readyz")
        self.assertEqual(response.status_code, 200)

    def test_mismatched_metric_names_are_not_ready(self):
        app, client = self.create_client()
        app.nic_runtime_state["metric_names"] = ["rx_bytes"]
        response = client.get("/readyz")
        payload = response.get_json()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(payload["status"], "not_ready")
        self.assertFalse(payload["checks"]["metric_names_match_axis_1"])


class CleanShutdownTest(unittest.TestCase):
    @staticmethod
    def free_port():
        with socket.socket() as server_socket:
            server_socket.bind(("127.0.0.1", 0))
            return server_socket.getsockname()[1]

    def test_server_answers_and_stops_on_sigterm(self):
        port = self.free_port()
        server_code = textwrap.dedent(
            f"""
            from app import create_app, run_app
            from synthetic_fixture import create_minimal_app_data

            app = create_app(**create_minimal_app_data())
            run_app(app, host="127.0.0.1", port={port})
            """
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(SOURCE_ROOT), str(RUNTIME_ROOT), environment.get("PYTHONPATH", "")]
        )
        process = subprocess.Popen(
            [sys.executable, "-c", server_code],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 10
            last_error = None
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/readyz", timeout=1
                    ) as response:
                        readiness = json.loads(response.read().decode("utf-8"))
                    if readiness["status"] == "ready":
                        break
                except (OSError, urllib.error.URLError) as exc:
                    last_error = exc
                time.sleep(0.1)
            else:
                self.fail(f"server did not become ready: {last_error}")

            process.terminate()
            process.wait(timeout=10)
            self.assertEqual(process.returncode, 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()


if __name__ == "__main__":
    unittest.main()
