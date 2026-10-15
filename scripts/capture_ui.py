#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
FIXTURE_ROOT = ROOT / "examples" / "legacy_fixture"
DEFAULT_OUTPUT = ROOT / "artifacts" / "legacy-ui.png"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class CaptureError(RuntimeError):
    pass


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Capture a reproducible local rendering of the legacy fixture UI."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Screenshot destination (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Explicit loopback port; otherwise a free ephemeral port is used.",
    )
    parser.add_argument("--width", type=int, default=1440, help="Viewport width.")
    parser.add_argument("--height", type=int, default=900, help="Viewport height.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Startup, page, and stability timeout in seconds.",
    )
    args = parser.parse_args(argv)
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def reserve_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return server_socket.getsockname()[1]


def process_output(process):
    return process.stdout.read() + process.stderr.read()


def wait_for_readiness(process, url, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process_output(process)
            raise CaptureError(
                f"The application exited before becoming ready "
                f"(status {process.returncode}).\n{output}"
            )
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if response.status == 200 and payload.get("status") == "ready":
                return
            last_error = RuntimeError(f"/readyz returned {payload}")
        except urllib.error.HTTPError as exc:
            last_error = exc
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(0.1)
    raise TimeoutError(f"/readyz did not report ready: {last_error}")


def wait_for_stable_element(page, selector, timeout_seconds):
    locator = page.locator(selector)
    locator.wait_for(state="visible", timeout=int(timeout_seconds * 1000))

    previous_box = locator.bounding_box()
    stable_checks = 0
    deadline = time.monotonic() + timeout_seconds
    while stable_checks < 2 and time.monotonic() < deadline:
        page.wait_for_timeout(250)
        current_box = locator.bounding_box()
        if previous_box is not None and current_box == previous_box:
            stable_checks += 1
        else:
            stable_checks = 0
        previous_box = current_box
    if stable_checks < 2:
        raise CaptureError(f"Layout element did not stabilize: {selector}")


def capture_page(root_url, output_path, width, height, timeout_seconds):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise CaptureError(
            "Playwright is required for browser rendering. Install the optional "
            "browser extra (for example: python -m pip install -e '.[browser]') "
            "and then run: python -m playwright install chromium"
        ) from exc

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=1,
        )
        page = context.new_page()
        try:
            page.goto(root_url, wait_until="domcontentloaded", timeout=timeout_ms(timeout_seconds))
            page.wait_for_load_state("networkidle", timeout=timeout_ms(timeout_seconds))
            wait_for_stable_element(page, "#metric-dropdown", timeout_seconds)
            page.wait_for_selector(
                "#heatmap-figure .js-plotly-plot",
                state="visible",
                timeout=timeout_ms(timeout_seconds),
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=output_path, full_page=False)
        finally:
            context.close()
            browser.close()
        return browser.version


def timeout_ms(timeout_seconds):
    return int(timeout_seconds * 1000)


def inspect_screenshot(path, expected_width, expected_height):
    with path.open("rb") as screenshot_file:
        header = screenshot_file.read(24)
    if len(header) != 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise CaptureError(f"The browser did not produce a valid PNG: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if (width, height) != (expected_width, expected_height):
        raise CaptureError(
            f"Unexpected screenshot dimensions: {width}x{height}; "
            f"expected {expected_width}x{expected_height}"
        )
    return width, height


def stop_process(process):
    if process.poll() is not None:
        return process.returncode
    process.terminate()
    try:
        return process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
        raise CaptureError("The application process did not terminate cleanly")
    finally:
        process.stdout.close()
        process.stderr.close()


def run_capture(args):
    if not FIXTURE_ROOT.is_dir():
        raise CaptureError(f"Legacy fixture is missing: {FIXTURE_ROOT}")

    output_path = args.output.resolve()
    port = args.port if args.port is not None else reserve_port()
    root_url = f"http://127.0.0.1:{port}/"

    with tempfile.TemporaryDirectory(prefix="nic-legacy-ui-") as temporary_directory:
        fixture_root = Path(temporary_directory) / "legacy_fixture"
        shutil.copytree(FIXTURE_ROOT, fixture_root)

        environment = os.environ.copy()
        environment["AURORA_APP_HOST"] = "127.0.0.1"
        environment["AURORA_APP_PORT"] = str(port)
        command = [
            sys.executable,
            str(SOURCE_ROOT / "main.py"),
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
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            wait_for_readiness(
                process, f"http://127.0.0.1:{port}/readyz", args.timeout
            )
            browser_version = capture_page(
                root_url, output_path, args.width, args.height, args.timeout
            )
        finally:
            server_status = stop_process(process)

    width, height = inspect_screenshot(output_path, args.width, args.height)
    screenshot_bytes = output_path.stat().st_size
    with output_path.open("rb") as screenshot_file:
        screenshot_hash = hashlib.file_digest(screenshot_file, "sha256").hexdigest()
    if process.poll() is None:
        raise CaptureError("The application process remains active")

    print(f"Browser: Chromium {browser_version}")
    print(f"Screenshot: {output_path}")
    print(f"Dimensions: {width}x{height}")
    print(f"Bytes: {screenshot_bytes}")
    print(f"SHA256: {screenshot_hash}")
    print(f"Server: terminated cleanly with status {server_status}")


def main(argv=None):
    args = parse_arguments(argv)
    try:
        run_capture(args)
    except (CaptureError, TimeoutError, OSError) as exc:
        print(f"capture_ui.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
