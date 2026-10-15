#!/usr/bin/env python3
"""Capture reproducible browser evidence for the synthetic legacy analysis UI."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import signal
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
TEST_ROOT = ROOT / "tests"
DEFAULT_1440_OUTPUT = ROOT / "artifacts" / "legacy-analysis-1440x900.png"
DEFAULT_1024_OUTPUT = ROOT / "artifacts" / "legacy-analysis-1024x768.png"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


class CaptureError(RuntimeError):
    pass


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Exercise and screenshot the synthetic question-driven analysis UI."
    )
    parser.add_argument(
        "--output-1440",
        type=Path,
        default=DEFAULT_1440_OUTPUT,
        help=f"1440x900 screenshot destination (default: {DEFAULT_1440_OUTPUT}).",
    )
    parser.add_argument(
        "--output-1024",
        type=Path,
        default=DEFAULT_1024_OUTPUT,
        help=f"1024x768 screenshot destination (default: {DEFAULT_1024_OUTPUT}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Explicit loopback port; otherwise a free ephemeral port is used.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="Server, page, interaction, and screenshot timeout in seconds.",
    )
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not args.serve and args.output_1440.resolve() == args.output_1024.resolve():
        parser.error("the two screenshot destinations must be different")
    return args


def reserve_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return server_socket.getsockname()[1]


def run_server(args):
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    if str(TEST_ROOT) not in sys.path:
        sys.path.insert(1, str(TEST_ROOT))

    from analysis_app import create_analysis_app
    from analysis_fixture import make_store
    from werkzeug.serving import make_server

    app = create_analysis_app(make_store())
    server = make_server(args.host, args.port, app.server, threaded=True)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    shutdown_requested = threading.Event()

    def request_shutdown(signum, frame):
        shutdown_requested.set()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    server_thread.start()
    try:
        shutdown_requested.wait()
    finally:
        server.shutdown()
        server_thread.join(timeout=5)


def process_output(process):
    return (process.stdout.read() or "") + (process.stderr.read() or "")


def wait_for_root(process, url, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise CaptureError(
                "The analysis application exited before rendering its root page "
                f"(status {process.returncode}).\n{process_output(process)}"
            )
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                body = response.read()
            if response.status == 200 and b"Aurora NIC Metrics" in body:
                return
            last_error = RuntimeError(f"root returned status {response.status}")
        except urllib.error.HTTPError as exc:
            last_error = exc
        except (OSError, ValueError) as exc:
            last_error = exc
        time.sleep(0.1)
    raise TimeoutError(f"the analysis root did not become ready: {last_error}")


def wait_for_filter_status(page, expected_text, timeout_seconds):
    try:
        page.wait_for_function(
            "expected => document.querySelector('#analysis-filter-status')"
            "?.textContent.includes(expected)",
            arg=expected_text,
            timeout=int(timeout_seconds * 1000),
        )
    except Exception as exc:
        current_status = page.locator("#analysis-filter-status").first.inner_text()
        raise CaptureError(
            f"filter status did not contain {expected_text!r}; "
            f"current status: {current_status!r}: {exc}"
        ) from exc


def verify_visible_state(page):
    initial_status = page.locator("#analysis-filter-status").first.inner_text()
    assert "node all" in initial_status
    assert "direction all" in initial_status

    node_control = page.locator("#analysis-node-filter")
    node_control.click()
    for _ in range(3):
        page.keyboard.press("ArrowDown")
        page.wait_for_timeout(100)
    page.keyboard.press("Enter")
    wait_for_filter_status(page, "node 2", 10.0)
    overview_text = page.locator("#analysis-overview").first.inner_text()
    node_table_text = page.locator("#analysis-node-table").inner_text()
    assert "1 / 1" in overview_text
    assert "node-a" not in node_table_text
    assert "node-b" in node_table_text

    rx_input = page.locator("#analysis-direction-filter input[value='rx']")
    rx_input.focus()
    assert page.evaluate("document.activeElement?.value") == "rx"
    page.keyboard.press("Space")
    wait_for_filter_status(page, "direction rx", 10.0)
    assert rx_input.is_checked()
    focused_style = page.evaluate(
        "getComputedStyle(document.activeElement).outlineStyle"
    )
    focused_width = page.evaluate(
        "getComputedStyle(document.activeElement).outlineWidth"
    )
    assert focused_style != "none"
    assert focused_width != "0px"
    updated_table_text = page.locator("#analysis-node-table").inner_text()
    assert "RX" in updated_table_text
    assert "TX" not in updated_table_text

    reset_button = page.locator("#analysis-filter-reset")
    reset_button.focus()
    page.keyboard.press("Enter")
    wait_for_filter_status(page, "node all", 10.0)
    wait_for_filter_status(page, "direction all", 10.0)
    reset_table_text = page.locator("#analysis-node-table").inner_text()
    assert "node-a" in reset_table_text
    assert "node-b" in reset_table_text

    nic_detail_tab = page.locator(".tab", has_text="NIC detail")
    nic_detail_tab.click()
    matching_heading = page.locator("#nic-03-matching-heading")
    matching_heading.wait_for(state="visible", timeout=10_000)
    matching_text = page.locator(".matching-panel").first.inner_text()
    assert "Missing" in matching_text
    assert "Receive-path evidence" in matching_text
    host_pressure_heading = page.locator("#nic-04-host-pressure-heading")
    host_pressure_heading.wait_for(state="visible", timeout=10_000)
    host_pressure_text = page.locator(".host-pressure-panel").first.inner_text()
    assert "Missing" in host_pressure_text
    assert "no DMA, dispatcher, or PCIe causal claim" in host_pressure_text
    operations_heading = page.locator("#nic-05-operations-heading")
    operations_heading.wait_for(state="visible", timeout=10_000)
    operations_text = page.locator(".operations-panel").first.inner_text()
    assert "No command-queue evidence" in operations_text
    assert "opcode mapping" in operations_text
    frames_heading = page.locator("#nic-06-frames-heading")
    frames_heading.wait_for(state="visible", timeout=10_000)
    frames_text = page.locator(".frames-panel").first.inner_text()
    assert "Missing Data" in frames_text
    assert "Wire-frame evidence only" in frames_text
    translation_heading = page.locator("#nic-07-translation-heading")
    translation_heading.wait_for(state="visible", timeout=10_000)
    translation_text = page.locator(".translation-panel").first.inner_text()
    assert "No address-translation evidence" in translation_text
    assert "no ATU latency" in translation_text
    latency_heading = page.locator("#nic-08-latency-heading")
    latency_heading.wait_for(state="visible", timeout=10_000)
    latency_text = page.locator(".latency-panel").first.inner_text()
    assert "No latency-histogram evidence" in latency_text
    assert "no ATU-latency" in latency_text
    flow_control_heading = page.locator("#nic-02-flow-control-heading")
    flow_control_heading.wait_for(state="visible", timeout=10_000)
    flow_control_text = page.locator(".flow-control-panel").first.inner_text()
    assert "Missing" in flow_control_text
    assert "no congestion, bottleneck, or ECN claim" in flow_control_text
    reliability_heading = page.locator("#nic-09-reliability-heading")
    reliability_heading.wait_for(state="visible", timeout=10_000)
    reliability_text = page.locator(".reliability-panel").first.inner_text()
    assert "Missing" in reliability_text
    assert "no causal reliability claim" in reliability_text
    page.locator(".tab", has_text="Overview and nodes").click()
    page.locator("#analysis-overview").first.wait_for(state="visible")


def verify_controls_fit(page, width, height):
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(100)
    page.locator(".tab", has_text="NIC detail").click()
    for heading_selector in (
        "#nic-03-matching-heading",
        "#nic-04-host-pressure-heading",
        "#nic-05-operations-heading",
        "#nic-06-frames-heading",
        "#nic-07-translation-heading",
        "#nic-08-latency-heading",
        "#nic-02-flow-control-heading",
        "#nic-09-reliability-heading",
    ):
        diagnostic_heading = page.locator(heading_selector)
        diagnostic_heading.wait_for(state="visible", timeout=10_000)
        diagnostic_heading.scroll_into_view_if_needed()
        diagnostic_box = diagnostic_heading.bounding_box()
        if (
            diagnostic_box is None
            or diagnostic_box["x"] < 0
            or diagnostic_box["y"] < 0
            or diagnostic_box["x"] + diagnostic_box["width"] > width + 0.5
            or diagnostic_box["y"] + diagnostic_box["height"] > height + 0.5
        ):
            raise CaptureError(
                f"diagnostic heading is clipped at {width}x{height}: "
                f"{heading_selector}; box={diagnostic_box}"
            )
    page.locator(".tab", has_text="Overview and nodes").click()
    page.locator("#analysis-overview").first.wait_for(state="visible")
    for selector in (
        "#analysis-node-filter",
        "#analysis-direction-filter",
        "#analysis-time-filter",
        "#analysis-filter-status",
        "#analysis-filter-reset",
        "#overview-heading",
    ):
        locator = page.locator(selector).first
        locator.wait_for(state="visible")
        box = locator.bounding_box()
        if box is None:
            raise CaptureError(f"control is not visible: {selector}")
        if (
            box["x"] < 0
            or box["y"] < 0
            or box["x"] + box["width"] > width + 0.5
            or box["y"] + box["height"] > height + 0.5
        ):
            raise CaptureError(
                f"control is clipped at {width}x{height}: {selector}; box={box}"
            )


def capture_page(root_url, output_paths, timeout_seconds):
    try:
        from playwright.sync_api import Error as PlaywrightError, sync_playwright
    except ImportError as exc:
        raise CaptureError(
            "Playwright is required for browser rendering. Install the optional "
            "browser extra (for example: python -m pip install -e '.[browser]') "
            "and then run: python -m playwright install chromium"
        ) from exc

    viewports = (
        (1440, 900, output_paths[0]),
        (1024, 768, output_paths[1]),
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": viewports[0][0], "height": viewports[0][1]},
            device_scale_factor=1,
        )
        page = context.new_page()
        try:
            page.goto(
                root_url,
                wait_until="domcontentloaded",
                timeout=int(timeout_seconds * 1000),
            )
            page.wait_for_load_state(
                "networkidle", timeout=int(timeout_seconds * 1000)
            )
            page.locator("h1", has_text="Aurora NIC Metrics").wait_for(
                state="visible", timeout=int(timeout_seconds * 1000)
            )
            page.locator("#analysis-overview").first.wait_for(
                state="visible", timeout=int(timeout_seconds * 1000)
            )
            verify_visible_state(page)

            for width, height, output_path in viewports:
                page.set_viewport_size({"width": width, "height": height})
                page.wait_for_timeout(250)
                verify_controls_fit(page, width, height)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=output_path, full_page=False)
        except PlaywrightError as exc:
            raise CaptureError(f"browser interaction failed: {exc}") from exc
        finally:
            context.close()
            browser.close()
        return browser.version


def _paeth_predictor(left, up, upper_left):
    distance = left + up - upper_left
    left_distance = abs(distance - left)
    up_distance = abs(distance - up)
    upper_left_distance = abs(distance - upper_left)
    if left_distance <= up_distance and left_distance <= upper_left_distance:
        return left
    if up_distance <= upper_left_distance:
        return up
    return upper_left


def _decode_png_pixels(data):
    if data[:8] != PNG_SIGNATURE:
        raise CaptureError("the browser did not produce a PNG")
    position = 8
    header = None
    compressed = bytearray()
    while position + 12 <= len(data):
        length, chunk_type = struct.unpack(">I4s", data[position : position + 8])
        position += 8
        payload = data[position : position + length]
        position += length + 4
        if chunk_type == b"IHDR":
            header = struct.unpack(">IIBBBBB", payload)
        elif chunk_type == b"IDAT":
            compressed.extend(payload)
        elif chunk_type == b"IEND":
            break
    if header is None:
        raise CaptureError("the PNG IHDR chunk is missing")
    width, height, bit_depth, color_type, compression, filtering, interlace = header
    if width <= 0 or height <= 0 or bit_depth != 8:
        raise CaptureError("the PNG has unsupported dimensions or bit depth")
    if color_type not in PNG_CHANNELS or compression != 0 or filtering != 0 or interlace != 0:
        raise CaptureError("the PNG encoding is unsupported")
    channels = PNG_CHANNELS[color_type]
    stride = width * channels
    expected_size = height * (1 + stride)
    try:
        raw = zlib.decompress(bytes(compressed))
    except zlib.error as exc:
        raise CaptureError(f"the PNG image data is malformed: {exc}") from exc
    if len(raw) != expected_size:
        raise CaptureError(
            f"the PNG payload size is invalid: {len(raw)}; expected {expected_size}"
        )

    pixels = bytearray()
    previous = bytearray(stride)
    offset = 0
    for _ in range(height):
        scanline_filter = raw[offset]
        offset += 1
        line = bytearray(raw[offset : offset + stride])
        offset += stride
        if scanline_filter == 1:
            for index in range(stride):
                prior = line[index - channels] if index >= channels else 0
                line[index] = (line[index] + prior) & 0xFF
        elif scanline_filter == 2:
            for index in range(stride):
                line[index] = (line[index] + previous[index]) & 0xFF
        elif scanline_filter == 3:
            for index in range(stride):
                prior = line[index - channels] if index >= channels else 0
                line[index] = (
                    line[index] + ((prior + previous[index]) // 2)
                ) & 0xFF
        elif scanline_filter == 4:
            for index in range(stride):
                prior = line[index - channels] if index >= channels else 0
                up = previous[index]
                upper_left = previous[index - channels] if index >= channels else 0
                line[index] = (
                    line[index] + _paeth_predictor(prior, up, upper_left)
                ) & 0xFF
        elif scanline_filter != 0:
            raise CaptureError(f"unsupported PNG scanline filter: {scanline_filter}")
        pixels.extend(line)
        previous = line

    if channels in (2, 4):
        sample_indices = range(0, len(pixels), channels)
    elif channels == 3:
        sample_indices = range(0, len(pixels), 3)
    else:
        sample_indices = range(0, len(pixels))
    luma_values = [
        0.2126 * pixels[index]
        + 0.7152 * pixels[index + 1]
        + 0.0722 * pixels[index + 2]
        if channels >= 3
        else pixels[index]
        for index in sample_indices
    ]
    return width, height, luma_values


def inspect_screenshot(path, expected_width, expected_height):
    data = path.read_bytes()
    width, height, luma_values = _decode_png_pixels(data)
    if (width, height) != (expected_width, expected_height):
        raise CaptureError(
            f"unexpected screenshot dimensions: {width}x{height}; "
            f"expected {expected_width}x{expected_height}"
        )
    if len(data) < 1024:
        raise CaptureError(f"the screenshot is suspiciously small: {path}")
    if len(set(luma_values)) < 2 or statistics.pstdev(luma_values) <= 1.0:
        raise CaptureError(f"the screenshot appears blank: {path}")
    if not math.isfinite(statistics.pstdev(luma_values)):
        raise CaptureError(f"the screenshot has invalid pixel values: {path}")
    return {
        "path": path,
        "width": width,
        "height": height,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def stop_process(process):
    if process.poll() is not None:
        return process.returncode
    process.terminate()
    try:
        return process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            return process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            raise CaptureError("the application process did not terminate cleanly")
    finally:
        process.stdout.close()
        process.stderr.close()


def run_capture(args):
    port = args.port if args.port is not None else reserve_port()
    root_url = f"http://127.0.0.1:{port}/"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for_root(process, root_url, args.timeout)
        browser_version = capture_page(
            root_url,
            (args.output_1440, args.output_1024),
            args.timeout,
        )
    finally:
        server_status = stop_process(process)
    if process.poll() is None:
        raise CaptureError("the application process remains active")
    if server_status != 0:
        raise CaptureError(
            f"the application process terminated with status {server_status}"
        )

    screenshot_records = [
        inspect_screenshot(args.output_1440, 1440, 900),
        inspect_screenshot(args.output_1024, 1024, 768),
    ]
    print(f"Browser: Chromium {browser_version}")
    print(
        "Interactions: root render, keyboard node filter to node-b, "
        "keyboard direction filter to RX, keyboard filter reset, "
        "NIC-detail matching, host-pressure, operations, frames, translation, latency, flow-control, and reliability missing states, "
        "visible status/table updates"
    )
    for record in screenshot_records:
        print(
            f"Screenshot: {record['path']} | {record['width']}x{record['height']} | "
            f"{record['bytes']} bytes | sha256:{record['sha256']}"
        )
    print(f"Server: terminated cleanly with status {server_status}")


def main(argv=None):
    args = parse_arguments(argv)
    if args.serve:
        run_server(args)
        return 0
    try:
        run_capture(args)
    except (CaptureError, TimeoutError, OSError, AssertionError) as exc:
        print(f"capture_analysis_ui.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
