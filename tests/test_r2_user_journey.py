"""Independent browser acceptance for focused diagnostics.

The fixtures intentionally separate observed zero from missing evidence:
``make_store(include_rx=True)`` contains observed-zero RX samples, while
``make_store(include_rx=False)`` contains no RX observations.
"""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import pytest
from playwright.sync_api import expect, sync_playwright
from werkzeug.serving import make_server

from analysis_app import create_analysis_app
from analysis_fixture import make_store


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return int(server_socket.getsockname()[1])


def start_server(store):
    port = reserve_port()
    app = create_analysis_app(store)
    server = make_server("127.0.0.1", port, app.server, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{port}/"


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        launched = playwright.chromium.launch(headless=True)
        try:
            yield launched
        finally:
            launched.close()


@pytest.fixture
def zero_rx_page(browser):
    server, thread, url = start_server(make_store(include_rx=True, sample_count=60))
    try:
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=15_000)
        page.wait_for_selector("#analysis-overview", state="visible", timeout=15_000)
        try:
            yield page
        finally:
            context.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture
def missing_rx_page(browser):
    server, thread, url = start_server(make_store(include_rx=False, sample_count=60))
    try:
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=15_000)
        page.wait_for_selector("#analysis-overview", state="visible", timeout=15_000)
        try:
            yield page
        finally:
            context.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)


def select_option(page, selector: str, label: str) -> None:
    page.locator(selector).click()
    option = page.get_by_role("option", name=label, exact=True)
    expect(option).to_be_visible(timeout=15_000)
    option.click()


def select_first_physical_nic(page) -> None:
    select_option(page, "#analysis-node-filter", "node-a (ID 1)")
    select_option(page, "#analysis-nic-filter", "cxi-0 on node 1 (ID 10)")


def open_questions(page) -> None:
    page.locator(".tab", has_text="Questions").click()
    expect(page.locator(".question-index").last).to_be_visible(timeout=15_000)


def download_export(page, tmp_path: Path) -> dict:
    output_path = tmp_path / "r2-diagnostic-export.json"
    with page.expect_download(timeout=15_000) as download_info:
        page.locator("#analysis-export-button").click()
    download_info.value.save_as(str(output_path))
    return json.loads(output_path.read_text(encoding="utf-8"))


def test_question_index_separates_capability_from_current_answerability(zero_rx_page):
    page = zero_rx_page
    open_questions(page)

    index = page.locator(".question-index").last
    expect(index).to_contain_text("Dataset capability")
    expect(index).to_contain_text("Current selection")
    expect(index).to_contain_text("NIC-01")
    expect(index).to_contain_text("Not answerable now")

    select_first_physical_nic(page)
    open_questions(page)
    expect(index).to_contain_text("Answerable now")


def test_supported_diagnostic_preserves_identity_units_window_and_export(
    zero_rx_page, tmp_path
):
    page = zero_rx_page
    select_first_physical_nic(page)
    open_questions(page)
    select_option(page, "#analysis-question-select", "NIC-01 — NIC traffic")

    result = page.locator("#analysis-question-result").last
    expect(result).to_contain_text("NIC-01: NIC traffic", timeout=15_000)
    expect(result).to_contain_text("node 1 / NIC 10")
    expect(result).to_contain_text("Exact window")
    expect(result).to_contain_text("bytes/s")
    expect(result).to_contain_text("Collector policy")

    exported = download_export(page, tmp_path)
    assert exported["run"]["run_id"] == "run-synthetic"
    assert exported["filters"]["node_id"] == 1
    assert exported["filters"]["nic_id"] == 10
    assert exported["methodology"]["units"]["traffic_totals"] == "bytes"
    assert exported["methodology"]["units"]["rates"] == "bytes/s"
    assert "collector_policy" in exported["nic_detail"]
    assert exported["nic_detail"]["collector_policy"]


def test_observed_zero_and_missing_rx_remain_distinct(
    zero_rx_page, missing_rx_page
):
    page = zero_rx_page
    select_first_physical_nic(page)
    open_questions(page)
    select_option(page, "#analysis-question-select", "NIC-01 — NIC traffic")
    result = page.locator("#analysis-question-result").last
    expect(result).to_contain_text("RX total", timeout=15_000)
    expect(result).to_contain_text("0 bytes")
    expect(result).not_to_contain_text("RX Missing")

    page = missing_rx_page
    select_first_physical_nic(page)
    open_questions(page)
    select_option(page, "#analysis-question-select", "NIC-01 — NIC traffic")
    result = page.locator("#analysis-question-result").last
    expect(result).to_contain_text("RX total", timeout=15_000)
    expect(result).to_contain_text("Missing")
    expect(result).not_to_contain_text("0 bytes")


def test_unsupported_diagnostic_and_named_interval_comparison(zero_rx_page):
    page = zero_rx_page
    select_first_physical_nic(page)
    open_questions(page)
    select_option(page, "#analysis-question-select", "NIC-09 — Reliability")

    result = page.locator("#analysis-question-result").last
    expect(result).to_contain_text("NIC-09: Reliability", timeout=15_000)
    expect(result).to_contain_text("Missing")
    expect(result).not_to_contain_text("0 reliability")

    page.locator("#analysis-question-interval-name").fill("early")
    page.locator("#analysis-question-interval-start").fill("0")
    page.locator("#analysis-question-interval-end").fill("1")
    page.locator("#analysis-question-interval-add").click()
    comparison = page.locator("#analysis-question-interval-result").last
    expect(comparison).to_contain_text("early TX total", timeout=15_000)
    expect(comparison).to_contain_text("exact window 0–1000000000 ns")

    page.locator("#analysis-question-interval-name").fill("late")
    page.locator("#analysis-question-interval-start").fill("2")
    page.locator("#analysis-question-interval-end").fill("3")
    page.locator("#analysis-question-interval-add").click()
    expect(comparison).to_contain_text("late TX total", timeout=15_000)
    expect(comparison).to_contain_text("exact window 2000000000–3000000000 ns")
