"""Browser acceptance for dataset open, switch, cancel, and failure recovery."""

from __future__ import annotations

import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from playwright.sync_api import expect, sync_playwright
from werkzeug.serving import make_server

from analysis_app import create_analysis_app
from analysis_fixture import make_store
from dataset_lifecycle import DatasetDescriptor, DatasetLifecycleManager, DatasetState


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return int(server_socket.getsockname()[1])


def start_server(app):
    port = reserve_port()
    server = make_server("127.0.0.1", port, app.server, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{port}/"


def make_descriptor(root: Path, dataset_id: str) -> DatasetDescriptor:
    trace_path = root / f"{dataset_id}.trace"
    trace_path.write_text(dataset_id, encoding="utf-8")
    args = SimpleNamespace(trace=str(trace_path), trace_cache=None)
    return DatasetDescriptor(
        dataset_id=dataset_id,
        trace_path=trace_path,
        cache_path=None,
        args=args,
    )


def make_manager(root: Path, loader) -> DatasetLifecycleManager:
    descriptors = [make_descriptor(root, "zero-rx"), make_descriptor(root, "missing-rx")]
    return DatasetLifecycleManager(root, descriptors, loader=loader)


def make_ready_manager(root: Path) -> DatasetLifecycleManager:
    stores = {
        "zero-rx": make_store(include_rx=True, sample_count=60),
        "missing-rx": make_store(include_rx=False, sample_count=30),
    }

    def loader(args, cancellation):
        dataset_id = Path(args.trace).stem
        return stores[dataset_id]

    return make_manager(root, loader)


def open_initial_dataset(manager: DatasetLifecycleManager) -> None:
    token = manager.open("zero-rx")
    status = manager.wait(token, timeout=2.0)
    assert status.state is DatasetState.READY


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        launched = playwright.chromium.launch(headless=True)
        try:
            yield launched
        finally:
            launched.close()


def test_dataset_switch_updates_current_result_without_stale_zero(
    browser, tmp_path
):
    manager = make_ready_manager(tmp_path)
    open_initial_dataset(manager)
    app = create_analysis_app(manager.current().store, dataset_manager=manager)
    server, thread, url = start_server(app)
    try:
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=15_000)
        page.wait_for_selector("#analysis-overview", state="visible", timeout=15_000)

        page.locator("#dataset-select").click()
        option = page.get_by_role("option", name="missing-rx", exact=True)
        expect(option).to_be_visible(timeout=15_000)
        option.click()
        page.locator("#dataset-request-button").click()

        status = page.locator("#dataset-status")
        expect(status).to_contain_text("Ready", timeout=15_000)
        expect(status).to_contain_text("missing-rx")

        overview = page.locator("#analysis-overview")
        expect(overview).to_contain_text("RX total", timeout=15_000)
        rx_value = overview.locator(".compact-card", has_text="RX total").locator(
            ".stat-value"
        )
        expect(rx_value).to_have_text("Missing", timeout=15_000)
        context.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        manager.close()


def test_cancel_and_failure_preserve_last_usable_dataset(browser, tmp_path):
    release = threading.Event()
    stores = {
        "zero-rx": make_store(include_rx=True, sample_count=60),
    }

    def cancel_loader(args, cancellation):
        dataset_id = Path(args.trace).stem
        if dataset_id == "missing-rx":
            while not cancellation.should_cancel():
                release.wait(timeout=0.01)
            cancellation.check("cancel test")
        return stores[dataset_id]

    cancel_manager = make_manager(tmp_path, cancel_loader)
    open_initial_dataset(cancel_manager)
    app = create_analysis_app(cancel_manager.current().store, dataset_manager=cancel_manager)
    server, thread, url = start_server(app)
    try:
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=15_000)
        page.wait_for_selector("#analysis-overview", state="visible", timeout=15_000)

        page.locator("#dataset-select").click()
        option = page.get_by_role("option", name="missing-rx", exact=True)
        expect(option).to_be_visible(timeout=15_000)
        option.click()
        page.locator("#dataset-request-button").click()
        expect(page.locator("#dataset-status")).to_contain_text("Loading", timeout=15_000)
        page.locator("#dataset-cancel-button").click()
        expect(page.locator("#dataset-status")).to_contain_text("Cancelled", timeout=15_000)
        rx_value = page.locator("#analysis-overview").locator(
            ".compact-card", has_text="RX total"
        ).locator(".stat-value")
        expect(rx_value).to_have_text("0 B", timeout=15_000)
        context.close()
    finally:
        release.set()
        server.shutdown()
        thread.join(timeout=5)
        cancel_manager.close()

    def failure_loader(args, cancellation):
        dataset_id = Path(args.trace).stem
        if dataset_id == "missing-rx":
            raise RuntimeError("synthetic dataset failure")
        return stores[dataset_id]

    failure_manager = make_manager(tmp_path, failure_loader)
    open_initial_dataset(failure_manager)
    app = create_analysis_app(failure_manager.current().store, dataset_manager=failure_manager)
    server, thread, url = start_server(app)
    try:
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=15_000)
        page.wait_for_selector("#analysis-overview", state="visible", timeout=15_000)

        page.locator("#dataset-select").click()
        option = page.get_by_role("option", name="missing-rx", exact=True)
        expect(option).to_be_visible(timeout=15_000)
        option.click()
        page.locator("#dataset-request-button").click()
        expect(page.locator("#dataset-error")).to_contain_text(
            "synthetic dataset failure", timeout=15_000
        )
        rx_value = page.locator("#analysis-overview").locator(
            ".compact-card", has_text="RX total"
        ).locator(".stat-value")
        expect(rx_value).to_have_text("0 B", timeout=15_000)
        context.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        failure_manager.close()
