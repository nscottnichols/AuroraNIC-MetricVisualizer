"""Browser performance acceptance for lifecycle and interaction stages."""

from __future__ import annotations

import socket
import threading
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest
from playwright.sync_api import expect, sync_playwright
from werkzeug.serving import make_server

from analysis_app import create_analysis_app
from analysis_fixture import make_store
from dataset_lifecycle import DatasetDescriptor, DatasetLifecycleManager, DatasetState
from performance_metrics import MetricIdentity, PerformanceMetricsRecorder


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


def make_descriptor(root: Path, dataset_id: str, *, cache: bool):
    trace_path = root / f"{dataset_id}.trace"
    trace_path.write_text(dataset_id, encoding="utf-8")
    cache_path = root / f"{dataset_id}.cache" if cache else None
    if cache_path is not None:
        cache_path.write_bytes(b"cache")
    args = SimpleNamespace(
        trace=str(trace_path),
        trace_cache=str(cache_path) if cache_path is not None else None,
    )
    return DatasetDescriptor(
        dataset_id=dataset_id,
        trace_path=trace_path,
        cache_path=cache_path,
        args=args,
    )


def test_lifecycle_and_interaction_performance_stages(tmp_path):
    stores = {
        "cached-zero-rx": make_store(include_rx=True, sample_count=60),
        "cold-missing-rx": make_store(include_rx=False, sample_count=30),
    }

    def loader(args, cancellation):
        dataset_id = Path(args.trace).stem
        return stores[dataset_id]

    descriptors = [
        make_descriptor(tmp_path, "cached-zero-rx", cache=True),
        make_descriptor(tmp_path, "cold-missing-rx", cache=False),
    ]
    recorder = PerformanceMetricsRecorder()
    manager = DatasetLifecycleManager(
        tmp_path, descriptors, loader=loader, performance_recorder=recorder
    )
    initial_token = manager.open("cached-zero-rx")
    assert manager.wait(initial_token, timeout=2.0).state is DatasetState.READY
    app = create_analysis_app(
        manager.current().store,
        dataset_manager=manager,
        performance_recorder=recorder,
    )
    server, thread, url = start_server(app)

    readiness_operation = recorder.begin(
        "app_readiness",
        dataset_token="cached-zero-rx",
        request_token=uuid.uuid4().hex,
        selection_token="initial",
    )
    render_operation = recorder.begin(
        "first_meaningful_render",
        dataset_token="cached-zero-rx",
        request_token=uuid.uuid4().hex,
        selection_token="initial",
    )

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=15_000)
            recorder.finish(
                readiness_operation,
                result_identity=MetricIdentity(
                    "cached-zero-rx",
                    readiness_operation.identity.request_token,
                    "initial",
                ),
                payload_bytes=len(page.content().encode("utf-8")),
            )
            page.wait_for_selector("#analysis-overview", state="visible", timeout=15_000)
            recorder.finish(
                render_operation,
                result_identity=MetricIdentity(
                    "cached-zero-rx",
                    render_operation.identity.request_token,
                    "initial",
                ),
                payload_bytes=len(page.content().encode("utf-8")),
            )

            page.locator("#dataset-select").click()
            option = page.get_by_role("option", name="cold-missing-rx", exact=True)
            expect(option).to_be_visible(timeout=15_000)
            option.click()
            page.locator("#dataset-request-button").click()
            expect(page.locator("#dataset-status")).to_contain_text(
                "Ready", timeout=15_000
            )

            filter_status = page.locator("#analysis-filter-status")
            for index in range(31):
                direction = "TX" if index % 2 == 0 else "All"
                page.locator(
                    "#analysis-direction-filter label", has_text=direction
                ).click()
                expect(filter_status).to_contain_text(
                    f"direction {direction.lower()}", timeout=15_000
                )

            context.close()
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        manager.close()

    summaries = {
        stage: recorder.stage_summary(stage)
        for stage in (
            "app_readiness",
            "first_meaningful_render",
            "cache_reopen",
            "cold_source_load",
            "warm_new_query",
            "warm_repeat",
        )
    }
    assert summaries["app_readiness"].success_count == 1
    assert summaries["first_meaningful_render"].success_count == 1
    assert summaries["cache_reopen"].success_count >= 1
    assert summaries["cold_source_load"].success_count >= 1
    assert summaries["warm_new_query"].success_count >= 1
    assert summaries["warm_repeat"].sample_count >= 30
    assert summaries["warm_repeat"].percentile_available is True
    assert summaries["warm_repeat"].p50_ms is not None
    assert summaries["warm_repeat"].p95_ms is not None
    print("Performance stage summaries:")
    for stage, summary in summaries.items():
        print(
            f"{stage}: samples={summary.sample_count}, "
            f"success={summary.success_count}, p50_ms={summary.p50_ms}, "
            f"p95_ms={summary.p95_ms}"
        )
