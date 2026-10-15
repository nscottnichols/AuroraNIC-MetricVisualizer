"""Focused regressions for legacy question navigation and diagnostic honesty."""

from __future__ import annotations

import socket
import threading

from playwright.sync_api import expect, sync_playwright

from analysis_app import AnalysisFilterState, build_view_model, create_analysis_app, default_filters
from analysis_fixture import make_store
from ui.diagnostics import diagnostic_payload, render_diagnostic
from ui.question_index import QUESTION_ROUTES, question_rows, render_question_index


def test_question_index_separates_capability_from_current_answerability():
    store = make_store()
    filters = default_filters(store)
    view = build_view_model(store, filters)
    rows = question_rows(view, {})

    assert len(rows) == 15
    assert [row["id"] for row in rows] == list(QUESTION_ROUTES)
    nic01 = next(row for row in rows if row["id"] == "NIC-01")
    nic09 = next(row for row in rows if row["id"] == "NIC-09")
    assert nic01["dataset_capability"] == "supported"
    assert nic01["current_selection"] == "Not answerable now"
    assert "Select one physical NIC" in nic01["reason"]
    assert nic09["dataset_capability"] == "unsupported"
    assert nic09["current_selection"] == "Not answerable now"

    rendered = render_question_index(view)
    rendered_text = str(rendered.to_plotly_json())
    assert "Dataset capability" in rendered_text
    assert "Current selection" in rendered_text


def test_focused_diagnostics_preserve_identity_window_units_and_missingness():
    store = make_store()
    filters = AnalysisFilterState(
        node_id=1,
        nic_id=10,
        direction="all",
        start_ns=0,
        end_ns=59_900_000_000,
    )
    view = build_view_model(store, filters)

    traffic = render_diagnostic(view, "NIC-01")
    traffic_text = str(traffic.to_plotly_json())
    assert "NIC-01: NIC traffic" in traffic_text
    assert "TX total" in traffic_text
    assert "bytes" in traffic_text
    assert "bytes/s" in traffic_text
    assert "node 1 / NIC 10" in traffic_text
    assert "0–59900000000 ns" in traffic_text

    reliability_payload = diagnostic_payload(view, "NIC-09")
    assert reliability_payload is not None
    assert reliability_payload["state"] == "missing"
    reliability = render_diagnostic(view, "NIC-09")
    reliability_text = str(reliability.to_plotly_json())
    assert "Missing reliability evidence" in reliability_text
    assert "No node/NIC reliability counter series is available" in reliability_text
    assert "Missing" in reliability_text
    assert "0 reliability" not in reliability_text


def test_every_catalog_question_has_a_focused_route():
    store = make_store()
    filters = AnalysisFilterState(
        node_id=1,
        nic_id=10,
        direction="tx",
        start_ns=0,
        end_ns=59_900_000_000,
    )
    view = build_view_model(store, filters)
    for question_id in QUESTION_ROUTES:
        rendered = render_diagnostic(view, question_id)
        rendered_text = str(rendered.to_plotly_json())
        assert question_id in rendered_text
        assert "Exact window" in rendered_text
        assert "Collector policy" in rendered_text


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return int(server_socket.getsockname()[1])


def test_browser_journey_opens_requested_diagnostic_and_preserves_selection():
    from tests.test_r1_user_journey import make_acceptance_store
    from werkzeug.serving import make_server

    port = reserve_port()
    app = create_analysis_app(make_acceptance_store())
    server = make_server("127.0.0.1", port, app.server, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()
            page.goto(f"http://127.0.0.1:{port}/", wait_until="domcontentloaded", timeout=15_000)
            page.wait_for_selector("#analysis-overview", state="visible", timeout=15_000)

            page.locator(".tab", has_text="Questions").click()
            index = page.locator(".question-index").last
            expect(index).to_contain_text("Dataset capability")
            expect(index).to_contain_text("Current selection")
            expect(index).to_contain_text("NIC-09")
            result = page.locator("#analysis-question-result").last
            expect(result).to_contain_text("No diagnostic requested yet")

            page.locator("#analysis-node-filter").click()
            page.get_by_role("option", name="node-a (ID 1)").click()
            page.locator("#analysis-nic-filter").click()
            page.get_by_role("option", name="cxi-0 on node 1 (ID 10)").click()
            page.locator("#analysis-direction-filter label", has_text="TX").click()
            page.locator("#analysis-question-select").click()
            page.get_by_role("option", name="NIC-09 — Reliability").click()

            expect(result).to_contain_text("NIC-09: Reliability", timeout=15_000)
            expect(result).to_contain_text("Missing reliability evidence")
            expect(result).to_contain_text("node 1 / NIC 10")
            expect(result).to_contain_text("Exact window")
            expect(result).to_contain_text("not zero-valued observations")

            page.locator("#analysis-question-select").click()
            page.get_by_role("option", name="NIC-01 — NIC traffic").click()
            expect(result).to_contain_text("NIC-01: NIC traffic", timeout=15_000)
            expect(result).to_contain_text("TX total")
            expect(result).to_contain_text("bytes/s")

            page.locator("#analysis-question-interval-name").fill("first-window")
            page.locator("#analysis-question-interval-start").fill("0")
            page.locator("#analysis-question-interval-end").fill("2")
            page.locator("#analysis-question-interval-add").click()
            comparison = page.locator("#analysis-question-interval-result").last
            expect(comparison).to_contain_text("first-window TX total", timeout=15_000)
            expect(comparison).to_contain_text("exact window 0–2000000000 ns")

            page.locator("#analysis-question-interval-name").fill("second-window")
            page.locator("#analysis-question-interval-start").fill("3")
            page.locator("#analysis-question-interval-end").fill("5")
            page.locator("#analysis-question-interval-add").click()
            expect(comparison).to_contain_text("second-window TX total", timeout=15_000)
            expect(comparison).to_contain_text("exact window 3000000000–5000000000 ns")
            expect(comparison).to_contain_text("direct comparison of two independently named windows")
            context.close()
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
