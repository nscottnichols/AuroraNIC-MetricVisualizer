"""Independent browser acceptance for traffic investigations.

The fixture is deliberately small and labeled synthetic. It uses the same NIC
label on two nodes, asymmetric traffic, and missing RX on the second node so a
passing journey also checks physical identity and explicit unavailable states.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
from pathlib import Path

import numpy as np
import pytest
from playwright.sync_api import expect, sync_playwright

from analysis_app import create_analysis_app
from counter_semantic_registry import COUNTER_SEMANTIC_REGISTRY_VERSION
from trace_model import (
    CollectorIdentity,
    CounterMetadata,
    CounterSemantics,
    Coverage,
    NicIdentity,
    NodeIdentity,
    NormalizedBatch,
    RunIdentity,
    ValueKind,
)
from trace_store import TraceStore


RUN = RunIdentity(
    run_id="r1-journey",
    source_fingerprint="sha256:r1-journey-fixture",
    source_format="synthetic-ctf",
    reader_version="acceptance-harness",
    conversion_lineage="direct",
    clock_domain="single-monotonic-clock",
    synchronization_evidence="single-clock fixture",
)
NODE_A = NodeIdentity(1, "node-a", "node-a")
NODE_B = NodeIdentity(2, "node-b", "node-b")
NIC_A = NicIdentity(10, NODE_A.node_id, "cxi-0", "cxi-0")
NIC_B = NicIdentity(11, NODE_B.node_id, "cxi-0", "cxi-0")
COLLECTOR = CollectorIdentity(20, "collector-a", rank=0)
AMBIGUOUS_COLLECTOR = CollectorIdentity(21, "collector-b", rank=1)
TX_COUNTER = CounterMetadata(
    30,
    "hni_sts_tx_ok_octets",
    ValueKind.EXACT_UINT64,
    CounterSemantics.CUMULATIVE,
    "bytes",
    baseline_relative=False,
    registry_version=COUNTER_SEMANTIC_REGISTRY_VERSION,
)
RX_COUNTER = CounterMetadata(
    31,
    "hni_sts_rx_ok_octets",
    ValueKind.EXACT_UINT64,
    CounterSemantics.CUMULATIVE,
    "bytes",
    baseline_relative=False,
    registry_version=COUNTER_SEMANTIC_REGISTRY_VERSION,
)


def make_acceptance_store() -> TraceStore:
    """Build two physical NICs with a shared readable label and known answers."""

    sample_count = 120
    timestamps_ns = np.arange(sample_count, dtype=np.int64) * 100_000_000
    rows: list[tuple[NodeIdentity, NicIdentity, CounterMetadata, CollectorIdentity, int, int]] = []
    for timestamp_ns, value in zip(timestamps_ns, np.arange(sample_count) * 1000):
        rows.append((NODE_A, NIC_A, TX_COUNTER, COLLECTOR, int(timestamp_ns), int(value)))
    for timestamp_ns, value in zip(timestamps_ns, np.arange(sample_count) * 600):
        rows.append((NODE_A, NIC_A, RX_COUNTER, COLLECTOR, int(timestamp_ns), int(value)))
    for timestamp_ns, value in zip(timestamps_ns, np.arange(sample_count) * 300):
        rows.append((NODE_B, NIC_B, TX_COUNTER, COLLECTOR, int(timestamp_ns), int(value)))
    for timestamp_ns, value in zip(timestamps_ns, np.arange(sample_count) * 250):
        rows.append(
            (
                NODE_B,
                NIC_B,
                TX_COUNTER,
                AMBIGUOUS_COLLECTOR,
                int(timestamp_ns),
                int(value),
            )
        )

    requested = (TX_COUNTER.name, RX_COUNTER.name)
    batch = NormalizedBatch(
        identity=RUN,
        nodes=(NODE_A, NODE_B),
        nics=(NIC_A, NIC_B),
        counters=(TX_COUNTER, RX_COUNTER),
        collectors=(COLLECTOR, AMBIGUOUS_COLLECTOR),
        coverage=Coverage(
            requested_counter_names=requested,
            available_counter_names=requested,
            readable_counter_names=requested,
            observed_counter_names=requested,
            acquisition_start_ns=int(timestamps_ns[0]),
            acquisition_end_ns=int(timestamps_ns[-1]),
            lost_record_count=0,
            gap_count=0,
        ),
        node_ids=np.asarray([row[0].node_id for row in rows], dtype=np.int64),
        nic_ids=np.asarray([row[1].nic_id for row in rows], dtype=np.int64),
        counter_ids=np.asarray([row[2].counter_id for row in rows], dtype=np.int64),
        collector_ids=np.asarray([row[3].collector_id for row in rows], dtype=np.int64),
        timestamps_ns=np.asarray([row[4] for row in rows], dtype=np.int64),
        record_ids=np.asarray([f"r1-{index}" for index in range(len(rows))]),
        uint64_values=np.asarray([row[5] for row in rows], dtype=np.uint64),
        float64_values=np.full(len(rows), np.nan, dtype=np.float64),
        quality_flags=np.zeros(len(rows), dtype=np.int64),
    )
    return TraceStore.from_batches([batch])


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return int(server_socket.getsockname()[1])


@pytest.fixture(scope="module")
def server_url():
    from werkzeug.serving import make_server

    port = reserve_port()
    app = create_analysis_app(make_acceptance_store())
    server = make_server("127.0.0.1", port, app.server, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        launched = playwright.chromium.launch(headless=True)
        try:
            yield launched
        finally:
            launched.close()


@pytest.fixture
def page(browser, server_url):
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    current_page = context.new_page()
    current_page.goto(server_url, wait_until="domcontentloaded", timeout=15_000)
    current_page.wait_for_selector("#analysis-overview", state="visible", timeout=15_000)
    try:
        yield current_page
    finally:
        context.close()


def select_dash_option_by_accessible_label(
    page, selector: str, option_label: str
) -> None:
    page.locator(selector).click()
    option = page.get_by_role("option", name=option_label, exact=True)
    expect(option).to_be_visible(timeout=15_000)
    option.click()


def select_physical_nic(page, node: str, nic: str) -> None:
    select_dash_option_by_accessible_label(
        page, "#analysis-node-filter", "node-a (ID 1)"
    )
    expect(page.locator("#analysis-filter-status").last).to_contain_text(
        "node 1", timeout=15_000
    )
    select_dash_option_by_accessible_label(
        page, "#analysis-nic-filter", "cxi-0 on node 1 (ID 10)"
    )
    expect(page.locator("#analysis-filter-status").last).to_contain_text(
        "NIC 10", timeout=15_000
    )


def set_shared_interval(page) -> None:
    handles = page.locator("#analysis-time-filter input")
    expect(handles).to_have_count(2)
    handles.nth(0).fill("2000000000")
    handles.nth(0).press("Tab")
    expect(page.locator("#analysis-filter-status").last).to_contain_text(
        "window 2000000000–11900000001 ns", timeout=15_000
    )
    handles = page.locator("#analysis-time-filter input")
    handles.nth(1).fill("4000000000")
    handles.nth(1).press("Tab")
    expect(page.locator("#analysis-filter-status").last).to_contain_text(
        "window 2000000000–4000000000 ns", timeout=15_000
    )


def download_export(page, tmp_path: Path) -> dict:
    output_path = tmp_path / "r1-journey-export.json"
    with page.expect_download(timeout=15_000) as download_info:
        page.locator("#analysis-export-button").click()
    download_info.value.save_as(str(output_path))
    return json.loads(output_path.read_text(encoding="utf-8"))


def assert_in_initial_viewport(locator) -> None:
    expect(locator).to_be_visible()
    bounding_box = locator.bounding_box()
    assert bounding_box is not None
    assert bounding_box["y"] >= 0
    assert bounding_box["y"] + bounding_box["height"] <= 900


def open_node_screen_with_keyboard(page, node_label: str):
    open_action = page.get_by_role(
        "button", name=r"open .*" + node_label + r".*cxi-0", exact=False
    ).first
    expect(open_action).to_have_count(1, timeout=15_000)
    open_action.focus()
    assert page.evaluate("document.activeElement?.tagName") in {"BUTTON", "A"}
    page.keyboard.press("Enter")
    node_heading = page.get_by_role("heading", name="Node comparison", exact=True)
    expect(node_heading).to_be_visible(timeout=15_000)
    return node_heading


def open_nic_screen_with_keyboard(page, node_label: str = "node-a"):
    open_action = page.get_by_role(
        "button", name=r"open .*" + node_label + r".*cxi-0", exact=False
    ).first
    expect(open_action).to_have_count(1, timeout=15_000)
    open_action.focus()
    assert page.evaluate("document.activeElement?.tagName") in {"BUTTON", "A"}
    page.keyboard.press("Enter")
    nic_heading = page.get_by_role("heading", name="Single-NIC traffic", exact=True)
    expect(nic_heading).to_be_visible(timeout=15_000)
    return nic_heading


def test_completed_slice_viewport_screenshots(page):
    open_action = page.get_by_role(
        "button", name=r"open .*node-a.*cxi-0", exact=False
    ).first
    open_action.focus()
    page.keyboard.press("Enter")
    expect(page.locator("#analysis-filter-status").last).to_contain_text(
        "node 1", timeout=15_000
    )
    expect(page.locator("#analysis-filter-status").last).to_contain_text(
        "NIC 10", timeout=15_000
    )
    set_shared_interval(page)
    page.locator(".tab", has_text="NIC detail").click()
    detail = page.locator("#analysis-nic-detail").last
    expect(detail).to_contain_text("node-a / cxi-0")
    expect(detail).to_contain_text("TX")
    expect(detail).to_contain_text("RX")

    evidence_root = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "ui-recovery"
        / "evidence"
    )
    evidence_root.mkdir(parents=True, exist_ok=True)
    outputs = (
        (
            evidence_root / "r1-user-journey-post-fix-1440x900.png",
            1440,
            900,
        ),
        (
            evidence_root / "r1-user-journey-post-fix-1024x768.png",
            1024,
            768,
        ),
    )
    for output_path, width, height in outputs:
        if output_path.exists():
            output_path.unlink()
        page.set_viewport_size({"width": width, "height": height})
        page.wait_for_timeout(250)
        page.screenshot(path=output_path, full_page=False)
        with output_path.open("rb") as screenshot_file:
            header = screenshot_file.read(24)
        assert header[:8] == b"\x89PNG\r\n\x1a\n"
        actual_width, actual_height = struct.unpack(">II", header[16:24])
        assert (actual_width, actual_height) == (width, height)


def test_component_ids_are_unique(page):
    identifiers = page.eval_on_selector_all("[id]", "elements => elements.map(e => e.id)")
    duplicated = sorted({identifier for identifier in identifiers if identifiers.count(identifier) > 1})
    assert not duplicated, f"Component IDs must be unique; duplicated IDs: {duplicated}"


def test_job_overview_orients_the_analyst(page):
    overview = page.locator("#analysis-overview").last
    expect(overview).to_contain_text("Run")
    expect(overview).to_contain_text("r1-journey")
    expect(overview).to_contain_text("TX total")
    expect(overview).to_contain_text("RX total")
    expect(overview).to_contain_text("Nodes / NICs")
    expect(overview).to_contain_text("Minimum traffic coverage")


def test_persistent_shell_and_initial_viewport_prioritize_investigation(page):
    shell = page.get_by_role("banner")
    expect(shell).to_be_visible()
    for shell_content in (
        "r1-journey",
        "synthetic-ctf",
        "sha256:r1-journey-fixture",
        "Synthetic fixture",
        "Direction",
        "Coverage",
        "0.000 s – 11.900 s",
    ):
        expect(shell).to_contain_text(shell_content)

    for control_name in ("Back", "Reset selection", "Reset interval", "Export"):
        control = page.get_by_role("button", name=control_name, exact=True)
        expect(control).to_have_count(1)
        control.focus()
        assert page.evaluate("document.activeElement?.textContent") == control_name

    primary_timeline = page.get_by_text("Aggregate traffic timeline", exact=True)
    ranked_selection = page.get_by_text("Ranked node/NIC selection", exact=True)
    evidence_matrix = page.get_by_text("Node×NIC evidence matrix", exact=True)
    assert_in_initial_viewport(primary_timeline)
    assert_in_initial_viewport(ranked_selection)
    assert_in_initial_viewport(evidence_matrix)

    tab_names = [tab.inner_text() for tab in page.get_by_role("tab").all()]
    assert "Questions and counters" not in tab_names, (
        "question/counter inventories and diagnostic tables must not be the "
        "initial navigation or substitute for the traffic investigation slice"
    )


def test_ranked_node_nic_selection_is_explicit_and_keyboard_operable(page):
    table = page.locator("#analysis-node-table")
    expect(table).to_contain_text("node-a")
    expect(table).to_contain_text("node-b")
    table_text = table.inner_text()
    assert table_text.index("node-a") < table_text.index("node-b"), (
        "node-a must rank above node-b for the fixture's asymmetric traffic"
    )

    open_actions = page.get_by_role("button", name=r"open .*node-a.*cxi-0", exact=False)
    assert open_actions.count() > 0, (
        "the ranked node/NIC result must expose an explicit open action; "
        "the investigation must include physical NIC selection"
    )
    open_actions.first.focus()
    assert page.evaluate("document.activeElement?.tagName") in {"BUTTON", "A"}


def test_node_and_nic_screens_are_distinct_and_preserve_shared_interval(page):
    set_shared_interval(page)
    node_heading = open_node_screen_with_keyboard(page, "node-a")
    shell = page.get_by_role("banner")
    expect(shell).to_contain_text("2.000 s – 4.000 s")
    expect(page.get_by_role("heading", name="Single-NIC traffic", exact=True)).to_have_count(0)

    open_nic_screen_with_keyboard(page)
    expect(node_heading).not_to_be_visible()
    nic_detail = page.locator("#analysis-nic-detail").last
    expect(nic_detail).to_contain_text("node-a / cxi-0")
    expect(nic_detail).to_contain_text("Rate (bytes/s)")
    expect(nic_detail).to_contain_text("Cumulative bytes")
    expect(shell).to_contain_text("2.000 s – 4.000 s")

    back = page.get_by_role("button", name="Back", exact=True)
    back.focus()
    page.keyboard.press("Enter")
    expect(page.get_by_role("heading", name="Node comparison", exact=True)).to_be_visible(
        timeout=15_000
    )
    expect(shell).to_contain_text("2.000 s – 4.000 s")


def test_reset_selection_and_reset_interval_have_separate_keyboard_semantics(page):
    set_shared_interval(page)
    open_node_screen_with_keyboard(page, "node-a")
    open_nic_screen_with_keyboard(page)
    shell = page.get_by_role("banner")

    reset_selection = page.get_by_role("button", name="Reset selection", exact=True)
    reset_selection.focus()
    page.keyboard.press("Enter")
    expect(page.get_by_role("heading", name="Job overview", exact=True)).to_be_visible(
        timeout=15_000
    )
    expect(shell).to_contain_text("2.000 s – 4.000 s")

    open_node_screen_with_keyboard(page, "node-a")
    open_nic_screen_with_keyboard(page)
    reset_interval = page.get_by_role("button", name="Reset interval", exact=True)
    reset_interval.focus()
    page.keyboard.press("Enter")
    expect(page.get_by_role("heading", name="Single-NIC traffic", exact=True)).to_be_visible(
        timeout=15_000
    )
    expect(page.locator("#analysis-nic-detail").last).to_contain_text("node-a / cxi-0")
    expect(shell).to_contain_text("0.000 s – 11.900 s")


def test_single_nic_tx_rx_matches_identity_window_and_export(page, tmp_path):
    select_physical_nic(page, "node-a", "cxi-0 on node 1")
    set_shared_interval(page)
    page.locator(".tab", has_text="NIC detail").click()
    expect(page.locator("#nic-detail-heading")).to_be_visible()
    detail = page.locator("#analysis-nic-detail").last
    expect(detail).to_contain_text("node-a / cxi-0")

    export = download_export(page, tmp_path)
    assert export["run"]["run_id"] == "r1-journey"
    assert export["filters"]["node_id"] == 1
    assert export["filters"]["nic_id"] == 10
    assert export["filters"]["start_ns"] == 2_000_000_000
    assert export["filters"]["end_ns"] == 4_000_000_000

    series = export["nic_detail"]["series"]
    assert len(series) == 2, (
        "single-NIC export must contain exactly one TX and one RX record for "
        f"node-a/cxi-0; got {len(series)} records"
    )
    assert {record["node"] for record in series} == {"node-a"}, [
        record["node"] for record in series
    ]
    assert {record["nic"] for record in series} == {"cxi-0"}
    assert {record["direction"].lower() for record in series} == {"tx", "rx"}
    totals = {
        record["direction"].lower(): record["total_bytes"] for record in series
    }
    assert totals == {"tx": 19_000, "rx": 11_400}, totals
    rates = {
        record["direction"].lower(): record["mean_rate_bytes_per_s"]
        for record in series
    }
    assert rates == {"tx": 10_000.0, "rx": 6_000.0}, rates
    assert all(record.get("coverage_fraction") == 0.95 for record in series)

    units = export["methodology"]["units"]
    assert units.get("traffic_totals") == "bytes"
    assert units.get("rates") == "bytes/s"
    assert "collector_policy" in export["nic_detail"], (
        "export must identify the collector policy, not silently merge streams"
    )


def test_export_matches_visible_directional_identity_and_methodology(page, tmp_path):
    select_physical_nic(page, "node-a", "cxi-0 on node 1")
    set_shared_interval(page)
    page.get_by_role("radio", name="TX", exact=True).check()
    page.locator(".tab", has_text="NIC detail").click()
    expect(page.locator("#analysis-nic-detail").last).to_contain_text(
        "node-a / cxi-0"
    )

    export = download_export(page, tmp_path)
    assert export["run"]["source_format"] == "synthetic-ctf"
    assert export["run"]["source_fingerprint"] == "sha256:r1-journey-fixture"
    assert export["filters"]["direction"] == "tx"
    assert export["filters"]["node_id"] == 1
    assert export["filters"]["nic_id"] == 10
    assert export["filters"]["start_ns"] == 2_000_000_000
    assert export["filters"]["end_ns"] == 4_000_000_000

    series = export["nic_detail"]["series"]
    assert len(series) == 1
    record = series[0]
    assert record["direction"].lower() == "tx"
    assert record["node"] == "node-a"
    assert record["nic"] == "cxi-0"
    assert record["node_id"] == 1
    assert record["nic_id"] == 10
    assert record["collector_id"] == 20
    assert record["total_bytes"] == 19_000
    assert record["mean_rate_bytes_per_s"] == 10_000.0
    assert record["p99_rate_bytes_per_s"] == 10_000.0
    assert record["coverage_fraction"] == 0.95

    collector_policy = export["nic_detail"]["collector_policy"]
    assert "collector-a" in str(collector_policy), collector_policy
    assert export["methodology"]["units"]["traffic_totals"] == "bytes"
    assert export["methodology"]["units"]["rates"] == "bytes/s"
    formulas = export["methodology"]["formulas"]
    assert formulas["interval_rate"]
    assert formulas["mean_rate"]
    assert formulas["p99_rate"]
    assert export["methodology"]["populations"]["valid_interval_count"] > 0


def test_missing_rx_is_explicit_and_never_zero(page, tmp_path):
    select_dash_option_by_accessible_label(
        page, "#analysis-node-filter", "node-b (ID 2)"
    )
    expect(page.locator("#analysis-filter-status").last).to_contain_text(
        "node 2", timeout=15_000
    )
    select_dash_option_by_accessible_label(
        page, "#analysis-nic-filter", "cxi-0 on node 2 (ID 11)"
    )
    expect(page.locator("#analysis-filter-status").last).to_contain_text(
        "NIC 11", timeout=15_000
    )
    page.locator("#analysis-direction-filter input[value='rx']").check()
    expect(page.locator("#analysis-filter-status").last).to_contain_text(
        "direction rx", timeout=15_000
    )

    page.locator(".tab", has_text="NIC detail").click()
    detail = page.locator("#analysis-nic-detail").last
    expect(detail).to_contain_text("node-b / cxi-0")
    detail_text = detail.inner_text()
    assert "Missing" in detail_text or "missing" in detail_text, (
        "the NIC chart region must explicitly state that RX is unavailable"
    )
    assert "0 B" not in detail_text, (
        "missing RX must not be rendered as an observed zero value"
    )

    export = download_export(page, tmp_path)
    series = export["nic_detail"]["series"]
    assert len(series) == 1, (
        "missing RX must be represented by one explicit record for "
        f"node-b/cxi-0; got {len(series)} records"
    )
    assert series[0]["node"] == "node-b"
    assert series[0]["nic"] == "cxi-0"
    assert series[0]["direction"].lower() == "rx"
    assert series[0]["total_bytes"] is None
    assert series[0]["state"] in {"missing", "partial"}


def test_multiple_collectors_are_ambiguous_not_merged(page, tmp_path):
    page.get_by_role("radio", name="TX", exact=True).check()
    set_shared_interval(page)
    open_node_screen_with_keyboard(page, "node-b")
    open_nic_screen_with_keyboard(page, "node-b")
    nic_detail = page.locator("#analysis-nic-detail").last
    expect(nic_detail).to_contain_text("node-b / cxi-0")
    expect(nic_detail).to_contain_text("Ambiguous")
    expect(nic_detail).to_contain_text("collector-a")
    expect(nic_detail).to_contain_text("collector-b")
    expect(nic_detail).to_contain_text("Select a collector")

    export = download_export(page, tmp_path)
    series = export["nic_detail"]["series"]
    assert len(series) == 1, (
        "an unresolved multi-collector selection must export one explicit "
        f"ambiguity record, not {len(series)} separate or merged values"
    )
    record = series[0]
    assert record["node"] == "node-b"
    assert record["nic"] == "cxi-0"
    assert record["direction"].lower() == "tx"
    assert record["state"] == "ambiguous"
    assert record["total_bytes"] is None
    assert record["mean_rate_bytes_per_s"] is None
    assert "collector-a" in str(record.get("collector_policy"))
    assert "collector-b" in str(record.get("collector_policy"))
