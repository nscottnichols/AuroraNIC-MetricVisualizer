"""Focused regressions for the UI recovery correctness findings.

Baseline evidence (2026-10-06, ``.venv/bin/python -m pytest
tests/test_ui_recovery_correctness.py -q``): 5 failed.

1. Repeated NIC labels: detail contains ``node-b/cxi-0`` in addition to selected
   ``node-a/cxi-0``.
2. Missing RX: plotted values are numeric zero instead of nonnumeric missing.
3. Callback payload IDs: duplicated IDs are ``analysis-overview``,
   ``analysis-filter-status``, and ``analysis-nic-detail``.
4. Stale dependent selection: the broad callback has no
   ``analysis-nic-filter.value`` output.
5. Multiple collectors: the partial stream record has no
   ``collector_identities`` field.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis_app import (
    NIC_FILTER_ID,
    OVERVIEW_ID,
    _render_filter_status,
    _render_nic_detail,
    _render_node_graph,
    _render_overview,
    build_export_document,
    build_view_model,
    clear_view_model_cache,
    create_analysis_app,
    default_filters,
)
from analysis_fixture import (
    NIC_A,
    NODE_A,
    RUN,
    TX_COUNTER,
    make_store,
)
from trace_model import CollectorIdentity, Coverage, NormalizedBatch
from trace_store import TraceStore


COLLECTOR_A = CollectorIdentity(20, "collector-a", rank=0)
COLLECTOR_B = CollectorIdentity(21, "collector-b", rank=1)


@pytest.fixture(autouse=True)
def isolate_view_model_cache():
    clear_view_model_cache()
    yield
    clear_view_model_cache()


def test_repeated_nic_labels_do_not_leak_across_nodes_in_detail_or_export():
    view = build_view_model(make_store(sample_count=3))
    detail = view["nic_detail"]
    expected = {(detail["selected_node"], detail["selected_nic"])}
    actual = {(record["node"], record["nic"]) for record in detail["series"]}

    assert actual == expected

    exported_detail = build_export_document(view)["nic_detail"]
    exported_actual = {
        (record["node"], record["nic"]) for record in exported_detail["series"]
    }
    assert exported_actual == expected


def test_missing_rx_is_not_plotted_as_observed_zero():
    view = build_view_model(make_store(include_rx=False, sample_count=3))
    figure = _render_node_graph(view).figure
    rx_traces = [trace for trace in figure.data if trace.name == "RX"]

    assert rx_traces
    assert all(value is None for trace in rx_traces for value in trace.y)


def test_callback_children_do_not_recreate_their_container_ids():
    view = build_view_model(make_store(sample_count=3))
    rendered = [
        (target, renderer(view))
        for target, renderer in (
            ("analysis-overview", _render_overview),
            ("analysis-filter-status", _render_filter_status),
            ("analysis-nic-detail", _render_nic_detail),
        )
    ]

    duplicated = [
        target for target, child in rendered if getattr(child, "id", None) == target
    ]
    assert not duplicated


def test_node_selection_normalizes_an_incompatible_nic_selection():
    store = make_store(sample_count=3)
    filters = default_filters(store)
    app = create_analysis_app(store)
    update_callback = next(
        callback
        for callback in app._callback_list
        if OVERVIEW_ID in callback["output"] and NIC_FILTER_ID in callback["output"]
    )

    assert f"{NIC_FILTER_ID}.value" in update_callback["output"]

    output_specs = [
        spec for spec in update_callback["output"].split("..") if spec
    ]
    raw_callback = app.callback_map[update_callback["output"]]["callback"].__wrapped__
    result = raw_callback(
        "node:2",
        "nic:10",
        "all",
        [0, filters.end_ns - filters.start_ns],
        {"acquisition_start_ns": filters.start_ns},
    )
    values_by_output = dict(zip(output_specs, result, strict=True))
    options = values_by_output[f"{NIC_FILTER_ID}.options"]
    selected_nic = values_by_output[f"{NIC_FILTER_ID}.value"]

    assert selected_nic in {"all", "nic:11"}
    assert selected_nic in {option["value"] for option in options}


def _make_multi_collector_store() -> TraceStore:
    samples_by_collector = {
        COLLECTOR_A: (0, 1_000, 3_000),
        COLLECTOR_B: (0, 500, 1_500),
    }
    rows = [
        (collector, timestamp, value)
        for collector in (COLLECTOR_A, COLLECTOR_B)
        for timestamp, value in zip(
            (0, 100_000_000, 200_000_000),
            samples_by_collector[collector],
            strict=True,
        )
    ]

    return TraceStore.from_batches(
        [
            NormalizedBatch(
                identity=RUN,
                nodes=(NODE_A,),
                nics=(NIC_A,),
                counters=(TX_COUNTER,),
                collectors=(COLLECTOR_A, COLLECTOR_B),
                coverage=Coverage(
                    requested_counter_names=(TX_COUNTER.name,),
                    available_counter_names=(TX_COUNTER.name,),
                    readable_counter_names=(TX_COUNTER.name,),
                    observed_counter_names=(TX_COUNTER.name,),
                    acquisition_start_ns=0,
                    acquisition_end_ns=200_000_000,
                    lost_record_count=0,
                    gap_count=0,
                ),
                node_ids=np.full(len(rows), NODE_A.node_id, dtype=np.int64),
                nic_ids=np.full(len(rows), NIC_A.nic_id, dtype=np.int64),
                counter_ids=np.full(len(rows), TX_COUNTER.counter_id, dtype=np.int64),
                collector_ids=np.asarray(
                    [row[0].collector_id for row in rows], dtype=np.int64
                ),
                timestamps_ns=np.asarray([row[1] for row in rows], dtype=np.int64),
                record_ids=np.asarray(
                    [f"collector-stream-{index}" for index in range(len(rows))]
                ),
                uint64_values=np.asarray([row[2] for row in rows], dtype=np.uint64),
                float64_values=np.full(len(rows), np.nan, dtype=np.float64),
                quality_flags=np.zeros(len(rows), dtype=np.int64),
            )
        ]
    )


def test_multiple_collector_streams_retain_explicit_collector_identity():
    detail = build_view_model(_make_multi_collector_store())["nic_detail"]
    partial_records = [
        record
        for record in detail["series"]
        if record["state"] == "partial" and record["stream_count"] == 2
    ]

    assert len(partial_records) == 1
    record = partial_records[0]
    assert record["total_bytes"] is None
    assert {
        (identity["collector_id"], identity["name"])
        for identity in record["collector_identities"]
    } == {
        (COLLECTOR_A.collector_id, COLLECTOR_A.name),
        (COLLECTOR_B.collector_id, COLLECTOR_B.name),
    }
