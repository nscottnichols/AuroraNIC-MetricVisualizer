import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pftrace_adapter import (
    PerfettoTraceProcessorUnavailableError,
    PftraceAdapterConfig,
    PftraceAdapterError,
    adapt_perfetto_trace_processor,
    adapt_perfetto_trace_processor_rows,
    extract_perfetto_trace_processor_rows,
    pftrace_rows_to_normalized_batch,
)
from trace_model import QualityFlag


CONFIG = PftraceAdapterConfig(
    run_id="run-1",
    source_fingerprint="sha256:test",
    reader_version="trace-processor-test-1",
    clock_domain="perfetto-trace-clock",
    synchronization_evidence="test metadata only",
    requested_counter_names=("rx_ok_octets",),
)


def track(track_id, parent_id, name, track_type):
    row = {
        "id": track_id,
        "parent_id": parent_id,
        "name": name,
        "type": track_type,
    }
    if track_type == "counter":
        row["collector"] = {"name": "cxi-sampler", "rank": 0, "process": None}
    return row


def counter_row(ts=100, track_id=3, value=12.0):
    return {"ts": ts, "track_id": track_id, "value": value}


def rows(*counter_rows):
    tracks = [
        track(1, None, "node-a", "node"),
        track(2, 1, "cxi0", "nic"),
        track(3, 2, "rx_ok_octets", "counter"),
    ]
    return {"tracks": tracks, "counter_rows": list(counter_rows)}


class FakeQueryResult:
    def __init__(self, columns, result_rows):
        self.column_names = list(columns)
        self._rows = [
            SimpleNamespace(**dict(zip(columns, row, strict=True)))
            for row in result_rows
        ]

    def __iter__(self):
        return iter(self._rows)


class FakeTraceProcessor:
    def __init__(self):
        self.queries = []

    def query(self, sql):
        self.queries.append(sql)
        if sql.startswith("SELECT id, parent_id, name, type"):
            return FakeQueryResult(
                ("id", "parent_id", "name", "source_type"),
                [
                    (1, None, "x3000c0s1b0n0", "async_track"),
                    (2, 1, "cxi sampling", "async_track"),
                    (3, 2, "NIC 0", "async_track"),
                    (4, 3, "cxi0", "async_track"),
                    (5, 4, "rx_ok_octets", "counter_track"),
                ],
            )
        if sql.startswith("SELECT id FROM counter_track"):
            return FakeQueryResult(("id",), [(5,)])
        if sql.startswith("SELECT ts, track_id, value FROM counter"):
            return FakeQueryResult(
                ("ts", "track_id", "value"), [(100, 5, 12.0), (200, 5, 17.0)]
            )
        raise AssertionError(f"unexpected query: {sql}")


def test_normalizes_rows_and_preserves_identity_and_limitations():
    batch = pftrace_rows_to_normalized_batch(
        rows(counter_row(100), counter_row(200, value=17.0)), CONFIG
    )

    assert batch.identity.source_format == "perfetto-structured-pftrace"
    assert batch.nodes[0].stable_device_id == "perfetto-track:1"
    assert batch.nics[0].stable_device_id == "perfetto-track:2:parent-track:1"
    assert batch.collectors[0].collector_type == "perfetto-track:3"
    assert batch.record_ids.tolist() == [
        "pftrace-row-0:track-3",
        "pftrace-row-1:track-3",
    ]
    assert batch.float64_values.tolist() == [12.0, 17.0]
    assert (batch.uint64_values == 0).all()
    assert np.all(batch.quality_flags & QualityFlag.BASELINE_RELATIVE)
    assert np.all(batch.quality_flags & QualityFlag.UNKNOWN_METADATA)
    assert not np.any(batch.quality_flags & QualityFlag.PRECISION_LOSS_POSSIBLE)
    assert batch.coverage.observed_counter_names == ("rx_ok_octets",)
    assert any("hidden initial sample" in note for note in batch.coverage.notes)
    assert any("omitted-sample count is unknown" in note for note in batch.coverage.notes)
    assert any("Native-cadence" in note for note in batch.coverage.notes)


def test_large_values_expose_possible_float_precision_loss():
    batch = pftrace_rows_to_normalized_batch(
        rows(counter_row(100, value=2**53 + 1)), CONFIG
    )

    assert batch.float64_values[0] == 9007199254740992.0
    assert batch.quality_flags[0] & QualityFlag.PRECISION_LOSS_POSSIBLE
    assert batch.counters[0].quality_flags & QualityFlag.PRECISION_LOSS_POSSIBLE


def test_exact_duplicate_is_removed_and_conflict_is_rejected():
    duplicate = pftrace_rows_to_normalized_batch(
        rows(counter_row(100), counter_row(100), counter_row(200)), CONFIG
    )
    assert len(duplicate.timestamps_ns) == 2
    assert duplicate.quality_flags[0] & QualityFlag.DUPLICATE_REMOVED
    assert any("duplicate source indices: [1]" in note for note in duplicate.coverage.notes)

    with pytest.raises(PftraceAdapterError, match="conflicting records"):
        pftrace_rows_to_normalized_batch(
            rows(counter_row(100), counter_row(100, value=13.0)), CONFIG
        )


def test_reordering_is_flagged_and_rows_are_sorted():
    batch = pftrace_rows_to_normalized_batch(
        rows(counter_row(200, value=2.0), counter_row(100, value=1.0)), CONFIG
    )
    assert batch.timestamps_ns.tolist() == [100, 200]
    assert np.all(batch.quality_flags & QualityFlag.REORDERED)
    assert np.all(batch.quality_flags & QualityFlag.NONMONOTONIC_TIMESTAMP)


def test_duplicate_collectors_remain_separate_streams():
    data = {
        "tracks": [
            track(1, None, "node-a", "node"),
            track(2, 1, "cxi0", "nic"),
            {
                **track(3, 2, "rx_ok_octets", "counter"),
                "collector": {"name": "cxi-sampler", "rank": 0, "process": None},
            },
            {
                **track(4, 2, "rx_ok_octets", "counter"),
                "collector": {"name": "cxi-sampler", "rank": 1, "process": None},
            },
        ],
        "counter_rows": [counter_row(100, 3), counter_row(100, 4, value=13.0)],
    }
    batch = pftrace_rows_to_normalized_batch(data, CONFIG)

    assert [collector.rank for collector in batch.collectors] == [0, 1]
    assert [collector.collector_type for collector in batch.collectors] == [
        "perfetto-track:3",
        "perfetto-track:4",
    ]
    assert sorted(batch.float64_values.tolist()) == [12.0, 13.0]


def test_multi_node_identity_is_preserved_without_merging_nics():
    data = {
        "tracks": [
            track(1, None, "node-a", "node"),
            track(2, 1, "cxi0", "nic"),
            track(3, 2, "rx_ok_octets", "counter"),
            track(4, None, "node-b", "node"),
            track(5, 4, "cxi0", "nic"),
            track(6, 5, "rx_ok_octets", "counter"),
        ],
        "counter_rows": [counter_row(100, 3), counter_row(100, 6, value=13.0)],
    }
    batch = pftrace_rows_to_normalized_batch(data, CONFIG)

    assert [node.label for node in batch.nodes] == ["node-a", "node-b"]
    assert [(nic.node_id, nic.label) for nic in batch.nics] == [
        (0, "cxi0"),
        (1, "cxi0"),
    ]
    assert batch.node_ids.tolist() == [0, 1]
    assert batch.nic_ids.tolist() == [0, 1]
    assert sorted(batch.float64_values.tolist()) == [12.0, 13.0]


def test_requested_counter_can_be_missing_without_discarding_events():
    config = replace(CONFIG, requested_counter_names=("rx_ok_octets", "absent"))
    batch = pftrace_rows_to_normalized_batch(rows(counter_row()), config)
    assert batch.coverage.missing_counter_names == ("absent",)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"extra": True}, "undocumented fields"),
        ({"traceEvents": []}, "Chrome JSON is not binary pftrace"),
    ],
)
def test_chrome_json_and_undocumented_top_level_fields_are_rejected(changes, message):
    data = rows(counter_row())
    data.update(changes)
    with pytest.raises(PftraceAdapterError, match=message):
        pftrace_rows_to_normalized_batch(data, CONFIG)


@pytest.mark.parametrize(
    "row_changes, message",
    [
        ({"ts": "100"}, "ts must be an integer"),
        ({"ts": 2**63}, "outside int64 range"),
        ({"track_id": 99}, "references unknown track"),
        ({"track_id": 1}, "references non-counter track"),
        ({"value": "12"}, "value must be a number"),
        ({"value": float("nan")}, "finite and nonnegative"),
        ({"value": -1.0}, "finite and nonnegative"),
        ({"value": True}, "value must be a number"),
        ({"extra": 1}, "undocumented fields"),
    ],
)
def test_counter_row_schema_and_values_are_strict(row_changes, message):
    row = counter_row()
    row.update(row_changes)
    with pytest.raises(PftraceAdapterError, match=message):
        pftrace_rows_to_normalized_batch(rows(row), CONFIG)


def test_track_schema_parent_and_duplicate_series_are_strict():
    with pytest.raises(PftraceAdapterError, match="undocumented fields"):
        pftrace_rows_to_normalized_batch(
            {
                    "tracks": [{**track(1, None, "node-a", "node"), "extra": 1}],
                "counter_rows": [counter_row()],
            },
            CONFIG,
        )

    with pytest.raises(PftraceAdapterError, match="unknown parent"):
        pftrace_rows_to_normalized_batch(
            {
                "tracks": [
                    track(1, None, "node-a", "node"),
                    track(2, 99, "cxi0", "nic"),
                    track(3, 2, "rx_ok_octets", "counter"),
                ],
                "counter_rows": [counter_row()],
            },
            CONFIG,
        )

    conflicting_tracks = {
        "tracks": [
            track(1, None, "node-a", "node"),
            track(2, 1, "cxi0", "nic"),
            track(3, 2, "rx_ok_octets", "counter"),
            track(4, 2, "rx_ok_octets", "counter"),
        ],
        "counter_rows": [counter_row(100, 3), counter_row(100, 4)],
    }
    with pytest.raises(
        PftraceAdapterError, match="conflicting duplicate counter tracks"
    ):
        pftrace_rows_to_normalized_batch(conflicting_tracks, CONFIG)


def test_config_rejects_processed_value_assumptions():
    with pytest.raises(PftraceAdapterError, match="source_format must identify pftrace"):
        replace(CONFIG, source_format="chrome-json")
    with pytest.raises(PftraceAdapterError, match="Perfetto reader"):
        replace(CONFIG, source_format="pftrace-unknown-reader")
    with pytest.raises(PftraceAdapterError, match="independent interval deltas"):
        replace(CONFIG, conversion_lineage="independent interval deltas")


def test_empty_counter_stream_is_rejected():
    with pytest.raises(PftraceAdapterError, match="at least one pftrace counter row"):
        pftrace_rows_to_normalized_batch(rows(), CONFIG)


def test_trace_processor_entrypoint_has_clear_dependency_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "perfetto.trace_processor", None)
    with pytest.raises(
        PerfettoTraceProcessorUnavailableError,
        match="Perfetto Trace Processor Python package",
    ):
        adapt_perfetto_trace_processor_rows(rows(counter_row()), CONFIG)


def test_trace_processor_extraction_uses_standard_tables_and_parent_chains():
    trace_processor = FakeTraceProcessor()
    extracted = extract_perfetto_trace_processor_rows(trace_processor)

    assert trace_processor.queries == [
        "SELECT id, parent_id, name, type AS source_type FROM track ORDER BY id",
        "SELECT id FROM counter_track ORDER BY id",
        "SELECT ts, track_id, value FROM counter ORDER BY ts, track_id",
    ]
    assert extracted["counter_rows"] == [
        {"ts": 100, "track_id": 5, "value": 12.0},
        {"ts": 200, "track_id": 5, "value": 17.0},
    ]
    assert [(track["id"], track["parent_id"], track["type"]) for track in extracted["tracks"]] == [
        (1, None, "node"),
        (2, 1, "other"),
        (3, 2, "other"),
        (4, 3, "nic"),
        (5, 4, "counter"),
    ]
    assert extracted["tracks"][4]["collector"] == {
        "name": "perfetto-track:5",
        "rank": None,
        "process": None,
    }

    batch = adapt_perfetto_trace_processor(trace_processor, CONFIG)
    assert [node.label for node in batch.nodes] == ["x3000c0s1b0n0"]
    assert [nic.label for nic in batch.nics] == ["cxi0"]
    assert batch.nics[0].stable_device_id == "perfetto-track:4:parent-track:3"
    assert batch.collectors[0].collector_type == "perfetto-track:5"
    assert batch.timestamps_ns.tolist() == [100, 200]
    assert batch.float64_values.tolist() == [12.0, 17.0]


def test_trace_processor_path_dependency_error(monkeypatch, tmp_path):
    trace_path = tmp_path / "trace.pftrace"
    trace_path.write_bytes(b"trace")
    monkeypatch.setitem(sys.modules, "perfetto", None)
    monkeypatch.setitem(sys.modules, "perfetto.trace_processor", None)

    with pytest.raises(
        PerfettoTraceProcessorUnavailableError,
        match="pftrace path extraction",
    ):
        adapt_perfetto_trace_processor(trace_path, CONFIG)


def test_trace_processor_missing_path_is_rejected():
    with pytest.raises(PftraceAdapterError, match="existing file"):
        adapt_perfetto_trace_processor("/does/not/exist.pftrace", CONFIG)
