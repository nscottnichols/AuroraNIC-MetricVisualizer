import sys
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from ctf_adapter import (
    Bt2UnavailableError,
    CtfAdapterConfig,
    CtfAdapterError,
    adapt_babeltrace2_events,
    ctf_events_to_normalized_batch,
    extract_babeltrace2_events,
    extract_babeltrace2_events_with_stats,
)
from trace_model import QualityFlag


CONFIG = CtfAdapterConfig(
    run_id="run-1",
    source_fingerprint="sha256:test",
    reader_version="structured-reader-1",
    clock_domain="ctf-clock",
    synchronization_evidence="test metadata only",
    requested_counter_names=("rx_ok_octets",),
)


def event(
    *,
    timestamp_ns=100,
    hostname="node-a",
    interface="cxi0",
    counter="rx_ok_octets",
    value=42,
    collector_name="cxi-sampler",
    rank=0,
):
    return {
        "name": "cxi",
        "provider": "lttng_ust_cxi_sampling",
        "kind": "raw_sampling",
        "timestamp_ns": timestamp_ns,
        "hostname": hostname,
        "collector": {"name": collector_name, "rank": rank, "process": None},
        "payload": {"interface_name": interface, "counter": counter, "value": value},
    }


def test_normalizes_raw_events_and_preserves_all_identities():
    events = [
        event(counter="rx_ok_octets", value=1),
        event(counter="tx_ok_octets", value=2),
        event(hostname="node-b", value=3),
        event(rank=1, value=4),
    ]
    batch = ctf_events_to_normalized_batch(events, CONFIG)

    assert batch.identity.source_format == "babeltrace2-structured-raw-ctf"
    assert [(node.label, node.node_id) for node in batch.nodes] == [
        ("node-a", 0),
        ("node-b", 1),
    ]
    assert [(nic.label, nic.node_id) for nic in batch.nics] == [
        ("cxi0", 0),
        ("cxi0", 1),
    ]
    assert [counter.name for counter in batch.counters] == [
        "rx_ok_octets",
        "tx_ok_octets",
    ]
    assert [(collector.rank, collector.name) for collector in batch.collectors] == [
        (0, repr(("node-a", "cxi-sampler", 0, None))),
        (1, repr(("node-a", "cxi-sampler", 1, None))),
        (0, repr(("node-b", "cxi-sampler", 0, None))),
    ]
    assert batch.uint64_values.tolist() == [1, 4, 2, 3]
    assert np.isnan(batch.float64_values).all()
    assert np.all(batch.quality_flags & QualityFlag.UNKNOWN_METADATA)
    assert batch.coverage.observed_counter_names == (
        "rx_ok_octets",
        "tx_ok_octets",
    )
    assert batch.coverage.missing_counter_names == ()
    assert any(
        "Observed counter 'tx_ok_octets'" in note for note in batch.coverage.notes
    )


def test_exact_duplicate_is_removed_without_losing_useful_records():
    events = [event(value=7), event(value=7), event(timestamp_ns=200, value=8)]
    batch = ctf_events_to_normalized_batch(events, CONFIG)

    assert len(batch.timestamps_ns) == 2
    assert batch.uint64_values.tolist() == [7, 8]
    assert batch.record_ids.tolist() == ["ctf-event-0", "ctf-event-2"]
    assert batch.quality_flags[0] & QualityFlag.DUPLICATE_REMOVED
    assert not batch.quality_flags[1] & QualityFlag.DUPLICATE_REMOVED
    assert any(
        "duplicate source indices: [1]" in note for note in batch.coverage.notes
    )


def test_exact_uint64_maximum_value_is_preserved():
    batch = ctf_events_to_normalized_batch([event(value=2**64 - 1)], CONFIG)

    assert batch.uint64_values.dtype == np.uint64
    assert int(batch.uint64_values[0]) == 2**64 - 1
    assert np.isnan(batch.float64_values[0])


def test_conflicting_duplicate_record_is_rejected():
    with pytest.raises(CtfAdapterError, match="conflicting records"):
        ctf_events_to_normalized_batch([event(value=7), event(value=8)], CONFIG)


def test_reordering_is_flagged_and_rows_are_sorted():
    events = [event(timestamp_ns=20, value=2), event(timestamp_ns=10, value=1)]
    batch = ctf_events_to_normalized_batch(events, CONFIG)

    assert batch.timestamps_ns.tolist() == [10, 20]
    assert np.all(batch.quality_flags & QualityFlag.REORDERED)
    assert np.all(batch.quality_flags & QualityFlag.NONMONOTONIC_TIMESTAMP)


def test_requested_counter_can_be_missing_without_discarding_events():
    config = replace(
        CONFIG, requested_counter_names=("rx_ok_octets", "absent_counter")
    )
    batch = ctf_events_to_normalized_batch([event()], config)

    assert batch.coverage.missing_counter_names == ("absent_counter",)
    assert batch.coverage.observed_counter_names == ("rx_ok_octets",)
    assert batch.uint64_values.tolist() == [42]


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"name": "unexpected"}, "unsupported event name"),
        ({"provider": "unexpected"}, "unsupported provider"),
        ({"kind": "analysis_interval"}, "analysis/interval CTF events"),
        ({"extra": True}, "undocumented fields"),
        ({"timestamp_ns": 2**63}, "outside int64 range"),
        ({"timestamp_ns": 1.5}, "timestamp_ns must be an integer"),
        (
            {"collector": {"name": "x", "rank": 0, "process": None, "extra": 1}},
            "undocumented fields",
        ),
        (
            {
                "payload": {
                    "interface_name": "cxi0",
                    "counter": "x",
                    "value": 1,
                    "extra": 1,
                }
            },
            "undocumented fields",
        ),
    ],
)
def test_undocumented_schema_and_event_names_are_rejected(changes, message):
    base = event(value=2**64 - 1)
    base.update(changes)
    with pytest.raises(CtfAdapterError, match=message):
        ctf_events_to_normalized_batch([base], CONFIG)


def test_uint64_value_validation_rejects_out_of_range_and_noninteger():
    with pytest.raises(CtfAdapterError, match="outside uint64 range"):
        ctf_events_to_normalized_batch([event(value=2**64)], CONFIG)
    with pytest.raises(CtfAdapterError, match="value must be an integer"):
        ctf_events_to_normalized_batch([event(value=1.5)], CONFIG)
    with pytest.raises(CtfAdapterError, match="value must be an integer"):
        ctf_events_to_normalized_batch([event(value=True)], CONFIG)


def test_config_rejects_non_raw_and_baseline_assumptions():
    with pytest.raises(CtfAdapterError, match="source_format must identify raw CTF"):
        replace(CONFIG, source_format="processed-ctf")
    with pytest.raises(CtfAdapterError, match="cannot be declared baseline-relative"):
        replace(CONFIG, conversion_lineage="baseline-relative converter")


def test_empty_event_stream_is_rejected():
    with pytest.raises(CtfAdapterError, match="at least one raw CTF event"):
        ctf_events_to_normalized_batch([], CONFIG)


def test_bt2_entrypoint_has_clear_optional_dependency_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "bt2", None)

    with pytest.raises(Bt2UnavailableError, match="Babeltrace 2 Python bindings"):
        adapt_babeltrace2_events([event()], CONFIG)


class FakeClockSnapshot:
    def __init__(self, ns_from_origin):
        self.ns_from_origin = ns_from_origin


class FakePayload(dict):
    pass


class FakeEvent:
    def __init__(self, name, stream, payload):
        self.name = name
        self.stream = stream
        self.payload = payload


class FakeEventMessage:
    def __init__(self, event, timestamp_ns):
        self.event = event
        self.default_clock_snapshot = FakeClockSnapshot(timestamp_ns)


class FakeMessage:
    pass


class FakeStream:
    def __init__(self, trace, name, stream_id, environment=None):
        self.trace = trace
        self.name = name
        self.id = stream_id
        self.environment = environment if environment is not None else {}


class FakeTrace:
    def __init__(self, name, trace_uuid, environment):
        self.name = name
        self.uuid = trace_uuid
        self.environment = environment


class FakeBt2:
    def __init__(self):
        self.inputs = []

    _EventMessage = FakeEventMessage


def install_fake_bt2(monkeypatch, messages):
    fake_bt2 = FakeBt2()

    def trace_collection_message_iterator(inputs):
        fake_bt2.inputs.append(inputs)
        return iter(messages)

    fake_bt2.TraceCollectionMessageIterator = trace_collection_message_iterator
    monkeypatch.setitem(sys.modules, "bt2", fake_bt2)
    return fake_bt2


def test_extract_babeltrace2_events_uses_message_and_trace_context(monkeypatch):
    trace = FakeTrace(
        "trace-a",
        UUID("12345678-1234-5678-1234-567812345678"),
        {"hostname": "node-a"},
    )
    stream = FakeStream(trace, "channel0", 7, {"hostname": "node-a"})
    event = FakeEvent(
        "lttng_ust_cxi_sampling:cxi",
        stream,
        FakePayload(interface_name="cxi0", counter="rx_ok_octets", value=42),
    )
    message = FakeEventMessage(event, 123456789)
    fake_bt2 = install_fake_bt2(monkeypatch, [FakeMessage(), message])
    path = Path("/traces/raw")

    events = extract_babeltrace2_events(path)

    assert fake_bt2.inputs == [str(path)]
    assert events == [
        {
            "name": "cxi",
            "provider": "lttng_ust_cxi_sampling",
            "kind": "raw_sampling",
            "timestamp_ns": 123456789,
            "hostname": "node-a",
            "collector": {
                "name": "babeltrace2",
                "rank": None,
                "process": (
                    "trace=uuid=12345678-1234-5678-1234-567812345678;"
                    "name='trace-a';stream=name='channel0';id=7"
                ),
            },
            "payload": {
                "interface_name": "cxi0",
                "counter": "rx_ok_octets",
                "value": 42,
            },
        }
    ]
    batch = ctf_events_to_normalized_batch(events, CONFIG)
    assert batch.uint64_values.tolist() == [42]


def test_extract_babeltrace2_events_has_clear_optional_dependency_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "bt2", None)

    with pytest.raises(Bt2UnavailableError, match="Babeltrace 2 Python bindings"):
        extract_babeltrace2_events(Path("/traces/raw"))


def test_extract_babeltrace2_events_filters_non_cxi_event(monkeypatch):
    trace = FakeTrace(
        "trace-a",
        UUID("12345678-1234-5678-1234-567812345678"),
        {"hostname": "node-a"},
    )
    stream = FakeStream(trace, "channel0", 7)
    event = FakeEvent(
        "lttng_ust_cxi_sampling:sampling_nic",
        stream,
        FakePayload(interface_name="cxi0", counter="rx_ok_octets", value=42),
    )
    install_fake_bt2(monkeypatch, [FakeEventMessage(event, 123456789)])

    cxi_event = FakeEvent(
        "lttng_ust_cxi_sampling:cxi",
        stream,
        FakePayload(interface_name="cxi0", counter="rx_ok_octets", value=42),
    )
    install_fake_bt2(
        monkeypatch,
        [
            FakeEventMessage(event, 123456789),
            FakeEventMessage(cxi_event, 123456790),
        ],
    )

    events, stats = extract_babeltrace2_events_with_stats(Path("/traces/raw"))

    assert len(events) == 1
    assert events[0]["timestamp_ns"] == 123456790
    assert stats.cxi_event_count == 1
    assert stats.filtered_event_count == 1


def test_extract_babeltrace2_events_rejects_missing_hostname_without_guessing(
    monkeypatch,
):
    trace = FakeTrace(
        "trace-a",
        UUID("12345678-1234-5678-1234-567812345678"),
        {"domain": "ust"},
    )
    stream = FakeStream(trace, "channel0", 7, {"vpid": 123})
    event = FakeEvent(
        "lttng_ust_cxi_sampling:cxi",
        stream,
        FakePayload(interface_name="cxi0", counter="rx_ok_octets", value=42),
    )
    install_fake_bt2(monkeypatch, [FakeEventMessage(event, 123456789)])

    with pytest.raises(
        CtfAdapterError, match="environment has no hostname"
    ):
        extract_babeltrace2_events(Path("/traces/raw"))
