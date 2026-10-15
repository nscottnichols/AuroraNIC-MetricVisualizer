from __future__ import annotations

from analysis_app import build_export_document, build_view_model, default_filters
from analysis_fixture import make_store
from counter_semantic_registry import (
    COUNTER_SEMANTIC_REGISTRY,
    COUNTER_SEMANTIC_REGISTRY_VERSION,
    CounterDirection,
    CounterQuality,
    CounterResetBehavior,
    CounterScope,
    counter_semantic_definition,
    ok_traffic_counter_names,
)
from ctf_adapter import CtfAdapterConfig, ctf_events_to_normalized_batch
from question_capabilities import (
    CounterDescriptor,
    QuestionCounterMetadata,
    build_question_capability_matrix,
)
from trace_model import CounterSemantics, QualityFlag
from trace_model import ValueKind as TraceValueKind
from traffic_analysis import (
    AnalysisStatus,
    MetricSeries,
    SeriesIdentity,
    SeriesQualityFlag,
    TrafficDirection,
    ValueKind,
    analyze_traffic_series,
)


EXPECTED_COUNTERS = {
    "hni_sts_tx_ok_octets",
    "hni_sts_rx_ok_octets",
    "hni_sts_tx_bad_octets",
    "hni_sts_rx_bad_octets",
}


def test_registry_contains_exactly_four_sourced_definitions():
    assert set(COUNTER_SEMANTIC_REGISTRY) == EXPECTED_COUNTERS
    assert ok_traffic_counter_names() == (
        "hni_sts_tx_ok_octets",
        "hni_sts_rx_ok_octets",
    )
    for name in EXPECTED_COUNTERS:
        definition = counter_semantic_definition(name)
        assert definition is not None
        assert definition.scope is CounterScope.INTERFACE_WIDE
        assert definition.semantics == "cumulative"
        assert definition.unit == "bytes"
        assert definition.measurement == "network-frame octets"
        assert definition.counter_width_bits == 56
        assert definition.wrap_modulus == 2**56
        assert definition.non_destructive is True
        assert definition.reset_behavior is CounterResetBehavior.NEW_ACCUMULATION_EPOCH
        assert definition.registry_version == COUNTER_SEMANTIC_REGISTRY_VERSION
        assert definition.source
        assert any("FCS/replay accounting" in caveat for caveat in definition.caveats)
        assert any("reset domains" in caveat for caveat in definition.caveats)

    tx_ok = counter_semantic_definition("hni_sts_tx_ok_octets")
    rx_bad = counter_semantic_definition("hni_sts_rx_bad_octets")
    assert tx_ok is not None and tx_ok.direction is CounterDirection.TX
    assert tx_ok.quality is CounterQuality.OK
    assert rx_bad is not None and rx_bad.direction is CounterDirection.RX
    assert rx_bad.quality is CounterQuality.BAD


def _ctf_event(counter: str, value: int, timestamp_ns: int = 100):
    return {
        "name": "cxi",
        "provider": "lttng_ust_cxi_sampling",
        "kind": "raw_sampling",
        "timestamp_ns": timestamp_ns,
        "hostname": "node-a",
        "collector": {"name": "cxi-sampler", "rank": 0, "process": None},
        "payload": {"interface_name": "cxi0", "counter": counter, "value": value},
    }


def test_ctf_adapter_attaches_sourced_metadata_to_four_counters():
    config = CtfAdapterConfig(
        run_id="run-registry",
        source_fingerprint="sha256:registry",
        reader_version="babeltrace2-test",
        clock_domain="ctf-clock",
        synchronization_evidence="test",
        requested_counter_names=tuple(EXPECTED_COUNTERS) + ("unknown_counter",),
    )
    events = [
        _ctf_event("hni_sts_tx_ok_octets", 10, 100),
        _ctf_event("hni_sts_rx_ok_octets", 20, 101),
        _ctf_event("hni_sts_tx_bad_octets", 1, 102),
        _ctf_event("hni_sts_rx_bad_octets", 2, 103),
        _ctf_event("unknown_counter", 3, 104),
    ]
    batch = ctf_events_to_normalized_batch(events, config)
    metadata_by_name = {counter.name: counter for counter in batch.counters}

    for name in EXPECTED_COUNTERS:
        metadata = metadata_by_name[name]
        assert metadata.semantics is CounterSemantics.CUMULATIVE
        assert metadata.unit == "bytes"
        assert metadata.value_kind is TraceValueKind.EXACT_UINT64
        assert metadata.baseline_relative is False
        assert metadata.registry_version == COUNTER_SEMANTIC_REGISTRY_VERSION
        assert metadata.quality_flags == QualityFlag.NONE

    unknown = metadata_by_name["unknown_counter"]
    assert unknown.semantics is CounterSemantics.UNKNOWN
    assert unknown.unit == "unknown"
    assert unknown.registry_version == "unvalidated-raw-ctf-v1"
    assert unknown.quality_flags == QualityFlag.UNKNOWN_METADATA


def test_registry_promotes_only_both_ok_traffic_counters():
    all_counters = tuple(EXPECTED_COUNTERS)
    metadata = QuestionCounterMetadata(
        requested_counter_names=all_counters,
        available_counter_names=all_counters,
        readable_counter_names=all_counters,
        observed_counter_names=all_counters,
        counter_metadata={
            name: CounterDescriptor(
                semantics="cumulative",
                unit="bytes",
                registry_version=COUNTER_SEMANTIC_REGISTRY_VERSION,
            )
            for name in all_counters
        },
    )
    rows = build_question_capability_matrix(metadata)
    nic01 = rows[0]
    assert nic01.status == "supported"
    assert nic01.required_counters == (
        "hni_sts_tx_ok_octets",
        "hni_sts_rx_ok_octets",
    )
    assert any(
        "HPE-sourced network-frame octet semantic registry" in reason
        for reason in nic01.reasons
    )

    missing_rx = QuestionCounterMetadata(
        requested_counter_names=all_counters,
        available_counter_names=(
            "hni_sts_tx_ok_octets",
            "hni_sts_tx_bad_octets",
            "hni_sts_rx_bad_octets",
        ),
        readable_counter_names=(
            "hni_sts_tx_ok_octets",
            "hni_sts_tx_bad_octets",
            "hni_sts_rx_bad_octets",
        ),
        observed_counter_names=(
            "hni_sts_tx_ok_octets",
            "hni_sts_tx_bad_octets",
            "hni_sts_rx_bad_octets",
        ),
        counter_metadata={
            name: CounterDescriptor(
                semantics="cumulative",
                unit="bytes",
                registry_version=COUNTER_SEMANTIC_REGISTRY_VERSION,
            )
            for name in all_counters
        },
    )
    partial = build_question_capability_matrix(missing_rx)[0]
    assert partial.status == "partial"
    assert partial.missing_required_counters == ("hni_sts_rx_ok_octets",)


def _metric_series(name: str, values, timestamps=(0, 1_000_000_000)):
    definition = counter_semantic_definition(name)
    assert definition is not None
    return MetricSeries(
        identity=SeriesIdentity(
            run_id="run-registry",
            node_id="node-a",
            nic_id="cxi0",
            collector_id="collector",
            counter_id=name,
            direction=TrafficDirection.TX
            if definition.direction is CounterDirection.TX
            else TrafficDirection.RX,
        ),
        timestamps_ns=tuple(timestamps),
        values=tuple(values),
        value_dtype="uint64",
        value_kind=ValueKind.CUMULATIVE,
        semantics=definition.measurement,
        unit=definition.unit,
        baseline_relative=False,
        counter_definition=definition,
    )


def test_56_bit_decrease_is_conservatively_excluded_as_wrap_suspect():
    series = _metric_series(
        "hni_sts_tx_ok_octets",
        (2**56 - 10, 5),
    )
    result = analyze_traffic_series(series)

    assert result.status is AnalysisStatus.INVALID_DATA
    assert result.total_bytes is None
    assert result.valid_interval_count == 0
    assert result.invalid_interval_count == 1
    assert result.wrap_suspect_interval_count == 1
    assert result.reset_epoch_boundary_count == 0
    assert result.accumulation_epoch_count == 1
    assert SeriesQualityFlag.WRAP_SUSPECT in result.quality_flags
    assert result.intervals[0].delta_bytes is None
    assert result.intervals[0].wrap_suspect is True
    assert any(
        "wrap remains unverified" in reason for reason in result.status_reasons
    )


def test_reset_to_zero_starts_new_accumulation_epoch_without_bridging():
    series = _metric_series(
        "hni_sts_rx_ok_octets",
        (100, 0, 50),
        timestamps=(0, 1_000_000_000, 2_000_000_000),
    )
    result = analyze_traffic_series(series)

    assert result.status is AnalysisStatus.PARTIAL_INVALID_DATA
    assert result.total_bytes == 50
    assert result.valid_interval_count == 1
    assert result.invalid_interval_count == 1
    assert result.accumulation_epoch_count == 2
    assert result.reset_epoch_boundary_count == 1
    assert SeriesQualityFlag.RESET_EPOCH in result.quality_flags
    assert result.intervals[0].reset_epoch is True
    assert result.intervals[1].delta_bytes == 50


def test_export_and_ui_expose_sourced_registry_status_and_bad_octet_separation():
    store = make_store(sample_count=2)
    view = build_view_model(store, default_filters(store))

    assert view["overview"]["counter_semantic_registry_version"] == (
        COUNTER_SEMANTIC_REGISTRY_VERSION
    )
    assert "network-frame octets" in view["overview"]["counter_semantics"]
    nic01 = next(
        row for row in view["question_coverage"] if row["id"] == "NIC-01"
    )
    assert nic01["state"] == "supported"
    assert "HPE-sourced network-frame octet semantic registry" in nic01["reason"]

    document = build_export_document(view)
    assert document["methodology"]["counter_semantic_registry_version"] == (
        COUNTER_SEMANTIC_REGISTRY_VERSION
    )
    assert document["methodology"]["counter_semantics"]
    assert document["methodology"]["counter_reset_behavior"] == (
        "new accumulation epoch after hardware reset"
    )
    assert document["methodology"]["counter_wrap_modulus"] == 2**56
    assert "decreases are conservatively excluded" in document["methodology"][
        "counter_semantics"
    ]
    assert any(
        "FCS/replay accounting" in limitation
        for limitation in document["methodology"]["limitations"]
    )
    assert any(
        "network-frame octets are not application payload"
        in limitation.lower()
        for limitation in document["methodology"]["limitations"]
    )
