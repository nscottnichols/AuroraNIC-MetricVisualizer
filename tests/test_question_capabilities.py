from collections import Counter

from analysis_fixture import make_store
from counter_semantic_registry import COUNTER_SEMANTIC_REGISTRY_VERSION
from question_capabilities import (
    CAPABILITY_MATRIX_VERSION,
    CounterDescriptor,
    QuestionCounterMetadata,
    build_question_capability_matrix,
)


EXPECTED_IDS = (
    "NIC-01",
    "NIC-02",
    "NIC-03",
    "NIC-04",
    "NIC-05",
    "NIC-06",
    "NIC-07",
    "NIC-08",
    "NIC-09",
    "NODE-01",
    "NODE-02",
    "JOB-01",
    "JOB-02",
    "JOB-03",
    "JOB-04",
)
EXPECTED_SCOPES = {
    **{question_id: "NIC" for question_id in EXPECTED_IDS[:9]},
    "NODE-01": "node",
    "NODE-02": "node",
    "JOB-01": "job",
    "JOB-02": "job",
    "JOB-03": "job",
    "JOB-04": "job",
}
TRAFFIC_COUNTERS = ("hni_sts_tx_ok_octets", "hni_sts_rx_ok_octets")
IMPLEMENTED_ANALYSIS_IDS = {
    "NIC-02": ("nic02.flow_control_analysis",),
    "NIC-03": ("nic03.matching_analysis",),
    "NIC-04": ("nic04.host_pressure_analysis",),
    "NIC-05": ("nic05.operation_analysis",),
    "NIC-06": ("nic06.frame_analysis",),
    "NIC-07": ("nic07.translation_analysis",),
    "NIC-08": ("nic08.latency_analysis",),
    "NIC-09": ("nic09.reliability_analysis",),
    "NODE-01": ("node01.nic_balance",),
    "NODE-02": ("node02.nic_traffic_consistency",),
    "JOB-01": ("job01.node_hotspots", "job01.optional_topology_groups"),
    "JOB-02": ("job02.node_balance",),
    "JOB-03": ("job03.manual_phase_analysis",),
}


def traffic_metadata(**overrides):
    values = {
        "requested_counter_names": TRAFFIC_COUNTERS,
        "available_counter_names": TRAFFIC_COUNTERS,
        "readable_counter_names": TRAFFIC_COUNTERS,
        "observed_counter_names": TRAFFIC_COUNTERS,
        "unreadable_counter_names": (),
        "counter_metadata": {
            name: CounterDescriptor(
                semantics="cumulative",
                unit="bytes",
                registry_version=COUNTER_SEMANTIC_REGISTRY_VERSION,
            )
            for name in TRAFFIC_COUNTERS
        },
    }
    values.update(overrides)
    return QuestionCounterMetadata(**values)


def test_matrix_covers_exactly_fifteen_unique_ids_and_scopes():
    rows = build_question_capability_matrix(make_store(sample_count=2))

    assert tuple(row.question_id for row in rows) == EXPECTED_IDS
    assert len({row.question_id for row in rows}) == 15
    assert all(row.contract_version == CAPABILITY_MATRIX_VERSION for row in rows)
    assert {row.question_id: row.scope for row in rows} == EXPECTED_SCOPES
    assert Counter(row.scope for row in rows) == Counter({"NIC": 9, "node": 2, "job": 4})


def test_trace_store_nic01_is_supported_and_preserves_zero_as_observed():
    rows = build_question_capability_matrix(make_store(sample_count=2))
    nic01 = rows[0]

    assert nic01.status == "supported"
    assert nic01.required_counters == TRAFFIC_COUNTERS
    assert nic01.available_required_counters == TRAFFIC_COUNTERS
    assert nic01.missing_required_counters == ()
    assert nic01.unreadable_required_counters == ()
    assert nic01.implemented_analysis_ids == (
        "nic01.traffic_totals",
        "nic01.weighted_rates",
        "nic01.rate_distribution",
        "nic01.time_series",
    )
    assert any(
        "Zero traffic is an observed zero only when valid samples exist"
        in limitation
        for limitation in nic01.scientific_limitations
    )


def test_required_counter_inventory_has_catalog_core_counters_without_duplicates():
    rows = build_question_capability_matrix(make_store(sample_count=2))
    required = tuple(
        dict.fromkeys(name for row in rows for name in row.required_counters)
    )

    assert len(required) == 195
    assert len(set(required)) == 195
    assert "hni_sts_tx_ok_octets" in required
    assert "hni_sts_rx_ok_octets" in required
    assert "hni_pcs_good_cw" in required


def test_nic01_supported_missing_and_unreadable_transitions():
    full = build_question_capability_matrix(traffic_metadata())[0]
    assert full.status == "supported"

    missing_rx = build_question_capability_matrix(
        traffic_metadata(
            available_counter_names=("hni_sts_tx_ok_octets",),
            readable_counter_names=("hni_sts_tx_ok_octets",),
            observed_counter_names=("hni_sts_tx_ok_octets",),
        )
    )[0]
    assert missing_rx.status == "partial"
    assert missing_rx.available_required_counters == ("hni_sts_tx_ok_octets",)
    assert missing_rx.missing_required_counters == ("hni_sts_rx_ok_octets",)
    assert missing_rx.unreadable_required_counters == ()
    assert any("Missing required counters" in reason for reason in missing_rx.reasons)

    unreadable_rx = build_question_capability_matrix(
        traffic_metadata(
            available_counter_names=("hni_sts_tx_ok_octets",),
            readable_counter_names=("hni_sts_tx_ok_octets",),
            observed_counter_names=("hni_sts_tx_ok_octets",),
            unreadable_counter_names=("hni_sts_rx_ok_octets",),
        )
    )[0]
    assert unreadable_rx.status == "partial"
    assert unreadable_rx.missing_required_counters == ()
    assert unreadable_rx.unreadable_required_counters == ("hni_sts_rx_ok_octets",)
    assert any(
        "Unreadable required counters" in reason
        for reason in unreadable_rx.reasons
    )

    no_evidence = build_question_capability_matrix(
        traffic_metadata(
            available_counter_names=(),
            readable_counter_names=(),
            observed_counter_names=(),
        )
    )[0]
    assert no_evidence.status == "unsupported"
    assert no_evidence.missing_required_counters == TRAFFIC_COUNTERS

    wrong_semantics = build_question_capability_matrix(
        traffic_metadata(
            counter_metadata={
                "hni_sts_tx_ok_octets": CounterDescriptor(
                    "cumulative", "bytes", COUNTER_SEMANTIC_REGISTRY_VERSION
                ),
                "hni_sts_rx_ok_octets": CounterDescriptor(
                    "delta", "bytes", COUNTER_SEMANTIC_REGISTRY_VERSION
                ),
            }
        )
    )[0]
    assert wrong_semantics.status == "partial"
    assert any(
        "does not identify both required counters as cumulative byte counters"
        in reason
        for reason in wrong_semantics.reasons
    )


def test_name_only_coverage_cannot_promote_nic01_to_supported():
    store = make_store(sample_count=2)
    rows = build_question_capability_matrix(store.coverage)
    nic01 = rows[0]

    assert nic01.status == "partial"
    assert nic01.available_required_counters == TRAFFIC_COUNTERS
    assert nic01.missing_required_counters == ()
    assert any(
        "Counter semantics were not supplied" in reason for reason in nic01.reasons
    )


def test_non_nic01_questions_never_become_supported_or_claim_missing_evidence():
    rows = build_question_capability_matrix(make_store(sample_count=2))
    all_counters = tuple(
        dict.fromkeys(
            name for row in rows for name in row.required_counters
        )
    )
    metadata = QuestionCounterMetadata(
        requested_counter_names=all_counters,
        available_counter_names=all_counters,
        readable_counter_names=all_counters,
        observed_counter_names=all_counters,
        counter_metadata={
            name: CounterDescriptor(
                semantics="cumulative" if name in TRAFFIC_COUNTERS else "unknown",
                unit="bytes" if name in TRAFFIC_COUNTERS else "count",
            )
            for name in all_counters
        },
    )
    observed_rows = build_question_capability_matrix(metadata)

    for row in observed_rows[1:]:
        assert row.status == "partial"
        assert row.implemented_analysis_ids == IMPLEMENTED_ANALYSIS_IDS.get(
            row.question_id, ()
        )
        assert row.reasons
        assert row.scientific_limitations
        assert (
            "support requires all required counters and authoritative metadata"
            in " ".join(row.reasons)
        )

    all_text = " ".join(
        " ".join(row.reasons + row.scientific_limitations)
        for row in observed_rows
    )
    assert "cannot be claimed without an identified ECN counter" in all_text
    assert "not MPI message sizes" in all_text
    assert "must not be relabeled as ATU latency" in all_text
    assert "not proof that the host/PCIe path is the limiting resource" in all_text
    assert "Correlations are not causes" in all_text


def test_empty_dataset_has_precise_unsupported_reasons():
    rows = build_question_capability_matrix(QuestionCounterMetadata())
    by_id = {row.question_id: row for row in rows}

    for question_id in EXPECTED_IDS[1:]:
        assert by_id[question_id].status == "unsupported"
        assert by_id[question_id].missing_required_counters
        assert any(
            "Missing required counters" in reason
            for reason in by_id[question_id].reasons
        )

    assert any(
        "no explicitly identified ECN counter" in reason
        for reason in by_id["NIC-02"].reasons
    )
    assert any(
        "Host blocked-counter units are unresolved" in reason
        for reason in by_id["NIC-04"].reasons
    )
    assert any(
        "authoritative opcode mapping is not supplied" in reason
        for reason in by_id["NIC-05"].reasons
    )
    assert any(
        "ATU-latency histogram group or bin mapping is supplied" in reason
        for reason in by_id["NIC-07"].reasons
    )
    assert any(
        "bin edges, bin semantics, and time conversion factors are not sourced"
        in reason
        for reason in by_id["NIC-08"].reasons
    )
    assert any(
        "Topology and communication-edge metadata are not available" in reason
        for reason in by_id["JOB-01"].reasons
    )
    assert any(
        "Verified clock alignment" in reason for reason in by_id["JOB-03"].reasons
    )


def test_capability_function_is_pure_and_does_not_mutate_metadata():
    metadata = traffic_metadata()
    first = build_question_capability_matrix(metadata)
    second = build_question_capability_matrix(metadata)

    assert first == second
    assert metadata.available_counter_names == TRAFFIC_COUNTERS
    assert metadata.observed_counter_names == TRAFFIC_COUNTERS
    assert set(metadata.counter_metadata) == set(TRAFFIC_COUNTERS)
