import pytest

from src.job_analysis import (
    JobBalanceState,
    JobBalanceStatus,
    JobTopologyState,
    analyze_job_node_balance,
)
from node_analysis import MaxMinRatioDenominator, analyze_node_nic_balance
from traffic_analysis import (
    MetricSeries,
    SeriesIdentity,
    TrafficDirection,
    ValueKind,
    analyze_traffic_series,
)


def make_node_result(
    node_id,
    total,
    *,
    values=None,
    timestamps=None,
    direction=TrafficDirection.TX,
    window=None,
    include_intervals=True,
):
    values = [0, total, total] if values is None else values
    identity = SeriesIdentity(
        run_id="run-1",
        node_id=node_id,
        nic_id="nic-1",
        collector_id="collector-1",
        counter_id="CXI_TX_BYTES",
        direction=direction,
    )
    series = MetricSeries(
        identity=identity,
        timestamps_ns=tuple(
            timestamps
            if timestamps is not None
            else [0, 1_000_000_000, 2_000_000_000]
        ),
        values=tuple(values),
        value_dtype="uint64",
        value_kind=ValueKind.CUMULATIVE,
        semantics="bytes",
        unit="bytes",
        baseline_relative=False,
    )
    analysis = analyze_traffic_series(
        series,
        window_start_ns=window[0] if window is not None else None,
        window_end_ns=window[1] if window is not None else None,
        include_intervals=include_intervals,
    )
    return analyze_node_nic_balance(node_id, {"nic-1": analysis}, ["nic-1"])


def make_results(totals):
    return {
        f"node-{index}": make_node_result(f"node-{index}", total)
        for index, total in enumerate(totals, start=1)
    }


def test_balanced_nodes_have_equal_shares_and_full_effective_count():
    result = analyze_job_node_balance(make_results([100, 100, 100, 100]))

    assert result.status is JobBalanceStatus.OK
    assert result.job_total_bytes == 400
    assert [node.traffic_share for node in result.per_node] == [0.25] * 4
    assert result.max_min_ratio == 1.0
    assert result.max_min_ratio_denominator is MaxMinRatioDenominator.POSITIVE
    assert result.effective_node_count == pytest.approx(4.0)
    assert result.spatial_p50_bytes == 100
    assert result.spatial_p99_bytes == 100
    assert result.hotspot_node_count == 1
    assert result.hotspot_concentration == pytest.approx(0.25)
    assert result.coverage_fraction == 1.0
    assert result.topology_state is JobTopologyState.ABSENT
    assert result.unsupported_dimensions == ("topology", "peer", "near_far")


def test_single_hot_node_has_one_hotspot_and_zero_denominator():
    result = analyze_job_node_balance(make_results([0, 1000, 0, 0]))

    assert result.status is JobBalanceStatus.OK
    assert result.job_total_bytes == 1000
    assert [node.traffic_share for node in result.per_node] == [0.0, 1.0, 0.0, 0.0]
    assert result.max_traffic_bytes == 1000
    assert result.min_traffic_bytes == 0
    assert result.max_min_ratio is None
    assert result.max_min_ratio_denominator is MaxMinRatioDenominator.ZERO
    assert result.effective_node_count == pytest.approx(1.0)
    assert result.spatial_p50_bytes == 0
    assert result.spatial_p99_bytes == 1000
    assert result.hotspot_concentration == 1.0


def test_spatial_p99_uses_node_totals_not_temporal_p99s():
    result = analyze_job_node_balance(make_results([100, 300]))

    temporal_p99s = [
        nic.temporal_p99_rate_bytes_per_s
        for node in result.per_node
        for nic in node.source_result.per_nic
    ]
    assert temporal_p99s == [100, 300]
    assert result.spatial_p99_bytes == 300
    assert result.spatial_p99_bytes != pytest.approx(sum(temporal_p99s) / 2)


def test_zero_traffic_job_has_undefined_derived_shares():
    result = analyze_job_node_balance(make_results([0, 0, 0]))

    assert result.status is JobBalanceStatus.ZERO_TRAFFIC
    assert JobBalanceState.ZERO_TRAFFIC in result.state_flags
    assert result.job_total_bytes == 0
    assert all(node.traffic_share is None for node in result.per_node)
    assert result.max_min_ratio is None
    assert result.max_min_ratio_denominator is MaxMinRatioDenominator.ZERO
    assert result.effective_node_count is None
    assert result.spatial_p50_bytes == 0
    assert result.spatial_p99_bytes == 0
    assert result.hotspot_concentration is None


def test_missing_expected_node_is_not_treated_as_zero_traffic():
    results = make_results([100, 200])

    result = analyze_job_node_balance(results, ["node-1", "node-2", "node-3"])

    assert result.status is JobBalanceStatus.PARTIAL_DATA
    assert result.missing_node_ids == ("node-3",)
    assert JobBalanceState.MISSING_NODE in result.state_flags
    assert result.known_node_count == 3
    assert result.observed_node_count == 2
    assert result.job_total_bytes == 300
    assert [node.traffic_share for node in result.per_node] == pytest.approx(
        [1 / 3, 2 / 3]
    )
    assert result.effective_node_count == pytest.approx(9 / 5)
    assert result.coverage_fraction is None


def test_partial_node_with_excluded_invalid_interval_is_incompatible():
    timestamps = [0, 500_000_000, 1_000_000_000, 1_500_000_000, 2_000_000_000]
    results = {
        "node-1": make_node_result(
            "node-1", None, values=[0, 40, 20, 60, 80], timestamps=timestamps
        ),
        "node-2": make_node_result(
            "node-2", None, values=[0, 0, 180, 300, 300], timestamps=timestamps
        ),
    }

    result = analyze_job_node_balance(results)

    assert result.status is JobBalanceStatus.UNSUPPORTED_DATA
    assert JobBalanceState.PARTIAL_NODE in result.state_flags
    assert result.job_total_bytes is None
    assert result.valid_interval_boundary_signature is None
    assert all(node.total_bytes is None for node in result.per_node)
    assert all(node.traffic_share is None for node in result.per_node)
    assert any(
        "exact valid interval boundaries" in reason
        for reason in result.status_reasons
    )


def test_valid_partial_topology_keeps_unmapped_node_visible():
    results = make_results([100, 300])

    result = analyze_job_node_balance(
        results,
        ["node-1", "node-2"],
        topology_groups={"group-a": ("node-1",)},
    )

    assert result.topology_state is JobTopologyState.VALID_PARTIAL
    assert result.topology_supported is True
    assert result.unknown_topology_node_ids == ("node-2",)
    assert len(result.topology_groups) == 1
    assert result.topology_groups[0].topology_group_id == "group-a"
    assert result.topology_groups[0].total_bytes == 100
    assert result.topology_groups[0].traffic_share == 0.25
    assert result.peer_near_far_supported is False
    assert "peer" in result.unsupported_dimensions
    assert "near_far" in result.unsupported_dimensions


def test_different_valid_interval_boundaries_do_not_produce_job_shares():
    results = {
        "node-1": make_node_result(
            "node-1",
            100,
            timestamps=[0, 1_000_000_000, 2_000_000_000],
            window=(0, 2_000_000_000),
        ),
        "node-2": make_node_result(
            "node-2",
            300,
            values=[0, 300],
            timestamps=[0, 1_000_000_000],
            window=(0, 2_000_000_000),
        ),
    }

    result = analyze_job_node_balance(
        results,
        ["node-1", "node-2"],
        topology_groups={"group-a": ("node-1", "node-2")},
    )

    assert result.status is JobBalanceStatus.UNSUPPORTED_DATA
    assert result.job_total_bytes is None
    assert result.common_window_ns is None
    assert result.represented_window_ns is None
    assert all(node.traffic_share is None for node in result.per_node)
    assert all(node.total_bytes is None for node in result.per_node)
    assert result.topology_groups[0].total_bytes is None
    assert result.topology_groups[0].traffic_share is None
    assert any(
        "valid interval boundaries" in reason for reason in result.status_reasons
    )


def test_summary_mode_rejects_same_outer_window_with_different_interior_boundaries():
    results = {
        "node-1": make_node_result(
            "node-1",
            300,
            timestamps=[0, 1_000_000_000, 2_000_000_000],
            window=(0, 2_000_000_000),
            include_intervals=False,
        ),
        "node-2": make_node_result(
            "node-2",
            300,
            values=[0, 150, 300],
            timestamps=[0, 500_000_000, 2_000_000_000],
            window=(0, 2_000_000_000),
            include_intervals=False,
        ),
    }

    result = analyze_job_node_balance(
        results,
        ["node-1", "node-2"],
        topology_groups={"group-a": ("node-1", "node-2")},
    )

    assert result.status is JobBalanceStatus.UNSUPPORTED_DATA
    assert result.job_total_bytes is None
    assert result.valid_interval_boundary_signature is None
    assert all(node.total_bytes is None for node in result.per_node)
    assert all(node.traffic_share is None for node in result.per_node)
    assert result.topology_groups[0].total_bytes is None
    assert result.topology_groups[0].traffic_share is None
    assert any(
        "exact valid interval boundaries" in reason
        for reason in result.status_reasons
    )


def test_summary_mode_matching_boundary_signatures_aggregate():
    timestamps = [0, 500_000_000, 2_000_000_000]
    results = {
        "node-1": make_node_result(
            "node-1",
            300,
            timestamps=timestamps,
            include_intervals=False,
        ),
        "node-2": make_node_result(
            "node-2",
            300,
            timestamps=timestamps,
            include_intervals=False,
        ),
    }

    result = analyze_job_node_balance(
        results,
        topology_groups={"group-a": ("node-1", "node-2")},
    )

    assert result.status is JobBalanceStatus.OK
    assert result.job_total_bytes == 600
    assert [node.traffic_share for node in result.per_node] == [0.5, 0.5]
    assert result.valid_interval_boundary_signature == (
        0,
        500_000_000,
        500_000_000,
        2_000_000_000,
    )
    assert result.topology_groups[0].total_bytes == 600
    assert result.topology_groups[0].traffic_share == 1.0


def test_duplicate_topology_mapping_falls_back_to_node_only_result():
    results = make_results([100, 300])

    result = analyze_job_node_balance(
        results,
        ["node-1", "node-2"],
        topology_groups={
            "group-a": ("node-1",),
            "group-b": ("node-1", "node-2"),
        },
    )

    assert result.topology_state is JobTopologyState.INVALID
    assert result.topology_supported is False
    assert result.duplicate_topology_node_ids == ("node-1",)
    assert result.topology_groups == ()
    assert result.job_total_bytes == 400
    assert [node.traffic_share for node in result.per_node] == [0.25, 0.75]
    assert "topology" in result.unsupported_dimensions


def test_absent_topology_returns_useful_node_only_result():
    results = make_results([100, 300])

    result = analyze_job_node_balance(results)

    assert result.topology_state is JobTopologyState.ABSENT
    assert result.topology_supported is False
    assert result.unknown_topology_node_ids == ()
    assert result.topology_groups == ()
    assert result.job_total_bytes == 400
    assert result.effective_node_count == pytest.approx(1.6)
    assert result.hotspot_concentration == 0.75
    assert result.peer_near_far_supported is False


def test_mixed_directions_are_unsupported_and_not_merged():
    results = {
        "node-1": make_node_result("node-1", 100),
        "node-2": make_node_result("node-2", 300, direction=TrafficDirection.RX),
    }

    result = analyze_job_node_balance(results)

    assert result.status is JobBalanceStatus.UNSUPPORTED_DATA
    assert result.direction is None
    assert result.job_total_bytes is None
    assert all(node.traffic_share is None for node in result.per_node)
