import math

import pytest

from src.node_analysis import (
    MaxMinRatioDenominator,
    NodeBalanceState,
    NodeBalanceStatus,
    analyze_node_nic_balance,
)
from src.traffic_analysis import (
    MetricSeries,
    SeriesIdentity,
    TrafficDirection,
    ValueKind,
    analyze_traffic_series,
)


def make_analysis(
    nic_id,
    values,
    *,
    timestamps=None,
    direction=TrafficDirection.TX,
    value_kind=ValueKind.CUMULATIVE,
    window=None,
    include_intervals=True,
):
    identity = SeriesIdentity(
        run_id="run-1",
        node_id="node-1",
        nic_id=nic_id,
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
        value_kind=value_kind,
        semantics="bytes",
        unit="bytes",
        baseline_relative=False,
    )
    return analyze_traffic_series(
        series,
        window_start_ns=window[0] if window is not None else None,
        window_end_ns=window[1] if window is not None else None,
        include_intervals=include_intervals,
    )


def make_results(totals, *, known=None):
    analyses = {
        f"nic-{index}": make_analysis(f"nic-{index}", [0, total, total])
        for index, total in enumerate(totals, start=1)
    }
    inventory = known if known is not None else list(analyses)
    return analyze_node_nic_balance("node-1", analyses, inventory)


def test_balanced_nics_have_unit_ratio_and_full_effective_count():
    result = make_results([100, 100, 100, 100])

    assert result.status is NodeBalanceStatus.OK
    assert result.node_total_bytes == 400
    assert type(result.node_total_bytes) is int
    assert [nic.traffic_share for nic in result.per_nic] == [0.25] * 4
    assert result.active_nic_fraction == 1.0
    assert result.max_min_ratio == 1.0
    assert result.max_min_ratio_denominator is MaxMinRatioDenominator.POSITIVE
    assert result.effective_nic_count == pytest.approx(4.0)
    assert result.spatial_p50_bytes == 100
    assert result.spatial_p99_bytes == 100
    assert result.cross_nic_coefficient_of_variation == 0.0


def test_single_hot_nic_keeps_idle_zero_denominator_visible():
    result = make_results([0, 1000, 0, 0])

    assert result.status is NodeBalanceStatus.OK
    assert result.node_total_bytes == 1000
    assert [nic.traffic_share for nic in result.per_nic] == [0.0, 1.0, 0.0, 0.0]
    assert result.active_nic_fraction == 0.25
    assert result.max_traffic_bytes == 1000
    assert result.min_traffic_bytes == 0
    assert result.max_min_ratio is None
    assert result.max_min_ratio_denominator is MaxMinRatioDenominator.ZERO
    assert result.effective_nic_count == pytest.approx(1.0)
    assert result.spatial_p50_bytes == 0
    assert result.spatial_p99_bytes == 1000
    assert result.cross_nic_coefficient_of_variation == pytest.approx(math.sqrt(3.0))


def test_zero_traffic_node_has_undefined_shares_and_effective_count():
    result = make_results([0, 0, 0, 0])

    assert result.status is NodeBalanceStatus.ZERO_TRAFFIC
    assert NodeBalanceState.ZERO_TRAFFIC in result.state_flags
    assert result.node_total_bytes == 0
    assert all(nic.traffic_share is None for nic in result.per_nic)
    assert result.active_nic_fraction == 0.0
    assert result.max_min_ratio is None
    assert result.max_min_ratio_denominator is MaxMinRatioDenominator.ZERO
    assert result.effective_nic_count is None
    assert result.spatial_p50_bytes == 0
    assert result.spatial_p99_bytes == 0
    assert result.cross_nic_coefficient_of_variation is None


def test_missing_known_nic_is_not_treated_as_zero_traffic():
    analyses = {
        "nic-1": make_analysis("nic-1", [0, 100, 100]),
        "nic-2": make_analysis("nic-2", [0, 200, 200]),
        "nic-3": make_analysis("nic-3", [0, 300, 300]),
    }
    result = analyze_node_nic_balance(
        "node-1", analyses, ["nic-1", "nic-2", "nic-3", "nic-4"]
    )

    assert result.status is NodeBalanceStatus.PARTIAL_DATA
    assert result.missing_nic_ids == ("nic-4",)
    assert NodeBalanceState.MISSING_NIC in result.state_flags
    assert result.node_total_bytes == 600
    assert [nic.traffic_share for nic in result.per_nic] == pytest.approx(
        [1 / 6, 1 / 3, 1 / 2]
    )
    assert result.active_nic_fraction == 0.75
    assert result.max_min_ratio == 3.0
    assert result.effective_nic_count == pytest.approx(18 / 7)
    assert result.spatial_p50_bytes == 200
    assert result.spatial_p99_bytes == 300
    assert result.cross_nic_coefficient_of_variation == pytest.approx(
        math.sqrt(1 / 6)
    )


def test_unknown_observed_nic_stays_outside_known_activity_denominator():
    analyses = {
        "nic-1": make_analysis("nic-1", [0, 100, 100]),
        "nic-2": make_analysis("nic-2", [0, 300, 300]),
    }
    result = analyze_node_nic_balance("node-1", analyses, ["nic-1"])

    assert result.status is NodeBalanceStatus.PARTIAL_DATA
    assert result.unknown_observed_nic_ids == ("nic-2",)
    assert NodeBalanceState.UNKNOWN_NIC in result.state_flags
    assert result.known_nic_count == 1
    assert result.observed_nic_count == 2
    assert result.active_known_nic_count == 1
    assert result.active_observed_nic_count == 2
    assert result.active_nic_fraction == 1.0
    assert result.node_total_bytes == 400
    assert [nic.traffic_share for nic in result.per_nic] == pytest.approx(
        [0.25, 0.75]
    )
    assert result.effective_nic_count == pytest.approx(1.6)


def test_partial_node_with_excluded_invalid_interval_is_incompatible():
    timestamps = [0, 500_000_000, 1_000_000_000, 1_500_000_000, 2_000_000_000]
    partial = make_analysis("nic-2", [0, 100, 50, 150, 200], timestamps=timestamps)
    assert partial.invalid_interval_count == 1
    analyses = {
        "nic-1": make_analysis(
            "nic-1", [0, 0, 150, 250, 250], timestamps=timestamps
        ),
        "nic-2": partial,
        "nic-3": make_analysis(
            "nic-3", [0, 0, 150, 250, 250], timestamps=timestamps
        ),
        "nic-4": make_analysis(
            "nic-4", [0, 0, 150, 250, 250], timestamps=timestamps
        ),
    }
    result = analyze_node_nic_balance(
        "node-1", analyses, ["nic-1", "nic-2", "nic-3", "nic-4"]
    )

    assert result.status is NodeBalanceStatus.UNSUPPORTED_DATA
    assert result.valid_interval_boundary_signature is None
    assert result.node_total_bytes is None
    assert result.invalid_interval_count == 1
    assert result.valid_interval_count == 15
    assert all(nic.traffic_share is None for nic in result.per_nic)
    assert any(
        "exact valid interval boundaries" in reason
        for reason in result.status_reasons
    )


def test_different_valid_interval_boundaries_do_not_produce_node_shares():
    analyses = {
        "nic-1": make_analysis(
            "nic-1",
            [0, 100, 100],
            timestamps=[0, 1_000_000_000, 2_000_000_000],
            window=(0, 2_000_000_000),
        ),
        "nic-2": make_analysis(
            "nic-2",
            [0, 300],
            timestamps=[0, 1_000_000_000],
            window=(0, 2_000_000_000),
        ),
    }

    result = analyze_node_nic_balance("node-1", analyses, list(analyses))

    assert result.status is NodeBalanceStatus.UNSUPPORTED_DATA
    assert result.node_total_bytes is None
    assert result.represented_window_ns is None
    assert all(nic.total_bytes is None for nic in result.per_nic)
    assert all(nic.traffic_share is None for nic in result.per_nic)
    assert any(
        "valid interval boundaries" in reason for reason in result.status_reasons
    )


def test_summary_mode_rejects_same_outer_window_with_different_interior_boundaries():
    analyses = {
        "nic-1": make_analysis(
            "nic-1",
            [0, 100, 300],
            timestamps=[0, 1_000_000_000, 2_000_000_000],
            window=(0, 2_000_000_000),
            include_intervals=False,
        ),
        "nic-2": make_analysis(
            "nic-2",
            [0, 100, 300],
            timestamps=[0, 500_000_000, 2_000_000_000],
            window=(0, 2_000_000_000),
            include_intervals=False,
        ),
    }

    result = analyze_node_nic_balance("node-1", analyses, list(analyses))

    assert result.status is NodeBalanceStatus.UNSUPPORTED_DATA
    assert result.node_total_bytes is None
    assert result.valid_interval_boundary_signature is None
    assert all(nic.total_bytes is None for nic in result.per_nic)
    assert all(nic.traffic_share is None for nic in result.per_nic)
    assert any(
        "exact valid interval boundaries" in reason
        for reason in result.status_reasons
    )


def test_summary_mode_matching_boundary_signatures_aggregate():
    timestamps = [0, 500_000_000, 2_000_000_000]
    analyses = {
        "nic-1": make_analysis(
            "nic-1",
            [0, 100, 300],
            timestamps=timestamps,
            include_intervals=False,
        ),
        "nic-2": make_analysis(
            "nic-2",
            [0, 100, 300],
            timestamps=timestamps,
            include_intervals=False,
        ),
    }

    result = analyze_node_nic_balance("node-1", analyses, list(analyses))

    assert result.status is NodeBalanceStatus.OK
    assert result.node_total_bytes == 600
    assert [nic.traffic_share for nic in result.per_nic] == [0.5, 0.5]
    assert result.valid_interval_boundary_signature == (
        0,
        500_000_000,
        500_000_000,
        2_000_000_000,
    )


def test_no_records_is_missing_not_zero_traffic():
    result = analyze_node_nic_balance("node-1", {}, ["nic-1", "nic-2"])

    assert result.status is NodeBalanceStatus.MISSING_DATA
    assert result.missing_nic_ids == ("nic-1", "nic-2")
    assert result.node_total_bytes is None
    assert result.active_nic_fraction == 0.0
    assert result.max_min_ratio_denominator is MaxMinRatioDenominator.MISSING
    assert result.effective_nic_count is None


def test_spatial_quantiles_use_nic_totals_not_temporal_p99s():
    analyses = {
        "nic-1": make_analysis("nic-1", [0, 50, 100]),
        "nic-2": make_analysis("nic-2", [0, 100, 200]),
        "nic-3": make_analysis("nic-3", [0, 150, 300]),
    }
    result = analyze_node_nic_balance(
        "node-1", analyses, list(analyses), activity_threshold_bytes_per_s=120
    )

    assert [nic.total_bytes for nic in result.per_nic] == [100, 200, 300]
    assert [nic.temporal_p99_rate_bytes_per_s for nic in result.per_nic] == [
        50,
        100,
        150,
    ]
    assert result.spatial_p50_bytes == 200
    assert result.spatial_p99_bytes == 300
    assert result.cross_nic_coefficient_of_variation == pytest.approx(
        math.sqrt(1 / 6)
    )
    assert result.effective_nic_count == pytest.approx(18 / 7)
    assert result.active_known_nic_count == 1
    assert result.active_nic_fraction == pytest.approx(1 / 3)


def test_mixed_directions_are_unsupported_and_not_merged():
    analyses = {
        "nic-1": make_analysis("nic-1", [0, 100, 100]),
        "nic-2": make_analysis(
            "nic-2", [0, 300, 300], direction=TrafficDirection.RX
        ),
    }
    result = analyze_node_nic_balance("node-1", analyses, list(analyses))

    assert result.status is NodeBalanceStatus.UNSUPPORTED_DATA
    assert result.direction is None
    assert NodeBalanceState.UNSUPPORTED_INPUT in result.state_flags
    assert result.node_total_bytes is None
    assert all(nic.traffic_share is None for nic in result.per_nic)
    assert result.effective_nic_count is None
    assert result.max_min_ratio_denominator is MaxMinRatioDenominator.MISSING


def test_unsupported_series_has_no_node_total():
    analyses = {
        "nic-1": make_analysis(
            "nic-1", [0, 100, 200], value_kind=ValueKind.DELTA
        )
    }
    result = analyze_node_nic_balance("node-1", analyses, ["nic-1"])

    assert result.status is NodeBalanceStatus.UNSUPPORTED_DATA
    assert result.unsupported_nic_ids == ("nic-1",)
    assert NodeBalanceState.UNSUPPORTED_INPUT in result.state_flags
    assert result.node_total_bytes is None
    assert result.per_nic[0].total_bytes is None
    assert result.effective_nic_count is None


def test_activity_threshold_must_be_finite_and_nonnegative():
    analyses = {"nic-1": make_analysis("nic-1", [0, 100, 100])}

    with pytest.raises(ValueError, match="finite and nonnegative"):
        analyze_node_nic_balance(
            "node-1", analyses, ["nic-1"], activity_threshold_bytes_per_s=-1
        )
