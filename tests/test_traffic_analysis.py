import math
from dataclasses import fields

import pytest

from src.traffic_analysis import (
    AnalysisStatus,
    GapPolicy,
    MetricSeries,
    SeriesIdentity,
    SeriesQualityFlag,
    TrafficDirection,
    TrafficAnalysis,
    ValueKind,
    analyze_traffic_series,
)


def make_series(
    timestamps,
    values,
    *,
    direction=TrafficDirection.TX,
    value_dtype="uint64",
    baseline_relative=False,
    quality_flags=(),
    value_kind=ValueKind.CUMULATIVE,
):
    identity = SeriesIdentity(
        run_id="run-1",
        node_id="node-1",
        nic_id="nic-1",
        collector_id="collector-rank-0",
        counter_id="CXI_RX_BYTES",
        direction=direction,
    )
    return MetricSeries(
        identity=identity,
        timestamps_ns=tuple(timestamps),
        values=tuple(values),
        value_dtype=value_dtype,
        value_kind=value_kind,
        semantics="bytes",
        unit="bytes",
        baseline_relative=baseline_relative,
        quality_flags=tuple(quality_flags),
    )


def test_regular_sampling_totals_and_statistics():
    series = make_series(
        [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
        [0, 100, 300, 600],
    )

    result = analyze_traffic_series(series)

    assert result.status is AnalysisStatus.OK
    assert result.total_bytes == 600
    assert type(result.total_bytes) is int
    assert result.covered_duration_ns == 3_000_000_000
    assert result.coverage_fraction == 1.0
    assert result.mean_rate_bytes_per_s == pytest.approx(200.0)
    assert result.p50_rate_bytes_per_s == 200.0
    assert result.p99_rate_bytes_per_s == 300.0
    assert result.weighted_variance == pytest.approx(20_000.0 / 3.0)
    assert result.weighted_standard_deviation == pytest.approx(
        math.sqrt(20_000.0 / 3.0)
    )
    assert result.coefficient_of_variation == pytest.approx(
        math.sqrt(20_000.0 / 3.0) / 200.0
    )
    assert result.quantile_interpolation.startswith("duration-weighted")


@pytest.mark.parametrize(
    ("timestamps", "values", "window", "gap_threshold", "gap_policy"),
    [
        pytest.param(
            [0, 1_000_000_000, 2_000_000_000],
            [0, 100, 300],
            None,
            None,
            GapPolicy.INCLUDE,
            id="regular",
        ),
        pytest.param(
            [0, 1_000_000_000, 4_000_000_000],
            [0, 100, 700],
            None,
            None,
            GapPolicy.INCLUDE,
            id="irregular",
        ),
        pytest.param(
            [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
            [0, 100, 20, 80],
            None,
            None,
            GapPolicy.INCLUDE,
            id="reset",
        ),
        pytest.param(
            [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
            [0, 10, 20, 30],
            (500_000_000, 2_500_000_000),
            None,
            GapPolicy.INCLUDE,
            id="boundary",
        ),
        pytest.param(
            [0, 1_000_000_000, 5_000_000_000],
            [0, 100, 500],
            None,
            500_000_000,
            GapPolicy.EXCLUDE_FROM_DISTRIBUTION,
            id="gap",
        ),
        pytest.param(
            [20, 10, 0],
            [3, 2, 1],
            None,
            None,
            GapPolicy.INCLUDE,
            id="reordered",
        ),
        pytest.param(
            [0, 10, 10, 20],
            [0, 1, 1, 2],
            None,
            None,
            GapPolicy.INCLUDE,
            id="duplicate-sample",
        ),
    ],
)
def test_summary_path_agrees_with_detailed_path(
    timestamps,
    values,
    window,
    gap_threshold,
    gap_policy,
):
    series = make_series(timestamps, values)
    arguments = {}
    if window is not None:
        arguments["window_start_ns"] = window[0]
        arguments["window_end_ns"] = window[1]
    if gap_threshold is not None:
        arguments["gap_threshold_ns"] = gap_threshold

    detailed = analyze_traffic_series(series, gap_policy=gap_policy, **arguments)
    summary = analyze_traffic_series(
        series,
        gap_policy=gap_policy,
        include_intervals=False,
        **arguments,
    )

    assert detailed.interval_detail_included is True
    assert detailed.intervals
    assert summary.interval_detail_included is False
    assert summary.intervals == ()
    assert summary.boundary_excluded_intervals == ()
    assert summary.sample_count == detailed.sample_count
    ignored_fields = {
        "intervals",
        "boundary_excluded_intervals",
        "interval_detail_included",
    }
    for field in fields(TrafficAnalysis):
        if field.name not in ignored_fields:
            assert getattr(summary, field.name) == getattr(
                detailed, field.name
            ), field.name


def test_summary_path_preserves_exact_uint64_arithmetic():
    start = 2**64 - 1_001
    end = 2**64 - 1
    series = make_series([0, 1_000_000_000], [start, end])

    detailed = analyze_traffic_series(series)
    summary = analyze_traffic_series(series, include_intervals=False)

    assert detailed.intervals[0].delta_bytes == 1_000
    assert summary.intervals == ()
    assert summary.total_bytes == detailed.total_bytes == 1_000
    assert type(summary.total_bytes) is int
    assert summary.mean_rate_bytes_per_s == detailed.mean_rate_bytes_per_s


def test_requested_window_sample_count_is_window_local():
    series = make_series(
        [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
        [0, 100, 300, 600],
    )
    window = (500_000_000, 2_500_000_000)

    detailed = analyze_traffic_series(
        series, window_start_ns=window[0], window_end_ns=window[1]
    )
    summary = analyze_traffic_series(
        series,
        window_start_ns=window[0],
        window_end_ns=window[1],
        include_intervals=False,
    )

    assert detailed.sample_count == summary.sample_count == 2


def test_detailed_uint64_total_survives_accumulation_across_resets():
    maximum = 2**64 - 1
    series = make_series(
        [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
        [0, maximum, 0, maximum],
    )

    detailed = analyze_traffic_series(series)
    summary = analyze_traffic_series(series, include_intervals=False)

    assert detailed.valid_interval_count == 2
    assert detailed.invalid_interval_count == 1
    assert detailed.total_bytes == 2 * maximum
    assert summary.total_bytes == detailed.total_bytes
    assert type(detailed.total_bytes) is int


def test_summary_boundary_signature_detects_interior_boundary_mismatch():
    regular = analyze_traffic_series(
        make_series(
            [0, 1_000_000_000, 2_000_000_000],
            [0, 100, 300],
        ),
        include_intervals=False,
    )
    alternate = analyze_traffic_series(
        make_series(
            [0, 500_000_000, 2_000_000_000],
            [0, 100, 300],
        ),
        include_intervals=False,
    )

    assert regular.represented_window_ns == alternate.represented_window_ns
    assert regular.intervals == alternate.intervals == ()
    assert regular.valid_interval_boundary_signature == (
        0,
        1_000_000_000,
        1_000_000_000,
        2_000_000_000,
    )
    assert alternate.valid_interval_boundary_signature == (
        0,
        500_000_000,
        500_000_000,
        2_000_000_000,
    )
    assert (
        regular.valid_interval_boundary_signature
        != alternate.valid_interval_boundary_signature
    )


def test_summary_path_preserves_interval_count_for_conflicting_timestamp():
    series = make_series(
        [0, 1_000_000_000, 1_000_000_000],
        [0, 10, 15],
    )

    detailed = analyze_traffic_series(series)
    summary = analyze_traffic_series(series, include_intervals=False)

    assert detailed.interval_count == summary.interval_count == 1
    assert summary.sample_count == detailed.sample_count == 1


def test_summary_path_uses_raw_extrema_for_conflicting_timestamp_window():
    series = make_series(
        [0, 1_000_000_000, 1_000_000_000],
        [0, 10, 15],
    )

    detailed = analyze_traffic_series(series)
    summary = analyze_traffic_series(series, include_intervals=False)

    assert summary.requested_window_ns == detailed.requested_window_ns == (
        0,
        1_000_000_000,
    )
    assert summary.covered_duration_ns == detailed.covered_duration_ns == 0
    assert summary.coverage_fraction == detailed.coverage_fraction == 0.0


def test_summary_path_preserves_int64_boundary_span_arithmetic():
    series = make_series(
        [-(2**63), 2**63 - 1],
        [0, 1],
    )

    detailed = analyze_traffic_series(series)
    summary = analyze_traffic_series(series, include_intervals=False)

    assert detailed.covered_duration_ns == 2**64 - 1
    assert summary.covered_duration_ns == detailed.covered_duration_ns
    assert summary.requested_duration_ns == detailed.requested_duration_ns
    assert summary.total_bytes == detailed.total_bytes == 1


def test_nonfinite_derived_rate_is_rejected_in_both_paths():
    series = make_series(
        [0, 1_000_000_000],
        [-1e308, 1e308],
        value_dtype="float64",
    )

    with pytest.raises(ValueError, match="non-finite rate"):
        analyze_traffic_series(series)

    with pytest.raises(ValueError, match="non-finite rate"):
        analyze_traffic_series(series, include_intervals=False)


def test_include_intervals_must_be_boolean():
    with pytest.raises(ValueError, match="include_intervals must be a boolean"):
        analyze_traffic_series(make_series([0, 1], [0, 1]), include_intervals=1)


def test_irregular_sampling_is_duration_weighted():
    series = make_series(
        [0, 1_000_000_000, 4_000_000_000],
        [0, 100, 700],
    )

    result = analyze_traffic_series(series)

    assert result.total_bytes == 700
    assert result.covered_duration_ns == 4_000_000_000
    assert result.mean_rate_bytes_per_s == pytest.approx(175.0)
    assert result.p50_rate_bytes_per_s == 200.0
    assert result.p99_rate_bytes_per_s == 200.0
    assert result.weighted_variance == pytest.approx(1_875.0)
    assert result.coefficient_of_variation == pytest.approx(
        math.sqrt(1_875.0) / 175.0
    )
    rates = [interval.rate_bytes_per_s for interval in result.intervals]
    assert rates == [100.0, 200.0]


def test_zero_traffic_is_distinct_from_missing_data():
    series = make_series(
        [0, 1_000_000_000, 2_000_000_000],
        [0, 0, 0],
    )

    result = analyze_traffic_series(series)

    assert result.status is AnalysisStatus.ZERO_TRAFFIC
    assert result.total_bytes == 0
    assert result.mean_rate_bytes_per_s == 0.0
    assert result.p50_rate_bytes_per_s == 0.0
    assert result.p99_rate_bytes_per_s == 0.0
    assert result.weighted_variance == 0.0
    assert result.coefficient_of_variation is None
    assert result.coverage_fraction == 1.0


def test_explicit_missing_state_has_no_numerical_results():
    series = make_series(
        [],
        [],
        quality_flags=(SeriesQualityFlag.MISSING,),
    )

    result = analyze_traffic_series(series)

    assert result.status is AnalysisStatus.MISSING_DATA
    assert result.total_bytes is None
    assert result.mean_rate_bytes_per_s is None
    assert result.coverage_fraction is None
    assert SeriesQualityFlag.MISSING in result.quality_flags


def test_decrease_or_reset_is_excluded_without_bridging():
    series = make_series(
        [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
        [0, 100, 20, 80],
    )

    result = analyze_traffic_series(series)

    assert result.status is AnalysisStatus.PARTIAL_INVALID_DATA
    assert result.total_bytes == 160
    assert result.covered_duration_ns == 2_000_000_000
    assert result.mean_rate_bytes_per_s == pytest.approx(80.0)
    assert result.valid_interval_count == 2
    assert result.invalid_interval_count == 1
    assert result.intervals[1].state.value == "decrease_or_reset"
    assert result.intervals[1].delta_bytes is None
    assert SeriesQualityFlag.DECREASE_OR_RESET in result.quality_flags


def test_uint64_subtraction_is_exact_before_float_conversion():
    start = 2**64 - 1_001
    end = 2**64 - 1
    series = make_series([0, 1_000_000_000], [start, end])

    result = analyze_traffic_series(series)

    assert result.total_bytes == 1_000
    assert type(result.total_bytes) is int
    assert result.intervals[0].delta_bytes == 1_000
    assert result.intervals[0].rate_bytes_per_s == pytest.approx(1_000.0)


def test_conflicting_equal_timestamp_is_invalid_zero_duration():
    series = make_series(
        [0, 1_000_000_000, 1_000_000_000],
        [0, 10, 15],
    )

    result = analyze_traffic_series(series)

    assert result.status is AnalysisStatus.INVALID_DATA
    assert result.total_bytes is None
    assert result.covered_duration_ns == 0
    assert result.coverage_fraction == 0.0
    assert result.invalid_interval_count == 1
    assert result.intervals[0].duration_ns == 0
    assert result.intervals[0].state.value == "conflicting_timestamp"
    assert SeriesQualityFlag.CONFLICTING_TIMESTAMP in result.quality_flags


def test_exact_window_excludes_straddling_boundary_intervals():
    series = make_series(
        [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
        [0, 10, 20, 30],
    )

    result = analyze_traffic_series(
        series,
        window_start_ns=500_000_000,
        window_end_ns=2_500_000_000,
    )

    assert result.status is AnalysisStatus.OK
    assert result.total_bytes == 10
    assert result.represented_window_ns == (
        1_000_000_000,
        2_000_000_000,
    )
    assert result.covered_duration_ns == 1_000_000_000
    assert result.coverage_fraction == 0.5
    assert result.valid_interval_count == 1
    assert result.boundary_excluded_interval_count == 2
    assert SeriesQualityFlag.BOUNDARY_INTERVAL_EXCLUDED in result.quality_flags


def test_tx_and_rx_remain_separate():
    tx = make_series(
        [0, 1_000_000_000],
        [0, 100],
        direction=TrafficDirection.TX,
    )
    rx = make_series(
        [0, 1_000_000_000],
        [0, 50],
        direction=TrafficDirection.RX,
    )

    tx_result = analyze_traffic_series(tx)
    rx_result = analyze_traffic_series(rx)

    assert tx_result.identity.direction is TrafficDirection.TX
    assert rx_result.identity.direction is TrafficDirection.RX
    assert tx_result.total_bytes == 100
    assert rx_result.total_bytes == 50
    assert tx_result.rate_unit == rx_result.rate_unit == "bytes/s"


def test_baseline_relative_does_not_invent_hidden_baseline():
    series = make_series(
        [0, 1_000_000_000, 2_000_000_000],
        [10, 110, 210],
        baseline_relative=True,
    )

    result = analyze_traffic_series(series)

    assert result.status is AnalysisStatus.OK
    assert result.total_bytes == 200
    assert result.intervals[0].delta_bytes == 100
    assert SeriesQualityFlag.BASELINE_RELATIVE in result.quality_flags
    assert result.baseline_relative is True


def test_long_gap_can_be_excluded_from_distribution_only():
    series = make_series(
        [0, 1_000_000_000, 5_000_000_000],
        [0, 100, 500],
    )

    result = analyze_traffic_series(
        series,
        gap_threshold_ns=500_000_000,
        gap_policy=GapPolicy.EXCLUDE_FROM_DISTRIBUTION,
    )

    assert result.status is AnalysisStatus.OK
    assert result.total_bytes == 500
    assert result.mean_rate_bytes_per_s == pytest.approx(100.0)
    assert result.p50_rate_bytes_per_s is None
    assert result.p99_rate_bytes_per_s is None
    assert result.distribution_excluded_duration_ns == 5_000_000_000
    assert SeriesQualityFlag.LONG_GAP in result.quality_flags


@pytest.mark.parametrize("gap_threshold_ns", [0, -1, 1.0, True])
def test_gap_threshold_must_be_positive_integer_or_none(gap_threshold_ns):
    series = make_series([0, 1_000_000_000], [0, 100])

    with pytest.raises(ValueError, match="positive integer"):
        analyze_traffic_series(series, gap_threshold_ns=gap_threshold_ns)


def test_excluding_gap_distribution_requires_threshold():
    series = make_series([0, 1_000_000_000], [0, 100])

    with pytest.raises(
        ValueError,
        match="EXCLUDE_FROM_DISTRIBUTION requires a positive gap_threshold_ns",
    ):
        analyze_traffic_series(
            series,
            gap_policy=GapPolicy.EXCLUDE_FROM_DISTRIBUTION,
        )


def test_delta_counters_are_unsupported_for_cumulative_analysis():
    series = make_series(
        [0, 1_000_000_000],
        [0, 100],
        value_kind=ValueKind.DELTA,
    )

    result = analyze_traffic_series(series)

    assert result.status is AnalysisStatus.UNSUPPORTED_SERIES
    assert result.total_bytes is None
    assert SeriesQualityFlag.UNSUPPORTED_SERIES in result.quality_flags


@pytest.mark.parametrize(
    ("timestamps", "values", "value_dtype"),
    [
        ((0, True), (0, 1), "uint64"),
        ((0, 2**64), (0, 1), "uint64"),
        ((0, 1), (0.0, 1.0), "uint64"),
        ((0, 1), (0, 1), "float64"),
        ((0, 1), (0.0, math.nan), "float64"),
    ],
)
def test_series_input_contract_validation(timestamps, values, value_dtype):
    with pytest.raises(ValueError):
        make_series(timestamps, values, value_dtype=value_dtype)
