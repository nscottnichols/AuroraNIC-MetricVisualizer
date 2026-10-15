import pytest

from latency_analysis import (
    LATENCY_INTERPRETATION,
    LatencyBinDefinition,
    LatencyBinEdge,
    LatencyBinEncoding,
    LatencyHistogramGroup,
    LatencyMeanStatus,
    LatencyQualityFlag,
    LatencyQuantileStatus,
    LatencyStatus,
    analyze_latency_histograms,
)
from traffic_analysis import (
    MetricSeries,
    SeriesIdentity,
    SeriesQualityFlag,
    TrafficDirection,
    ValueKind,
)


NANOSECOND = 1_000_000_000


def make_series(
    group,
    bin_index,
    timestamps,
    values,
    *,
    quality_flags=(),
    value_kind=ValueKind.CUMULATIVE,
):
    prefix = (
        "pct_req_rsp_latency_"
        if group is LatencyHistogramGroup.REQUEST_RESPONSE
        else "pct_host_access_latency_"
    )
    return MetricSeries(
        identity=SeriesIdentity(
            run_id="run-1",
            node_id="node-1",
            nic_id="nic-1",
            collector_id="collector-1",
            counter_id=f"{prefix}{bin_index}",
            direction=TrafficDirection.TX,
        ),
        timestamps_ns=tuple(timestamps),
        values=tuple(values),
        value_dtype="uint64",
        value_kind=value_kind,
        semantics="cumulative histogram bin",
        unit="observations",
        baseline_relative=False,
        quality_flags=tuple(quality_flags),
    )


def two_bin_definition(
    encoding,
    *,
    edges=True,
    group=LatencyHistogramGroup.REQUEST_RESPONSE,
):
    return LatencyBinDefinition(
        definition_version=1,
        group=group,
        bin_names=("bin-0", "bin-1"),
        encoding=encoding,
        source="hand-check fixture definition",
        bin_edges=(
            (
                LatencyBinEdge(lower=0, upper=10, open_tailed=False),
                LatencyBinEdge(lower=10, upper=20, open_tailed=False),
            )
            if edges
            else None
        ),
        unit="nanoseconds" if edges else None,
    )


def test_disjoint_bins_use_exact_deltas_and_edge_bounds():
    definition = two_bin_definition(LatencyBinEncoding.DISJOINT)
    series = (
        make_series(
            LatencyHistogramGroup.REQUEST_RESPONSE,
            0,
            [0, NANOSECOND, 2 * NANOSECOND],
            [0, 3, 7],
        ),
        make_series(
            LatencyHistogramGroup.REQUEST_RESPONSE,
            1,
            [0, NANOSECOND, 2 * NANOSECOND],
            [0, 1, 4],
        ),
    )

    result = analyze_latency_histograms(series, {definition.group: definition})[0]

    assert result.status is LatencyStatus.OBSERVED_NONZERO
    assert result.bins_are_disjoint is True
    assert [interval.bin_deltas for interval in result.intervals] == [(3, 1), (4, 3)]
    assert [bin_result.count for bin_result in result.bin_results] == [7, 4]
    assert result.total_observations == 11
    assert [bin_result.normalized_fraction for bin_result in result.bin_results] == [
        pytest.approx(7 / 11),
        pytest.approx(4 / 11),
    ]
    assert result.quantile_results[0].status is LatencyQuantileStatus.BOUNDED
    assert result.quantile_results[0].bin_name == "bin-0"
    assert result.quantile_results[0].upper_edge == 10
    assert result.quantile_results[1].status is LatencyQuantileStatus.BOUNDED
    assert result.quantile_results[1].bin_name == "bin-1"
    assert result.quantile_results[1].upper_edge == 20
    assert result.mean_result.status is LatencyMeanStatus.ESTIMATED
    assert result.mean_result.value == pytest.approx((7 * 5 + 4 * 15) / 11)


def test_boundary_cumulative_encoding_is_decumulated_after_time_deltas():
    definition = two_bin_definition(LatencyBinEncoding.BOUNDARY_CUMULATIVE)
    series = (
        make_series(
            LatencyHistogramGroup.REQUEST_RESPONSE,
            0,
            [0, NANOSECOND, 2 * NANOSECOND],
            [0, 6, 8],
        ),
        make_series(
            LatencyHistogramGroup.REQUEST_RESPONSE,
            1,
            [0, NANOSECOND, 2 * NANOSECOND],
            [0, 10, 15],
        ),
    )

    result = analyze_latency_histograms(series, {definition.group: definition})[0]

    assert [interval.bin_deltas for interval in result.intervals] == [(6, 10), (2, 5)]
    assert [interval.disjoint_bin_counts for interval in result.intervals] == [
        (6, 4),
        (2, 3),
    ]
    assert [bin_result.count for bin_result in result.bin_results] == [8, 7]
    assert result.total_observations == 15
    assert result.boundary_decumulation_formula is not None
    assert [bin_result.normalized_fraction for bin_result in result.bin_results] == [
        pytest.approx(8 / 15),
        pytest.approx(7 / 15),
    ]
    assert result.mean_result.value == pytest.approx((8 * 5 + 7 * 15) / 15)


def test_requested_window_sample_count_is_window_local():
    definition = two_bin_definition(LatencyBinEncoding.DISJOINT)
    timestamps = [0, NANOSECOND, 2 * NANOSECOND, 3 * NANOSECOND]
    series = (
        make_series(
            LatencyHistogramGroup.REQUEST_RESPONSE, 0, timestamps, [0, 3, 7, 9]
        ),
        make_series(
            LatencyHistogramGroup.REQUEST_RESPONSE, 1, timestamps, [0, 1, 4, 5]
        ),
    )

    result = analyze_latency_histograms(
        series,
        {definition.group: definition},
        requested_window_ns=(500_000_000, 2_500_000_000),
    )[0]

    assert result.sample_count == 2
    assert result.interval_count == 1
    assert result.valid_interval_count == 1
    assert result.boundary_excluded_interval_count == 2
    assert result.total_observations == 7
    assert result.covered_duration_ns == NANOSECOND
    assert result.coverage_fraction == 0.5


def test_missing_edges_keeps_named_counts_but_blocks_latency_summaries():
    definition = two_bin_definition(LatencyBinEncoding.DISJOINT, edges=False)
    series = (
        make_series(
            LatencyHistogramGroup.REQUEST_RESPONSE, 0, [0, NANOSECOND], [0, 7]
        ),
        make_series(
            LatencyHistogramGroup.REQUEST_RESPONSE, 1, [0, NANOSECOND], [0, 4]
        ),
    )

    result = analyze_latency_histograms(series, {definition.group: definition})[0]

    assert [bin_result.count for bin_result in result.bin_results] == [7, 4]
    assert result.total_observations == 11
    assert result.normalized_distribution_available is True
    assert LatencyQualityFlag.MISSING_BIN_EDGES in result.quality_flags
    assert all(
        quantile.status is LatencyQuantileStatus.UNSUPPORTED
        for quantile in result.quantile_results
    )
    assert result.mean_result.status is LatencyMeanStatus.UNSUPPORTED


def test_open_tail_reports_lower_bound_and_blocks_mean_estimate():
    definition = LatencyBinDefinition(
        definition_version=1,
        group=LatencyHistogramGroup.HOST_ACCESS,
        bin_names=("bin-0", "bin-1", "bin-2"),
        encoding=LatencyBinEncoding.DISJOINT,
        source="hand-check open-tail fixture",
        bin_edges=(
            LatencyBinEdge(lower=0, upper=10, open_tailed=False),
            LatencyBinEdge(lower=10, upper=20, open_tailed=False),
            LatencyBinEdge(lower=20, upper=None, open_tailed=True),
        ),
        unit="nanoseconds",
    )
    timestamps = [0, NANOSECOND]
    series = tuple(
        make_series(
            LatencyHistogramGroup.HOST_ACCESS,
            bin_index,
            timestamps,
            [0, count],
        )
        for bin_index, count in enumerate((2, 3, 5))
    )

    result = analyze_latency_histograms(series, {definition.group: definition})[0]

    assert result.total_observations == 10
    assert result.quantile_results[0].status is LatencyQuantileStatus.BOUNDED
    assert result.quantile_results[0].bin_name == "bin-1"
    assert result.quantile_results[1].status is LatencyQuantileStatus.OPEN_TAIL
    assert result.quantile_results[1].lower_edge == 20
    assert result.quantile_results[1].upper_edge is None
    assert result.mean_result.status is LatencyMeanStatus.OPEN_TAIL
    assert result.atu_latency_substitute is False
    assert "ATU-latency" in LATENCY_INTERPRETATION


def test_zero_observations_are_distinct_from_missing_bins():
    definition = two_bin_definition(LatencyBinEncoding.DISJOINT)
    zero_series = make_series(
        LatencyHistogramGroup.REQUEST_RESPONSE,
        0,
        [0, NANOSECOND, 2 * NANOSECOND],
        [0, 0, 0],
    )

    result = analyze_latency_histograms(
        (zero_series,), {definition.group: definition}
    )[0]

    assert result.status is LatencyStatus.PARTIAL_MISSING_DATA
    assert result.bin_results[0].count == 0
    assert result.bin_results[1].count is None
    assert result.total_observations is None
    assert result.normalized_distribution_available is False
    assert result.mean_result.status is LatencyMeanStatus.UNSUPPORTED


def test_explicitly_missing_supplied_bins_are_missing_not_insufficient():
    definition = two_bin_definition(LatencyBinEncoding.DISJOINT)
    series = (
        make_series(
            LatencyHistogramGroup.REQUEST_RESPONSE,
            0,
            [0, NANOSECOND],
            [0, 7],
            quality_flags=(SeriesQualityFlag.MISSING,),
        ),
        make_series(
            LatencyHistogramGroup.REQUEST_RESPONSE,
            1,
            [0, NANOSECOND],
            [0, 4],
            quality_flags=(SeriesQualityFlag.MISSING,),
        ),
    )

    result = analyze_latency_histograms(series, {definition.group: definition})[0]

    assert result.status is LatencyStatus.MISSING_DATA
    assert LatencyQualityFlag.MISSING_BIN in result.quality_flags
    assert all(bin_result.count is None for bin_result in result.bin_results)
    assert result.total_observations is None
    assert result.normalized_distribution_available is False


def test_reset_interval_is_excluded_from_counts_and_coverage():
    definition = two_bin_definition(LatencyBinEncoding.DISJOINT)
    timestamps = [0, NANOSECOND, 2 * NANOSECOND, 3 * NANOSECOND]
    series = (
        make_series(
            LatencyHistogramGroup.REQUEST_RESPONSE,
            0,
            timestamps,
            [0, 5, 2, 7],
        ),
        make_series(
            LatencyHistogramGroup.REQUEST_RESPONSE,
            1,
            timestamps,
            [0, 3, 3, 6],
        ),
    )

    result = analyze_latency_histograms(series, {definition.group: definition})[0]

    assert result.status is LatencyStatus.PARTIAL_INVALID_DATA
    assert [bin_result.count for bin_result in result.bin_results] == [10, 6]
    assert result.total_observations == 16
    assert result.valid_interval_count == 2
    assert result.invalid_interval_count == 1
    assert result.reset_excluded_interval_count == 1
    assert result.covered_duration_ns == 2 * NANOSECOND
    assert LatencyQualityFlag.DECREASE_OR_RESET in result.quality_flags


def test_incompatible_boundary_encoding_is_rejected():
    definition = two_bin_definition(LatencyBinEncoding.BOUNDARY_CUMULATIVE)
    series = (
        make_series(
            LatencyHistogramGroup.REQUEST_RESPONSE, 0, [0, NANOSECOND], [0, 10]
        ),
        make_series(
            LatencyHistogramGroup.REQUEST_RESPONSE, 1, [0, NANOSECOND], [0, 5]
        ),
    )

    result = analyze_latency_histograms(series, {definition.group: definition})[0]

    assert result.status is LatencyStatus.UNSUPPORTED_ENCODING
    assert result.valid_interval_count == 0
    assert all(bin_result.count is None for bin_result in result.bin_results)
    assert result.total_observations is None
    assert LatencyQualityFlag.INCOMPATIBLE_ENCODING in result.quality_flags


def test_exact_boundary_quantile_selects_lower_bin():
    definition = two_bin_definition(LatencyBinEncoding.DISJOINT)
    timestamps = [0, NANOSECOND]
    series = (
        make_series(
            LatencyHistogramGroup.REQUEST_RESPONSE, 0, timestamps, [0, 5]
        ),
        make_series(
            LatencyHistogramGroup.REQUEST_RESPONSE, 1, timestamps, [0, 5]
        ),
    )

    result = analyze_latency_histograms(
        series, {definition.group: definition}, quantiles=(0.5,)
    )[0]

    assert result.quantile_results[0].bin_name == "bin-0"
    assert result.quantile_results[0].lower_edge == 0
    assert result.quantile_results[0].upper_edge == 10


def test_unknown_encoding_cannot_claim_sourced_edges():
    with pytest.raises(ValueError, match="unknown encoding"):
        two_bin_definition(LatencyBinEncoding.UNKNOWN)


def test_latency_population_is_not_directional():
    series = make_series(
        LatencyHistogramGroup.REQUEST_RESPONSE, 0, [0, 1], [0, 1]
    )

    result = analyze_latency_histograms(
        (series,),
        {LatencyHistogramGroup.REQUEST_RESPONSE: two_bin_definition(
            LatencyBinEncoding.DISJOINT
        )},
    )[0]

    assert result.population.direction == "not_directional"
