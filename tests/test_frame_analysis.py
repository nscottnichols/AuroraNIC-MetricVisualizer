from frame_analysis import (
    FrameBinEdge,
    FrameBinEdgeSet,
    FrameBytePopulationCompatibility,
    FrameCounterIdentity,
    FrameCounterKind,
    FrameCounterSeries,
    FrameCounterSpec,
    FrameDataset,
    FrameDirection,
    FrameMeanStatus,
    FramePopulation,
    FrameQualityFlag,
    FrameQuantileStatus,
    FrameStatus,
    FrameValueKind,
    analyze_frames,
    byte_count_spec,
    frame_bin_spec,
)


SECOND = 1_000_000_000
POPULATION = FramePopulation(
    run_id="run-1",
    node_id="node-1",
    nic_id="nic-1",
    collector_id="collector-1",
)


def series(spec, timestamps, values, population_id=None, baseline_relative=False):
    return FrameCounterSeries(
        spec=spec,
        identity=FrameCounterIdentity(
            run_id=POPULATION.run_id,
            node_id=POPULATION.node_id,
            nic_id=POPULATION.nic_id,
            collector_id=POPULATION.collector_id,
            source_counter_id=spec.counter_id,
        ),
        timestamps_ns=tuple(timestamps),
        values=tuple(values),
        value_kind=FrameValueKind.CUMULATIVE,
        semantics="cumulative counter",
        baseline_relative=baseline_relative,
    )


def bin_spec(counter_id, direction, bin_id, population_id="wire-population"):
    return frame_bin_spec(counter_id, direction, bin_id, population_id)


def edges(direction, entries, source="fixture sourced histogram edges"):
    return FrameBinEdgeSet(
        direction=direction,
        source=source,
        edges=tuple(
            FrameBinEdge(
                bin_id=bin_id,
                lower_bytes=lower_bytes,
                upper_bytes=upper_bytes,
                open_ended=open_ended,
            )
            for bin_id, lower_bytes, upper_bytes, open_ended in entries
        ),
    )




def analyze(
    specs,
    observations,
    edge_sets=(),
    compatibility=(),
    window=(0, 2 * SECOND),
):
    return analyze_frames(
        FrameDataset(
            population=POPULATION,
            requested_counters=tuple(specs),
            series=tuple(observations),
            bin_edges=tuple(edge_sets),
            byte_count_compatibility=tuple(compatibility),
            window_ns=window,
        )
    )


def direction_result(result, direction):
    return next(item for item in result.direction_results if item.direction is direction)


def bin_result(direction, bin_id):
    return next(item for item in direction.bin_results if item.spec.bin_id == bin_id)


def test_regular_sampling_direction_shares_and_exact_byte_count_mean():
    tx_small = bin_spec("tx_small", FrameDirection.TX, "small")
    tx_large = bin_spec("tx_large", FrameDirection.TX, "large")
    rx_small = bin_spec("rx_small", FrameDirection.RX, "small")
    rx_large = bin_spec("rx_large", FrameDirection.RX, "large")
    tx_bytes = byte_count_spec("tx_bytes", FrameDirection.TX, "wire-population")
    timestamps = (0, SECOND, 2 * SECOND)

    result = analyze(
        [tx_small, tx_large, rx_small, rx_large, tx_bytes],
        [
            series(tx_small, timestamps, (0, 3, 7)),
            series(tx_large, timestamps, (0, 2, 5)),
            series(rx_small, timestamps, (0, 1, 1)),
            series(rx_large, timestamps, (0, 1, 4)),
            series(tx_bytes, timestamps, (0, 5000, 14000)),
        ],
        edge_sets=[
            edges(FrameDirection.TX, [("small", 0, 1024, False), ("large", 1024, None, True)]),
            edges(FrameDirection.RX, [("small", 0, 1024, False), ("large", 1024, 1518, False)]),
        ],
        compatibility=[
            FrameBytePopulationCompatibility(
                direction=FrameDirection.TX,
                byte_counter_id="tx_bytes",
                frame_population_id="wire-population",
                byte_population_id="wire-population",
                compatible=True,
                definition_evidence="fixture: both counters count the same TX wire population",
            )
        ],
    )

    tx = direction_result(result, FrameDirection.TX)
    rx = direction_result(result, FrameDirection.RX)

    assert [interval.delta for interval in bin_result(tx, "small").intervals] == [3, 4]
    assert [interval.delta for interval in bin_result(tx, "large").intervals] == [2, 3]
    assert tx.bin_total == 12
    assert rx.bin_total == 5
    assert bin_result(tx, "small").bin_share == 7 / 12
    assert bin_result(tx, "large").bin_share == 5 / 12
    assert tx.mean_status is FrameMeanStatus.SUPPORTED
    assert tx.mean_wire_frame_size_bytes == 14000 / 12
    assert tx.mean_coverage_fraction == 1.0
    assert rx.mean_status is FrameMeanStatus.UNSUPPORTED
    assert result.status is FrameStatus.OBSERVED_NONZERO


def test_irregular_sampling_uses_exact_interval_deltas_and_coverage():
    small = bin_spec("tx_small", FrameDirection.TX, "small")
    large = bin_spec("tx_large", FrameDirection.TX, "large")
    timestamps = (0, 400_000_000, 1_700_000_000)

    result = analyze(
        [small, large],
        [
            series(small, timestamps, (0, 2, 5)),
            series(large, timestamps, (0, 1, 3)),
        ],
        edge_sets=[
            edges(
                FrameDirection.TX,
                [("small", 0, 1024, False), ("large", 1024, 1518, False)],
            )
        ],
        window=(0, 2 * SECOND),
    )
    tx = direction_result(result, FrameDirection.TX)

    assert [interval.duration_ns for interval in tx.temporal_intervals] == [
        400_000_000,
        1_300_000_000,
    ]
    assert [interval.total_frames for interval in tx.temporal_intervals] == [3, 5]
    assert tx.temporal_intervals[0].bin_deltas == (("small", 2), ("large", 1))
    assert tx.temporal_intervals[1].bin_deltas == (("small", 3), ("large", 2))
    assert tx.bin_total == 8
    assert tx.covered_duration_ns == 1_700_000_000
    assert tx.coverage_fraction == 0.85
    assert bin_result(tx, "small").rate_per_second == 5 / 1.7


def test_requested_window_sample_count_is_window_local():
    small = bin_spec("tx_small", FrameDirection.TX, "small")
    timestamps = (0, SECOND, 2 * SECOND, 3 * SECOND)
    result = analyze(
        [small],
        [series(small, timestamps, (0, 2, 5, 9))],
        edge_sets=[
            edges(FrameDirection.TX, [("small", 0, 1024, False)])
        ],
        window=(500_000_000, 2_500_000_000),
    )
    tx = direction_result(result, FrameDirection.TX)

    assert bin_result(tx, "small").sample_count == 2


def test_observed_zero_is_distinct_from_a_missing_bin():
    small = bin_spec("tx_small", FrameDirection.TX, "small")
    absent = bin_spec("tx_absent", FrameDirection.TX, "absent")

    result = analyze(
        [small, absent],
        [series(small, (0, SECOND, 2 * SECOND), (0, 0, 0))],
        edge_sets=[
            edges(
                FrameDirection.TX,
                [("small", 0, 1024, False), ("absent", 1024, 1518, False)],
            )
        ],
    )
    tx = direction_result(result, FrameDirection.TX)
    zero = bin_result(tx, "small")
    missing = bin_result(tx, "absent")

    assert zero.total == 0
    assert zero.zero_observed is True
    assert zero.persistent_zero is True
    assert zero.status is FrameStatus.OBSERVED_ZERO
    assert missing.total is None
    assert missing.zero_observed is False
    assert missing.status is FrameStatus.MISSING_DATA
    assert tx.status is FrameStatus.PARTIAL_MISSING_DATA
    assert tx.bin_total is None


def test_baseline_relative_series_cannot_produce_numeric_totals():
    small = bin_spec("tx_small", FrameDirection.TX, "small")
    large = bin_spec("tx_large", FrameDirection.TX, "large")
    tx_bytes = byte_count_spec("tx_bytes", FrameDirection.TX, "wire-population")
    timestamps = (0, SECOND, 2 * SECOND)

    result = analyze(
        [small, large, tx_bytes],
        [
            series(
                small,
                timestamps,
                (0, 3, 7),
                baseline_relative=True,
            ),
            series(large, timestamps, (0, 1, 3)),
            series(tx_bytes, timestamps, (0, 5000, 14000), baseline_relative=True),
        ],
        edge_sets=[
            edges(
                FrameDirection.TX,
                [("small", 0, 1024, False), ("large", 1024, 1518, False)],
            )
        ],
        compatibility=[
            FrameBytePopulationCompatibility(
                direction=FrameDirection.TX,
                byte_counter_id="tx_bytes",
                frame_population_id="wire-population",
                byte_population_id="wire-population",
                compatible=True,
                definition_evidence="fixture: same TX wire population",
            )
        ],
    )
    tx = direction_result(result, FrameDirection.TX)

    assert result.status is FrameStatus.PARTIAL_UNSUPPORTED_DATA
    assert bin_result(tx, "small").status is FrameStatus.UNSUPPORTED
    assert bin_result(tx, "small").total is None
    assert tx.bin_total is None
    assert tx.byte_total is None
    assert tx.mean_wire_frame_size_bytes is None
    assert FrameQualityFlag.BASELINE_RELATIVE in tx.quality_flags
    assert any(
        "Baseline-relative counters cannot support numeric frame totals."
        in reason
        for reason in result.status_reasons
    )


def test_reset_interval_is_excluded_from_total_coverage_and_shares():
    small = bin_spec("tx_small", FrameDirection.TX, "small")
    large = bin_spec("tx_large", FrameDirection.TX, "large")
    timestamps = (0, 400_000_000, SECOND)

    result = analyze(
        [small, large],
        [
            series(small, timestamps, (0, 5, 3)),
            series(large, timestamps, (0, 2, 4)),
        ],
        edge_sets=[
            edges(
                FrameDirection.TX,
                [("small", 0, 1024, False), ("large", 1024, 1518, False)],
            )
        ],
        window=(0, SECOND),
    )
    tx = direction_result(result, FrameDirection.TX)
    reset_bin = bin_result(tx, "small")

    assert [interval.delta for interval in reset_bin.intervals] == [5, -2]
    assert [interval.valid for interval in reset_bin.intervals] == [True, False]
    assert reset_bin.total == 5
    assert reset_bin.covered_duration_ns == 400_000_000
    assert reset_bin.coverage_fraction == 0.4
    assert reset_bin.status is FrameStatus.PARTIAL_INVALID_DATA
    assert FrameQualityFlag.DECREASE_OR_RESET in reset_bin.quality_flags
    assert tx.bin_total is None
    assert tx.temporal_intervals == ()
    assert all(
        quantile.status is FrameQuantileStatus.UNSUPPORTED
        for quantile in tx.quantile_results
    )


def test_open_ended_tail_blocks_bin_mean_and_upper_quantile_bound():
    small = bin_spec("tx_small", FrameDirection.TX, "small")
    large = bin_spec("tx_large", FrameDirection.TX, "large")
    tx_bytes = byte_count_spec("tx_bytes", FrameDirection.TX, "wire-population")
    timestamps = (0, SECOND, 2 * SECOND)

    result = analyze(
        [small, large, tx_bytes],
        [
            series(small, timestamps, (0, 3, 7)),
            series(large, timestamps, (0, 1, 3)),
            series(tx_bytes, timestamps, (0, 4000, 13000)),
        ],
        edge_sets=[
            edges(
                FrameDirection.TX,
                [("small", 0, 1024, False), ("large", 1024, None, True)],
            )
        ],
        compatibility=[
            FrameBytePopulationCompatibility(
                direction=FrameDirection.TX,
                byte_counter_id="tx_bytes",
                frame_population_id="wire-population",
                byte_population_id="wire-population",
                compatible=True,
                definition_evidence="fixture: exact same TX wire population",
            )
        ],
    )
    tx = direction_result(result, FrameDirection.TX)

    assert tx.open_ended_tail is True
    assert FrameQualityFlag.OPEN_ENDED_BIN in tx.quality_flags
    assert tx.approximate_bin_midpoint_mean_wire_frame_size_bytes is None
    assert tx.mean_wire_frame_size_bytes == 13000 / 10
    assert tx.quantile_results[0].status is FrameQuantileStatus.BOUNDED
    assert tx.quantile_results[0].lower_bytes == 0
    assert tx.quantile_results[0].upper_bytes == 1024
    assert tx.quantile_results[1].status is FrameQuantileStatus.OPEN_ENDED
    assert tx.quantile_results[1].lower_bytes == 1024
    assert tx.quantile_results[1].upper_bytes is None


def test_incompatible_byte_and_frame_populations_do_not_produce_a_mean():
    small = bin_spec("tx_small", FrameDirection.TX, "small")
    large = bin_spec("tx_large", FrameDirection.TX, "large")
    tx_bytes = byte_count_spec(
        "tx_bytes", FrameDirection.TX, "different-byte-population"
    )
    timestamps = (0, SECOND, 2 * SECOND)

    result = analyze(
        [small, large, tx_bytes],
        [
            series(small, timestamps, (0, 2, 5)),
            series(large, timestamps, (0, 1, 3)),
            series(tx_bytes, timestamps, (0, 1000, 5000)),
        ],
        edge_sets=[
            edges(
                FrameDirection.TX,
                [("small", 0, 1024, False), ("large", 1024, 1518, False)],
            )
        ],
        compatibility=[
            FrameBytePopulationCompatibility(
                direction=FrameDirection.TX,
                byte_counter_id="tx_bytes",
                frame_population_id="wire-population",
                byte_population_id="different-byte-population",
                compatible=False,
                definition_evidence="fixture: byte counter includes framing overhead",
            )
        ],
    )
    tx = direction_result(result, FrameDirection.TX)

    assert tx.byte_total == 5000
    assert tx.bin_total == 8
    assert tx.mean_status is FrameMeanStatus.UNSUPPORTED
    assert tx.mean_wire_frame_size_bytes is None
    assert (
        FrameQualityFlag.INCOMPATIBLE_BYTE_COUNT_POPULATION in tx.quality_flags
    )


def test_compatible_populations_still_require_identical_valid_coverage():
    small = bin_spec("tx_small", FrameDirection.TX, "small")
    large = bin_spec("tx_large", FrameDirection.TX, "large")
    tx_bytes = byte_count_spec("tx_bytes", FrameDirection.TX, "wire-population")

    result = analyze(
        [small, large, tx_bytes],
        [
            series(small, (0, SECOND, 2 * SECOND), (0, 2, 5)),
            series(large, (0, SECOND, 2 * SECOND), (0, 1, 3)),
            series(tx_bytes, (0, 500_000_000, 2 * SECOND), (0, 1000, 5000)),
        ],
        edge_sets=[
            edges(
                FrameDirection.TX,
                [("small", 0, 1024, False), ("large", 1024, 1518, False)],
            )
        ],
        compatibility=[
            FrameBytePopulationCompatibility(
                direction=FrameDirection.TX,
                byte_counter_id="tx_bytes",
                frame_population_id="wire-population",
                byte_population_id="wire-population",
                compatible=True,
                definition_evidence="fixture: same TX wire population",
            )
        ],
    )
    tx = direction_result(result, FrameDirection.TX)

    assert tx.bin_total == 8
    assert tx.byte_total == 5000
    assert tx.mean_status is FrameMeanStatus.UNSUPPORTED
    assert tx.mean_wire_frame_size_bytes is None
    assert FrameQualityFlag.INCOMPATIBLE_COVERAGE in tx.quality_flags


def test_absent_edges_return_named_bin_counts_but_no_mean_or_quantiles():
    small = bin_spec("tx_small", FrameDirection.TX, "small")
    large = bin_spec("tx_large", FrameDirection.TX, "large")
    tx_bytes = byte_count_spec("tx_bytes", FrameDirection.TX, "wire-population")
    timestamps = (0, SECOND, 2 * SECOND)

    result = analyze(
        [small, large, tx_bytes],
        [
            series(small, timestamps, (0, 2, 5)),
            series(large, timestamps, (0, 1, 3)),
            series(tx_bytes, timestamps, (0, 1000, 5000)),
        ],
        compatibility=[
            FrameBytePopulationCompatibility(
                direction=FrameDirection.TX,
                byte_counter_id="tx_bytes",
                frame_population_id="wire-population",
                byte_population_id="wire-population",
                compatible=True,
                definition_evidence="fixture: same TX wire population",
            )
        ],
    )
    tx = direction_result(result, FrameDirection.TX)

    assert bin_result(tx, "small").total == 5
    assert bin_result(tx, "large").total == 3
    assert bin_result(tx, "small").bin_share == 5 / 8
    assert bin_result(tx, "large").bin_share == 3 / 8
    assert tx.bin_edges_source is None
    assert tx.bin_edges_complete is False
    assert FrameQualityFlag.MISSING_BIN_EDGES in tx.quality_flags
    assert tx.mean_status is FrameMeanStatus.UNSUPPORTED
    assert tx.mean_wire_frame_size_bytes is None
    assert all(
        quantile.status is FrameQuantileStatus.UNSUPPORTED
        for quantile in tx.quantile_results
    )
