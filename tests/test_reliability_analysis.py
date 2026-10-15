from reliability_analysis import (
    CounterIdentity,
    CounterRole,
    ReliabilityCounterSeries,
    ReliabilityCounterSpec,
    ReliabilityDataset,
    ReliabilityPopulation,
    ReliabilityQualityFlag,
    ReliabilityStatus,
    ValueKind,
    analyze_reliability,
)


SECOND = 1_000_000_000
POPULATION = ReliabilityPopulation(
    run_id="run-1",
    node_id="node-1",
    nic_id="nic-1",
    collector_id="collector-1",
)


def counter_spec(counter_id, role=CounterRole.ERROR, unit="events", aggregation_group=None):
    return ReliabilityCounterSpec(
        counter_id=counter_id,
        role=role,
        unit=unit,
        aggregation_group=aggregation_group,
    )


def counter_series(spec, timestamps, values, **overrides):
    arguments = {
        "spec": spec,
        "identity": CounterIdentity(
            run_id=POPULATION.run_id,
            node_id=POPULATION.node_id,
            nic_id=POPULATION.nic_id,
            collector_id=POPULATION.collector_id,
            source_counter_id=spec.counter_id,
        ),
        "timestamps_ns": tuple(timestamps),
        "values": tuple(values),
        "value_kind": ValueKind.CUMULATIVE,
        "semantics": "cumulative counter",
        "baseline_relative": False,
    }
    arguments.update(overrides)
    return ReliabilityCounterSeries(**arguments)


def analyze(specs, series, window=(0, 2 * SECOND)):
    return analyze_reliability(
        ReliabilityDataset(
            population=POPULATION,
            requested_counters=tuple(specs),
            series=tuple(series),
            window_ns=window,
        )
    )


def test_exact_nonzero_deltas_rate_and_first_nonzero():
    spec = counter_spec("pct_sct_timeouts", aggregation_group="errors")
    result = analyze([spec], [counter_series(spec, (0, SECOND, 2 * SECOND), (0, 2, 5))])

    counter = result.counter_results[0]
    assert [interval.delta for interval in counter.intervals] == [2, 3]
    assert counter.total == 5
    assert counter.covered_duration_ns == 2 * SECOND
    assert counter.coverage_fraction == 1.0
    assert counter.rate_per_second == 2.5
    assert counter.first_nonzero_sample_ns == SECOND
    assert counter.first_nonzero_increment_ns == SECOND
    assert counter.status is ReliabilityStatus.OBSERVED_NONZERO
    assert result.error_total == 5
    assert result.aggregate_error_rate_per_second == 2.5


def test_all_zero_is_an_observed_zero_not_missing():
    spec = counter_spec("hni_pcs_uncorrected_cw", aggregation_group="errors")
    result = analyze([spec], [counter_series(spec, (0, SECOND, 2 * SECOND), (0, 0, 0))])
    counter = result.counter_results[0]

    assert counter.total == 0
    assert counter.rate_per_second == 0.0
    assert counter.zero_observed is True
    assert counter.persistent_zero is True
    assert counter.status is ReliabilityStatus.OBSERVED_ZERO
    assert result.status is ReliabilityStatus.OBSERVED_ZERO
    assert result.error_total == 0
    assert result.persistent_zero is True


def test_requested_window_sample_count_is_window_local():
    spec = counter_spec("hni_pcs_uncorrected_cw")
    result = analyze(
        [spec],
        [counter_series(spec, (0, SECOND, 2 * SECOND, 3 * SECOND), (0, 0, 0, 0))],
        window=(500_000_000, 2_500_000_000),
    )

    assert result.counter_results[0].sample_count == 2


def test_decrease_or_reset_is_excluded_from_total_and_denominator():
    spec = counter_spec("hni_llr_rx_replay_event", aggregation_group="errors")
    result = analyze([spec], [counter_series(spec, (0, SECOND, 2 * SECOND), (0, 3, 1))])
    counter = result.counter_results[0]

    assert [interval.delta for interval in counter.intervals] == [3, -2]
    assert [interval.valid for interval in counter.intervals] == [True, False]
    assert counter.total == 3
    assert counter.covered_duration_ns == SECOND
    assert counter.coverage_fraction == 0.5
    assert counter.rate_per_second == 3.0
    assert counter.invalid_interval_count == 1
    assert counter.status is ReliabilityStatus.PARTIAL_INVALID_DATA
    assert ReliabilityQualityFlag.DECREASE_OR_RESET in counter.quality_flags


def test_missing_counter_is_distinct_from_zero():
    present = counter_spec("pct_sct_timeouts", aggregation_group="errors")
    absent = counter_spec("pct_spt_timeouts", aggregation_group="errors")
    result = analyze(
        [present, absent],
        [counter_series(present, (0, SECOND, 2 * SECOND), (0, 0, 0))],
    )
    missing = result.counter_results[1]

    assert missing.total is None
    assert missing.rate_per_second is None
    assert missing.zero_observed is False
    assert missing.persistent_zero is False
    assert missing.status is ReliabilityStatus.MISSING_DATA
    assert result.status is ReliabilityStatus.PARTIAL_MISSING_DATA
    assert result.error_total is None
    assert result.aggregate_error_rate_per_second is None


def test_good_codeword_counter_is_not_an_error_counter():
    error = counter_spec("pct_sct_timeouts", aggregation_group="errors")
    good = counter_spec(
        "hni_pcs_good_cw",
        role=CounterRole.GOOD,
        unit="codewords",
        aggregation_group="good_codewords",
    )
    result = analyze(
        [error, good],
        [
            counter_series(error, (0, SECOND, 2 * SECOND), (0, 0, 0)),
            counter_series(good, (0, SECOND, 2 * SECOND), (0, 7, 7)),
        ],
    )

    assert result.error_total == 0
    assert result.good_total == 7
    assert result.aggregate_error_rate_per_second == 0.0
    assert result.counter_results[1].status is ReliabilityStatus.OBSERVED_NONZERO
    assert result.status is ReliabilityStatus.OBSERVED_ZERO


def test_mislabeled_hni_pcs_good_cw_is_unsupported_as_error():
    spec = counter_spec(
        "hni_pcs_good_cw", role=CounterRole.ERROR, aggregation_group="errors"
    )
    result = analyze([spec], [counter_series(spec, (0, SECOND), (0, 7))])

    assert result.counter_results[0].status is ReliabilityStatus.UNSUPPORTED
    assert ReliabilityQualityFlag.KNOWN_GOOD_COUNTER_MISLABELED in result.quality_flags
    assert result.error_total is None
    assert result.status is ReliabilityStatus.UNSUPPORTED


def test_aggregate_rate_uses_common_measured_duration():
    first = counter_spec("pct_sct_timeouts", aggregation_group="errors")
    second = counter_spec("pct_spt_timeouts", aggregation_group="errors")
    result = analyze(
        [first, second],
        [
            counter_series(first, (0, SECOND, 2 * SECOND), (0, 3, 5)),
            counter_series(second, (0, SECOND, 2 * SECOND), (0, 4, 4)),
        ],
    )

    assert result.error_total == 9
    assert result.covered_duration_ns == 2 * SECOND
    assert result.aggregate_error_rate_per_second == 4.5
    assert result.aggregate_rate_unit == "events/s"


def test_incompatible_coverage_does_not_sum_interval_durations():
    first = counter_spec("pct_sct_timeouts", aggregation_group="errors")
    second = counter_spec("pct_spt_timeouts", aggregation_group="errors")
    result = analyze(
        [first, second],
        [
            counter_series(first, (0, SECOND), (0, 3)),
            counter_series(second, (SECOND, 3 * SECOND), (0, 4)),
        ],
        window=(0, 3 * SECOND),
    )

    assert result.counter_results[0].rate_per_second == 3.0
    assert result.counter_results[1].rate_per_second == 2.0
    assert result.error_total is None
    assert result.aggregate_error_rate_per_second is None
    assert ReliabilityQualityFlag.INCOMPATIBLE_COVERAGE in result.quality_flags


def test_unsupported_semantics_returns_no_total_or_rate():
    spec = counter_spec("hni_discard_cntr_0", aggregation_group="errors")
    result = analyze(
        [spec],
        [counter_series(spec, (0, SECOND), (0, 3), value_kind=ValueKind.DELTA)],
    )
    counter = result.counter_results[0]

    assert counter.status is ReliabilityStatus.UNSUPPORTED
    assert counter.total is None
    assert counter.rate_per_second is None
    assert result.status is ReliabilityStatus.UNSUPPORTED


def test_window_contains_only_complete_valid_intervals():
    spec = counter_spec("ixe_rx_pkt_drop_pct", aggregation_group="errors")
    result = analyze(
        [spec],
        [counter_series(spec, (0, SECOND, 2 * SECOND, 3 * SECOND), (0, 2, 5, 9))],
        window=(500_000_000, 2 * SECOND),
    )
    counter = result.counter_results[0]

    assert [interval.delta for interval in counter.intervals] == [3]
    assert counter.total == 3
    assert counter.covered_duration_ns == SECOND
    assert counter.coverage_fraction == 2 / 3
    assert counter.rate_per_second == 3.0
    assert counter.outside_window_interval_count == 2
    assert counter.first_nonzero_sample_ns == SECOND
    assert counter.first_nonzero_increment_ns == 2 * SECOND
