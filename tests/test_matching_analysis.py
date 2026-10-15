from matching_analysis import (
    MatchingCounterIdentity,
    MatchingCounterSeries,
    MatchingCounterSpec,
    MatchingDataset,
    MatchingPopulation,
    MatchingQualityFlag,
    MatchingRole,
    MatchingStatus,
    ValueKind,
    analyze_matching,
)


SECOND = 1_000_000_000
POPULATION = MatchingPopulation(
    run_id="run-1",
    node_id="node-1",
    nic_id="nic-1",
    collector_id="collector-1",
)


def spec(role, priority=0, unit="events", population_id="receive-matches"):
    return MatchingCounterSpec(
        counter_id=f"lpe_net_match_{role.value}_{priority}",
        role=role,
        priority=priority,
        unit=unit,
        population_id=population_id,
    )


def series(counter_spec, timestamps, values, **overrides):
    arguments = {
        "spec": counter_spec,
        "identity": MatchingCounterIdentity(
            run_id=POPULATION.run_id,
            node_id=POPULATION.node_id,
            nic_id=POPULATION.nic_id,
            collector_id=POPULATION.collector_id,
            source_counter_id=counter_spec.counter_id,
        ),
        "timestamps_ns": tuple(timestamps),
        "values": tuple(values),
        "value_kind": ValueKind.CUMULATIVE,
        "semantics": "cumulative counter",
        "baseline_relative": False,
    }
    arguments.update(overrides)
    return MatchingCounterSeries(**arguments)


def analyze(specs, observations, window=(0, 3 * SECOND)):
    return analyze_matching(
        MatchingDataset(
            population=POPULATION,
            requested_counters=tuple(specs),
            series=tuple(observations),
            window_ns=window,
        )
    )


def test_exact_nonzero_deltas_totals_fractions_and_temporal_changes():
    timestamps = (0, SECOND, 2 * SECOND, 3 * SECOND)
    priority_0 = spec(MatchingRole.PRIORITY, 0)
    priority_1 = spec(MatchingRole.PRIORITY, 1)
    overflow_0 = spec(MatchingRole.OVERFLOW, 0)
    overflow_1 = spec(MatchingRole.OVERFLOW, 1)
    result = analyze(
        [priority_0, priority_1, overflow_0, overflow_1],
        [
            series(priority_0, timestamps, (0, 2, 2, 5)),
            series(priority_1, timestamps, (10, 10, 12, 12)),
            series(overflow_0, timestamps, (0, 1, 4, 4)),
            series(overflow_1, timestamps, (7, 7, 7, 8)),
        ],
    )

    assert [interval.delta for interval in result.counter_results[0].intervals] == [2, 0, 3]
    assert [interval.delta for interval in result.counter_results[1].intervals] == [0, 2, 0]
    assert [interval.delta for interval in result.counter_results[2].intervals] == [1, 3, 0]
    assert [interval.delta for interval in result.counter_results[3].intervals] == [0, 0, 1]
    assert result.priority_total == 7
    assert result.overflow_total == 5
    assert result.total == 12
    assert result.priority_fraction == 7 / 12
    assert result.overflow_fraction == 5 / 12
    assert [(item.priority_delta, item.overflow_delta) for item in result.temporal_intervals] == [
        (2, 1),
        (2, 3),
        (3, 1),
    ]
    assert [item.priority_fraction for item in result.temporal_intervals] == [
        2 / 3,
        2 / 5,
        3 / 4,
    ]
    assert result.status is MatchingStatus.OBSERVED_NONZERO
    assert result.population_compatible is True
    assert result.covered_duration_ns == 3 * SECOND
    assert result.coverage_fraction == 1.0
    assert "receive-path evidence" in result.interpretation


def test_requested_window_sample_count_is_window_local():
    priority = spec(MatchingRole.PRIORITY)
    overflow = spec(MatchingRole.OVERFLOW)
    timestamps = (0, SECOND, 2 * SECOND, 3 * SECOND)
    result = analyze(
        [priority, overflow],
        [
            series(priority, timestamps, (0, 2, 2, 5)),
            series(overflow, timestamps, (0, 1, 4, 4)),
        ],
        window=(500_000_000, 2_500_000_000),
    )

    assert result.counter_results[0].sample_count == 2
    assert result.counter_results[1].sample_count == 2


def test_zero_counts_are_observed_but_fraction_is_undefined():
    priority = spec(MatchingRole.PRIORITY)
    overflow = spec(MatchingRole.OVERFLOW)
    result = analyze(
        [priority, overflow],
        [
            series(priority, (0, SECOND, 2 * SECOND), (0, 0, 0)),
            series(overflow, (0, SECOND, 2 * SECOND), (0, 0, 0)),
        ],
    )

    assert result.priority_total == 0
    assert result.overflow_total == 0
    assert result.total == 0
    assert result.priority_fraction is None
    assert result.overflow_fraction is None
    assert result.status is MatchingStatus.OBSERVED_ZERO
    assert result.counter_results[0].zero_observed is True
    assert result.counter_results[0].persistent_zero is True
    assert result.counter_results[0].status is MatchingStatus.OBSERVED_ZERO
    assert "denominator is zero" in result.status_reasons[0]


def test_reset_interval_is_excluded_and_disables_combined_fraction():
    priority = spec(MatchingRole.PRIORITY)
    overflow = spec(MatchingRole.OVERFLOW)
    result = analyze(
        [priority, overflow],
        [
            series(priority, (0, SECOND, 2 * SECOND), (0, 5, 2)),
            series(overflow, (0, SECOND, 2 * SECOND), (0, 1, 1)),
        ],
    )

    priority_result = result.counter_results[0]
    assert [interval.delta for interval in priority_result.intervals] == [5, -3]
    assert [interval.valid for interval in priority_result.intervals] == [True, False]
    assert priority_result.total == 5
    assert priority_result.covered_duration_ns == SECOND
    assert priority_result.status is MatchingStatus.PARTIAL_INVALID_DATA
    assert MatchingQualityFlag.DECREASE_OR_RESET in priority_result.quality_flags
    assert result.priority_total is None
    assert result.overflow_total is None
    assert result.total is None
    assert result.priority_fraction is None
    assert result.temporal_intervals == ()
    assert result.status is MatchingStatus.PARTIAL_INVALID_DATA


def test_missing_priority_is_not_zero_and_prevents_fraction():
    priority = spec(MatchingRole.PRIORITY)
    overflow = spec(MatchingRole.OVERFLOW)
    result = analyze(
        [priority, overflow],
        [series(overflow, (0, SECOND, 2 * SECOND), (0, 0, 1))],
    )

    assert result.missing_counter_ids == (priority.counter_id,)
    assert result.counter_results[0].status is MatchingStatus.MISSING_DATA
    assert result.counter_results[0].total is None
    assert result.counter_results[0].zero_observed is False
    assert result.counter_results[1].total == 1
    assert result.priority_total is None
    assert result.overflow_total is None
    assert result.priority_fraction is None
    assert result.status is MatchingStatus.PARTIAL_MISSING_DATA


def test_missing_overflow_is_not_zero_and_prevents_fraction():
    priority = spec(MatchingRole.PRIORITY)
    overflow = spec(MatchingRole.OVERFLOW)
    result = analyze(
        [priority, overflow],
        [series(priority, (0, SECOND, 2 * SECOND), (0, 0, 3))],
    )

    assert result.missing_counter_ids == (overflow.counter_id,)
    assert result.counter_results[0].total == 3
    assert result.counter_results[1].total is None
    assert result.counter_results[1].zero_observed is False
    assert result.priority_total is None
    assert result.overflow_total is None
    assert result.total is None
    assert result.priority_fraction is None
    assert result.status is MatchingStatus.PARTIAL_MISSING_DATA


def test_incompatible_units_do_not_sum_or_form_a_fraction():
    priority = spec(MatchingRole.PRIORITY, unit="events")
    overflow = spec(MatchingRole.OVERFLOW, unit="packets")
    result = analyze(
        [priority, overflow],
        [
            series(priority, (0, SECOND, 2 * SECOND), (0, 0, 3)),
            series(overflow, (0, SECOND, 2 * SECOND), (0, 0, 3)),
        ],
    )

    assert result.counter_results[0].total == 3
    assert result.counter_results[1].total == 3
    assert result.priority_total is None
    assert result.overflow_total is None
    assert result.total is None
    assert result.priority_fraction is None
    assert MatchingQualityFlag.INCOMPATIBLE_UNIT in result.quality_flags
    assert result.status is MatchingStatus.UNSUPPORTED


def test_population_id_and_exact_interval_boundaries_are_required():
    priority = spec(MatchingRole.PRIORITY, population_id="priority-only")
    overflow = spec(MatchingRole.OVERFLOW, population_id="overflow-only")
    result = analyze(
        [priority, overflow],
        [
            series(priority, (0, SECOND, 2 * SECOND), (0, 0, 4)),
            series(overflow, (SECOND, 2 * SECOND, 3 * SECOND), (0, 1, 3)),
        ],
    )

    assert MatchingQualityFlag.INCOMPATIBLE_POPULATION in result.quality_flags
    assert MatchingQualityFlag.INCOMPATIBLE_COVERAGE in result.quality_flags
    assert result.priority_total is None
    assert result.overflow_total is None
    assert result.status is MatchingStatus.UNSUPPORTED


def test_baseline_relative_change_only_series_is_unsupported():
    priority = spec(MatchingRole.PRIORITY)
    overflow = spec(MatchingRole.OVERFLOW)
    result = analyze(
        [priority, overflow],
        [
            series(
                priority,
                (0, SECOND, 2 * SECOND),
                (0, 5, 7),
                baseline_relative=True,
            ),
            series(overflow, (0, SECOND, 2 * SECOND), (0, 1, 3)),
        ],
    )

    assert result.counter_results[0].status is MatchingStatus.UNSUPPORTED
    assert MatchingQualityFlag.BASELINE_RELATIVE in result.quality_flags
    assert result.counter_results[0].total is None
    assert result.status is MatchingStatus.PARTIAL_UNSUPPORTED_DATA


def test_boundary_selection_uses_only_complete_intervals_inside_window():
    priority = spec(MatchingRole.PRIORITY)
    overflow = spec(MatchingRole.OVERFLOW)
    result = analyze(
        [priority, overflow],
        [
            series(priority, (0, SECOND, 2 * SECOND, 3 * SECOND), (0, 10, 20, 30)),
            series(overflow, (0, SECOND, 2 * SECOND, 3 * SECOND), (0, 5, 5, 5)),
        ],
        window=(500_000_000, 2 * SECOND),
    )

    priority_result = result.counter_results[0]
    assert [(interval.start_ns, interval.end_ns) for interval in priority_result.intervals] == [
        (SECOND, 2 * SECOND)
    ]
    assert [interval.delta for interval in priority_result.intervals] == [10]
    assert priority_result.total == 10
    assert result.overflow_total == 0
    assert result.total == 10
    assert result.priority_fraction == 1.0
    assert result.covered_duration_ns == SECOND
    assert result.requested_duration_ns == 1_500_000_000
    assert result.coverage_fraction == 2 / 3
    assert MatchingQualityFlag.BOUNDARY_INTERVAL_EXCLUDED in result.quality_flags
    assert MatchingQualityFlag.PARTIAL_WINDOW in result.quality_flags
