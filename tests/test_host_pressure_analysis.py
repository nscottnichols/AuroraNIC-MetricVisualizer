from host_pressure_analysis import (
    BlockedCounterUnit,
    BlockedUnitResolution,
    HostPressureCounterKind,
    HostPressureDataset,
    HostPressureDenominatorState,
    HostPressurePath,
    HostPressurePopulation,
    HostPressureQualityFlag,
    HostPressureSeries,
    HostPressureSeriesIdentity,
    HostPressureSeriesSpec,
    HostPressureStatus,
    HostPressureValueKind,
    analyze_host_pressure,
)


SECOND = 1_000_000_000
POPULATION = HostPressurePopulation(
    run_id="run-1",
    node_id="node-1",
    nic_id="nic-1",
    collector_id="collector-1",
)
BLOCKED_UNITS = BlockedUnitResolution(
    unit=BlockedCounterUnit.EVENTS,
    evidence="authoritative hardware documentation fixture",
)


def spec(counter_id, kind, unit, path=None):
    return HostPressureSeriesSpec(
        counter_id=counter_id,
        counter_kind=kind,
        unit=unit,
        path=path,
    )


def series(series_spec, timestamps, values, **overrides):
    arguments = {
        "spec": series_spec,
        "identity": HostPressureSeriesIdentity(
            run_id=POPULATION.run_id,
            node_id=POPULATION.node_id,
            nic_id=POPULATION.nic_id,
            collector_id=POPULATION.collector_id,
            source_counter_id=series_spec.counter_id,
        ),
        "timestamps_ns": tuple(timestamps),
        "values": tuple(values),
        "value_kind": HostPressureValueKind.CUMULATIVE,
        "semantics": "cumulative counter",
        "baseline_relative": False,
    }
    arguments.update(overrides)
    return HostPressureSeries(**arguments)


def dataset(series_items, blocked_units=BLOCKED_UNITS, window=(0, 2 * SECOND)):
    return HostPressureDataset(
        population=POPULATION,
        series=tuple(series_items),
        window_ns=window,
        blocked_unit_resolution=blocked_units,
    )


def pair(
    packet_values,
    blocked_values,
    packet_unit="packets",
    blocked_unit="events",
    path=HostPressurePath.POSTED,
):
    packet_id, blocked_id = {
        HostPressurePath.POSTED: (
            "parbs_tarb_pi_posted_pkts",
            "parbs_tarb_pi_posted_blocked_cnt",
        ),
        HostPressurePath.NONPOSTED: (
            "parbs_tarb_pi_non_posted_pkts",
            "parbs_tarb_pi_non_posted_blocked_cnt",
        ),
    }[path]
    return (
        series(
            spec(
                packet_id,
                HostPressureCounterKind.PACKET,
                packet_unit,
                path,
            ),
            (0, SECOND, 2 * SECOND),
            packet_values,
        ),
        series(
            spec(
                blocked_id,
                HostPressureCounterKind.BLOCKED,
                blocked_unit,
                path,
            ),
            (0, SECOND, 2 * SECOND),
            blocked_values,
        ),
    )


def test_zero_observations_are_not_missing_and_have_zero_rates():
    result = analyze_host_pressure(
        dataset(
            [
                *pair((0, 0, 0), (0, 0, 0), path=HostPressurePath.POSTED),
                *pair((0, 0, 0), (0, 0, 0), path=HostPressurePath.NONPOSTED),
            ]
        )
    )
    posted = result.path_results[0]
    nonposted = result.path_results[1]

    assert posted.status is HostPressureStatus.OBSERVED_ZERO
    assert posted.packet_result.zero_observed is True
    assert posted.blocked_result.zero_observed is True
    assert posted.packet_total == 0
    assert posted.blocked_total == 0
    assert posted.packet_rate_per_second == 0.0
    assert posted.blocked_rate_per_second == 0.0
    assert posted.blocked_per_packet is None
    assert posted.denominator_state is HostPressureDenominatorState.ZERO_PACKETS
    assert posted.blocked_per_packet_unit is None
    assert posted.coverage_fraction == 1.0
    assert nonposted.status is HostPressureStatus.OBSERVED_ZERO
    assert nonposted.packet_total == 0
    assert result.status is HostPressureStatus.OBSERVED_ZERO


def test_requested_window_sample_count_is_window_local():
    timestamps = (0, SECOND, 2 * SECOND, 3 * SECOND)
    packet = series(
        spec(
            "posted_packets",
            HostPressureCounterKind.PACKET,
            "packets",
            HostPressurePath.POSTED,
        ),
        timestamps,
        (0, 10, 20, 30),
    )
    blocked = series(
        spec(
            "posted_blocked",
            HostPressureCounterKind.BLOCKED,
            "events",
            HostPressurePath.POSTED,
        ),
        timestamps,
        (0, 2, 4, 6),
    )
    result = analyze_host_pressure(
        dataset(
            [packet, blocked],
            window=(500_000_000, 2_500_000_000),
        )
    )
    posted = result.path_results[0]

    assert posted.packet_result.sample_count == 2
    assert posted.blocked_result.sample_count == 2


def test_nonzero_deltas_rates_and_ratio_are_hand_checkable():
    result = analyze_host_pressure(dataset(pair((10, 22, 37), (0, 3, 11))))
    posted = result.path_results[0]

    assert [interval.delta for interval in posted.packet_result.intervals] == [12, 15]
    assert [interval.delta for interval in posted.blocked_result.intervals] == [3, 8]
    assert posted.packet_total == 27
    assert posted.blocked_total == 11
    assert posted.common_valid_interval_count == 2
    assert posted.covered_duration_ns == 2 * SECOND
    assert posted.packet_rate_per_second == 13.5
    assert posted.blocked_rate_per_second == 5.5
    assert posted.blocked_per_packet == 11 / 27
    assert posted.denominator_state is HostPressureDenominatorState.AVAILABLE
    assert posted.status is HostPressureStatus.OBSERVED_NONZERO


def test_reset_interval_is_excluded_from_totals_and_coverage():
    result = analyze_host_pressure(
        dataset(pair((0, 10, 3), (0, 2, 4)))
    )
    posted = result.path_results[0]

    assert [interval.delta for interval in posted.packet_result.intervals] == [10, -7]
    assert [interval.valid for interval in posted.packet_result.intervals] == [True, False]
    assert posted.packet_total == 10
    assert posted.blocked_total == 2
    assert posted.covered_duration_ns == SECOND
    assert posted.coverage_fraction == 0.5
    assert posted.common_valid_interval_count == 1
    assert posted.reset_excluded_interval_count == 1
    assert HostPressureQualityFlag.DECREASE_OR_RESET in posted.quality_flags
    assert posted.status is HostPressureStatus.PARTIAL_INVALID_DATA
    assert posted.blocked_per_packet == 2 / 10


def test_missing_packet_denominator_is_distinct_from_zero():
    blocked = series(
        spec(
            "posted_blocked",
            HostPressureCounterKind.BLOCKED,
            "events",
            HostPressurePath.POSTED,
        ),
        (0, SECOND, 2 * SECOND),
        (0, 2, 5),
    )
    result = analyze_host_pressure(dataset([blocked]))
    posted = result.path_results[0]

    assert posted.packet_result.total is None
    assert posted.packet_result.zero_observed is False
    assert posted.packet_total is None
    assert posted.blocked_total is None
    assert posted.blocked_per_packet is None
    assert posted.denominator_state is HostPressureDenominatorState.MISSING_PACKET
    assert HostPressureQualityFlag.MISSING_PACKET in posted.quality_flags
    assert posted.status is HostPressureStatus.PARTIAL_MISSING_DATA
    assert "parbs_tarb_pi_posted_pkts" in result.missing_counter_ids
    assert "parbs_tarb_pi_non_posted_pkts" in result.missing_counter_ids


def test_incompatible_units_do_not_produce_a_ratio():
    result = analyze_host_pressure(
        dataset(
            pair(
                (0, 10, 30),
                (0, 2, 5),
                packet_unit="messages",
                blocked_unit="cycles",
            )
        )
    )
    posted = result.path_results[0]

    assert posted.status is HostPressureStatus.UNSUPPORTED_UNITS
    assert posted.denominator_state is HostPressureDenominatorState.INCOMPATIBLE_UNITS
    assert HostPressureQualityFlag.INCOMPATIBLE_UNITS in posted.quality_flags
    assert posted.blocked_per_packet is None
    assert result.status is HostPressureStatus.UNSUPPORTED_UNITS


def test_unresolved_blocked_units_remain_unsupported_until_resolved():
    result = analyze_host_pressure(
        dataset(
            pair((0, 10, 30), (0, 2, 5), blocked_unit="unresolved"),
            blocked_units=None,
        )
    )
    posted = result.path_results[0]

    assert result.blocked_unit == "unresolved"
    assert posted.status is HostPressureStatus.UNSUPPORTED_UNITS
    assert posted.denominator_state is HostPressureDenominatorState.UNRESOLVED_BLOCKED_UNITS
    assert HostPressureQualityFlag.UNRESOLVED_BLOCKED_UNITS in posted.quality_flags
    assert posted.blocked_per_packet is None
    assert posted.blocked_per_packet_unit is None
    assert result.status is HostPressureStatus.UNSUPPORTED_UNITS
