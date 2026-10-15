import pytest
import numpy as np

from flow_control_analysis import (
    FlowControlMetricKind,
    FlowControlStatus,
    SeriesQualityFlag,
    TrafficDirection,
    VerifiedPauseCycleClock,
    analyze_flow_control,
    pause_cycle_seconds,
)
from traffic_analysis import MetricSeries, SeriesIdentity, ValueKind


def make_series(
    counter_name,
    timestamps,
    values,
    *,
    direction=TrafficDirection.TX,
    semantics="cycles",
    unit="cycles",
    value_dtype="uint64",
    value_kind=ValueKind.CUMULATIVE,
    baseline_relative=False,
    quality_flags=(),
):
    timestamps_arg = (
        timestamps if isinstance(timestamps, np.ndarray) else tuple(timestamps)
    )
    values_arg = values if isinstance(values, np.ndarray) else tuple(values)
    return MetricSeries(
        identity=SeriesIdentity(
            run_id="run-1",
            node_id="node-1",
            nic_id="nic-1",
            collector_id="collector-1",
            counter_id=counter_name,
            direction=direction,
        ),
        timestamps_ns=timestamps_arg,
        values=values_arg,
        value_dtype=value_dtype,
        value_kind=value_kind,
        semantics=semantics,
        unit=unit,
        baseline_relative=baseline_relative,
        quality_flags=tuple(quality_flags),
    )


def result_for(result, counter_name):
    return next(
        series_result
        for series_result in result.series_results
        if series_result.counter_name == counter_name
    )


def test_regular_sampling_and_mixed_directions_remain_separate():
    tx = make_series(
        "hni_tx_paused_0",
        [0, 1_000_000_000, 2_000_000_000],
        [0, 10, 30],
    )
    rx = make_series(
        "hni_rx_paused_0",
        [0, 1_000_000_000, 2_000_000_000],
        [0, 5, 5],
        direction=TrafficDirection.RX,
    )
    xoff = make_series(
        "hni_pause_xoff_sent_0",
        [0, 1_000_000_000, 2_000_000_000],
        [0, 2, 3],
        semantics="events",
        unit="events",
    )
    fgfc_discard = make_series(
        "hni_fgfc_discard",
        [0, 1_000_000_000, 2_000_000_000],
        [0, 2, 2],
        direction=None,
        semantics="events",
        unit="events",
    )
    fgfc_xoff = make_series(
        "hni_fgfc_event_xoff",
        [0, 1_000_000_000, 2_000_000_000],
        [0, 3, 5],
        direction=None,
        semantics="events",
        unit="events",
    )

    result = analyze_flow_control(
        [tx, rx, xoff, fgfc_discard, fgfc_xoff],
        expected_counter_names=(
            "hni_tx_paused_0",
            "hni_rx_paused_0",
            "hni_pause_xoff_sent_0",
            "hni_fgfc_discard",
            "hni_fgfc_event_xoff",
        ),
        expected_pause_cycle_priorities=(0,),
        expected_pause_xoff_sent_priorities=(0,),
        expected_pause_received_priorities=(0,),
    )
    tx_result = result_for(result, "hni_tx_paused_0")
    rx_result = result_for(result, "hni_rx_paused_0")
    xoff_result = result_for(result, "hni_pause_xoff_sent_0")
    fgfc_discard_result = result_for(result, "hni_fgfc_discard")
    fgfc_xoff_result = result_for(result, "hni_fgfc_event_xoff")

    assert result.status is FlowControlStatus.OK
    assert tx_result.total == 30
    assert tx_result.mean_rate_per_second == pytest.approx(15.0)
    assert [interval.rate_per_second for interval in tx_result.intervals] == [
        10.0,
        20.0,
    ]
    assert tx_result.first_nonzero_ns == 1_000_000_000
    assert tx_result.first_nonzero_delta == 10
    assert rx_result.total == 5
    assert rx_result.mean_rate_per_second == pytest.approx(2.5)
    assert rx_result.first_nonzero_ns == 1_000_000_000
    assert xoff_result.metric_kind is FlowControlMetricKind.PAUSE_XOFF_SENT
    assert xoff_result.total == 3
    assert xoff_result.mean_rate_per_second == pytest.approx(1.5)
    assert fgfc_discard_result.total == 2
    assert fgfc_discard_result.mean_rate_per_second == pytest.approx(1.0)
    assert fgfc_xoff_result.total == 5
    assert fgfc_xoff_result.mean_rate_per_second == pytest.approx(2.5)
    assert len(result.pause_cycle_results) == 2
    assert len(result.xoff_event_results) == 2
    assert len(result.fgfc_event_results) == 2
    assert result.coverage_fraction == 1.0


def test_irregular_sampling_uses_measured_intervals():
    series = make_series(
        "hni_tx_paused_1",
        [0, 1_000_000_000, 4_000_000_000],
        [0, 100, 700],
    )

    result = analyze_flow_control(
        [series], expected_pause_cycle_priorities=(1,)
    )
    series_result = result_for(result, "hni_tx_paused_1")

    assert series_result.total == 700
    assert series_result.covered_duration_ns == 4_000_000_000
    assert series_result.mean_rate_per_second == pytest.approx(175.0)
    assert [interval.rate_per_second for interval in series_result.intervals] == [
        100.0,
        200.0,
    ]
    assert series_result.first_nonzero_ns == 1_000_000_000
    assert series_result.first_nonzero_delta == 100


def test_zero_observations_are_distinct_from_missing_priority():
    zero = make_series(
        "hni_tx_paused_0",
        [0, 1_000_000_000, 2_000_000_000],
        [0, 0, 0],
    )
    priority_two = make_series(
        "hni_tx_paused_2",
        [0, 1_000_000_000, 2_000_000_000],
        [0, 4, 6],
    )

    result = analyze_flow_control(
        [zero, priority_two],
        expected_counter_names=(
            "hni_tx_paused_0",
            "hni_tx_paused_1",
            "hni_tx_paused_2",
        ),
        expected_pause_cycle_priorities=(0, 1, 2),
    )
    zero_result = result_for(result, "hni_tx_paused_0")

    assert result.status is FlowControlStatus.PARTIAL_MISSING_DATA
    assert result.missing_counter_names == ("hni_tx_paused_1",)
    assert zero_result.status is FlowControlStatus.ZERO_OBSERVATIONS
    assert zero_result.total == 0
    assert zero_result.mean_rate_per_second == 0.0
    assert zero_result.first_nonzero_ns is None
    assert result.zero_observed_counter_names == ("hni_tx_paused_0",)


def test_reset_interval_is_excluded_without_bridging():
    series = make_series(
        "hni_rx_paused_3",
        [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
        [0, 100, 20, 80],
        direction=TrafficDirection.RX,
    )

    result = analyze_flow_control(
        [series], expected_pause_cycle_priorities=(3,)
    )
    series_result = result_for(result, "hni_rx_paused_3")

    assert series_result.status is FlowControlStatus.PARTIAL_INVALID_DATA
    assert series_result.total == 160
    assert series_result.valid_interval_count == 2
    assert series_result.invalid_interval_count == 1
    assert series_result.covered_duration_ns == 2_000_000_000
    assert series_result.coverage_fraction == pytest.approx(2.0 / 3.0)
    assert series_result.first_nonzero_ns == 1_000_000_000
    assert series_result.first_nonzero_delta == 100
    assert SeriesQualityFlag.DECREASE_OR_RESET in series_result.quality_flags


def test_direct_uint64_array_decrease_is_reset_not_wraparound():
    series = make_series(
        "hni_tx_paused_0",
        np.asarray([0, 1_000_000_000], dtype=np.int64),
        np.asarray([2, 1], dtype=np.uint64),
    )

    result = analyze_flow_control(
        [series], expected_pause_cycle_priorities=(0,)
    )
    series_result = result_for(result, "hni_tx_paused_0")

    assert series_result.status is FlowControlStatus.INVALID_DATA
    assert series_result.total is None
    assert series_result.invalid_interval_count == 1
    assert series_result.intervals[0].state.value == "decrease_or_reset"
    assert SeriesQualityFlag.DECREASE_OR_RESET in series_result.quality_flags


def test_baseline_relative_values_use_observed_changes():
    series = make_series(
        "hni_tx_paused_0",
        [0, 1_000_000_000, 2_000_000_000],
        [0.0, 10.0, 30.0],
        value_dtype="float64",
        baseline_relative=True,
    )

    result = analyze_flow_control(
        [series],
        expected_counter_names=("hni_tx_paused_0",),
        expected_pause_cycle_priorities=(0,),
    )
    series_result = result_for(result, "hni_tx_paused_0")

    assert series_result.total == pytest.approx(30.0)
    assert series_result.first_nonzero_ns == 1_000_000_000
    assert series_result.first_nonzero_delta == pytest.approx(10.0)
    assert SeriesQualityFlag.BASELINE_RELATIVE in series_result.quality_flags


def test_missing_input_series_is_not_zero():
    series = make_series(
        "hni_tx_paused_0",
        [],
        [],
        quality_flags=(SeriesQualityFlag.MISSING,),
    )

    result = analyze_flow_control(
        [series],
        expected_counter_names=("hni_tx_paused_0",),
        expected_pause_cycle_priorities=(0,),
    )
    series_result = result_for(result, "hni_tx_paused_0")

    assert series_result.status is FlowControlStatus.MISSING_DATA
    assert result.status is FlowControlStatus.MISSING_DATA
    assert series_result.total is None
    assert series_result.mean_rate_per_second is None
    assert series_result.first_nonzero_ns is None
    assert result.missing_counter_names == ()


def test_unsupported_semantics_do_not_become_numerical_evidence():
    series = make_series(
        "hni_fgfc_discard",
        [0, 1_000_000_000],
        [0, 5],
        semantics="nanoseconds",
        unit="nanoseconds",
    )

    result = analyze_flow_control([series])
    series_result = result_for(result, "hni_fgfc_discard")

    assert series_result.status is FlowControlStatus.UNSUPPORTED_SERIES
    assert series_result.total is None
    assert result.unsupported_counter_names == ("hni_fgfc_discard",)


def test_unknown_counter_is_unsupported_without_numerical_results():
    series = make_series(
        "hni_not_a_flow_control_counter",
        [0, 1_000_000_000],
        [0, 99],
    )

    result = analyze_flow_control(
        [series],
        expected_counter_names=(),
    )
    series_result = result_for(result, "hni_not_a_flow_control_counter")

    assert result.status is FlowControlStatus.UNSUPPORTED_SERIES
    assert series_result.total is None
    assert result.unsupported_counter_names == (
        "hni_not_a_flow_control_counter",
    )


def test_time_conversion_requires_verified_pause_cycle_clock():
    cycle = make_series(
        "hni_tx_paused_0",
        [0, 1_000_000_000],
        [0, 20],
    )
    event = make_series(
        "hni_pause_sent",
        [0, 1_000_000_000],
        [0, 7],
        semantics="events",
        unit="events",
    )
    result = analyze_flow_control(
        [cycle, event],
        expected_pause_cycle_priorities=(0,),
    )
    cycle_result = result_for(result, "hni_tx_paused_0")
    event_result = result_for(result, "hni_pause_sent")
    clock = VerifiedPauseCycleClock(
        frequency_hz=2,
        verification_evidence="fixture-verified clock metadata",
        applies_to_counter_names=("hni_tx_paused_0",),
    )

    with pytest.raises(ValueError, match="verified clock frequency"):
        pause_cycle_seconds(cycle_result)
    with pytest.raises(ValueError, match="Only pause-cycle counters"):
        pause_cycle_seconds(event_result, clock)
    with pytest.raises(ValueError, match="does not apply"):
        pause_cycle_seconds(
            result_for(
                analyze_flow_control(
                    [make_series("hni_rx_paused_0", [0, 1], [0, 6])],
                    expected_pause_cycle_priorities=(0,),
                ),
                "hni_rx_paused_0",
            ),
            clock,
        )

    assert pause_cycle_seconds(cycle_result, clock) == pytest.approx(10.0)
    assert not hasattr(result, "pause_seconds")
    assert not hasattr(result, "wall_clock_pause_fraction")
    assert "never summed into wall-clock pause time" in " ".join(result.limitations)
