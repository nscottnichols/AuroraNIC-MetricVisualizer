import math

import pytest

from src.phase_analysis import (
    ManualPhaseAnalysisResult,
    PhaseAnalysisStatus,
    PhaseComparisonStatus,
    PhaseGapPolicy,
    PhaseState,
    PhaseTrafficRecord,
    PhaseWindow,
    analyze_manual_phase_windows,
)
from traffic_analysis import (
    AnalysisStatus,
    MetricSeries,
    SeriesIdentity,
    TrafficDirection,
    ValueKind,
    analyze_traffic_series,
)


def make_record(
    values,
    *,
    timestamps=None,
    window=(0, 3_000_000_000),
    clock_id="clock-1",
    run_id="run-1",
    node_id="node-1",
    nic_id="nic-1",
    collector_id="collector-1",
    direction=TrafficDirection.TX,
    counter_id="CXI_TX_BYTES",
):
    if timestamps is None:
        timestamps = list(
            range(0, (len(values)) * 1_000_000_000, 1_000_000_000)
        )
    identity = SeriesIdentity(
        run_id=run_id,
        node_id=node_id,
        nic_id=nic_id,
        collector_id=collector_id,
        counter_id=counter_id,
        direction=direction,
    )
    series = MetricSeries(
        identity=identity,
        timestamps_ns=tuple(timestamps),
        values=tuple(values),
        value_dtype="uint64",
        value_kind=ValueKind.CUMULATIVE,
        semantics="bytes",
        unit="bytes",
        baseline_relative=False,
    )
    analysis = analyze_traffic_series(
        series,
        window_start_ns=window[0],
        window_end_ns=window[1],
    )
    return PhaseTrafficRecord(clock_id=clock_id, analysis=analysis)


def phases_for_result(result: ManualPhaseAnalysisResult):
    return {phase.phase_name: phase for phase in result.phases}


def test_regular_manual_phases_have_hand_checkable_totals_and_changes():
    record = make_record([0, 100, 300, 600])
    phases = (
        PhaseWindow("warmup", 0, 1_000_000_000),
        PhaseWindow("steady", 1_000_000_000, 3_000_000_000),
    )

    result = analyze_manual_phase_windows(
        phases, {"warmup": record, "steady": record}
    )
    by_name = phases_for_result(result)

    assert result.status is PhaseAnalysisStatus.OK
    assert by_name["warmup"].total_bytes == 100
    assert by_name["steady"].total_bytes == 500
    assert by_name["warmup"].mean_rate_bytes_per_s == pytest.approx(100.0)
    assert by_name["steady"].mean_rate_bytes_per_s == pytest.approx(250.0)
    assert by_name["warmup"].coverage_fraction == 1.0
    assert by_name["steady"].coverage_fraction == 1.0
    assert result.coverage_fraction == 1.0

    change = result.changes[0]
    assert change.total_change_bytes == 400
    assert change.total_change_fraction == pytest.approx(4.0)
    assert change.total_change_percent == pytest.approx(400.0)
    assert change.rate_change_bytes_per_s == pytest.approx(150.0)
    assert change.rate_ratio == pytest.approx(2.5)
    assert change.comparison_status is PhaseComparisonStatus.COMPARABLE
    assert change.is_comparable is True
    assert change.uncertainty_reasons == ()
    assert "do not infer workload labels" in result.interpretation_limits


def test_irregular_phase_boundaries_exclude_straddling_interval():
    record = make_record([0, 100, 200, 500])
    phases = (
        PhaseWindow("first", 0, 1_500_000_000),
        PhaseWindow("second", 1_500_000_000, 3_000_000_000),
    )

    result = analyze_manual_phase_windows(
        phases, {"first": record, "second": record}
    )
    by_name = phases_for_result(result)

    assert by_name["first"].total_bytes == 100
    assert by_name["second"].total_bytes == 300
    assert by_name["first"].covered_duration_ns == 1_000_000_000
    assert by_name["second"].covered_duration_ns == 1_000_000_000
    assert by_name["first"].coverage_fraction == pytest.approx(2 / 3)
    assert by_name["second"].coverage_fraction == pytest.approx(2 / 3)
    assert by_name["first"].boundary_excluded_interval_count == 1
    assert by_name["second"].boundary_excluded_interval_count == 1
    assert by_name["first"].mean_rate_bytes_per_s == pytest.approx(100.0)
    assert by_name["second"].mean_rate_bytes_per_s == pytest.approx(300.0)

    change = result.changes[0]
    assert change.total_change_bytes == 200
    assert change.rate_change_bytes_per_s == pytest.approx(200.0)
    assert change.comparison_status is PhaseComparisonStatus.UNCERTAIN
    assert change.is_comparable is False
    assert "incomplete coverage" in " ".join(change.uncertainty_reasons)


def test_overlapping_phase_windows_are_rejected():
    record = make_record([0, 100, 200])
    phases = (
        PhaseWindow("first", 0, 2_000_000_000),
        PhaseWindow("second", 1_000_000_000, 3_000_000_000),
    )

    with pytest.raises(ValueError, match="overlap"):
        analyze_manual_phase_windows(
            phases, {"first": record, "second": record}
        )


def test_gap_policy_rejects_by_default_and_can_preserve_gap():
    record = make_record([0, 100, 300, 600])
    phases = (
        PhaseWindow("before", 0, 1_000_000_000),
        PhaseWindow("after", 2_000_000_000, 3_000_000_000),
    )
    records = {"before": record, "after": record}

    with pytest.raises(ValueError, match="1000000000 ns gap"):
        analyze_manual_phase_windows(phases, records)

    result = analyze_manual_phase_windows(
        phases, records, gap_policy=PhaseGapPolicy.ALLOW_GAPS
    )
    change = result.changes[0]

    assert result.gap_policy is PhaseGapPolicy.ALLOW_GAPS
    assert result.inter_phase_gap_duration_ns == 1_000_000_000
    assert PhaseState.INTER_PHASE_GAP in result.state_flags
    assert change.total_change_bytes == 200
    assert change.comparison_status is PhaseComparisonStatus.NOT_ADJACENT
    assert change.is_comparable is False
    assert "unassigned time" in " ".join(change.uncertainty_reasons)


def test_zero_traffic_phase_change_is_explicit_not_missing():
    record = make_record([0, 0, 0, 0])
    phases = (
        PhaseWindow("first", 0, 1_000_000_000),
        PhaseWindow("second", 1_000_000_000, 3_000_000_000),
    )

    result = analyze_manual_phase_windows(
        phases, {"first": record, "second": record}
    )
    by_name = phases_for_result(result)
    change = result.changes[0]

    assert result.status is PhaseAnalysisStatus.ZERO_TRAFFIC
    assert by_name["first"].status is AnalysisStatus.ZERO_TRAFFIC
    assert by_name["second"].status is AnalysisStatus.ZERO_TRAFFIC
    assert by_name["first"].total_bytes == 0
    assert by_name["second"].total_bytes == 0
    assert change.total_change_bytes == 0
    assert change.rate_change_bytes_per_s == 0.0
    assert change.rate_ratio is None
    assert change.comparison_status is PhaseComparisonStatus.COMPARABLE
    assert "zero traffic" in " ".join(change.uncertainty_reasons)


def test_missing_phase_data_is_not_treated_as_zero():
    record = make_record([0, 100, 300, 600])
    phases = (
        PhaseWindow("observed", 0, 2_000_000_000),
        PhaseWindow("missing", 2_000_000_000, 3_000_000_000),
    )

    result = analyze_manual_phase_windows(
        phases, {"observed": record}
    )
    by_name = phases_for_result(result)
    change = result.changes[0]

    assert result.status is PhaseAnalysisStatus.PARTIAL_DATA
    assert result.missing_phase_names == ("missing",)
    assert by_name["missing"].status is AnalysisStatus.MISSING_DATA
    assert by_name["missing"].total_bytes is None
    assert by_name["missing"].coverage_fraction == 0.0
    assert change.total_change_bytes is None
    assert change.comparison_status is PhaseComparisonStatus.MISSING
    assert PhaseState.MISSING_PHASE_DATA in by_name["missing"].state_flags


def test_reset_in_phase_is_excluded_with_partial_coverage():
    record = make_record([0, 100, 20, 120])
    phases = (
        PhaseWindow("before", 0, 1_000_000_000),
        PhaseWindow("after", 1_000_000_000, 3_000_000_000),
    )

    result = analyze_manual_phase_windows(
        phases, {"before": record, "after": record}
    )
    by_name = phases_for_result(result)
    change = result.changes[0]

    assert by_name["before"].total_bytes == 100
    assert by_name["after"].total_bytes == 100
    assert by_name["after"].status is AnalysisStatus.PARTIAL_INVALID_DATA
    assert by_name["after"].invalid_interval_count == 1
    assert by_name["after"].covered_duration_ns == 1_000_000_000
    assert by_name["after"].coverage_fraction == 0.5
    assert PhaseState.INVALID_DATA in by_name["after"].state_flags
    assert change.total_change_bytes == 0
    assert change.comparison_status is PhaseComparisonStatus.UNCERTAIN
    assert change.is_comparable is False


def test_clock_mismatch_is_rejected_before_numerical_analysis():
    first = make_record([0, 100, 200, 300])
    second = make_record(
        [0, 100, 200, 300],
        clock_id="clock-2",
    )
    phases = (
        PhaseWindow("first", 0, 1_000_000_000),
        PhaseWindow("second", 1_000_000_000, 3_000_000_000),
    )

    with pytest.raises(ValueError, match="does not share common clock"):
        analyze_manual_phase_windows(
            phases, {"first": first, "second": second}
        )


def test_common_window_mismatch_is_rejected():
    first = make_record(
        [0, 100, 200, 300],
        window=(0, 2_000_000_000),
    )
    second = make_record(
        [0, 100, 200, 300],
        window=(0, 3_000_000_000),
    )
    phases = (
        PhaseWindow("first", 0, 1_000_000_000),
        PhaseWindow("second", 1_000_000_000, 3_000_000_000),
    )

    with pytest.raises(ValueError, match="does not share common window"):
        analyze_manual_phase_windows(
            phases, {"first": first, "second": second}
        )


def test_run_direction_and_counter_mismatches_are_rejected():
    base = make_record([0, 100, 200, 300])
    mismatched_run = make_record(
        [0, 100, 200, 300],
        run_id="run-2",
    )
    mismatched_direction = make_record(
        [0, 100, 200, 300],
        direction=TrafficDirection.RX,
    )
    mismatched_counter = make_record(
        [0, 100, 200, 300],
        counter_id="CXI_RX_BYTES",
    )
    phases = (
        PhaseWindow("first", 0, 1_000_000_000),
        PhaseWindow("second", 1_000_000_000, 3_000_000_000),
    )

    for second in (mismatched_run, mismatched_direction, mismatched_counter):
        with pytest.raises(ValueError, match="does not share common"):
            analyze_manual_phase_windows(
                phases, {"first": base, "second": second}
            )
