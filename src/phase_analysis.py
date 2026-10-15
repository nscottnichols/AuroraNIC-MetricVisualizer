"""Manual phase-window analysis over detailed traffic results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import math

from traffic_analysis import (
    AnalysisStatus,
    IntervalState,
    NANOSECONDS_PER_SECOND,
    SeriesQualityFlag,
    TrafficAnalysis,
    TrafficDirection,
    ValueKind,
)


TOTAL_FORMULA = (
    "sum(delta_bytes for valid intervals with both endpoints inside the phase)"
)
MEAN_RATE_FORMULA = "phase_total_bytes / covered_duration_seconds"
COVERAGE_FORMULA = "covered_duration_ns / phase_duration_ns"
TOTAL_CHANGE_FORMULA = "current_phase_total_bytes - previous_phase_total_bytes"
RATE_CHANGE_FORMULA = (
    "current_phase_mean_rate_bytes_per_s - "
    "previous_phase_mean_rate_bytes_per_s"
)
INTERPRETATION_LIMITS = (
    "Phase names and boundaries are user-provided. Changes describe observed "
    "counter differences only; they do not infer workload labels, phase causes, "
    "or causal relationships."
)


class PhaseGapPolicy(str, Enum):
    REJECT_GAPS = "reject_gaps"
    ALLOW_GAPS = "allow_gaps"


class PhaseState(str, Enum):
    ZERO_TRAFFIC = "zero_traffic"
    MISSING_PHASE_DATA = "missing_phase_data"
    PARTIAL_COVERAGE = "partial_coverage"
    INVALID_DATA = "invalid_data"
    UNSUPPORTED_DATA = "unsupported_data"
    BASELINE_RELATIVE = "baseline_relative"
    BOUNDARY_INTERVAL_EXCLUDED = "boundary_interval_excluded"
    INTER_PHASE_GAP = "inter_phase_gap"


class PhaseAnalysisStatus(str, Enum):
    OK = "ok"
    ZERO_TRAFFIC = "zero_traffic"
    PARTIAL_DATA = "partial_data"
    MISSING_DATA = "missing_data"
    UNSUPPORTED_DATA = "unsupported_data"


class PhaseComparisonStatus(str, Enum):
    COMPARABLE = "comparable"
    UNCERTAIN = "uncertain"
    NOT_ADJACENT = "not_adjacent"
    MISSING = "missing"


@dataclass(frozen=True)
class PhaseWindow:
    """A user-provided, named interval on the common trace clock."""

    name: str
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class PhaseTrafficRecord:
    """A detailed traffic result and the clock it was measured on."""

    clock_id: str
    analysis: TrafficAnalysis


@dataclass(frozen=True)
class PhaseMetric:
    """Numerical results and limitations for one user-defined phase."""

    phase_name: str
    window_ns: tuple[int, int]
    status: AnalysisStatus
    state_flags: frozenset[PhaseState]
    status_reasons: tuple[str, ...]
    quality_flags: frozenset[SeriesQualityFlag]
    source_quality_flags: frozenset[SeriesQualityFlag]
    source_counter: str
    total_bytes: int | float | None
    covered_duration_ns: int
    requested_duration_ns: int
    uncovered_duration_ns: int
    coverage_fraction: float | None
    mean_rate_bytes_per_s: float | None
    valid_interval_count: int
    invalid_interval_count: int
    boundary_excluded_interval_count: int
    represented_window_ns: tuple[int, int] | None
    baseline_relative: bool


@dataclass(frozen=True)
class PhaseChange:
    """Observed change between two adjacent user-defined phases."""

    from_phase_name: str
    to_phase_name: str
    from_window_ns: tuple[int, int]
    to_window_ns: tuple[int, int]
    inter_phase_gap_duration_ns: int
    total_change_bytes: int | float | None
    total_change_fraction: float | None
    total_change_percent: float | None
    rate_change_bytes_per_s: float | None
    rate_ratio: float | None
    comparison_status: PhaseComparisonStatus
    is_comparable: bool
    uncertainty_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ManualPhaseAnalysisResult:
    """Explicit, non-causal comparison of ordered manual phase windows."""

    run_id: str
    node_id: str
    nic_id: str
    collector_id: str
    direction: TrafficDirection
    source_counter: str
    unit: str
    rate_unit: str
    semantics: str
    value_kind: ValueKind
    clock_id: str
    common_window_ns: tuple[int, int]
    gap_policy: PhaseGapPolicy
    status: PhaseAnalysisStatus
    state_flags: frozenset[PhaseState]
    status_reasons: tuple[str, ...]
    phases: tuple[PhaseMetric, ...]
    changes: tuple[PhaseChange, ...]
    phase_count: int
    missing_phase_count: int
    covered_duration_ns: int
    requested_duration_ns: int
    uncovered_duration_ns: int
    coverage_fraction: float | None
    valid_interval_count: int
    invalid_interval_count: int
    boundary_excluded_interval_count: int
    inter_phase_gap_duration_ns: int
    missing_phase_names: tuple[str, ...]
    partial_phase_names: tuple[str, ...]
    zero_phase_names: tuple[str, ...]
    unsupported_phase_names: tuple[str, ...]
    baseline_relative: bool
    total_formula: str
    mean_rate_formula: str
    coverage_formula: str
    total_change_formula: str
    rate_change_formula: str
    interpretation_limits: str


def _validate_windows(
    phases: Sequence[PhaseWindow],
    gap_policy: PhaseGapPolicy,
) -> None:
    if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)):
        raise ValueError("phases must be a nonempty sequence of PhaseWindow values")
    if not phases:
        raise ValueError("at least one phase window is required")

    names: set[str] = set()
    for phase in phases:
        if not isinstance(phase, PhaseWindow):
            raise ValueError("every phase must be a PhaseWindow")
        if not isinstance(phase.name, str) or not phase.name:
            raise ValueError("every phase name must be a nonempty string")
        if phase.name in names:
            raise ValueError("phase names must be unique")
        if type(phase.start_ns) is not int or type(phase.end_ns) is not int:
            raise ValueError("phase boundaries must be integer nanoseconds")
        if phase.start_ns >= phase.end_ns:
            raise ValueError("every phase window must have positive duration")
        names.add(phase.name)

    for previous, current in zip(phases, phases[1:]):
        if current.start_ns < previous.end_ns:
            raise ValueError(
                f"phase windows {previous.name!r} and {current.name!r} overlap"
            )
        gap = current.start_ns - previous.end_ns
        if gap > 0 and gap_policy is PhaseGapPolicy.REJECT_GAPS:
            raise ValueError(
                f"phase windows {previous.name!r} and {current.name!r} have a "
                f"{gap} ns gap; supply PhaseGapPolicy.ALLOW_GAPS to preserve it"
            )


def _selected_records(
    phases: Sequence[PhaseWindow],
    records: Mapping[str, PhaseTrafficRecord],
) -> list[tuple[PhaseWindow, PhaseTrafficRecord]]:
    if not isinstance(records, Mapping):
        raise ValueError("records must be a mapping keyed by phase name")
    selected: list[tuple[PhaseWindow, PhaseTrafficRecord]] = []
    for phase in phases:
        record = records.get(phase.name)
        if record is None:
            continue
        if not isinstance(record, PhaseTrafficRecord):
            raise ValueError("every record value must be a PhaseTrafficRecord")
        if not isinstance(record.clock_id, str) or not record.clock_id:
            raise ValueError("every record clock_id must be a nonempty string")
        if record.analysis.interval_detail_included is not True:
            raise ValueError(
                "phase analysis requires TrafficAnalysis interval_detail_included=True"
            )
        selected.append((phase, record))

    if not selected:
        raise ValueError("at least one phase must have a TrafficAnalysis record")
    return selected


def _validate_common_identity(
    selected: Sequence[tuple[PhaseWindow, PhaseTrafficRecord]],
) -> None:
    first_phase, first_record = selected[0]
    first_analysis = first_record.analysis
    first_window = first_analysis.requested_window_ns
    if first_window is None:
        raise ValueError("TrafficAnalysis records must have a requested window")
    if first_analysis.source_counter != first_analysis.identity.counter_id:
        raise ValueError("TrafficAnalysis source and identity counters disagree")

    for phase, record in selected:
        analysis = record.analysis
        identity = analysis.identity
        mismatches: list[str] = []
        if record.clock_id != selected[0][1].clock_id:
            mismatches.append("clock")
        if identity.run_id != first_analysis.identity.run_id:
            mismatches.append("run")
        if identity.node_id != first_analysis.identity.node_id:
            mismatches.append("node")
        if identity.nic_id != first_analysis.identity.nic_id:
            mismatches.append("NIC")
        if identity.collector_id != first_analysis.identity.collector_id:
            mismatches.append("collector")
        if identity.direction != first_analysis.identity.direction:
            mismatches.append("direction")
        if analysis.source_counter != first_analysis.source_counter:
            mismatches.append("counter")
        if analysis.requested_window_ns != first_window:
            mismatches.append("window")
        if analysis.unit != first_analysis.unit:
            mismatches.append("units")
        if analysis.semantics != first_analysis.semantics:
            mismatches.append("semantics")
        if analysis.value_kind != first_analysis.value_kind:
            mismatches.append("value kind")
        if analysis.baseline_relative != first_analysis.baseline_relative:
            mismatches.append("baseline-relative semantics")
        if mismatches:
            raise ValueError(
                f"phase {phase.name!r} does not share common "
                f"{', '.join(mismatches)}"
            )

        common_start, common_end = first_window
        if phase.start_ns < common_start or phase.end_ns > common_end:
            raise ValueError(
                f"phase {phase.name!r} is outside the common analysis window"
            )


def _sum_deltas(values: Sequence[int | float]) -> int | float:
    if any(type(value) is float for value in values):
        return math.fsum(float(value) for value in values)
    return sum(values, start=0)


def _phase_metric(
    phase: PhaseWindow,
    record: PhaseTrafficRecord | None,
) -> PhaseMetric:
    if record is None:
        return PhaseMetric(
            phase_name=phase.name,
            window_ns=(phase.start_ns, phase.end_ns),
            status=AnalysisStatus.MISSING_DATA,
            state_flags=frozenset({PhaseState.MISSING_PHASE_DATA}),
            status_reasons=(
                "No TrafficAnalysis record was supplied for this phase.",
            ),
            quality_flags=frozenset({SeriesQualityFlag.MISSING}),
            source_quality_flags=frozenset(),
            source_counter="",
            total_bytes=None,
            covered_duration_ns=0,
            requested_duration_ns=phase.end_ns - phase.start_ns,
            uncovered_duration_ns=phase.end_ns - phase.start_ns,
            coverage_fraction=0.0,
            mean_rate_bytes_per_s=None,
            valid_interval_count=0,
            invalid_interval_count=0,
            boundary_excluded_interval_count=0,
            represented_window_ns=None,
            baseline_relative=False,
        )

    analysis = record.analysis
    selected_intervals = []
    boundary_excluded_count = 0
    for interval in analysis.intervals:
        inside = (
            interval.start_ns >= phase.start_ns
            and interval.end_ns <= phase.end_ns
        )
        overlaps = (
            interval.end_ns > phase.start_ns
            and interval.start_ns < phase.end_ns
        )
        if inside:
            selected_intervals.append(interval)
        elif overlaps:
            boundary_excluded_count += 1

    valid_intervals = [
        interval
        for interval in selected_intervals
        if interval.valid and interval.delta_bytes is not None
    ]
    invalid_count = len(selected_intervals) - len(valid_intervals)
    covered_duration_ns = sum(
        interval.duration_ns for interval in valid_intervals
    )
    requested_duration_ns = phase.end_ns - phase.start_ns
    uncovered_duration_ns = max(
        0, requested_duration_ns - covered_duration_ns
    )
    coverage_fraction = (
        covered_duration_ns / requested_duration_ns
        if requested_duration_ns > 0
        else None
    )
    total_bytes = (
        _sum_deltas(
            [interval.delta_bytes for interval in valid_intervals]
        )
        if valid_intervals
        else None
    )
    mean_rate = (
        float(total_bytes)
        / (covered_duration_ns / NANOSECONDS_PER_SECOND)
        if total_bytes is not None and covered_duration_ns > 0
        else None
    )

    states: set[PhaseState] = set()
    quality: set[SeriesQualityFlag] = set()
    reasons: list[str] = []
    for flag in analysis.quality_flags:
        if flag in {
            SeriesQualityFlag.BASELINE_RELATIVE,
            SeriesQualityFlag.PRECISION_LOSS_POSSIBLE,
            SeriesQualityFlag.UNSUPPORTED_SERIES,
            SeriesQualityFlag.MISSING,
            SeriesQualityFlag.MULTIPLE_OBSERVATION_STREAMS,
        }:
            quality.add(flag)
    if any(
        interval.state is IntervalState.DECREASE_OR_RESET
        for interval in selected_intervals
    ):
        quality.add(SeriesQualityFlag.DECREASE_OR_RESET)
        states.add(PhaseState.INVALID_DATA)
    if any(
        interval.state is IntervalState.CONFLICTING_TIMESTAMP
        for interval in selected_intervals
    ):
        quality.add(SeriesQualityFlag.CONFLICTING_TIMESTAMP)
        states.add(PhaseState.INVALID_DATA)
    if boundary_excluded_count:
        quality.add(SeriesQualityFlag.BOUNDARY_INTERVAL_EXCLUDED)
        states.add(PhaseState.BOUNDARY_INTERVAL_EXCLUDED)
        reasons.append(
            "Intervals straddling a phase boundary are excluded, not allocated."
        )
    if coverage_fraction is not None and coverage_fraction < 1.0:
        quality.add(SeriesQualityFlag.PARTIAL_WINDOW)
        states.add(PhaseState.PARTIAL_COVERAGE)
        reasons.append(
            "The mean rate covers observed duration only; missing duration "
            "is not zero-filled."
        )
    if analysis.baseline_relative:
        states.add(PhaseState.BASELINE_RELATIVE)
        quality.add(SeriesQualityFlag.BASELINE_RELATIVE)
        reasons.append(
            "Baseline-relative totals cover observed increments, not an "
            "absolute pre-phase baseline."
        )

    if analysis.status is AnalysisStatus.UNSUPPORTED_SERIES:
        status = AnalysisStatus.UNSUPPORTED_SERIES
        states.add(PhaseState.UNSUPPORTED_DATA)
        reasons.append("The source traffic analysis is unsupported.")
    elif analysis.status is AnalysisStatus.MISSING_DATA:
        status = AnalysisStatus.MISSING_DATA
        states.add(PhaseState.MISSING_PHASE_DATA)
        reasons.append("The source traffic analysis is explicitly missing.")
    elif not valid_intervals:
        if invalid_count:
            status = AnalysisStatus.INVALID_DATA
            states.add(PhaseState.INVALID_DATA)
        else:
            status = AnalysisStatus.INSUFFICIENT_DATA
        reasons.append(
            "The phase contains no complete valid interval at source resolution."
        )
    elif total_bytes == 0:
        status = AnalysisStatus.ZERO_TRAFFIC
        states.add(PhaseState.ZERO_TRAFFIC)
        reasons.append("All valid intervals in this phase have zero traffic.")
    elif invalid_count:
        status = AnalysisStatus.PARTIAL_INVALID_DATA
        states.add(PhaseState.INVALID_DATA)
        reasons.append(
            "Invalid intervals are excluded; the total covers valid intervals only."
        )
    else:
        status = AnalysisStatus.OK

    represented_window = (
        (
            min(interval.start_ns for interval in valid_intervals),
            max(interval.end_ns for interval in valid_intervals),
        )
        if valid_intervals
        else None
    )
    return PhaseMetric(
        phase_name=phase.name,
        window_ns=(phase.start_ns, phase.end_ns),
        status=status,
        state_flags=frozenset(states),
        status_reasons=tuple(reasons),
        quality_flags=frozenset(quality),
        source_quality_flags=frozenset(analysis.quality_flags),
        source_counter=analysis.source_counter,
        total_bytes=total_bytes,
        covered_duration_ns=covered_duration_ns,
        requested_duration_ns=requested_duration_ns,
        uncovered_duration_ns=uncovered_duration_ns,
        coverage_fraction=coverage_fraction,
        mean_rate_bytes_per_s=mean_rate,
        valid_interval_count=len(valid_intervals),
        invalid_interval_count=invalid_count,
        boundary_excluded_interval_count=boundary_excluded_count,
        represented_window_ns=represented_window,
        baseline_relative=analysis.baseline_relative,
    )


def _phase_change(
    previous: PhaseMetric,
    current: PhaseMetric,
) -> PhaseChange:
    gap_duration = max(
        0, current.window_ns[0] - previous.window_ns[1]
    )
    uncertainty_reasons: list[str] = []

    if previous.total_bytes is None or current.total_bytes is None:
        total_change = None
        total_change_fraction = None
        total_change_percent = None
        comparison_status = PhaseComparisonStatus.MISSING
        uncertainty_reasons.append(
            "At least one phase has no valid traffic total."
        )
    else:
        total_change = current.total_bytes - previous.total_bytes
        total_change_fraction = (
            float(total_change / previous.total_bytes)
            if previous.total_bytes > 0
            else None
        )
        total_change_percent = (
            100.0 * total_change_fraction
            if total_change_fraction is not None
            else None
        )
        comparison_status = PhaseComparisonStatus.COMPARABLE
        if previous.total_bytes == 0:
            uncertainty_reasons.append(
                "The change fraction is undefined because the previous phase "
                "has zero traffic."
            )

    if (
        previous.mean_rate_bytes_per_s is None
        or current.mean_rate_bytes_per_s is None
    ):
        rate_change = None
        rate_ratio = None
        if comparison_status is PhaseComparisonStatus.COMPARABLE:
            comparison_status = PhaseComparisonStatus.MISSING
        uncertainty_reasons.append(
            "At least one phase has no valid covered-duration mean rate."
        )
    else:
        rate_change = (
            current.mean_rate_bytes_per_s
            - previous.mean_rate_bytes_per_s
        )
        rate_ratio = (
            current.mean_rate_bytes_per_s / previous.mean_rate_bytes_per_s
            if previous.mean_rate_bytes_per_s > 0
            else None
        )
        if previous.mean_rate_bytes_per_s == 0:
            uncertainty_reasons.append(
                "The rate ratio is undefined because the previous phase has "
                "zero traffic."
            )

    if gap_duration > 0:
        comparison_status = PhaseComparisonStatus.NOT_ADJACENT
        uncertainty_reasons.append(
            f"The phases are separated by {gap_duration} ns of unassigned time."
        )

    uncertain_states = {
        PhaseState.PARTIAL_COVERAGE,
        PhaseState.INVALID_DATA,
        PhaseState.BASELINE_RELATIVE,
        PhaseState.BOUNDARY_INTERVAL_EXCLUDED,
    }
    if uncertain_states.intersection(
        previous.state_flags | current.state_flags
    ):
        comparison_status = PhaseComparisonStatus.UNCERTAIN
        uncertainty_reasons.append(
            "At least one phase has incomplete coverage, invalid data, "
            "excluded boundary intervals, or baseline-relative values."
        )

    return PhaseChange(
        from_phase_name=previous.phase_name,
        to_phase_name=current.phase_name,
        from_window_ns=previous.window_ns,
        to_window_ns=current.window_ns,
        inter_phase_gap_duration_ns=gap_duration,
        total_change_bytes=total_change,
        total_change_fraction=total_change_fraction,
        total_change_percent=total_change_percent,
        rate_change_bytes_per_s=rate_change,
        rate_ratio=rate_ratio,
        comparison_status=comparison_status,
        is_comparable=comparison_status is PhaseComparisonStatus.COMPARABLE,
        uncertainty_reasons=tuple(uncertainty_reasons),
    )


def analyze_manual_phase_windows(
    phases: Sequence[PhaseWindow],
    records: Mapping[str, PhaseTrafficRecord],
    gap_policy: PhaseGapPolicy = PhaseGapPolicy.REJECT_GAPS,
) -> ManualPhaseAnalysisResult:
    """Analyze ordered manual phases without inventing sub-sample behavior.

    Intervals are assigned only when both sample endpoints lie in a phase.
    Straddling intervals remain visible as boundary exclusions. A phase with
    no supplied record is missing, never zero.
    """

    if not isinstance(gap_policy, PhaseGapPolicy):
        raise ValueError("gap_policy must be a PhaseGapPolicy")
    _validate_windows(phases, gap_policy)
    selected = _selected_records(phases, records)
    _validate_common_identity(selected)

    phase_results = tuple(
        _phase_metric(phase, records.get(phase.name)) for phase in phases
    )
    change_results = tuple(
        _phase_change(previous, current)
        for previous, current in zip(phase_results, phase_results[1:])
    )

    missing = tuple(
        metric.phase_name
        for metric in phase_results
        if PhaseState.MISSING_PHASE_DATA in metric.state_flags
    )
    partial = tuple(
        metric.phase_name
        for metric in phase_results
        if PhaseState.PARTIAL_COVERAGE in metric.state_flags
        or PhaseState.INVALID_DATA in metric.state_flags
        or PhaseState.BOUNDARY_INTERVAL_EXCLUDED in metric.state_flags
    )
    zero = tuple(
        metric.phase_name
        for metric in phase_results
        if PhaseState.ZERO_TRAFFIC in metric.state_flags
    )
    unsupported = tuple(
        metric.phase_name
        for metric in phase_results
        if PhaseState.UNSUPPORTED_DATA in metric.state_flags
    )
    inter_phase_gaps = sum(
        change.inter_phase_gap_duration_ns for change in change_results
    )
    covered_duration = sum(
        metric.covered_duration_ns for metric in phase_results
    )
    requested_duration = sum(
        metric.requested_duration_ns for metric in phase_results
    )
    uncovered_duration = sum(
        metric.uncovered_duration_ns for metric in phase_results
    )

    states: set[PhaseState] = set()
    for metric in phase_results:
        states.update(metric.state_flags)
    if inter_phase_gaps:
        states.add(PhaseState.INTER_PHASE_GAP)

    reasons: list[str] = []
    if missing:
        reasons.append("Missing phase records remain missing, not zero traffic.")
    if partial:
        reasons.append(
            "Some phase totals or rates have incomplete or invalid coverage."
        )
    if inter_phase_gaps:
        reasons.append(
            "Inter-phase gaps are retained as unassigned time and make "
            "adjacent changes non-comparable."
        )
    if unsupported:
        reasons.append("Some phase source analyses are unsupported.")

    if missing:
        status = PhaseAnalysisStatus.PARTIAL_DATA
    elif unsupported:
        status = (
            PhaseAnalysisStatus.UNSUPPORTED_DATA
            if len(unsupported) == len(phase_results)
            else PhaseAnalysisStatus.PARTIAL_DATA
        )
    elif all(
        metric.status is AnalysisStatus.ZERO_TRAFFIC
        for metric in phase_results
    ):
        status = PhaseAnalysisStatus.ZERO_TRAFFIC
    elif any(
        metric.status
        not in {AnalysisStatus.OK, AnalysisStatus.ZERO_TRAFFIC}
        for metric in phase_results
    ):
        status = PhaseAnalysisStatus.PARTIAL_DATA
    else:
        status = PhaseAnalysisStatus.OK

    first_record = selected[0][1]
    first_analysis = first_record.analysis
    first_identity = first_analysis.identity
    return ManualPhaseAnalysisResult(
        run_id=first_identity.run_id,
        node_id=first_identity.node_id,
        nic_id=first_identity.nic_id,
        collector_id=first_identity.collector_id,
        direction=first_identity.direction,
        source_counter=first_analysis.source_counter,
        unit=first_analysis.unit,
        rate_unit=first_analysis.rate_unit,
        semantics=first_analysis.semantics,
        value_kind=first_analysis.value_kind,
        clock_id=first_record.clock_id,
        common_window_ns=first_analysis.requested_window_ns,
        gap_policy=gap_policy,
        status=status,
        state_flags=frozenset(states),
        status_reasons=tuple(reasons),
        phases=phase_results,
        changes=change_results,
        phase_count=len(phase_results),
        missing_phase_count=len(missing),
        covered_duration_ns=covered_duration,
        requested_duration_ns=requested_duration,
        uncovered_duration_ns=uncovered_duration,
        coverage_fraction=(
            covered_duration / requested_duration
            if requested_duration > 0
            else None
        ),
        valid_interval_count=sum(
            metric.valid_interval_count for metric in phase_results
        ),
        invalid_interval_count=sum(
            metric.invalid_interval_count for metric in phase_results
        ),
        boundary_excluded_interval_count=sum(
            metric.boundary_excluded_interval_count
            for metric in phase_results
        ),
        inter_phase_gap_duration_ns=inter_phase_gaps,
        missing_phase_names=missing,
        partial_phase_names=partial,
        zero_phase_names=zero,
        unsupported_phase_names=unsupported,
        baseline_relative=first_analysis.baseline_relative,
        total_formula=TOTAL_FORMULA,
        mean_rate_formula=MEAN_RATE_FORMULA,
        coverage_formula=COVERAGE_FORMULA,
        total_change_formula=TOTAL_CHANGE_FORMULA,
        rate_change_formula=RATE_CHANGE_FORMULA,
        interpretation_limits=INTERPRETATION_LIMITS,
    )
