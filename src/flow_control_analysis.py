"""Pure flow-control pressure evidence for cumulative NIC counters."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import groupby
import math

from traffic_analysis import (
    IntervalState,
    MetricSeries,
    NANOSECONDS_PER_SECOND,
    SeriesQualityFlag,
    TrafficDirection,
    ValueKind,
)


FLOW_CONTROL_ANALYSIS_VERSION = 1
DELTA_FORMULA = "end_value - start_value"
MEAN_RATE_FORMULA = "sum(valid delta) / sum(valid duration_seconds)"
INTERVAL_RATE_FORMULA = "delta / duration_seconds"
PAUSE_CYCLE_SECONDS_FORMULA = "pause_cycle_delta / verified_clock_hz"
DEFAULT_FLOW_CONTROL_PRIORITIES = tuple(range(8))

FLOW_CONTROL_LIMITATIONS = (
    "Pause-cycle totals remain in cycles; no wall-clock pause duration or "
    "percentage is claimed without verified clock evidence.",
    "Per-direction and per-priority results remain separate and are never "
    "summed into wall-clock pause time.",
    "Flow-control events and cycles are pressure evidence, not proof of "
    "congestion, a unique bottleneck, or ECN.",
    "Rates use complete intervals between observed samples and do not "
    "establish a fixed cadence or sub-interval burst shape.",
    "Missing counters and priorities remain missing; they are not treated as zero.",
)


class FlowControlMetricKind(str, Enum):
    PAUSE_CYCLES = "pause_cycles"
    PAUSE_REFRESH = "pause_refresh"
    PAUSE_SENT = "pause_sent"
    PAUSE_XOFF_SENT = "pause_xoff_sent"
    PAUSE_RECEIVED = "pause_received"
    FGFC_DISCARD = "fgfc_discard"
    FGFC_XOFF = "fgfc_xoff"


class FlowControlStatus(str, Enum):
    OK = "ok"
    ZERO_OBSERVATIONS = "zero_observations"
    PARTIAL_ZERO_OBSERVATIONS = "partial_zero_observations"
    PARTIAL_MISSING_DATA = "partial_missing_data"
    MISSING_DATA = "missing_data"
    INSUFFICIENT_DATA = "insufficient_data"
    PARTIAL_INSUFFICIENT_DATA = "partial_insufficient_data"
    PARTIAL_INVALID_DATA = "partial_invalid_data"
    INVALID_DATA = "invalid_data"
    UNSUPPORTED_SERIES = "unsupported_series"
    PARTIAL_UNSUPPORTED_SERIES = "partial_unsupported_series"


@dataclass(frozen=True)
class FlowControlCounterKey:
    metric_kind: FlowControlMetricKind
    direction: TrafficDirection | None
    priority: int | None


@dataclass(frozen=True)
class _CounterDefinition:
    key: FlowControlCounterKey
    semantics: str
    unit: str


_COUNTER_DEFINITIONS: dict[str, _CounterDefinition] = {}


def _register_counter(
    counter_name: str,
    metric_kind: FlowControlMetricKind,
    direction: TrafficDirection | None,
    priority: int | None,
    semantics: str,
) -> None:
    _COUNTER_DEFINITIONS[counter_name] = _CounterDefinition(
        key=FlowControlCounterKey(
            metric_kind=metric_kind,
            direction=direction,
            priority=priority,
        ),
        semantics=semantics,
        unit=semantics,
    )


for _priority in DEFAULT_FLOW_CONTROL_PRIORITIES:
    _register_counter(
        f"hni_tx_paused_{_priority}",
        FlowControlMetricKind.PAUSE_CYCLES,
        TrafficDirection.TX,
        _priority,
        "cycles",
    )
    _register_counter(
        f"hni_rx_paused_{_priority}",
        FlowControlMetricKind.PAUSE_CYCLES,
        TrafficDirection.RX,
        _priority,
        "cycles",
    )
    _register_counter(
        f"hni_pause_xoff_sent_{_priority}",
        FlowControlMetricKind.PAUSE_XOFF_SENT,
        TrafficDirection.TX,
        _priority,
        "events",
    )
    _register_counter(
        f"hni_pause_recv_{_priority}",
        FlowControlMetricKind.PAUSE_RECEIVED,
        TrafficDirection.RX,
        _priority,
        "events",
    )

_register_counter(
    "hni_pause_refresh",
    FlowControlMetricKind.PAUSE_REFRESH,
    TrafficDirection.TX,
    None,
    "events",
)
_register_counter(
    "hni_pause_sent",
    FlowControlMetricKind.PAUSE_SENT,
    TrafficDirection.TX,
    None,
    "events",
)
_register_counter(
    "hni_fgfc_discard",
    FlowControlMetricKind.FGFC_DISCARD,
    None,
    None,
    "events",
)
_register_counter(
    "hni_fgfc_event_xoff",
    FlowControlMetricKind.FGFC_XOFF,
    None,
    None,
    "events",
)


@dataclass(frozen=True)
class VerifiedPauseCycleClock:
    """Clock evidence required before any pause-cycle time conversion."""

    frequency_hz: int
    verification_evidence: str
    applies_to_counter_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.frequency_hz) is not int or self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be a positive integer")
        if not isinstance(self.verification_evidence, str) or not self.verification_evidence:
            raise ValueError("verification_evidence must be a nonempty string")
        if not self.applies_to_counter_names:
            raise ValueError("applies_to_counter_names must be nonempty")
        if len(set(self.applies_to_counter_names)) != len(self.applies_to_counter_names):
            raise ValueError("applies_to_counter_names must be unique")
        for counter_name in self.applies_to_counter_names:
            definition = _COUNTER_DEFINITIONS.get(counter_name)
            if definition is None or definition.key.metric_kind is not (
                FlowControlMetricKind.PAUSE_CYCLES
            ):
                raise ValueError("clock evidence can only apply to pause-cycle counters")


@dataclass(frozen=True)
class FlowControlInterval:
    """One adjacent cumulative-counter transition and its disposition."""

    start_ns: int
    end_ns: int
    duration_ns: int
    start_value: int | float
    end_value: int | float
    delta: int | float | None
    rate_per_second: float | None
    valid: bool
    state: IntervalState


@dataclass(frozen=True)
class FlowControlSeriesResult:
    """Flow-control evidence for one counter, direction, and priority."""

    counter_name: str
    metric_kind: FlowControlMetricKind
    direction: TrafficDirection | None
    priority: int | None
    status: FlowControlStatus
    quality_flags: frozenset[SeriesQualityFlag]
    status_reasons: tuple[str, ...]
    requested_window_ns: tuple[int, int] | None
    represented_window_ns: tuple[int, int] | None
    source_counter: str
    value_kind: ValueKind
    semantics: str
    unit: str
    baseline_relative: bool
    total: int | float | None
    mean_rate_per_second: float | None
    first_nonzero_ns: int | None
    first_nonzero_delta: int | float | None
    first_nonzero_value: int | float | None
    covered_duration_ns: int
    requested_duration_ns: int
    uncovered_requested_duration_ns: int
    coverage_fraction: float | None
    sample_count: int
    interval_count: int
    valid_interval_count: int
    invalid_interval_count: int
    boundary_excluded_interval_count: int
    intervals: tuple[FlowControlInterval, ...]
    delta_formula: str
    interval_rate_formula: str
    mean_rate_formula: str
    aggregation_population: str


@dataclass(frozen=True)
class FlowControlIdentity:
    run_id: str
    node_id: str
    nic_id: str
    collector_id: str


@dataclass(frozen=True)
class FlowControlResult:
    """Per-NIC flow-control evidence without unsupported causal claims."""

    contract_version: int
    identity: FlowControlIdentity | None
    status: FlowControlStatus
    quality_flags: frozenset[SeriesQualityFlag]
    status_reasons: tuple[str, ...]
    requested_window_ns: tuple[int, int] | None
    represented_window_ns: tuple[int, int] | None
    covered_duration_ns: int
    requested_duration_ns: int
    uncovered_requested_duration_ns: int
    coverage_fraction: float | None
    series_results: tuple[FlowControlSeriesResult, ...]
    pause_cycle_results: tuple[FlowControlSeriesResult, ...]
    event_results: tuple[FlowControlSeriesResult, ...]
    xoff_event_results: tuple[FlowControlSeriesResult, ...]
    fgfc_event_results: tuple[FlowControlSeriesResult, ...]
    observed_counter_names: tuple[str, ...]
    missing_counter_names: tuple[str, ...]
    unsupported_counter_names: tuple[str, ...]
    zero_observed_counter_names: tuple[str, ...]
    first_nonzero_counter_names: tuple[str, ...]
    limitations: tuple[str, ...]


def _ordered_samples(
    series: MetricSeries,
) -> tuple[
    list[tuple[int, int | float]],
    list[FlowControlInterval],
    frozenset[SeriesQualityFlag],
]:
    if series.value_dtype == "uint64":
        records = [
            (int(timestamp), int(value))
            for timestamp, value in zip(
                series.timestamps_ns, series.values, strict=True
            )
        ]
    else:
        records = [
            (int(timestamp), float(value))
            for timestamp, value in zip(
                series.timestamps_ns, series.values, strict=True
            )
        ]
    ordered = sorted(records, key=lambda record: record[0])
    flags: set[SeriesQualityFlag] = set(series.quality_flags)
    if records != ordered:
        flags.add(SeriesQualityFlag.REORDERED)

    samples: list[tuple[int, int | float]] = []
    intervals: list[FlowControlInterval] = []
    previous_sample: tuple[int, int | float] | None = None

    for timestamp, grouped_records in groupby(ordered, key=lambda record: record[0]):
        values = [value for _, value in grouped_records]
        if len(set(values)) != 1:
            first_value, second_value = values[:2]
            intervals.append(
                FlowControlInterval(
                    start_ns=timestamp,
                    end_ns=timestamp,
                    duration_ns=0,
                    start_value=first_value,
                    end_value=second_value,
                    delta=None,
                    rate_per_second=None,
                    valid=False,
                    state=IntervalState.CONFLICTING_TIMESTAMP,
                )
            )
            flags.add(SeriesQualityFlag.CONFLICTING_TIMESTAMP)
            previous_sample = None
            continue

        sample = (timestamp, values[0])
        samples.append(sample)
        if len(values) > 1:
            flags.add(SeriesQualityFlag.DUPLICATE)
        if previous_sample is not None:
            start_ns, start_value = previous_sample
            end_ns, end_value = sample
            duration_ns = end_ns - start_ns
            delta = end_value - start_value
            if delta < 0:
                intervals.append(
                    FlowControlInterval(
                        start_ns=start_ns,
                        end_ns=end_ns,
                        duration_ns=duration_ns,
                        start_value=start_value,
                        end_value=end_value,
                        delta=None,
                        rate_per_second=None,
                        valid=False,
                        state=IntervalState.DECREASE_OR_RESET,
                    )
                )
            else:
                rate = float(delta) / (duration_ns / NANOSECONDS_PER_SECOND)
                intervals.append(
                    FlowControlInterval(
                        start_ns=start_ns,
                        end_ns=end_ns,
                        duration_ns=duration_ns,
                        start_value=start_value,
                        end_value=end_value,
                        delta=delta,
                        rate_per_second=rate,
                        valid=True,
                        state=IntervalState.VALID,
                    )
                )
        previous_sample = sample

    return samples, intervals, frozenset(flags)


def _empty_series_result(
    series: MetricSeries,
    definition: _CounterDefinition | None,
    status: FlowControlStatus,
    requested_window_ns: tuple[int, int] | None,
    quality_flags: frozenset[SeriesQualityFlag],
    reasons: tuple[str, ...],
    sample_count: int,
) -> FlowControlSeriesResult:
    requested_duration = (
        max(0, requested_window_ns[1] - requested_window_ns[0])
        if requested_window_ns is not None
        else 0
    )
    key = definition.key if definition is not None else FlowControlCounterKey(
        metric_kind=FlowControlMetricKind.PAUSE_CYCLES,
        direction=None,
        priority=None,
    )
    return FlowControlSeriesResult(
        counter_name=series.identity.counter_id,
        metric_kind=key.metric_kind,
        direction=key.direction,
        priority=key.priority,
        status=status,
        quality_flags=quality_flags,
        status_reasons=reasons,
        requested_window_ns=requested_window_ns,
        represented_window_ns=None,
        source_counter=series.identity.counter_id,
        value_kind=series.value_kind,
        semantics=series.semantics,
        unit=series.unit,
        baseline_relative=series.baseline_relative,
        total=None,
        mean_rate_per_second=None,
        first_nonzero_ns=None,
        first_nonzero_delta=None,
        first_nonzero_value=None,
        covered_duration_ns=0,
        requested_duration_ns=requested_duration,
        uncovered_requested_duration_ns=requested_duration,
        coverage_fraction=0.0 if requested_duration > 0 else None,
        sample_count=sample_count,
        interval_count=0,
        valid_interval_count=0,
        invalid_interval_count=0,
        boundary_excluded_interval_count=0,
        intervals=(),
        delta_formula=DELTA_FORMULA,
        interval_rate_formula=INTERVAL_RATE_FORMULA,
        mean_rate_formula=MEAN_RATE_FORMULA,
        aggregation_population="No valid complete intervals",
    )


def _series_result(
    series: MetricSeries,
    definition: _CounterDefinition,
    requested_window_ns: tuple[int, int] | None,
) -> FlowControlSeriesResult:
    if definition is None:
        return _empty_series_result(
            series,
            None,
            FlowControlStatus.UNSUPPORTED_SERIES,
            requested_window_ns,
            frozenset(series.quality_flags)
            | {SeriesQualityFlag.UNSUPPORTED_SERIES},
            ("The counter is not in the flow-control catalog.",),
            len(series.timestamps_ns),
        )

    quality_flags: set[SeriesQualityFlag] = set(series.quality_flags)
    unsupported_reasons: list[str] = []
    if series.value_kind is not ValueKind.CUMULATIVE:
        quality_flags.add(SeriesQualityFlag.UNSUPPORTED_SERIES)
        unsupported_reasons.append("The counter must be cumulative.")
    if series.semantics != definition.semantics or series.unit != definition.unit:
        quality_flags.add(SeriesQualityFlag.UNSUPPORTED_SERIES)
        unsupported_reasons.append(
            f"The counter must have {definition.semantics} semantics and units."
        )
    if series.identity.direction != definition.key.direction:
        quality_flags.add(SeriesQualityFlag.UNSUPPORTED_SERIES)
        unsupported_reasons.append(
            "The counter's series direction does not match its catalog definition."
        )
    if unsupported_reasons:
        return _empty_series_result(
            series,
            definition,
            FlowControlStatus.UNSUPPORTED_SERIES,
            requested_window_ns,
            frozenset(quality_flags),
            tuple(unsupported_reasons),
            len(series.timestamps_ns),
        )

    if SeriesQualityFlag.MISSING in quality_flags:
        return _empty_series_result(
            series,
            definition,
            FlowControlStatus.MISSING_DATA,
            requested_window_ns,
            frozenset(quality_flags),
            ("The input series is explicitly marked missing.",),
            len(series.timestamps_ns),
        )
    if SeriesQualityFlag.INVALID in quality_flags:
        return _empty_series_result(
            series,
            definition,
            FlowControlStatus.INVALID_DATA,
            requested_window_ns,
            frozenset(quality_flags),
            ("The input series is explicitly marked invalid.",),
            len(series.timestamps_ns),
        )

    samples, intervals, ordered_quality = _ordered_samples(series)
    quality_flags.update(ordered_quality)
    if series.baseline_relative:
        quality_flags.add(SeriesQualityFlag.BASELINE_RELATIVE)

    selected_intervals = intervals
    boundary_excluded_count = 0
    if requested_window_ns is not None:
        window_start_ns, window_end_ns = requested_window_ns
        selected_intervals = []
        for interval in intervals:
            inside = (
                window_start_ns <= interval.start_ns
                and interval.end_ns <= window_end_ns
            )
            overlaps = (
                interval.end_ns > window_start_ns
                and interval.start_ns < window_end_ns
            )
            if inside:
                selected_intervals.append(interval)
            elif overlaps:
                boundary_excluded_count += 1
        if boundary_excluded_count:
            quality_flags.add(SeriesQualityFlag.BOUNDARY_INTERVAL_EXCLUDED)
        if samples and (
            samples[0][0] < window_start_ns or samples[-1][0] > window_end_ns
        ):
            quality_flags.add(SeriesQualityFlag.PARTIAL_WINDOW)

    selected_intervals = list(selected_intervals)
    valid_intervals = [
        interval for interval in selected_intervals if interval.valid
    ]
    invalid_intervals = [
        interval for interval in selected_intervals if not interval.valid
    ]
    if any(
        interval.state is IntervalState.DECREASE_OR_RESET
        for interval in selected_intervals
    ):
        quality_flags.add(SeriesQualityFlag.DECREASE_OR_RESET)

    covered_duration_ns = sum(
        interval.duration_ns for interval in valid_intervals
    )
    requested_duration_ns = (
        requested_window_ns[1] - requested_window_ns[0]
        if requested_window_ns is not None
        else 0
    )
    uncovered_duration_ns = max(0, requested_duration_ns - covered_duration_ns)
    coverage_fraction = (
        covered_duration_ns / requested_duration_ns
        if requested_duration_ns > 0
        else None
    )

    total: int | float | None
    if not valid_intervals:
        total = None
    elif series.value_dtype == "uint64":
        total = sum(
            interval.delta
            for interval in valid_intervals
            if interval.delta is not None
        )
    else:
        total = math.fsum(
            float(interval.delta)
            for interval in valid_intervals
            if interval.delta is not None
        )

    represented_window_ns = (
        (
            min(interval.start_ns for interval in valid_intervals),
            max(interval.end_ns for interval in valid_intervals),
        )
        if valid_intervals
        else None
    )
    first_nonzero_interval = next(
        (
            interval
            for interval in valid_intervals
            if interval.delta is not None and interval.delta > 0
        ),
        None,
    )
    mean_rate = (
        float(total) / (covered_duration_ns / NANOSECONDS_PER_SECOND)
        if total is not None
        else None
    )

    reasons: list[str] = []
    if not valid_intervals:
        if invalid_intervals:
            status = FlowControlStatus.INVALID_DATA
            reasons.append("No valid intervals remain after quality exclusions.")
        else:
            status = FlowControlStatus.INSUFFICIENT_DATA
            reasons.append(
                "At least two distinct timestamps are required for an interval."
            )
    elif total == 0:
        if invalid_intervals:
            status = FlowControlStatus.PARTIAL_INVALID_DATA
            reasons.append("Valid intervals are zero, but invalid intervals were excluded.")
        else:
            status = FlowControlStatus.ZERO_OBSERVATIONS
            reasons.append("All valid intervals have zero flow-control increments.")
    elif invalid_intervals:
        status = FlowControlStatus.PARTIAL_INVALID_DATA
        reasons.append(
            "Numerical results include only valid intervals; resets and conflicts are excluded."
        )
    else:
        status = FlowControlStatus.OK

    if series.baseline_relative:
        reasons.append(
            "Values are baseline-relative; totals use differences between observed changed samples."
        )

    return FlowControlSeriesResult(
        counter_name=series.identity.counter_id,
        metric_kind=definition.key.metric_kind,
        direction=definition.key.direction,
        priority=definition.key.priority,
        status=status,
        quality_flags=frozenset(quality_flags),
        status_reasons=tuple(reasons),
        requested_window_ns=requested_window_ns,
        represented_window_ns=represented_window_ns,
        source_counter=series.identity.counter_id,
        value_kind=series.value_kind,
        semantics=series.semantics,
        unit=series.unit,
        baseline_relative=series.baseline_relative,
        total=total,
        mean_rate_per_second=mean_rate,
        first_nonzero_ns=(
            first_nonzero_interval.end_ns
            if first_nonzero_interval is not None
            else None
        ),
        first_nonzero_delta=(
            first_nonzero_interval.delta
            if first_nonzero_interval is not None
            else None
        ),
        first_nonzero_value=(
            first_nonzero_interval.end_value
            if first_nonzero_interval is not None
            else None
        ),
        covered_duration_ns=covered_duration_ns,
        requested_duration_ns=requested_duration_ns,
        uncovered_requested_duration_ns=uncovered_duration_ns,
        coverage_fraction=coverage_fraction,
        sample_count=len(samples),
        interval_count=len(intervals),
        valid_interval_count=len(valid_intervals),
        invalid_interval_count=len(invalid_intervals),
        boundary_excluded_interval_count=boundary_excluded_count,
        intervals=tuple(selected_intervals),
        delta_formula=DELTA_FORMULA,
        interval_rate_formula=INTERVAL_RATE_FORMULA,
        mean_rate_formula=MEAN_RATE_FORMULA,
        aggregation_population=(
            f"{len(valid_intervals)} valid complete intervals; "
            f"{len(invalid_intervals)} excluded intervals"
        ),
    )


def _priorities_argument(
    priorities: Sequence[int], argument_name: str
) -> tuple[int, ...]:
    result = tuple(priorities)
    if not result or any(type(priority) is not int or priority < 0 for priority in result):
        raise ValueError(f"{argument_name} must contain nonnegative integers")
    if len(set(result)) != len(result):
        raise ValueError(f"{argument_name} must not contain duplicates")
    return result


def _expected_counter_names(
    pause_cycle_priorities: tuple[int, ...],
    pause_xoff_sent_priorities: tuple[int, ...],
    pause_received_priorities: tuple[int, ...],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *(f"hni_tx_paused_{priority}" for priority in pause_cycle_priorities),
                *(f"hni_rx_paused_{priority}" for priority in pause_cycle_priorities),
                "hni_pause_refresh",
                "hni_pause_sent",
                *(
                    f"hni_pause_xoff_sent_{priority}"
                    for priority in pause_xoff_sent_priorities
                ),
                *(
                    f"hni_pause_recv_{priority}"
                    for priority in pause_received_priorities
                ),
                "hni_fgfc_discard",
                "hni_fgfc_event_xoff",
            )
        )
    )


def _union_duration(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    total = 0
    current_start: int | None = None
    current_end: int | None = None
    for start, end in ordered:
        if current_start is None or current_end is None:
            current_start, current_end = start, end
            continue
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_start is not None and current_end is not None:
        total += current_end - current_start
    return total


def _overall_status(
    series_results: tuple[FlowControlSeriesResult, ...],
    missing_counter_names: tuple[str, ...],
) -> FlowControlStatus:
    if not series_results:
        return FlowControlStatus.MISSING_DATA

    statuses = [result.status for result in series_results]
    if all(status is FlowControlStatus.UNSUPPORTED_SERIES for status in statuses):
        return FlowControlStatus.UNSUPPORTED_SERIES
    if any(status is FlowControlStatus.UNSUPPORTED_SERIES for status in statuses):
        return FlowControlStatus.PARTIAL_UNSUPPORTED_SERIES
    if all(status is FlowControlStatus.MISSING_DATA for status in statuses):
        return FlowControlStatus.MISSING_DATA
    if any(status is FlowControlStatus.MISSING_DATA for status in statuses):
        return FlowControlStatus.PARTIAL_MISSING_DATA
    if all(
        status in (FlowControlStatus.INVALID_DATA, FlowControlStatus.PARTIAL_INVALID_DATA)
        for status in statuses
    ):
        return FlowControlStatus.INVALID_DATA
    if any(
        status in (FlowControlStatus.INVALID_DATA, FlowControlStatus.PARTIAL_INVALID_DATA)
        for status in statuses
    ):
        return FlowControlStatus.PARTIAL_INVALID_DATA
    if all(status is FlowControlStatus.INSUFFICIENT_DATA for status in statuses):
        return FlowControlStatus.INSUFFICIENT_DATA
    if any(status is FlowControlStatus.INSUFFICIENT_DATA for status in statuses):
        return FlowControlStatus.PARTIAL_INSUFFICIENT_DATA
    if missing_counter_names:
        return FlowControlStatus.PARTIAL_MISSING_DATA
    if all(status is FlowControlStatus.ZERO_OBSERVATIONS for status in statuses):
        return FlowControlStatus.ZERO_OBSERVATIONS
    if any(status is FlowControlStatus.ZERO_OBSERVATIONS for status in statuses):
        return FlowControlStatus.PARTIAL_ZERO_OBSERVATIONS
    return FlowControlStatus.OK


def analyze_flow_control(
    series: Iterable[MetricSeries],
    window_start_ns: int | None = None,
    window_end_ns: int | None = None,
    expected_counter_names: Sequence[str] | None = None,
    expected_pause_cycle_priorities: Sequence[int] = DEFAULT_FLOW_CONTROL_PRIORITIES,
    expected_pause_xoff_sent_priorities: Sequence[int] = DEFAULT_FLOW_CONTROL_PRIORITIES,
    expected_pause_received_priorities: Sequence[int] = DEFAULT_FLOW_CONTROL_PRIORITIES,
) -> FlowControlResult:
    """Analyze one NIC's flow-control counters without causal overclaims.

    All input series must come from the same run, node, NIC, and collector
    stream. This prevents duplicate observations of one physical NIC from being
    summed. The exact window policy includes only intervals whose endpoints are
    both inside the requested window.
    """

    input_series = tuple(series)
    if (window_start_ns is None) != (window_end_ns is None):
        raise ValueError("window_start_ns and window_end_ns must be supplied together")
    if window_start_ns is not None and window_start_ns >= window_end_ns:
        raise ValueError("The requested window must have a strictly positive duration")

    pause_cycle_priorities = _priorities_argument(
        expected_pause_cycle_priorities, "expected_pause_cycle_priorities"
    )
    pause_xoff_priorities = _priorities_argument(
        expected_pause_xoff_sent_priorities,
        "expected_pause_xoff_sent_priorities",
    )
    pause_received_priorities = _priorities_argument(
        expected_pause_received_priorities, "expected_pause_received_priorities"
    )
    if expected_counter_names is None:
        expected_counter_names = _expected_counter_names(
            pause_cycle_priorities,
            pause_xoff_priorities,
            pause_received_priorities,
        )
    else:
        expected_counter_names = tuple(expected_counter_names)
        if len(set(expected_counter_names)) != len(expected_counter_names):
            raise ValueError("expected_counter_names must be unique")
        if any(name not in _COUNTER_DEFINITIONS for name in expected_counter_names):
            raise ValueError(
                "expected_counter_names contains unknown flow-control counters"
            )

    identities = {
        (
            metric_series.identity.run_id,
            metric_series.identity.node_id,
            metric_series.identity.nic_id,
            metric_series.identity.collector_id,
        )
        for metric_series in input_series
    }
    if len(identities) > 1:
        raise ValueError(
            "Flow-control analysis requires one run, node, NIC, and collector stream"
        )
    if len({metric_series.identity.counter_id for metric_series in input_series}) != len(
        input_series
    ):
        raise ValueError("Each flow-control counter may be supplied only once")

    if input_series:
        identity = input_series[0].identity
        flow_identity = FlowControlIdentity(
            run_id=identity.run_id,
            node_id=identity.node_id,
            nic_id=identity.nic_id,
            collector_id=identity.collector_id,
        )
    else:
        flow_identity = None

    all_timestamps = [
        timestamp
        for metric_series in input_series
        for timestamp in metric_series.timestamps_ns
    ]
    requested_window_ns = (
        (window_start_ns, window_end_ns)
        if window_start_ns is not None
        else (
            (min(all_timestamps), max(all_timestamps))
            if all_timestamps
            else None
        )
    )

    series_results = tuple(
        _series_result(
            metric_series,
            _COUNTER_DEFINITIONS.get(metric_series.identity.counter_id, None),
            requested_window_ns,
        )
        for metric_series in input_series
    )
    series_results = tuple(
        sorted(
            series_results,
            key=lambda result: (
                result.metric_kind.value,
                result.direction.value if result.direction is not None else "",
                result.priority if result.priority is not None else -1,
                result.counter_name,
            ),
        )
    )

    observed_counter_names = tuple(
        dict.fromkeys(
            metric_series.identity.counter_id for metric_series in input_series
        )
    )
    missing_counter_names = tuple(
        name
        for name in expected_counter_names
        if name not in set(observed_counter_names)
    )
    unsupported_counter_names = tuple(
        result.counter_name
        for result in series_results
        if result.status is FlowControlStatus.UNSUPPORTED_SERIES
    )
    zero_observed_counter_names = tuple(
        result.counter_name
        for result in series_results
        if result.status is FlowControlStatus.ZERO_OBSERVATIONS
    )
    first_nonzero_counter_names = tuple(
        result.counter_name
        for result in series_results
        if result.first_nonzero_ns is not None
    )

    pause_cycle_results = tuple(
        result
        for result in series_results
        if result.metric_kind is FlowControlMetricKind.PAUSE_CYCLES
    )
    event_results = tuple(
        result
        for result in series_results
        if result.metric_kind
        not in (FlowControlMetricKind.PAUSE_CYCLES,)
    )
    xoff_event_results = tuple(
        result
        for result in event_results
        if result.metric_kind
        in (FlowControlMetricKind.PAUSE_XOFF_SENT, FlowControlMetricKind.FGFC_XOFF)
    )
    fgfc_event_results = tuple(
        result
        for result in event_results
        if result.metric_kind
        in (FlowControlMetricKind.FGFC_DISCARD, FlowControlMetricKind.FGFC_XOFF)
    )

    quality_flags: set[SeriesQualityFlag] = set()
    for result in series_results:
        quality_flags.update(result.quality_flags)
    if missing_counter_names:
        quality_flags.add(SeriesQualityFlag.MISSING)

    represented_window_values = [
        result.represented_window_ns
        for result in series_results
        if result.represented_window_ns is not None
    ]
    represented_window_ns = (
        (
            min(window[0] for window in represented_window_values),
            max(window[1] for window in represented_window_values),
        )
        if represented_window_values
        else None
    )
    covered_duration_ns = _union_duration(
        (
            interval.start_ns,
            interval.end_ns,
        )
        for result in series_results
        for interval in result.intervals
        if interval.valid
    )
    requested_duration_ns = (
        requested_window_ns[1] - requested_window_ns[0]
        if requested_window_ns is not None
        else 0
    )
    uncovered_duration_ns = max(
        0, requested_duration_ns - covered_duration_ns
    )
    coverage_fraction = (
        covered_duration_ns / requested_duration_ns
        if requested_duration_ns > 0
        else None
    )

    status = _overall_status(series_results, missing_counter_names)
    status_reasons: list[str] = []
    if missing_counter_names:
        status_reasons.append(
            "Expected flow-control counters or priorities are missing; missing is not zero."
        )
    if unsupported_counter_names:
        status_reasons.append(
            "Unsupported series were excluded from numerical results."
        )
    if any(
        result.status in (FlowControlStatus.INVALID_DATA, FlowControlStatus.PARTIAL_INVALID_DATA)
        for result in series_results
    ):
        status_reasons.append(
            "Reset, wrap, or conflicting-timestamp intervals were excluded."
        )
    if zero_observed_counter_names:
        status_reasons.append(
            "At least one available counter observed zero increments."
        )

    return FlowControlResult(
        contract_version=FLOW_CONTROL_ANALYSIS_VERSION,
        identity=flow_identity,
        status=status,
        quality_flags=frozenset(quality_flags),
        status_reasons=tuple(status_reasons),
        requested_window_ns=requested_window_ns,
        represented_window_ns=represented_window_ns,
        covered_duration_ns=covered_duration_ns,
        requested_duration_ns=requested_duration_ns,
        uncovered_requested_duration_ns=uncovered_duration_ns,
        coverage_fraction=coverage_fraction,
        series_results=series_results,
        pause_cycle_results=pause_cycle_results,
        event_results=event_results,
        xoff_event_results=xoff_event_results,
        fgfc_event_results=fgfc_event_results,
        observed_counter_names=observed_counter_names,
        missing_counter_names=missing_counter_names,
        unsupported_counter_names=unsupported_counter_names,
        zero_observed_counter_names=zero_observed_counter_names,
        first_nonzero_counter_names=first_nonzero_counter_names,
        limitations=FLOW_CONTROL_LIMITATIONS,
    )


def pause_cycle_seconds(
    result: FlowControlSeriesResult,
    clock: VerifiedPauseCycleClock | None = None,
) -> float | None:
    """Convert one pause-cycle total only with explicit verified evidence."""

    if clock is None:
        raise ValueError(
            "Pause-cycle time conversion requires a verified clock frequency."
        )
    if result.metric_kind is not FlowControlMetricKind.PAUSE_CYCLES:
        raise ValueError("Only pause-cycle counters can be converted to seconds.")
    if result.counter_name not in clock.applies_to_counter_names:
        raise ValueError(
            "The verified clock evidence does not apply to this pause-cycle counter."
        )
    if result.total is None:
        return None
    return float(result.total) / clock.frequency_hz
