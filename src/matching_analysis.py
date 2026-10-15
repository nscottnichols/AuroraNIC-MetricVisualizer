"""Pure receive-path match analysis for cumulative NIC counters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import groupby


MATCHING_ANALYSIS_VERSION = 1
UINT64_MAX = 2**64 - 1
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
NANOSECONDS_PER_SECOND = 1_000_000_000
CUMULATIVE_COUNTER_SEMANTICS = "cumulative counter"
PRIORITY_FRACTION_FORMULA = "priority_total / (priority_total + overflow_total)"
OVERFLOW_FRACTION_FORMULA = "overflow_total / (priority_total + overflow_total)"
DELTA_FORMULA = "end_value - start_value"

MATCHING_INTERPRETATION = (
    "Priority and overflow matches are receive-path evidence. They do not "
    "measure copy time or establish exact MPI message/receive semantics."
)

MATCHING_LIMITATIONS = (
    "Priority and overflow subcounters are summed only after explicit "
    "population compatibility is supplied and exact interval boundaries match.",
    "A zero priority/overflow denominator is undefined, not a zero or one fraction.",
    "Missing, reset, baseline-relative, and non-cumulative counters remain "
    "unsupported or partial rather than being treated as zero.",
    "Matching evidence suggests receive handling; it does not prove copy cost "
    "or a causal network bottleneck.",
)


class ValueKind(str, Enum):
    CUMULATIVE = "cumulative"
    DELTA = "delta"
    GAUGE = "gauge"
    UNKNOWN = "unknown"


class MatchingRole(str, Enum):
    PRIORITY = "priority"
    OVERFLOW = "overflow"
    UNKNOWN = "unknown"


class MatchingStatus(str, Enum):
    OBSERVED_ZERO = "observed_zero"
    OBSERVED_NONZERO = "observed_nonzero"
    PARTIAL_MISSING_DATA = "partial_missing_data"
    PARTIAL_INVALID_DATA = "partial_invalid_data"
    PARTIAL_UNSUPPORTED_DATA = "partial_unsupported_data"
    MISSING_DATA = "missing_data"
    INSUFFICIENT_DATA = "insufficient_data"
    UNSUPPORTED = "unsupported"


class MatchingQualityFlag(str, Enum):
    MISSING_COUNTER = "missing_counter"
    UNSUPPORTED_SERIES = "unsupported_series"
    BASELINE_RELATIVE = "baseline_relative"
    DECREASE_OR_RESET = "decrease_or_reset"
    REORDERED = "reordered"
    DUPLICATE = "duplicate"
    CONFLICTING_TIMESTAMP = "conflicting_timestamp"
    PARTIAL_WINDOW = "partial_window"
    BOUNDARY_INTERVAL_EXCLUDED = "boundary_interval_excluded"
    INCOMPATIBLE_UNIT = "incompatible_unit"
    INCOMPATIBLE_POPULATION = "incompatible_population"
    INCOMPATIBLE_COVERAGE = "incompatible_coverage"


class IntervalState(str, Enum):
    VALID = "valid"
    DECREASE_OR_RESET = "decrease_or_reset"
    CONFLICTING_TIMESTAMP = "conflicting_timestamp"


@dataclass(frozen=True)
class MatchingPopulation:
    run_id: str
    node_id: str
    nic_id: str
    collector_id: str


@dataclass(frozen=True)
class MatchingCounterIdentity:
    run_id: str
    node_id: str
    nic_id: str
    collector_id: str
    source_counter_id: str


@dataclass(frozen=True)
class MatchingCounterSpec:
    counter_id: str
    role: MatchingRole
    priority: int
    unit: str
    population_id: str | None = None

    def __post_init__(self) -> None:
        if not self.counter_id:
            raise ValueError("counter_id must be non-empty")
        if type(self.priority) is not int or self.priority < 0:
            raise ValueError("priority must be a non-negative integer")
        if not self.unit:
            raise ValueError("counter unit must be non-empty")


@dataclass(frozen=True)
class MatchingCounterSeries:
    spec: MatchingCounterSpec
    identity: MatchingCounterIdentity
    timestamps_ns: tuple[int, ...]
    values: tuple[int, ...]
    value_kind: ValueKind
    semantics: str
    baseline_relative: bool
    quality_flags: tuple[MatchingQualityFlag, ...] = ()

    def __post_init__(self) -> None:
        if len(self.timestamps_ns) != len(self.values):
            raise ValueError("timestamps_ns and values must have equal length")
        if not self.identity.source_counter_id:
            raise ValueError("source_counter_id must be non-empty")
        if not isinstance(self.baseline_relative, bool):
            raise ValueError("baseline_relative must be boolean")

        for field_name in ("run_id", "node_id", "nic_id", "collector_id"):
            if not getattr(self.identity, field_name):
                raise ValueError(f"identity {field_name} must be non-empty")

        for timestamp in self.timestamps_ns:
            if type(timestamp) is not int or not INT64_MIN <= timestamp <= INT64_MAX:
                raise ValueError("every timestamp must be an int64 integer")

        for value in self.values:
            if type(value) is not int or not 0 <= value <= UINT64_MAX:
                raise ValueError("counter values must be uint64 integers")


@dataclass(frozen=True)
class MatchingDataset:
    population: MatchingPopulation
    requested_counters: tuple[MatchingCounterSpec, ...]
    series: tuple[MatchingCounterSeries, ...]
    window_ns: tuple[int, int]

    def __post_init__(self) -> None:
        requested_ids = [spec.counter_id for spec in self.requested_counters]
        if len(requested_ids) != len(set(requested_ids)):
            raise ValueError("requested counter IDs must be unique")
        role_priorities = [
            (spec.role, spec.priority) for spec in self.requested_counters
        ]
        if len(role_priorities) != len(set(role_priorities)):
            raise ValueError("each role and priority may be requested only once")
        series_ids = [series.spec.counter_id for series in self.series]
        if len(series_ids) != len(set(series_ids)):
            raise ValueError("each counter may have only one observation series")
        if len(self.window_ns) != 2:
            raise ValueError("window_ns must contain start and end")
        if type(self.window_ns[0]) is not int or type(self.window_ns[1]) is not int:
            raise ValueError("window bounds must be integers")
        if not self.window_ns[0] < self.window_ns[1]:
            raise ValueError("window end must be greater than window start")

        for observation in self.series:
            if observation.spec not in self.requested_counters:
                raise ValueError(
                    f"series counter {observation.spec.counter_id!r} was not requested"
                )
            identity = observation.identity
            population = self.population
            if (
                identity.run_id != population.run_id
                or identity.node_id != population.node_id
                or identity.nic_id != population.nic_id
                or identity.collector_id != population.collector_id
            ):
                raise ValueError("series identity must match the dataset population")


@dataclass(frozen=True)
class MatchingInterval:
    start_ns: int
    end_ns: int
    duration_ns: int
    start_value: int
    end_value: int
    delta: int
    valid: bool
    state: IntervalState
    inside_window: bool


@dataclass(frozen=True)
class MatchingCounterResult:
    spec: MatchingCounterSpec
    source_identity: MatchingCounterIdentity | None
    status: MatchingStatus
    quality_flags: frozenset[MatchingQualityFlag]
    status_reasons: tuple[str, ...]
    requested_window_ns: tuple[int, int]
    represented_window_ns: tuple[int, int] | None
    covered_duration_ns: int
    requested_duration_ns: int
    coverage_fraction: float | None
    sample_count: int
    valid_interval_count: int
    invalid_interval_count: int
    outside_window_interval_count: int
    total: int | None
    unit: str
    first_nonzero_sample_ns: int | None
    first_nonzero_increment_ns: int | None
    zero_observed: bool
    persistent_zero: bool
    intervals: tuple[MatchingInterval, ...]


@dataclass(frozen=True)
class MatchingPriorityResult:
    priority: int
    priority_counter_id: str | None
    overflow_counter_id: str | None
    priority_total: int | None
    overflow_total: int | None
    total: int | None
    priority_fraction: float | None
    overflow_fraction: float | None
    status: MatchingStatus
    quality_flags: frozenset[MatchingQualityFlag]
    status_reasons: tuple[str, ...]


@dataclass(frozen=True)
class MatchingTemporalInterval:
    start_ns: int
    end_ns: int
    duration_ns: int
    priority_delta: int
    overflow_delta: int
    total_delta: int
    priority_fraction: float | None
    valid: bool
    state: IntervalState


@dataclass(frozen=True)
class MatchingAnalysis:
    contract_version: int
    population: MatchingPopulation
    status: MatchingStatus
    quality_flags: frozenset[MatchingQualityFlag]
    status_reasons: tuple[str, ...]
    requested_window_ns: tuple[int, int]
    represented_window_ns: tuple[int, int] | None
    covered_duration_ns: int | None
    requested_duration_ns: int
    coverage_fraction: float | None
    counter_results: tuple[MatchingCounterResult, ...]
    priority_results: tuple[MatchingPriorityResult, ...]
    priority_total: int | None
    overflow_total: int | None
    total: int | None
    priority_fraction: float | None
    overflow_fraction: float | None
    population_compatible: bool
    compatible_counter_ids: tuple[str, ...]
    incompatible_counter_ids: tuple[str, ...]
    missing_counter_ids: tuple[str, ...]
    temporal_intervals: tuple[MatchingTemporalInterval, ...]
    priority_fraction_formula: str
    overflow_fraction_formula: str
    delta_formula: str
    aggregation_population: str
    interpretation: str
    limitations: tuple[str, ...]


def _missing_result(
    spec: MatchingCounterSpec,
    window_ns: tuple[int, int],
) -> MatchingCounterResult:
    requested_duration_ns = window_ns[1] - window_ns[0]
    return MatchingCounterResult(
        spec=spec,
        source_identity=None,
        status=MatchingStatus.MISSING_DATA,
        quality_flags=frozenset({MatchingQualityFlag.MISSING_COUNTER}),
        status_reasons=("The requested counter has no observation series.",),
        requested_window_ns=window_ns,
        represented_window_ns=None,
        covered_duration_ns=0,
        requested_duration_ns=requested_duration_ns,
        coverage_fraction=0.0,
        sample_count=0,
        valid_interval_count=0,
        invalid_interval_count=0,
        outside_window_interval_count=0,
        total=None,
        unit=spec.unit,
        first_nonzero_sample_ns=None,
        first_nonzero_increment_ns=None,
        zero_observed=False,
        persistent_zero=False,
        intervals=(),
    )


def _unsupported_result(
    series: MatchingCounterSeries,
    window_ns: tuple[int, int],
    quality_flags: set[MatchingQualityFlag],
    reasons: list[str],
) -> MatchingCounterResult:
    requested_duration_ns = window_ns[1] - window_ns[0]
    window_sample_count = sum(
        1
        for timestamp in series.timestamps_ns
        if window_ns[0] <= timestamp <= window_ns[1]
    )
    return MatchingCounterResult(
        spec=series.spec,
        source_identity=series.identity,
        status=MatchingStatus.UNSUPPORTED,
        quality_flags=frozenset(quality_flags | set(series.quality_flags)),
        status_reasons=tuple(reasons),
        requested_window_ns=window_ns,
        represented_window_ns=None,
        covered_duration_ns=0,
        requested_duration_ns=requested_duration_ns,
        coverage_fraction=0.0,
        sample_count=window_sample_count,
        valid_interval_count=0,
        invalid_interval_count=0,
        outside_window_interval_count=0,
        total=None,
        unit=series.spec.unit,
        first_nonzero_sample_ns=None,
        first_nonzero_increment_ns=None,
        zero_observed=False,
        persistent_zero=False,
        intervals=(),
    )


def _analyze_series(
    series: MatchingCounterSeries,
    window_ns: tuple[int, int],
) -> MatchingCounterResult:
    window_sample_count = sum(
        1
        for timestamp in series.timestamps_ns
        if window_ns[0] <= timestamp <= window_ns[1]
    )
    quality_flags: set[MatchingQualityFlag] = set(series.quality_flags)
    reasons: list[str] = []

    if series.spec.role is MatchingRole.UNKNOWN:
        quality_flags.add(MatchingQualityFlag.UNSUPPORTED_SERIES)
        reasons.append("The counter role is unknown.")
    if series.baseline_relative:
        quality_flags.add(MatchingQualityFlag.BASELINE_RELATIVE)
        reasons.append(
            "Baseline-relative pftrace change-only samples do not preserve "
            "exact cumulative-counter intervals."
        )
    if series.value_kind is not ValueKind.CUMULATIVE:
        quality_flags.add(MatchingQualityFlag.UNSUPPORTED_SERIES)
        reasons.append(f"Unsupported value kind: {series.value_kind.value}.")
    if series.semantics != CUMULATIVE_COUNTER_SEMANTICS:
        quality_flags.add(MatchingQualityFlag.UNSUPPORTED_SERIES)
        reasons.append(f"Unsupported counter semantics: {series.semantics!r}.")
    if reasons:
        return _unsupported_result(series, window_ns, quality_flags, reasons)

    records = list(zip(series.timestamps_ns, series.values, strict=True))
    ordered = sorted(records, key=lambda record: record[0])
    if records != ordered:
        quality_flags.add(MatchingQualityFlag.REORDERED)

    samples: list[tuple[int, int]] = []
    intervals: list[MatchingInterval] = []
    valid_intervals: list[MatchingInterval] = []
    invalid_interval_count = 0
    outside_window_interval_count = 0

    for timestamp, grouped_records in groupby(ordered, key=lambda record: record[0]):
        values = [value for _, value in grouped_records]
        if len(set(values)) != 1:
            first_value, second_value = values[:2]
            inside_window = window_ns[0] <= timestamp <= window_ns[1]
            interval = MatchingInterval(
                start_ns=timestamp,
                end_ns=timestamp,
                duration_ns=0,
                start_value=first_value,
                end_value=second_value,
                delta=0,
                valid=False,
                state=IntervalState.CONFLICTING_TIMESTAMP,
                inside_window=inside_window,
            )
            intervals.append(interval)
            quality_flags.add(MatchingQualityFlag.CONFLICTING_TIMESTAMP)
            if inside_window:
                invalid_interval_count += 1
            else:
                outside_window_interval_count += 1
            continue

        sample = (timestamp, values[0])
        if samples and samples[-1] == sample:
            quality_flags.add(MatchingQualityFlag.DUPLICATE)
        samples.append(sample)
        if len(values) > 1:
            quality_flags.add(MatchingQualityFlag.DUPLICATE)

    for (start_ns, start_value), (end_ns, end_value) in zip(
        samples, samples[1:], strict=False
    ):
        delta = end_value - start_value
        state = (
            IntervalState.VALID if delta >= 0 else IntervalState.DECREASE_OR_RESET
        )
        inside_window = window_ns[0] <= start_ns and end_ns <= window_ns[1]
        interval = MatchingInterval(
            start_ns=start_ns,
            end_ns=end_ns,
            duration_ns=end_ns - start_ns,
            start_value=start_value,
            end_value=end_value,
            delta=delta,
            valid=delta >= 0,
            state=state,
            inside_window=inside_window,
        )
        intervals.append(interval)
        if not inside_window:
            outside_window_interval_count += 1
            if start_ns < window_ns[0] < end_ns or start_ns < window_ns[1] < end_ns:
                quality_flags.add(MatchingQualityFlag.BOUNDARY_INTERVAL_EXCLUDED)
        elif delta >= 0:
            valid_intervals.append(interval)
        else:
            invalid_interval_count += 1
            quality_flags.add(MatchingQualityFlag.DECREASE_OR_RESET)

    selected_intervals = tuple(
        interval for interval in intervals if interval.inside_window
    )
    covered_duration_ns = sum(interval.duration_ns for interval in valid_intervals)
    requested_duration_ns = window_ns[1] - window_ns[0]
    total = sum(interval.delta for interval in valid_intervals)
    window_samples = tuple(
        (timestamp, value)
        for timestamp, value in samples
        if window_ns[0] <= timestamp <= window_ns[1]
    )
    first_nonzero_sample_ns = next(
        (
            timestamp
            for timestamp, value in window_samples
            if value != 0
        ),
        None,
    )
    first_nonzero_increment_ns = next(
        (interval.end_ns for interval in valid_intervals if interval.delta > 0),
        None,
    )
    all_samples_zero = bool(window_samples) and all(
        value == 0 for _, value in window_samples
    )
    all_valid_deltas_zero = bool(valid_intervals) and all(
        interval.delta == 0 for interval in valid_intervals
    )
    persistent_zero = (
        all_samples_zero
        and all_valid_deltas_zero
        and invalid_interval_count == 0
    )
    if covered_duration_ns < requested_duration_ns:
        quality_flags.add(MatchingQualityFlag.PARTIAL_WINDOW)

    if not valid_intervals:
        if invalid_interval_count:
            status = MatchingStatus.PARTIAL_INVALID_DATA
            reasons.append("The window contains no complete valid counter interval.")
        else:
            status = MatchingStatus.INSUFFICIENT_DATA
            reasons.append(
                "At least two distinct in-window timestamps are required."
            )
    elif invalid_interval_count:
        status = MatchingStatus.PARTIAL_INVALID_DATA
        reasons.append("Valid results exclude one or more decreases or resets.")
    elif total == 0:
        status = MatchingStatus.OBSERVED_ZERO
    else:
        status = MatchingStatus.OBSERVED_NONZERO

    represented_window_ns = (
        (
            min(interval.start_ns for interval in valid_intervals),
            max(interval.end_ns for interval in valid_intervals),
        )
        if valid_intervals
        else None
    )

    return MatchingCounterResult(
        spec=series.spec,
        source_identity=series.identity,
        status=status,
        quality_flags=frozenset(quality_flags),
        status_reasons=tuple(reasons),
        requested_window_ns=window_ns,
        represented_window_ns=represented_window_ns,
        covered_duration_ns=covered_duration_ns,
        requested_duration_ns=requested_duration_ns,
        coverage_fraction=(
            covered_duration_ns / requested_duration_ns
            if requested_duration_ns > 0
            else None
        ),
        sample_count=window_sample_count,
        valid_interval_count=len(valid_intervals),
        invalid_interval_count=invalid_interval_count,
        outside_window_interval_count=outside_window_interval_count,
        total=total,
        unit=series.spec.unit,
        first_nonzero_sample_ns=first_nonzero_sample_ns,
        first_nonzero_increment_ns=first_nonzero_increment_ns,
        zero_observed=all_samples_zero,
        persistent_zero=persistent_zero,
        intervals=selected_intervals,
    )


def _interval_boundaries(
    result: MatchingCounterResult,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (interval.start_ns, interval.end_ns)
        for interval in result.intervals
        if interval.valid
    )


def _results_for_role(
    results: tuple[MatchingCounterResult, ...], role: MatchingRole
) -> tuple[MatchingCounterResult, ...]:
    return tuple(result for result in results if result.spec.role is role)


def _population_is_compatible(
    results: tuple[MatchingCounterResult, ...],
) -> bool:
    if not results:
        return False
    observed_statuses = {
        MatchingStatus.OBSERVED_ZERO,
        MatchingStatus.OBSERVED_NONZERO,
    }
    if any(result.status not in observed_statuses for result in results):
        return False
    units = {result.spec.unit for result in results}
    population_ids = {result.spec.population_id for result in results}
    if len(units) != 1 or len(population_ids) != 1:
        return False
    if None in population_ids:
        return False
    boundaries = [_interval_boundaries(result) for result in results]
    return bool(boundaries[0]) and all(
        boundary == boundaries[0] for boundary in boundaries
    )


def _aggregate_role(
    results: tuple[MatchingCounterResult, ...],
) -> int | None:
    if not _population_is_compatible(results):
        return None
    return sum(result.total or 0 for result in results)


def _priority_result(
    priority: int,
    priority_result: MatchingCounterResult | None,
    overflow_result: MatchingCounterResult | None,
) -> MatchingPriorityResult:
    quality_flags: set[MatchingQualityFlag] = set()
    reasons: list[str] = []
    priority_total = priority_result.total if priority_result is not None else None
    overflow_total = overflow_result.total if overflow_result is not None else None

    if priority_result is None:
        reasons.append("The requested priority counter is missing.")
        quality_flags.add(MatchingQualityFlag.MISSING_COUNTER)
    if overflow_result is None:
        reasons.append("The requested overflow counter is missing.")
        quality_flags.add(MatchingQualityFlag.MISSING_COUNTER)
    if priority_result is not None:
        quality_flags.update(priority_result.quality_flags)
    if overflow_result is not None:
        quality_flags.update(overflow_result.quality_flags)

    compatible = bool(priority_result and overflow_result) and _population_is_compatible(
        (priority_result, overflow_result)
    )
    if priority_result is not None and overflow_result is not None and not compatible:
        if priority_result.spec.unit != overflow_result.spec.unit:
            quality_flags.add(MatchingQualityFlag.INCOMPATIBLE_UNIT)
            reasons.append("Priority and overflow units are incompatible.")
        if (
            priority_result.spec.population_id is None
            or priority_result.spec.population_id != overflow_result.spec.population_id
        ):
            quality_flags.add(MatchingQualityFlag.INCOMPATIBLE_POPULATION)
            reasons.append("Priority and overflow population IDs are incompatible.")
        if _interval_boundaries(priority_result) != _interval_boundaries(
            overflow_result
        ):
            quality_flags.add(MatchingQualityFlag.INCOMPATIBLE_COVERAGE)
            reasons.append("Priority and overflow interval coverage is incompatible.")
        priority_total = None
        overflow_total = None

    total = (
        priority_total + overflow_total
        if priority_total is not None and overflow_total is not None
        else None
    )
    priority_fraction = (
        priority_total / total if total is not None and total > 0 else None
    )
    overflow_fraction = (
        overflow_total / total if total is not None and total > 0 else None
    )
    if total == 0:
        reasons.append("The priority/overflow denominator is zero and undefined.")
        status = MatchingStatus.OBSERVED_ZERO
    elif total is None:
        status = (
            MatchingStatus.UNSUPPORTED
            if priority_result is not None and overflow_result is not None
            else MatchingStatus.PARTIAL_MISSING_DATA
        )
    else:
        status = MatchingStatus.OBSERVED_NONZERO

    return MatchingPriorityResult(
        priority=priority,
        priority_counter_id=(
            priority_result.spec.counter_id if priority_result is not None else None
        ),
        overflow_counter_id=(
            overflow_result.spec.counter_id if overflow_result is not None else None
        ),
        priority_total=priority_total,
        overflow_total=overflow_total,
        total=total,
        priority_fraction=priority_fraction,
        overflow_fraction=overflow_fraction,
        status=status,
        quality_flags=frozenset(quality_flags),
        status_reasons=tuple(reasons),
    )


def _temporal_intervals(
    results: tuple[MatchingCounterResult, ...],
) -> tuple[MatchingTemporalInterval, ...]:
    priority_results = _results_for_role(results, MatchingRole.PRIORITY)
    overflow_results = _results_for_role(results, MatchingRole.OVERFLOW)
    if not _population_is_compatible(priority_results + overflow_results):
        return ()

    boundaries = _interval_boundaries(priority_results[0])
    priority_intervals_by_boundary = {
        (interval.start_ns, interval.end_ns): []
        for result in priority_results
        for interval in result.intervals
        if interval.valid
    }
    for result in priority_results:
        for interval in result.intervals:
            if interval.valid:
                priority_intervals_by_boundary[
                    (interval.start_ns, interval.end_ns)
                ].append(interval.delta)
    overflow_intervals_by_boundary = {
        (interval.start_ns, interval.end_ns): []
        for result in overflow_results
        for interval in result.intervals
        if interval.valid
    }
    for result in overflow_results:
        for interval in result.intervals:
            if interval.valid:
                overflow_intervals_by_boundary[
                    (interval.start_ns, interval.end_ns)
                ].append(interval.delta)
    temporal_intervals: list[MatchingTemporalInterval] = []
    for start_ns, end_ns in boundaries:
        priority_delta = sum(priority_intervals_by_boundary[(start_ns, end_ns)])
        overflow_delta = sum(overflow_intervals_by_boundary[(start_ns, end_ns)])
        total_delta = priority_delta + overflow_delta
        temporal_intervals.append(
            MatchingTemporalInterval(
                start_ns=start_ns,
                end_ns=end_ns,
                duration_ns=end_ns - start_ns,
                priority_delta=priority_delta,
                overflow_delta=overflow_delta,
                total_delta=total_delta,
                priority_fraction=(
                    priority_delta / total_delta if total_delta > 0 else None
                ),
                valid=True,
                state=IntervalState.VALID,
            )
        )
    return tuple(temporal_intervals)


def analyze_matching(dataset: MatchingDataset) -> MatchingAnalysis:
    """Analyze cumulative matching evidence without unsupported causal claims."""

    series_by_counter = {
        series.spec.counter_id: series for series in dataset.series
    }
    results = tuple(
        _analyze_series(series_by_counter[spec.counter_id], dataset.window_ns)
        if spec.counter_id in series_by_counter
        else _missing_result(spec, dataset.window_ns)
        for spec in dataset.requested_counters
    )

    priorities = sorted(
        {
            spec.priority
            for spec in dataset.requested_counters
            if spec.role in {MatchingRole.PRIORITY, MatchingRole.OVERFLOW}
        }
    )
    priority_results = tuple(
        _priority_result(
            priority,
            next(
                (
                    result
                    for result in results
                    if result.spec.role is MatchingRole.PRIORITY
                    and result.spec.priority == priority
                ),
                None,
            ),
            next(
                (
                    result
                    for result in results
                    if result.spec.role is MatchingRole.OVERFLOW
                    and result.spec.priority == priority
                ),
                None,
            ),
        )
        for priority in priorities
    )

    priority_counter_results = _results_for_role(results, MatchingRole.PRIORITY)
    overflow_counter_results = _results_for_role(results, MatchingRole.OVERFLOW)
    population_compatible = _population_is_compatible(
        priority_counter_results + overflow_counter_results
    )
    priority_total = (
        _aggregate_role(priority_counter_results) if population_compatible else None
    )
    overflow_total = (
        _aggregate_role(overflow_counter_results) if population_compatible else None
    )
    total = (
        priority_total + overflow_total
        if priority_total is not None and overflow_total is not None
        else None
    )
    priority_fraction = (
        priority_total / total if total is not None and total > 0 else None
    )
    overflow_fraction = (
        overflow_total / total if total is not None and total > 0 else None
    )

    quality_flags: set[MatchingQualityFlag] = set()
    status_reasons: list[str] = []
    for result in results:
        quality_flags.update(result.quality_flags)

    missing_counter_ids = tuple(
        result.spec.counter_id
        for result in results
        if result.status is MatchingStatus.MISSING_DATA
    )
    unsupported_counter_results = tuple(
        result for result in results if result.status is MatchingStatus.UNSUPPORTED
    )
    incompatible_counter_ids = tuple(
        result.spec.counter_id
        for result in results
        if result.status
        in {
            MatchingStatus.OBSERVED_ZERO,
            MatchingStatus.OBSERVED_NONZERO,
        }
        and not _population_is_compatible((result,))
    )
    observed_statuses = {
        MatchingStatus.OBSERVED_ZERO,
        MatchingStatus.OBSERVED_NONZERO,
    }
    compatible_counter_ids = tuple(
        result.spec.counter_id
        for result in results
        if result.status in observed_statuses
        and _population_is_compatible((result,))
    )

    if not priority_counter_results or not overflow_counter_results:
        status = MatchingStatus.UNSUPPORTED
        status_reasons.append(
            "Both priority and overflow counter roles must be requested."
        )
    elif all(result.status is MatchingStatus.MISSING_DATA for result in results):
        status = MatchingStatus.MISSING_DATA
    elif missing_counter_ids:
        status = MatchingStatus.PARTIAL_MISSING_DATA
        status_reasons.append("At least one requested matching counter is missing.")
    elif unsupported_counter_results:
        if len(unsupported_counter_results) == len(results):
            status = MatchingStatus.UNSUPPORTED
        else:
            status = MatchingStatus.PARTIAL_UNSUPPORTED_DATA
        status_reasons.append("At least one matching counter has unsupported data.")
    elif any(
        result.status is MatchingStatus.PARTIAL_INVALID_DATA for result in results
    ):
        status = MatchingStatus.PARTIAL_INVALID_DATA
        status_reasons.append(
            "Valid intervals are reported after excluding decreases or resets."
        )
    elif not any(result.valid_interval_count for result in results):
        status = MatchingStatus.INSUFFICIENT_DATA
        status_reasons.append("No matching counter has a complete valid interval.")
    elif not population_compatible:
        status = MatchingStatus.UNSUPPORTED
        status_reasons.append(
            "Priority/overflow totals and fractions require one compatible unit, "
            "population ID, and exact interval-boundary set."
        )
    elif total == 0:
        status = MatchingStatus.OBSERVED_ZERO
        status_reasons.append(
            "The priority/overflow denominator is zero and is undefined."
        )
    else:
        status = MatchingStatus.OBSERVED_NONZERO

    if priority_counter_results and overflow_counter_results and not population_compatible:
        if len({result.spec.unit for result in results}) > 1:
            quality_flags.add(MatchingQualityFlag.INCOMPATIBLE_UNIT)
        if len({result.spec.population_id for result in results}) > 1:
            quality_flags.add(MatchingQualityFlag.INCOMPATIBLE_POPULATION)
        boundaries = [_interval_boundaries(result) for result in results]
        if any(boundary != boundaries[0] for boundary in boundaries[1:]):
            quality_flags.add(MatchingQualityFlag.INCOMPATIBLE_COVERAGE)

    represented_bounds = [
        result.represented_window_ns
        for result in results
        if result.represented_window_ns is not None
    ]
    represented_window_ns = (
        (
            min(start for start, _ in represented_bounds),
            max(end for _, end in represented_bounds),
        )
        if represented_bounds
        else None
    )
    boundaries = _interval_boundaries(results[0]) if results else ()
    if population_compatible and boundaries:
        covered_duration_ns = sum(end_ns - start_ns for start_ns, end_ns in boundaries)
    else:
        covered_duration_ns = None

    return MatchingAnalysis(
        contract_version=MATCHING_ANALYSIS_VERSION,
        population=dataset.population,
        status=status,
        quality_flags=frozenset(quality_flags),
        status_reasons=tuple(status_reasons),
        requested_window_ns=dataset.window_ns,
        represented_window_ns=represented_window_ns,
        covered_duration_ns=covered_duration_ns,
        requested_duration_ns=dataset.window_ns[1] - dataset.window_ns[0],
        coverage_fraction=(
            covered_duration_ns / (dataset.window_ns[1] - dataset.window_ns[0])
            if covered_duration_ns is not None
            else None
        ),
        counter_results=results,
        priority_results=priority_results,
        priority_total=priority_total,
        overflow_total=overflow_total,
        total=total,
        priority_fraction=priority_fraction,
        overflow_fraction=overflow_fraction,
        population_compatible=population_compatible,
        compatible_counter_ids=compatible_counter_ids,
        incompatible_counter_ids=incompatible_counter_ids,
        missing_counter_ids=missing_counter_ids,
        temporal_intervals=_temporal_intervals(results),
        priority_fraction_formula=PRIORITY_FRACTION_FORMULA,
        overflow_fraction_formula=OVERFLOW_FRACTION_FORMULA,
        delta_formula=DELTA_FORMULA,
        aggregation_population=(
            "Requested priority and overflow counters with one explicit "
            "compatible population, one unit, and one exact interval-boundary set."
        ),
        interpretation=MATCHING_INTERPRETATION,
        limitations=MATCHING_LIMITATIONS,
    )
