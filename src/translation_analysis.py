"""Pure address-translation analysis for cumulative NIC counters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import groupby
import math


TRANSLATION_ANALYSIS_VERSION = 1
UINT64_MAX = 2**64 - 1
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
CUMULATIVE_COUNTER_SEMANTICS = "cumulative counter"
DELTA_FORMULA = "end_value - start_value"
HIT_RATIO_FORMULA = "hit_total / (hit_total + miss_total)"
MISSES_PER_REQUEST_FORMULA = "miss_total / request_total"
MISSES_PER_GB_FORMULA = "miss_total / traffic_total_gb"
TRAFFIC_BYTES_UNIT = "bytes"
TRAFFIC_GB_UNIT = "GB"
BYTES_PER_GB = 1_000_000_000

TRANSLATION_LIMITATIONS = (
    "Pool-view and client-view misses are reported separately and are never summed.",
    "Hit ratios require explicitly compatible hit, miss, unit, population, and interval definitions.",
    "A zero request or traffic denominator is unavailable, not zero.",
    "Traffic normalization specifies TX, RX, or both and must cover the same translation population and window.",
    "The supplied inventory does not identify an ATU-latency histogram; request/response latency is not substituted.",
)


class ValueKind(str, Enum):
    CUMULATIVE = "cumulative"
    DELTA = "delta"
    GAUGE = "gauge"
    UNKNOWN = "unknown"


class TranslationCounterRole(str, Enum):
    REQUEST = "request"
    HIT = "hit"
    MISS = "miss"
    EVICTION = "eviction"


class TranslationMissView(str, Enum):
    POOL = "pool"
    CLIENT = "client"


class TranslationStatus(str, Enum):
    OBSERVED_ZERO = "observed_zero"
    OBSERVED_NONZERO = "observed_nonzero"
    PARTIAL_MISSING_DATA = "partial_missing_data"
    PARTIAL_INVALID_DATA = "partial_invalid_data"
    PARTIAL_UNSUPPORTED_DATA = "partial_unsupported_data"
    MISSING_DATA = "missing_data"
    INSUFFICIENT_DATA = "insufficient_data"
    UNSUPPORTED = "unsupported"


class TranslationQualityFlag(str, Enum):
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
    INCOMPATIBLE_TRAFFIC_UNIT = "incompatible_traffic_unit"
    INCOMPATIBLE_TRAFFIC_POPULATION = "incompatible_traffic_population"
    INCOMPATIBLE_TRAFFIC_WINDOW = "incompatible_traffic_window"
    ZERO_DENOMINATOR = "zero_denominator"


class IntervalState(str, Enum):
    VALID = "valid"
    DECREASE_OR_RESET = "decrease_or_reset"
    CONFLICTING_TIMESTAMP = "conflicting_timestamp"


class TrafficDirection(str, Enum):
    TX = "tx"
    RX = "rx"
    BOTH = "both"


class TrafficStatus(str, Enum):
    NOT_SUPPLIED = "not_supplied"
    COMPATIBLE = "compatible"
    ZERO_DENOMINATOR = "zero_denominator"
    INCOMPATIBLE_UNIT = "incompatible_unit"
    INCOMPATIBLE_POPULATION = "incompatible_population"
    INCOMPATIBLE_WINDOW = "incompatible_window"


@dataclass(frozen=True)
class TranslationPopulation:
    run_id: str
    node_id: str
    nic_id: str
    collector_id: str


@dataclass(frozen=True)
class TranslationCounterIdentity:
    run_id: str
    node_id: str
    nic_id: str
    collector_id: str
    source_counter_id: str


@dataclass(frozen=True)
class TranslationCounterSpec:
    counter_id: str
    role: TranslationCounterRole
    unit: str
    population_id: str | None = None
    miss_view: TranslationMissView | None = None

    def __post_init__(self) -> None:
        if not self.counter_id:
            raise ValueError("counter_id must be non-empty")
        if not self.unit:
            raise ValueError("counter unit must be non-empty")
        if self.role is TranslationCounterRole.MISS and self.miss_view is None:
            raise ValueError("miss counters require a miss_view")
        if self.role is not TranslationCounterRole.MISS and self.miss_view is not None:
            raise ValueError("only miss counters may define a miss_view")


@dataclass(frozen=True)
class TranslationCounterSeries:
    spec: TranslationCounterSpec
    identity: TranslationCounterIdentity
    timestamps_ns: tuple[int, ...]
    values: tuple[int | float, ...]
    value_kind: ValueKind
    semantics: str
    baseline_relative: bool
    quality_flags: tuple[TranslationQualityFlag, ...] = ()

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
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
            ):
                raise ValueError("counter values must be finite numbers")


@dataclass(frozen=True)
class TranslationTrafficTotal:
    total: int | float
    unit: str
    direction: TrafficDirection
    population_id: str
    window_ns: tuple[int, int]
    source_counter_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.total) not in (int, float) or isinstance(self.total, bool):
            raise ValueError("traffic total must be numeric")
        if not math.isfinite(self.total) or self.total < 0:
            raise ValueError("traffic total must be finite and non-negative")
        if not self.unit:
            raise ValueError("traffic unit must be non-empty")
        if not self.population_id:
            raise ValueError("traffic population_id must be non-empty")
        if len(self.window_ns) != 2:
            raise ValueError("window_ns must contain start and end")
        if type(self.window_ns[0]) is not int or type(self.window_ns[1]) is not int:
            raise ValueError("traffic window bounds must be integers")
        if not self.window_ns[0] < self.window_ns[1]:
            raise ValueError("traffic window end must be greater than start")


@dataclass(frozen=True)
class TranslationDataset:
    population: TranslationPopulation
    requested_counters: tuple[TranslationCounterSpec, ...]
    series: tuple[TranslationCounterSeries, ...]
    window_ns: tuple[int, int]
    traffic_total: TranslationTrafficTotal | None = None

    def __post_init__(self) -> None:
        requested_ids = [spec.counter_id for spec in self.requested_counters]
        if len(requested_ids) != len(set(requested_ids)):
            raise ValueError("requested counter IDs must be unique")
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
class TranslationInterval:
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
class TranslationCounterResult:
    spec: TranslationCounterSpec
    source_identity: TranslationCounterIdentity | None
    status: TranslationStatus
    quality_flags: frozenset[TranslationQualityFlag]
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
    zero_observed: bool
    persistent_zero: bool
    intervals: tuple[TranslationInterval, ...]


@dataclass(frozen=True)
class TranslationMissViewResult:
    miss_view: TranslationMissView
    request_total: int | None
    hit_total: int | None
    miss_total: int | None
    hit_ratio: float | None
    hit_denominator: int | None
    misses_per_request: float | None
    request_denominator: int | None
    misses_per_gb: float | None
    traffic_total_gb: float | None
    traffic_direction: TrafficDirection | None
    traffic_status: TrafficStatus
    covered_duration_ns: int | None
    represented_window_ns: tuple[int, int] | None
    coverage_fraction: float | None
    zero_observed: bool
    persistent_zero: bool
    status: TranslationStatus
    quality_flags: frozenset[TranslationQualityFlag]
    status_reasons: tuple[str, ...]
    request_counter_ids: tuple[str, ...]
    hit_counter_ids: tuple[str, ...]
    miss_counter_ids: tuple[str, ...]
    missing_counter_ids: tuple[str, ...]


@dataclass(frozen=True)
class TranslationAnalysis:
    contract_version: int
    population: TranslationPopulation
    status: TranslationStatus
    quality_flags: frozenset[TranslationQualityFlag]
    status_reasons: tuple[str, ...]
    requested_window_ns: tuple[int, int]
    represented_window_ns: tuple[int, int] | None
    covered_duration_ns: int | None
    requested_duration_ns: int
    coverage_fraction: float | None
    counter_results: tuple[TranslationCounterResult, ...]
    view_results: tuple[TranslationMissViewResult, ...]
    eviction_total: int | None
    eviction_status: TranslationStatus
    eviction_counter_ids: tuple[str, ...]
    eviction_missing_counter_ids: tuple[str, ...]
    eviction_zero_observed: bool
    hit_ratio_formula: str
    misses_per_request_formula: str
    misses_per_gb_formula: str
    delta_formula: str
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class _CounterAggregate:
    role: TranslationCounterRole
    miss_view: TranslationMissView | None
    total: int | None
    unit: str | None
    population_id: str | None
    boundaries: tuple[tuple[int, int], ...]
    status: TranslationStatus
    quality_flags: frozenset[TranslationQualityFlag]
    status_reasons: tuple[str, ...]
    counter_ids: tuple[str, ...]
    missing_counter_ids: tuple[str, ...]
    zero_observed: bool
    persistent_zero: bool


def _missing_result(
    spec: TranslationCounterSpec,
    window_ns: tuple[int, int],
) -> TranslationCounterResult:
    requested_duration_ns = window_ns[1] - window_ns[0]
    return TranslationCounterResult(
        spec=spec,
        source_identity=None,
        status=TranslationStatus.MISSING_DATA,
        quality_flags=frozenset({TranslationQualityFlag.MISSING_COUNTER}),
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
        zero_observed=False,
        persistent_zero=False,
        intervals=(),
    )


def _unsupported_result(
    series: TranslationCounterSeries,
    window_ns: tuple[int, int],
    quality_flags: set[TranslationQualityFlag],
    reasons: list[str],
) -> TranslationCounterResult:
    requested_duration_ns = window_ns[1] - window_ns[0]
    return TranslationCounterResult(
        spec=series.spec,
        source_identity=series.identity,
        status=TranslationStatus.UNSUPPORTED,
        quality_flags=frozenset(quality_flags | set(series.quality_flags)),
        status_reasons=tuple(reasons),
        requested_window_ns=window_ns,
        represented_window_ns=None,
        covered_duration_ns=0,
        requested_duration_ns=requested_duration_ns,
        coverage_fraction=0.0,
        sample_count=_window_sample_count(series, window_ns),
        valid_interval_count=0,
        invalid_interval_count=0,
        outside_window_interval_count=0,
        total=None,
        unit=series.spec.unit,
        zero_observed=False,
        persistent_zero=False,
        intervals=(),
    )


def _window_sample_count(
    series: TranslationCounterSeries, window_ns: tuple[int, int]
) -> int:
    return sum(
        1
        for timestamp in series.timestamps_ns
        if window_ns[0] <= timestamp <= window_ns[1]
    )


def _analyze_series(
    series: TranslationCounterSeries,
    window_ns: tuple[int, int],
) -> TranslationCounterResult:
    quality_flags: set[TranslationQualityFlag] = set(series.quality_flags)
    reasons: list[str] = []

    if series.baseline_relative:
        quality_flags.add(TranslationQualityFlag.BASELINE_RELATIVE)
        reasons.append(
            "Baseline-relative pftrace change-only samples do not preserve "
            "exact cumulative-counter intervals."
        )
    if series.value_kind is not ValueKind.CUMULATIVE:
        quality_flags.add(TranslationQualityFlag.UNSUPPORTED_SERIES)
        reasons.append(f"Unsupported value kind: {series.value_kind.value}.")
    if series.semantics != CUMULATIVE_COUNTER_SEMANTICS:
        quality_flags.add(TranslationQualityFlag.UNSUPPORTED_SERIES)
        reasons.append(f"Unsupported counter semantics: {series.semantics!r}.")
    if reasons:
        return _unsupported_result(series, window_ns, quality_flags, reasons)

    records = list(zip(series.timestamps_ns, series.values, strict=True))
    ordered = sorted(records, key=lambda record: record[0])
    if records != ordered:
        quality_flags.add(TranslationQualityFlag.REORDERED)

    samples: list[tuple[int, int]] = []
    intervals: list[TranslationInterval] = []
    valid_intervals: list[TranslationInterval] = []
    invalid_interval_count = 0
    outside_window_interval_count = 0

    for timestamp, grouped_records in groupby(ordered, key=lambda record: record[0]):
        values = [value for _, value in grouped_records]
        if len(set(values)) != 1:
            first_value, second_value = values[:2]
            inside_window = window_ns[0] <= timestamp <= window_ns[1]
            interval = TranslationInterval(
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
            quality_flags.add(TranslationQualityFlag.CONFLICTING_TIMESTAMP)
            if inside_window:
                invalid_interval_count += 1
            else:
                outside_window_interval_count += 1
            continue

        sample = (timestamp, values[0])
        if samples and samples[-1] == sample:
            quality_flags.add(TranslationQualityFlag.DUPLICATE)
        samples.append(sample)
        if len(values) > 1:
            quality_flags.add(TranslationQualityFlag.DUPLICATE)

    for (start_ns, start_value), (end_ns, end_value) in zip(
        samples, samples[1:], strict=False
    ):
        delta = end_value - start_value
        state = (
            IntervalState.VALID if delta >= 0 else IntervalState.DECREASE_OR_RESET
        )
        inside_window = window_ns[0] <= start_ns and end_ns <= window_ns[1]
        interval = TranslationInterval(
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
                quality_flags.add(TranslationQualityFlag.BOUNDARY_INTERVAL_EXCLUDED)
        elif delta >= 0:
            valid_intervals.append(interval)
        else:
            invalid_interval_count += 1
            quality_flags.add(TranslationQualityFlag.DECREASE_OR_RESET)

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
        quality_flags.add(TranslationQualityFlag.PARTIAL_WINDOW)

    if not valid_intervals:
        if invalid_interval_count:
            status = TranslationStatus.PARTIAL_INVALID_DATA
            reasons.append("The window contains no complete valid counter interval.")
        else:
            status = TranslationStatus.INSUFFICIENT_DATA
            reasons.append("At least two distinct in-window timestamps are required.")
    elif invalid_interval_count:
        status = TranslationStatus.PARTIAL_INVALID_DATA
        reasons.append("Valid results exclude one or more decreases or resets.")
    elif total == 0:
        status = TranslationStatus.OBSERVED_ZERO
    else:
        status = TranslationStatus.OBSERVED_NONZERO

    represented_window_ns = (
        (
            min(interval.start_ns for interval in valid_intervals),
            max(interval.end_ns for interval in valid_intervals),
        )
        if valid_intervals
        else None
    )

    return TranslationCounterResult(
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
        sample_count=_window_sample_count(series, window_ns),
        valid_interval_count=len(valid_intervals),
        invalid_interval_count=invalid_interval_count,
        outside_window_interval_count=outside_window_interval_count,
        total=total,
        unit=series.spec.unit,
        zero_observed=all_samples_zero,
        persistent_zero=persistent_zero,
        intervals=selected_intervals,
    )


def _valid_interval_boundaries(
    result: TranslationCounterResult,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (interval.start_ns, interval.end_ns)
        for interval in result.intervals
        if interval.valid
    )


def _observed(status: TranslationStatus) -> bool:
    return status in {
        TranslationStatus.OBSERVED_ZERO,
        TranslationStatus.OBSERVED_NONZERO,
    }


def _has_scientific_total(status: TranslationStatus) -> bool:
    return _observed(status) or status is TranslationStatus.PARTIAL_INVALID_DATA


def _aggregate_status(
    statuses: set[TranslationStatus],
) -> TranslationStatus:
    if TranslationStatus.MISSING_DATA in statuses and len(statuses) == 1:
        return TranslationStatus.MISSING_DATA
    if TranslationStatus.MISSING_DATA in statuses:
        return TranslationStatus.PARTIAL_MISSING_DATA
    if any(
        status in statuses
        for status in (
            TranslationStatus.UNSUPPORTED,
            TranslationStatus.PARTIAL_UNSUPPORTED_DATA,
        )
    ):
        return TranslationStatus.PARTIAL_UNSUPPORTED_DATA
    if any(
        status in statuses
        for status in (
            TranslationStatus.PARTIAL_INVALID_DATA,
            TranslationStatus.INSUFFICIENT_DATA,
        )
    ):
        return TranslationStatus.PARTIAL_INVALID_DATA
    observed_statuses = {
        TranslationStatus.OBSERVED_ZERO,
        TranslationStatus.OBSERVED_NONZERO,
    }
    if statuses == {TranslationStatus.OBSERVED_ZERO}:
        return TranslationStatus.OBSERVED_ZERO
    if statuses <= observed_statuses:
        return TranslationStatus.OBSERVED_NONZERO
    return TranslationStatus.UNSUPPORTED


def _aggregate_role(
    results: tuple[TranslationCounterResult, ...],
    role: TranslationCounterRole,
    miss_view: TranslationMissView | None = None,
) -> _CounterAggregate:
    selected = tuple(
        result
        for result in results
        if result.spec.role is role
        and (miss_view is None or result.spec.miss_view is miss_view)
    )
    counter_ids = tuple(result.spec.counter_id for result in selected)
    missing_counter_ids = tuple(
        result.spec.counter_id
        for result in selected
        if result.status is TranslationStatus.MISSING_DATA
    )
    quality_flags: set[TranslationQualityFlag] = set()
    reasons: list[str] = []
    for result in selected:
        quality_flags.update(result.quality_flags)

    if not selected:
        return _CounterAggregate(
            role=role,
            miss_view=miss_view,
            total=None,
            unit=None,
            population_id=None,
            boundaries=(),
            status=TranslationStatus.MISSING_DATA,
            quality_flags=frozenset({TranslationQualityFlag.MISSING_COUNTER}),
            status_reasons=(f"No {role.value} counter was requested.",),
            counter_ids=(),
            missing_counter_ids=(),
            zero_observed=False,
            persistent_zero=False,
        )

    usable = tuple(
        result for result in selected if _has_scientific_total(result.status)
    )
    compatible = bool(usable)
    if usable:
        units = {result.unit for result in usable}
        population_ids = {result.spec.population_id for result in usable}
        boundaries = [_valid_interval_boundaries(result) for result in usable]
        if len(units) != 1 or len(population_ids) != 1 or None in population_ids:
            compatible = False
            if len(units) != 1:
                quality_flags.add(TranslationQualityFlag.INCOMPATIBLE_UNIT)
                reasons.append(f"{role.value.capitalize()} units are incompatible.")
            if len(population_ids) != 1 or None in population_ids:
                quality_flags.add(TranslationQualityFlag.INCOMPATIBLE_POPULATION)
                reasons.append(
                    f"{role.value.capitalize()} population IDs are missing or incompatible."
                )
        if compatible and (
            not boundaries[0] or any(boundary != boundaries[0] for boundary in boundaries)
        ):
            compatible = False
            quality_flags.add(TranslationQualityFlag.INCOMPATIBLE_COVERAGE)
            reasons.append(f"{role.value.capitalize()} interval coverage is incompatible.")

    total = (
        sum(result.total or 0 for result in usable)
        if compatible and usable
        else None
    )
    aggregate_status = _aggregate_status({result.status for result in selected})
    if usable and not compatible:
        aggregate_status = TranslationStatus.UNSUPPORTED

    return _CounterAggregate(
        role=role,
        miss_view=miss_view,
        total=total,
        unit=next(iter({result.unit for result in usable}), None),
        population_id=next(
            iter({result.spec.population_id for result in usable}), None
        ),
        boundaries=(
            _valid_interval_boundaries(usable[0])
            if usable and compatible
            else ()
        ),
        status=aggregate_status,
        quality_flags=frozenset(quality_flags),
        status_reasons=tuple(reasons),
        counter_ids=counter_ids,
        missing_counter_ids=missing_counter_ids,
        zero_observed=bool(usable) and all(result.zero_observed for result in usable),
        persistent_zero=bool(usable)
        and all(result.persistent_zero for result in usable),
    )


def _aggregates_are_compatible(
    first: _CounterAggregate, second: _CounterAggregate
) -> bool:
    return (
        first.total is not None
        and second.total is not None
        and first.unit == second.unit
        and first.population_id is not None
        and first.population_id == second.population_id
        and bool(first.boundaries)
        and first.boundaries == second.boundaries
    )


def _traffic_details(
    traffic: TranslationTrafficTotal | None,
    population_id: str | None,
    boundaries: tuple[tuple[int, int], ...],
) -> tuple[float | None, TrafficStatus, TrafficDirection | None]:
    if traffic is None:
        return None, TrafficStatus.NOT_SUPPLIED, None
    if traffic.unit not in (TRAFFIC_BYTES_UNIT, TRAFFIC_GB_UNIT):
        return None, TrafficStatus.INCOMPATIBLE_UNIT, traffic.direction
    if traffic.population_id != population_id:
        return None, TrafficStatus.INCOMPATIBLE_POPULATION, traffic.direction
    if not boundaries or traffic.window_ns != (
        min(start for start, _ in boundaries),
        max(end for _, end in boundaries),
    ):
        return None, TrafficStatus.INCOMPATIBLE_WINDOW, traffic.direction
    if traffic.total == 0:
        return 0.0, TrafficStatus.ZERO_DENOMINATOR, traffic.direction

    total_gb = (
        traffic.total / BYTES_PER_GB
        if traffic.unit == TRAFFIC_BYTES_UNIT
        else traffic.total
    )
    return total_gb, TrafficStatus.COMPATIBLE, traffic.direction


def _view_result(
    miss_view: TranslationMissView,
    request: _CounterAggregate,
    hit: _CounterAggregate,
    miss: _CounterAggregate,
    traffic: TranslationTrafficTotal | None,
    requested_duration_ns: int,
) -> TranslationMissViewResult:
    quality_flags = set()
    quality_flags.update(request.quality_flags)
    quality_flags.update(hit.quality_flags)
    quality_flags.update(miss.quality_flags)
    reasons = list(request.status_reasons)
    reasons.extend(hit.status_reasons)
    reasons.extend(miss.status_reasons)

    hit_denominator = (
        hit.total + miss.total
        if hit.total is not None and miss.total is not None
        else None
    )
    hit_ratio = hit.total / hit_denominator if hit_denominator else None
    request_denominator = request.total
    misses_per_request = (
        miss.total / request.total
        if miss.total is not None and request.total is not None and request.total > 0
        else None
    )

    if hit.total is not None and miss.total is not None:
        if not _aggregates_are_compatible(hit, miss):
            hit_denominator = None
            hit_ratio = None
            quality_flags.update(
                {
                    TranslationQualityFlag.INCOMPATIBLE_UNIT,
                    TranslationQualityFlag.INCOMPATIBLE_POPULATION,
                    TranslationQualityFlag.INCOMPATIBLE_COVERAGE,
                }
            )
            reasons.append("Hit and miss populations are incompatible.")
        elif hit_denominator == 0:
            quality_flags.add(TranslationQualityFlag.ZERO_DENOMINATOR)
            reasons.append("The hit-plus-miss denominator is zero and undefined.")

    if miss.total is not None and request.total is not None:
        if not _aggregates_are_compatible(miss, request):
            request_denominator = None
            misses_per_request = None
            quality_flags.update(
                {
                    TranslationQualityFlag.INCOMPATIBLE_UNIT,
                    TranslationQualityFlag.INCOMPATIBLE_POPULATION,
                    TranslationQualityFlag.INCOMPATIBLE_COVERAGE,
                }
            )
            reasons.append("Miss and request populations are incompatible.")
        elif request.total == 0:
            quality_flags.add(TranslationQualityFlag.ZERO_DENOMINATOR)
            reasons.append("The request denominator is zero and undefined.")

    coverage_aggregate = next(
        (
            aggregate
            for aggregate in (request, hit, miss)
            if aggregate.boundaries
        ),
        None,
    )
    represented_window_ns = (
        (
            min(start for start, _ in coverage_aggregate.boundaries),
            max(end for _, end in coverage_aggregate.boundaries),
        )
        if coverage_aggregate is not None
        else None
    )
    covered_duration_ns = (
        sum(end - start for start, end in coverage_aggregate.boundaries)
        if coverage_aggregate is not None
        else None
    )
    coverage_fraction = (
        covered_duration_ns / requested_duration_ns
        if covered_duration_ns is not None and requested_duration_ns > 0
        else None
    )

    traffic_total_gb, traffic_status, traffic_direction = _traffic_details(
        traffic,
        coverage_aggregate.population_id if coverage_aggregate is not None else None,
        coverage_aggregate.boundaries if coverage_aggregate is not None else (),
    )
    if traffic_status is TrafficStatus.INCOMPATIBLE_UNIT:
        quality_flags.add(TranslationQualityFlag.INCOMPATIBLE_TRAFFIC_UNIT)
        reasons.append("The supplied traffic unit is not bytes or GB.")
    elif traffic_status is TrafficStatus.INCOMPATIBLE_POPULATION:
        quality_flags.add(TranslationQualityFlag.INCOMPATIBLE_TRAFFIC_POPULATION)
        reasons.append("The traffic population is incompatible with translation counters.")
    elif traffic_status is TrafficStatus.INCOMPATIBLE_WINDOW:
        quality_flags.add(TranslationQualityFlag.INCOMPATIBLE_TRAFFIC_WINDOW)
        reasons.append("The traffic window is incompatible with translation intervals.")
    elif traffic_status is TrafficStatus.ZERO_DENOMINATOR:
        quality_flags.add(TranslationQualityFlag.ZERO_DENOMINATOR)
        reasons.append("The traffic denominator is zero and undefined.")

    misses_per_gb = (
        miss.total / traffic_total_gb
        if miss.total is not None
        and traffic_total_gb is not None
        and traffic_total_gb > 0
        else None
    )

    status = _aggregate_status({request.status, hit.status, miss.status})
    if all(
        aggregate.status
        in {TranslationStatus.OBSERVED_ZERO, TranslationStatus.OBSERVED_NONZERO}
        for aggregate in (request, hit, miss)
    ):
        status = (
            TranslationStatus.OBSERVED_ZERO
            if all(
                aggregate.total == 0
                for aggregate in (request, hit, miss)
            )
            else TranslationStatus.OBSERVED_NONZERO
        )

    missing_counter_ids = tuple(
        dict.fromkeys(
            request.missing_counter_ids + hit.missing_counter_ids + miss.missing_counter_ids
        )
    )
    return TranslationMissViewResult(
        miss_view=miss_view,
        request_total=request.total,
        hit_total=hit.total,
        miss_total=miss.total,
        hit_ratio=hit_ratio,
        hit_denominator=hit_denominator,
        misses_per_request=misses_per_request,
        request_denominator=request_denominator,
        misses_per_gb=misses_per_gb,
        traffic_total_gb=traffic_total_gb,
        traffic_direction=traffic_direction,
        traffic_status=traffic_status,
        covered_duration_ns=covered_duration_ns,
        represented_window_ns=represented_window_ns,
        coverage_fraction=coverage_fraction,
        zero_observed=(
            request.zero_observed and hit.zero_observed and miss.zero_observed
        ),
        persistent_zero=(
            request.persistent_zero
            and hit.persistent_zero
            and miss.persistent_zero
        ),
        status=status,
        quality_flags=frozenset(quality_flags),
        status_reasons=tuple(dict.fromkeys(reasons)),
        request_counter_ids=request.counter_ids,
        hit_counter_ids=hit.counter_ids,
        miss_counter_ids=miss.counter_ids,
        missing_counter_ids=missing_counter_ids,
    )


def _overall_status(
    statuses: tuple[TranslationStatus, ...],
) -> TranslationStatus:
    if not statuses:
        return TranslationStatus.MISSING_DATA
    return _aggregate_status(set(statuses))


def analyze_translation(dataset: TranslationDataset) -> TranslationAnalysis:
    series_by_id = {series.spec.counter_id: series for series in dataset.series}
    counter_results = tuple(
        _analyze_series(series_by_id[spec.counter_id], dataset.window_ns)
        if spec.counter_id in series_by_id
        else _missing_result(spec, dataset.window_ns)
        for spec in dataset.requested_counters
    )

    miss_views = tuple(
        dict.fromkeys(
            result.spec.miss_view
            for result in counter_results
            if result.spec.miss_view is not None
        )
    )
    view_results = tuple(
        _view_result(
            miss_view=miss_view,
            request=_aggregate_role(
                counter_results, TranslationCounterRole.REQUEST
            ),
            hit=_aggregate_role(counter_results, TranslationCounterRole.HIT),
            miss=_aggregate_role(
                counter_results,
                TranslationCounterRole.MISS,
                miss_view=miss_view,
            ),
            traffic=dataset.traffic_total,
            requested_duration_ns=dataset.window_ns[1] - dataset.window_ns[0],
        )
        for miss_view in miss_views
    )

    eviction = _aggregate_role(counter_results, TranslationCounterRole.EVICTION)
    represented_windows = tuple(
        view_result.represented_window_ns
        for view_result in view_results
        if view_result.represented_window_ns is not None
    )
    represented_window_ns = (
        (
            min(start for start, _ in represented_windows),
            max(end for _, end in represented_windows),
        )
        if represented_windows
        else None
    )
    covered_durations = tuple(
        view_result.covered_duration_ns
        for view_result in view_results
        if view_result.covered_duration_ns is not None
    )
    covered_duration_ns = min(covered_durations) if covered_durations else None
    requested_duration_ns = dataset.window_ns[1] - dataset.window_ns[0]

    quality_flags = {
        flag
        for result in counter_results
        for flag in result.quality_flags
    }
    quality_flags.update(eviction.quality_flags)
    for view_result in view_results:
        quality_flags.update(view_result.quality_flags)

    status_reasons = tuple(
        dict.fromkeys(
            reason
            for source in (view_results, counter_results)
            for result in source
            for reason in result.status_reasons
        )
    )
    status = _overall_status(
        tuple(view_result.status for view_result in view_results)
        + (eviction.status,)
    )

    return TranslationAnalysis(
        contract_version=TRANSLATION_ANALYSIS_VERSION,
        population=dataset.population,
        status=status,
        quality_flags=frozenset(quality_flags),
        status_reasons=status_reasons,
        requested_window_ns=dataset.window_ns,
        represented_window_ns=represented_window_ns,
        covered_duration_ns=covered_duration_ns,
        requested_duration_ns=requested_duration_ns,
        coverage_fraction=(
            covered_duration_ns / requested_duration_ns
            if covered_duration_ns is not None and requested_duration_ns > 0
            else None
        ),
        counter_results=counter_results,
        view_results=view_results,
        eviction_total=eviction.total,
        eviction_status=eviction.status,
        eviction_counter_ids=eviction.counter_ids,
        eviction_missing_counter_ids=eviction.missing_counter_ids,
        eviction_zero_observed=eviction.zero_observed,
        hit_ratio_formula=HIT_RATIO_FORMULA,
        misses_per_request_formula=MISSES_PER_REQUEST_FORMULA,
        misses_per_gb_formula=MISSES_PER_GB_FORMULA,
        delta_formula=DELTA_FORMULA,
        limitations=TRANSLATION_LIMITATIONS,
    )
