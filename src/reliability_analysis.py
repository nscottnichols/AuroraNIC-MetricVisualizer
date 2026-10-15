"""Pure analysis for cumulative NIC reliability and error counters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
UINT64_MAX = 2**64 - 1
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
NANOSECONDS_PER_SECOND = 1_000_000_000
CUMULATIVE_COUNTER_SEMANTICS = "cumulative counter"
ERROR_RATE_FORMULA = "sum(error interval deltas) / common measured covered duration"
HISTORICAL_ZERO_NOTE = (
    "An all-zero observation is source-history evidence, not a guarantee that "
    "future reliability counters will remain zero."
)


class ValueKind(str, Enum):
    CUMULATIVE = "cumulative"
    DELTA = "delta"
    GAUGE = "gauge"
    UNKNOWN = "unknown"


class CounterRole(str, Enum):
    ERROR = "error"
    GOOD = "good"
    UNKNOWN = "unknown"


class ReliabilityStatus(str, Enum):
    OBSERVED_ZERO = "observed_zero"
    OBSERVED_NONZERO = "observed_nonzero"
    PARTIAL_MISSING_DATA = "partial_missing_data"
    PARTIAL_INVALID_DATA = "partial_invalid_data"
    PARTIAL_UNSUPPORTED_DATA = "partial_unsupported_data"
    MISSING_DATA = "missing_data"
    INSUFFICIENT_DATA = "insufficient_data"
    UNSUPPORTED = "unsupported"


class ReliabilityQualityFlag(str, Enum):
    MISSING_COUNTER = "missing_counter"
    UNSUPPORTED_SERIES = "unsupported_series"
    BASELINE_RELATIVE = "baseline_relative"
    DECREASE_OR_RESET = "decrease_or_reset"
    REORDERED = "reordered"
    DUPLICATE = "duplicate"
    CONFLICTING_TIMESTAMP = "conflicting_timestamp"
    PARTIAL_WINDOW = "partial_window"
    BOUNDARY_INTERVAL_EXCLUDED = "boundary_interval_excluded"
    INCOMPATIBLE_COVERAGE = "incompatible_coverage"
    INCOMPATIBLE_AGGREGATION = "incompatible_aggregation"
    KNOWN_GOOD_COUNTER_MISLABELED = "known_good_counter_mislabeled"


class IntervalState(str, Enum):
    VALID = "valid"
    DECREASE_OR_RESET = "decrease_or_reset"
    CONFLICTING_TIMESTAMP = "conflicting_timestamp"


@dataclass(frozen=True)
class ReliabilityPopulation:
    run_id: str
    node_id: str
    nic_id: str
    collector_id: str


@dataclass(frozen=True)
class CounterIdentity:
    run_id: str
    node_id: str
    nic_id: str
    collector_id: str
    source_counter_id: str


@dataclass(frozen=True)
class ReliabilityCounterSpec:
    counter_id: str
    role: CounterRole
    unit: str
    aggregation_group: str | None = None


@dataclass(frozen=True)
class ReliabilityCounterSeries:
    spec: ReliabilityCounterSpec
    identity: CounterIdentity
    timestamps_ns: tuple[int, ...]
    values: tuple[int, ...]
    value_kind: ValueKind
    semantics: str
    baseline_relative: bool
    quality_flags: tuple[ReliabilityQualityFlag, ...] = ()

    def __post_init__(self) -> None:
        if len(self.timestamps_ns) != len(self.values):
            raise ValueError("timestamps_ns and values must have equal length")
        if not self.spec.counter_id:
            raise ValueError("counter_id must be non-empty")
        if not self.spec.unit:
            raise ValueError("counter unit must be non-empty")
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
class ReliabilityDataset:
    population: ReliabilityPopulation
    requested_counters: tuple[ReliabilityCounterSpec, ...]
    series: tuple[ReliabilityCounterSeries, ...]
    window_ns: tuple[int, int]

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

        for series in self.series:
            if series.spec not in self.requested_counters:
                raise ValueError(
                    f"series counter {series.spec.counter_id!r} was not requested"
                )
            identity = series.identity
            population = self.population
            if (
                identity.run_id != population.run_id
                or identity.node_id != population.node_id
                or identity.nic_id != population.nic_id
                or identity.collector_id != population.collector_id
            ):
                raise ValueError("series identity must match the dataset population")


@dataclass(frozen=True)
class ReliabilityInterval:
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
class ReliabilityCounterResult:
    spec: ReliabilityCounterSpec
    source_identity: CounterIdentity | None
    status: ReliabilityStatus
    quality_flags: frozenset[ReliabilityQualityFlag]
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
    rate_unit: str
    rate_per_second: float | None
    first_nonzero_sample_ns: int | None
    first_nonzero_increment_ns: int | None
    persistent_zero: bool
    zero_observed: bool
    intervals: tuple[ReliabilityInterval, ...]


@dataclass(frozen=True)
class ReliabilityAnalysis:
    population: ReliabilityPopulation
    status: ReliabilityStatus
    quality_flags: frozenset[ReliabilityQualityFlag]
    status_reasons: tuple[str, ...]
    requested_window_ns: tuple[int, int]
    represented_window_ns: tuple[int, int] | None
    covered_duration_ns: int | None
    requested_duration_ns: int
    coverage_fraction: float | None
    counter_results: tuple[ReliabilityCounterResult, ...]
    error_total: int | None
    good_total: int | None
    aggregate_error_rate_per_second: float | None
    aggregate_rate_unit: str | None
    compatible_counter_ids: tuple[str, ...]
    incompatible_counter_ids: tuple[str, ...]
    first_nonzero_error_sample_ns: int | None
    first_nonzero_error_increment_ns: int | None
    persistent_zero: bool
    zero_observed: bool
    rate_formula: str
    aggregation_population: str
    historical_zero_note: str


def _missing_result(
    spec: ReliabilityCounterSpec,
    window_ns: tuple[int, int],
) -> ReliabilityCounterResult:
    return ReliabilityCounterResult(
        spec=spec,
        source_identity=None,
        status=ReliabilityStatus.MISSING_DATA,
        quality_flags=frozenset({ReliabilityQualityFlag.MISSING_COUNTER}),
        status_reasons=("The requested counter has no observation series.",),
        requested_window_ns=window_ns,
        represented_window_ns=None,
        covered_duration_ns=0,
        requested_duration_ns=window_ns[1] - window_ns[0],
        coverage_fraction=0.0,
        sample_count=0,
        valid_interval_count=0,
        invalid_interval_count=0,
        outside_window_interval_count=0,
        total=None,
        unit=spec.unit,
        rate_unit=f"{spec.unit}/s",
        rate_per_second=None,
        first_nonzero_sample_ns=None,
        first_nonzero_increment_ns=None,
        persistent_zero=False,
        zero_observed=False,
        intervals=(),
    )


def _unsupported_result(
    series: ReliabilityCounterSeries,
    window_ns: tuple[int, int],
    quality_flags: set[ReliabilityQualityFlag],
    reasons: list[str],
) -> ReliabilityCounterResult:
    window_sample_count = sum(
        1
        for timestamp in series.timestamps_ns
        if window_ns[0] <= timestamp <= window_ns[1]
    )
    return ReliabilityCounterResult(
        spec=series.spec,
        source_identity=series.identity,
        status=ReliabilityStatus.UNSUPPORTED,
        quality_flags=frozenset(quality_flags | set(series.quality_flags)),
        status_reasons=tuple(reasons),
        requested_window_ns=window_ns,
        represented_window_ns=None,
        covered_duration_ns=0,
        requested_duration_ns=window_ns[1] - window_ns[0],
        coverage_fraction=0.0,
        sample_count=window_sample_count,
        valid_interval_count=0,
        invalid_interval_count=0,
        outside_window_interval_count=0,
        total=None,
        unit=series.spec.unit,
        rate_unit=f"{series.spec.unit}/s",
        rate_per_second=None,
        first_nonzero_sample_ns=None,
        first_nonzero_increment_ns=None,
        persistent_zero=False,
        zero_observed=False,
        intervals=(),
    )


def _analyze_series(
    series: ReliabilityCounterSeries,
    window_ns: tuple[int, int],
) -> ReliabilityCounterResult:
    quality_flags: set[ReliabilityQualityFlag] = set(series.quality_flags)
    reasons: list[str] = []

    if series.spec.counter_id == "hni_pcs_good_cw" and series.spec.role != CounterRole.GOOD:
        quality_flags.add(ReliabilityQualityFlag.KNOWN_GOOD_COUNTER_MISLABELED)
        reasons.append(
            "hni_pcs_good_cw is a good-codeword statistic and cannot be classified as an error."
        )
    if series.spec.role == CounterRole.UNKNOWN:
        quality_flags.add(ReliabilityQualityFlag.UNSUPPORTED_SERIES)
        reasons.append("The counter role is unknown.")
    if series.baseline_relative:
        quality_flags.add(ReliabilityQualityFlag.BASELINE_RELATIVE)
        reasons.append(
            "Baseline-relative pftrace change-only samples do not preserve "
            "exact cumulative-counter intervals."
        )
    if series.value_kind is not ValueKind.CUMULATIVE:
        quality_flags.add(ReliabilityQualityFlag.UNSUPPORTED_SERIES)
        reasons.append(f"Unsupported value kind: {series.value_kind.value}.")
    if series.semantics != CUMULATIVE_COUNTER_SEMANTICS:
        quality_flags.add(ReliabilityQualityFlag.UNSUPPORTED_SERIES)
        reasons.append(f"Unsupported counter semantics: {series.semantics!r}.")
    if reasons:
        return _unsupported_result(series, window_ns, quality_flags, reasons)

    records = list(zip(series.timestamps_ns, series.values, strict=True))
    ordered = sorted(records, key=lambda record: record[0])
    if records != ordered:
        quality_flags.add(ReliabilityQualityFlag.REORDERED)

    intervals: list[ReliabilityInterval] = []
    valid_intervals: list[ReliabilityInterval] = []
    invalid_interval_count = 0
    outside_window_interval_count = 0
    index = 0
    while index < len(ordered) - 1:
        start_timestamp, start_value = ordered[index]
        end_timestamp, end_value = ordered[index + 1]

        if start_timestamp == end_timestamp:
            if start_value == end_value:
                quality_flags.add(ReliabilityQualityFlag.DUPLICATE)
                index += 1
                continue

            quality_flags.add(ReliabilityQualityFlag.CONFLICTING_TIMESTAMP)
            inside_window = window_ns[0] <= start_timestamp <= window_ns[1]
            interval = ReliabilityInterval(
                start_ns=start_timestamp,
                end_ns=end_timestamp,
                duration_ns=0,
                start_value=start_value,
                end_value=end_value,
                delta=0,
                valid=False,
                state=IntervalState.CONFLICTING_TIMESTAMP,
                inside_window=inside_window,
            )
            intervals.append(interval)
            if inside_window:
                invalid_interval_count += 1
            else:
                outside_window_interval_count += 1
            index += 1
            continue

        delta = end_value - start_value
        state = IntervalState.VALID if delta >= 0 else IntervalState.DECREASE_OR_RESET
        if delta < 0:
            quality_flags.add(ReliabilityQualityFlag.DECREASE_OR_RESET)

        inside_window = window_ns[0] <= start_timestamp and end_timestamp <= window_ns[1]
        interval = ReliabilityInterval(
            start_ns=start_timestamp,
            end_ns=end_timestamp,
            duration_ns=end_timestamp - start_timestamp,
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
            if (
                start_timestamp < window_ns[0] < end_timestamp
                or start_timestamp < window_ns[1] < end_timestamp
            ):
                quality_flags.add(ReliabilityQualityFlag.BOUNDARY_INTERVAL_EXCLUDED)
        elif delta >= 0:
            valid_intervals.append(interval)
        else:
            invalid_interval_count += 1
        index += 1

    selected_intervals = tuple(interval for interval in intervals if interval.inside_window)
    covered_duration_ns = sum(
        interval.duration_ns for interval in valid_intervals
    )
    requested_duration_ns = window_ns[1] - window_ns[0]
    total = sum(interval.delta for interval in valid_intervals)
    rate_per_second = (
        total / (covered_duration_ns / NANOSECONDS_PER_SECOND)
        if covered_duration_ns > 0
        else None
    )

    window_samples = tuple(
        (timestamp, value)
        for timestamp, value in ordered
        if window_ns[0] <= timestamp <= window_ns[1]
    )
    first_nonzero_sample_ns = next(
        (
            timestamp
            for (timestamp, value) in ordered
            if value != 0 and window_ns[0] <= timestamp <= window_ns[1]
        ),
        None,
    )
    first_nonzero_increment_ns = next(
        (
            interval.end_ns
            for interval in valid_intervals
            if interval.delta > 0
        ),
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

    if valid_intervals:
        represented_window_ns = (
            min(interval.start_ns for interval in valid_intervals),
            max(interval.end_ns for interval in valid_intervals),
        )
    else:
        represented_window_ns = None

    if covered_duration_ns < requested_duration_ns:
        quality_flags.add(ReliabilityQualityFlag.PARTIAL_WINDOW)

    if not valid_intervals:
        status = ReliabilityStatus.INSUFFICIENT_DATA
        reasons.append("The window contains no complete valid counter interval.")
    elif invalid_interval_count:
        status = ReliabilityStatus.PARTIAL_INVALID_DATA
        reasons.append("One or more intervals were excluded for decrease or reset.")
    elif total == 0:
        status = ReliabilityStatus.OBSERVED_ZERO
    else:
        status = ReliabilityStatus.OBSERVED_NONZERO

    return ReliabilityCounterResult(
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
        sample_count=len(window_samples),
        valid_interval_count=len(valid_intervals),
        invalid_interval_count=invalid_interval_count,
        outside_window_interval_count=outside_window_interval_count,
        total=total,
        unit=series.spec.unit,
        rate_unit=f"{series.spec.unit}/s",
        rate_per_second=rate_per_second,
        first_nonzero_sample_ns=first_nonzero_sample_ns,
        first_nonzero_increment_ns=first_nonzero_increment_ns,
        persistent_zero=persistent_zero,
        zero_observed=all_samples_zero,
        intervals=selected_intervals,
    )


def _interval_boundaries(result: ReliabilityCounterResult) -> tuple[tuple[int, int], ...]:
    return tuple(
        (interval.start_ns, interval.end_ns)
        for interval in result.intervals
        if interval.valid
    )


def analyze_reliability(dataset: ReliabilityDataset) -> ReliabilityAnalysis:
    """Analyze requested cumulative reliability counters without hidden assumptions."""

    series_by_counter = {series.spec.counter_id: series for series in dataset.series}
    results = tuple(
        _analyze_series(series_by_counter[spec.counter_id], dataset.window_ns)
        if spec.counter_id in series_by_counter
        else _missing_result(spec, dataset.window_ns)
        for spec in dataset.requested_counters
    )

    error_results = tuple(
        result for result in results if result.spec.role is CounterRole.ERROR
    )
    good_results = tuple(
        result for result in results if result.spec.role is CounterRole.GOOD
    )

    error_boundaries = [_interval_boundaries(result) for result in error_results]
    error_units = {result.spec.unit for result in error_results}
    error_aggregation_groups = {
        result.spec.aggregation_group for result in error_results
    }
    error_aggregation_compatible = (
        len(error_results) == 1
        or (
            len(error_units) == 1
            and len(error_aggregation_groups) == 1
            and next(iter(error_aggregation_groups), None) is not None
        )
    )
    error_coverage_compatible = (
        bool(error_results)
        and bool(error_boundaries[0])
        and all(boundaries == error_boundaries[0] for boundaries in error_boundaries)
    )
    error_total: int | None = None
    aggregate_error_rate: float | None = None
    common_error_duration_ns: int | None = None
    if error_coverage_compatible and error_aggregation_compatible:
        all_error_results_supported = all(
            result.status
            in {ReliabilityStatus.OBSERVED_ZERO, ReliabilityStatus.OBSERVED_NONZERO}
            for result in error_results
        )
        if all_error_results_supported:
            common_error_duration_ns = sum(
                end_ns - start_ns for start_ns, end_ns in error_boundaries[0]
            )
            error_total = sum(result.total or 0 for result in error_results)
            aggregate_error_rate = error_total / (
                common_error_duration_ns / NANOSECONDS_PER_SECOND
            )

    good_units = {result.spec.unit for result in good_results}
    good_aggregation_groups = {
        result.spec.aggregation_group for result in good_results
    }
    good_aggregation_compatible = (
        len(good_results) == 1
        or (
            len(good_units) == 1
            and len(good_aggregation_groups) == 1
            and next(iter(good_aggregation_groups), None) is not None
        )
    )
    good_boundaries = [_interval_boundaries(result) for result in good_results]
    good_coverage_compatible = (
        bool(good_results)
        and bool(good_boundaries[0])
        and all(boundaries == good_boundaries[0] for boundaries in good_boundaries)
        and good_aggregation_compatible
        and all(
            result.status
            in {ReliabilityStatus.OBSERVED_ZERO, ReliabilityStatus.OBSERVED_NONZERO}
            for result in good_results
        )
    )
    good_total = (
        sum(result.total or 0 for result in good_results)
        if good_coverage_compatible
        else None
    )

    supported_results = tuple(
        result
        for result in results
        if result.status
        in {ReliabilityStatus.OBSERVED_ZERO, ReliabilityStatus.OBSERVED_NONZERO}
    )
    represented_bounds = [
        result.represented_window_ns for result in supported_results if result.represented_window_ns
    ]
    represented_window_ns = (
        (min(start for start, _ in represented_bounds), max(end for _, end in represented_bounds))
        if represented_bounds
        else None
    )

    quality_flags: set[ReliabilityQualityFlag] = set()
    status_reasons: list[str] = []
    for result in results:
        quality_flags.update(result.quality_flags)

    missing_count = sum(
        result.status is ReliabilityStatus.MISSING_DATA for result in results
    )
    unsupported_count = sum(
        result.status is ReliabilityStatus.UNSUPPORTED for result in results
    )
    invalid_count = sum(
        result.status is ReliabilityStatus.PARTIAL_INVALID_DATA for result in results
    )

    if not error_results:
        status = ReliabilityStatus.UNSUPPORTED
        status_reasons.append("No error-role counter was requested.")
    elif missing_count == len(results):
        status = ReliabilityStatus.MISSING_DATA
    elif missing_count:
        status = ReliabilityStatus.PARTIAL_MISSING_DATA
        status_reasons.append("At least one requested reliability counter is missing.")
    elif unsupported_count == len(results):
        status = ReliabilityStatus.UNSUPPORTED
    elif unsupported_count:
        status = ReliabilityStatus.PARTIAL_UNSUPPORTED_DATA
        status_reasons.append("At least one counter has unsupported semantics.")
    elif not any(result.valid_interval_count for result in error_results):
        status = ReliabilityStatus.INSUFFICIENT_DATA
        status_reasons.append("No error counter has a complete valid interval.")
    elif invalid_count:
        status = ReliabilityStatus.PARTIAL_INVALID_DATA
        status_reasons.append("Valid intervals are reported after excluding decreases or resets.")
    elif any(result.total for result in error_results):
        status = ReliabilityStatus.OBSERVED_NONZERO
    else:
        status = ReliabilityStatus.OBSERVED_ZERO

    if error_results and not error_coverage_compatible:
        quality_flags.add(ReliabilityQualityFlag.INCOMPATIBLE_COVERAGE)
        status_reasons.append(
            "Aggregate error totals and rates require identical measured "
            "intervals across error counters."
        )
    if error_results and not error_aggregation_compatible:
        quality_flags.add(ReliabilityQualityFlag.INCOMPATIBLE_AGGREGATION)
        status_reasons.append(
            "Aggregate error totals require one common unit and an explicitly "
            "compatible aggregation group."
        )

    all_error_zero = bool(error_results) and all(
        result.persistent_zero for result in error_results
    )
    any_error_zero = bool(error_results) and all(
        result.status is ReliabilityStatus.OBSERVED_ZERO for result in error_results
    )
    first_nonzero_samples = [
        result.first_nonzero_sample_ns
        for result in error_results
        if result.first_nonzero_sample_ns is not None
    ]
    first_nonzero_increments = [
        result.first_nonzero_increment_ns
        for result in error_results
        if result.first_nonzero_increment_ns is not None
    ]

    compatible_counter_ids: list[str] = []
    incompatible_counter_ids: list[str] = []
    if supported_results:
        reference_boundaries = _interval_boundaries(supported_results[0])
        for result in results:
            incompatible = result.status in {
                ReliabilityStatus.MISSING_DATA,
                ReliabilityStatus.UNSUPPORTED,
                ReliabilityStatus.INSUFFICIENT_DATA,
                ReliabilityStatus.PARTIAL_INVALID_DATA,
            }
            boundaries_match = _interval_boundaries(result) == reference_boundaries
            if result.status in {
                ReliabilityStatus.OBSERVED_ZERO,
                ReliabilityStatus.OBSERVED_NONZERO,
            } and boundaries_match:
                compatible_counter_ids.append(result.spec.counter_id)
            elif incompatible or not boundaries_match:
                incompatible_counter_ids.append(result.spec.counter_id)

    return ReliabilityAnalysis(
        population=dataset.population,
        status=status,
        quality_flags=frozenset(quality_flags),
        status_reasons=tuple(status_reasons),
        requested_window_ns=dataset.window_ns,
        represented_window_ns=represented_window_ns,
        covered_duration_ns=common_error_duration_ns,
        requested_duration_ns=dataset.window_ns[1] - dataset.window_ns[0],
        coverage_fraction=(
            common_error_duration_ns / (dataset.window_ns[1] - dataset.window_ns[0])
            if common_error_duration_ns is not None
            else None
        ),
        counter_results=results,
        error_total=error_total,
        good_total=good_total,
        aggregate_error_rate_per_second=aggregate_error_rate,
        aggregate_rate_unit=(
            f"{next(iter(error_units))}/s" if error_total is not None else None
        ),
        compatible_counter_ids=tuple(compatible_counter_ids),
        incompatible_counter_ids=incompatible_counter_ids,
        first_nonzero_error_sample_ns=min(first_nonzero_samples) if first_nonzero_samples else None,
        first_nonzero_error_increment_ns=(
            min(first_nonzero_increments) if first_nonzero_increments else None
        ),
        persistent_zero=all_error_zero,
        zero_observed=any_error_zero,
        rate_formula=ERROR_RATE_FORMULA,
        aggregation_population=(
            "All requested error counters over one identical set of exact valid intervals."
        ),
        historical_zero_note=HISTORICAL_ZERO_NOTE,
    )
