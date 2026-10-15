"""Pure wire-frame size analysis for cumulative NIC counters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


FRAME_ANALYSIS_VERSION = 1
UINT64_MAX = 2**64 - 1
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
NANOSECONDS_PER_SECOND = 1_000_000_000
FRAME_UNIT = "frames"
BYTE_UNIT = "bytes"
CUMULATIVE_COUNTER_SEMANTICS = "cumulative counter"
DELTA_FORMULA = "end_value - start_value"
SHARE_FORMULA = "bin frame delta total / direction frame delta total"
MEAN_FORMULA = "compatible direction byte delta total / direction frame delta total"
APPROXIMATE_BIN_MEAN_FORMULA = (
    "sum(bin frame delta total * bin midpoint bytes) / direction frame delta total"
)
QUANTILES = (0.5, 0.99)

FRAME_INTERPRETATION = (
    "Frame-size bins describe wire-frame populations. They do not measure "
    "MPI message, payload, or allreduce sizes."
)

FRAME_LIMITATIONS = (
    "Bin shares require all requested bins for one direction over identical valid intervals.",
    "An exact byte/count mean requires sourced bin edges and explicitly compatible byte and frame populations.",
    "A bin-midpoint mean is approximate and is unavailable with an open-ended tail bin.",
    "Missing bins and missing edges are distinct from observed zero frames.",
    "Decreases and resets exclude the affected interval from totals, rates, and coverage.",
    "Frame counts and byte counts may suggest correlations but do not prove causality.",
)


class FrameValueKind(str, Enum):
    CUMULATIVE = "cumulative"
    DELTA = "delta"
    GAUGE = "gauge"
    UNKNOWN = "unknown"


class FrameDirection(str, Enum):
    TX = "tx"
    RX = "rx"


class FrameCounterKind(str, Enum):
    FRAME_BIN = "frame_bin"
    BYTE_COUNT = "byte_count"


class FrameStatus(str, Enum):
    OBSERVED_ZERO = "observed_zero"
    OBSERVED_NONZERO = "observed_nonzero"
    PARTIAL_MISSING_DATA = "partial_missing_data"
    PARTIAL_INVALID_DATA = "partial_invalid_data"
    PARTIAL_UNSUPPORTED_DATA = "partial_unsupported_data"
    MISSING_DATA = "missing_data"
    INSUFFICIENT_DATA = "insufficient_data"
    UNSUPPORTED = "unsupported"


class FrameQualityFlag(str, Enum):
    MISSING_COUNTER = "missing_counter"
    MISSING_BIN_EDGE = "missing_bin_edge"
    MISSING_BIN_EDGES = "missing_bin_edges"
    UNSUPPORTED_SERIES = "unsupported_series"
    BASELINE_RELATIVE = "baseline_relative"
    DECREASE_OR_RESET = "decrease_or_reset"
    REORDERED = "reordered"
    DUPLICATE = "duplicate"
    CONFLICTING_TIMESTAMP = "conflicting_timestamp"
    PARTIAL_WINDOW = "partial_window"
    BOUNDARY_INTERVAL_EXCLUDED = "boundary_interval_excluded"
    INCOMPATIBLE_COVERAGE = "incompatible_coverage"
    INCOMPATIBLE_BYTE_COUNT_POPULATION = "incompatible_byte_count_population"
    OPEN_ENDED_BIN = "open_ended_bin"


class FrameIntervalState(str, Enum):
    VALID = "valid"
    DECREASE_OR_RESET = "decrease_or_reset"
    CONFLICTING_TIMESTAMP = "conflicting_timestamp"


class FrameMeanStatus(str, Enum):
    SUPPORTED = "supported"
    ZERO_DENOMINATOR = "zero_denominator"
    UNSUPPORTED = "unsupported"


class FrameQuantileStatus(str, Enum):
    BOUNDED = "bounded"
    OPEN_ENDED = "open_ended"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class FramePopulation:
    run_id: str
    node_id: str
    nic_id: str
    collector_id: str


@dataclass(frozen=True)
class FrameCounterIdentity:
    run_id: str
    node_id: str
    nic_id: str
    collector_id: str
    source_counter_id: str


@dataclass(frozen=True)
class FrameCounterSpec:
    counter_id: str
    direction: FrameDirection
    counter_kind: FrameCounterKind
    unit: str
    population_id: str
    bin_id: str | None = None

    def __post_init__(self) -> None:
        if not self.counter_id:
            raise ValueError("counter_id must be nonempty")
        if not self.unit:
            raise ValueError("counter unit must be nonempty")
        if not self.population_id:
            raise ValueError("population_id must be nonempty")
        if self.counter_kind is FrameCounterKind.FRAME_BIN:
            if not self.bin_id:
                raise ValueError("frame-bin counters require a nonempty bin_id")
            if self.unit != FRAME_UNIT:
                raise ValueError("frame-bin counters must use the frames unit")
        else:
            if self.bin_id is not None:
                raise ValueError("byte counters cannot have a bin_id")
            if self.unit != BYTE_UNIT:
                raise ValueError("byte counters must use the bytes unit")


@dataclass(frozen=True)
class FrameCounterSeries:
    spec: FrameCounterSpec
    identity: FrameCounterIdentity
    timestamps_ns: tuple[int, ...]
    values: tuple[int, ...]
    value_kind: FrameValueKind
    semantics: str
    baseline_relative: bool
    quality_flags: tuple[FrameQualityFlag, ...] = ()

    def __post_init__(self) -> None:
        if len(self.timestamps_ns) != len(self.values):
            raise ValueError("timestamps_ns and values must have equal length")
        if not self.semantics:
            raise ValueError("counter semantics must be nonempty")
        if not isinstance(self.baseline_relative, bool):
            raise ValueError("baseline_relative must be boolean")
        for field_name in ("run_id", "node_id", "nic_id", "collector_id"):
            if not getattr(self.identity, field_name):
                raise ValueError(f"identity {field_name} must be nonempty")
        if not self.identity.source_counter_id:
            raise ValueError("source_counter_id must be nonempty")
        for timestamp in self.timestamps_ns:
            if type(timestamp) is not int or not INT64_MIN <= timestamp <= INT64_MAX:
                raise ValueError("timestamps must be int64 integers")
        for value in self.values:
            if type(value) is not int or not 0 <= value <= UINT64_MAX:
                raise ValueError("counter values must be uint64 integers")


@dataclass(frozen=True)
class FrameBinEdge:
    bin_id: str
    lower_bytes: int
    upper_bytes: int | None
    open_ended: bool

    def __post_init__(self) -> None:
        if not self.bin_id:
            raise ValueError("bin_id must be nonempty")
        if type(self.lower_bytes) is not int or self.lower_bytes < 0:
            raise ValueError("lower_bytes must be a nonnegative integer")
        if self.open_ended:
            if self.upper_bytes is not None:
                raise ValueError("an open-ended bin cannot have an upper edge")
        elif type(self.upper_bytes) is not int or self.upper_bytes <= self.lower_bytes:
            raise ValueError("closed-bin upper_bytes must exceed lower_bytes")


@dataclass(frozen=True)
class FrameBinEdgeSet:
    direction: FrameDirection
    source: str
    edges: tuple[FrameBinEdge, ...]

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("bin edge source must be nonempty")
        if not self.edges:
            raise ValueError("bin edge sets must contain at least one edge")
        edge_ids = [edge.bin_id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("bin edge IDs must be unique")
        ordered = sorted(self.edges, key=lambda edge: edge.lower_bytes)
        if ordered[0].lower_bytes != 0:
            raise ValueError("the first bin lower edge must be zero")
        previous_upper: int | None = None
        open_ended_seen = False
        for edge in ordered:
            if open_ended_seen:
                raise ValueError("an open-ended bin must be the final bin")
            if previous_upper is not None:
                if edge.lower_bytes < previous_upper:
                    raise ValueError("bin edges must not overlap")
                if edge.lower_bytes > previous_upper:
                    raise ValueError("bin edges must be adjacent with no size-domain gap")
            if edge.open_ended:
                open_ended_seen = True
            previous_upper = edge.upper_bytes


@dataclass(frozen=True)
class FrameBytePopulationCompatibility:
    direction: FrameDirection
    byte_counter_id: str
    frame_population_id: str
    byte_population_id: str
    compatible: bool
    definition_evidence: str

    def __post_init__(self) -> None:
        if not self.byte_counter_id:
            raise ValueError("byte_counter_id must be nonempty")
        if not self.frame_population_id or not self.byte_population_id:
            raise ValueError("byte/count population IDs must be nonempty")
        if not self.definition_evidence:
            raise ValueError("definition_evidence must be nonempty")
        if not isinstance(self.compatible, bool):
            raise ValueError("compatible must be boolean")


@dataclass(frozen=True)
class FrameCounterInterval:
    start_ns: int
    end_ns: int
    duration_ns: int
    start_value: int
    end_value: int
    delta: int
    valid: bool
    state: FrameIntervalState
    inside_window: bool


@dataclass(frozen=True)
class FrameCounterResult:
    spec: FrameCounterSpec
    source_identity: FrameCounterIdentity | None
    status: FrameStatus
    quality_flags: frozenset[FrameQualityFlag]
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
    rate_per_second: float | None
    zero_observed: bool
    persistent_zero: bool
    intervals: tuple[FrameCounterInterval, ...]
    bin_edge: FrameBinEdge | None = None
    bin_share: float | None = None


@dataclass(frozen=True)
class FrameTemporalInterval:
    start_ns: int
    end_ns: int
    duration_ns: int
    bin_deltas: tuple[tuple[str, int], ...]
    total_frames: int


@dataclass(frozen=True)
class FrameQuantileResult:
    quantile: float
    status: FrameQuantileStatus
    lower_bytes: int | None
    upper_bytes: int | None
    status_reasons: tuple[str, ...]


@dataclass(frozen=True)
class FrameDirectionResult:
    direction: FrameDirection
    status: FrameStatus
    quality_flags: frozenset[FrameQualityFlag]
    status_reasons: tuple[str, ...]
    requested_window_ns: tuple[int, int]
    represented_window_ns: tuple[int, int] | None
    covered_duration_ns: int | None
    requested_duration_ns: int
    coverage_fraction: float | None
    bin_results: tuple[FrameCounterResult, ...]
    byte_result: FrameCounterResult | None
    bin_total: int | None
    byte_total: int | None
    temporal_intervals: tuple[FrameTemporalInterval, ...]
    bin_edges_source: str | None
    bin_edges_complete: bool
    open_ended_tail: bool
    mean_wire_frame_size_bytes: float | None
    mean_status: FrameMeanStatus
    mean_status_reasons: tuple[str, ...]
    mean_coverage_fraction: float | None
    approximate_bin_midpoint_mean_wire_frame_size_bytes: float | None
    quantile_results: tuple[FrameQuantileResult, ...]


@dataclass(frozen=True)
class FrameAnalysis:
    contract_version: int
    population: FramePopulation
    status: FrameStatus
    quality_flags: frozenset[FrameQualityFlag]
    status_reasons: tuple[str, ...]
    requested_window_ns: tuple[int, int]
    direction_results: tuple[FrameDirectionResult, ...]
    interpretation: str
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class FrameDataset:
    population: FramePopulation
    requested_counters: tuple[FrameCounterSpec, ...]
    series: tuple[FrameCounterSeries, ...]
    bin_edges: tuple[FrameBinEdgeSet, ...]
    byte_count_compatibility: tuple[FrameBytePopulationCompatibility, ...]
    window_ns: tuple[int, int]

    def __post_init__(self) -> None:
        counter_ids = [spec.counter_id for spec in self.requested_counters]
        if len(counter_ids) != len(set(counter_ids)):
            raise ValueError("requested counter IDs must be unique")
        series_ids = [series.spec.counter_id for series in self.series]
        if len(series_ids) != len(set(series_ids)):
            raise ValueError("series counter IDs must be unique")
        if len(self.window_ns) != 2:
            raise ValueError("window_ns must contain start and end")
        if type(self.window_ns[0]) is not int or type(self.window_ns[1]) is not int:
            raise ValueError("window bounds must be integers")
        if not self.window_ns[0] < self.window_ns[1]:
            raise ValueError("window end must be greater than window start")
        requested_specs = set(self.requested_counters)
        for series in self.series:
            if series.spec not in requested_specs:
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
        edge_directions = [edge_set.direction for edge_set in self.bin_edges]
        if len(edge_directions) != len(set(edge_directions)):
            raise ValueError("each direction may have only one bin edge set")
        compatibility_directions = [
            compatibility.direction
            for compatibility in self.byte_count_compatibility
        ]
        if len(compatibility_directions) != len(set(compatibility_directions)):
            raise ValueError("each direction may have only one compatibility record")
        for compatibility in self.byte_count_compatibility:
            byte_specs = [
                spec
                for spec in self.requested_counters
                if spec.counter_id == compatibility.byte_counter_id
                and spec.counter_kind is FrameCounterKind.BYTE_COUNT
            ]
            if len(byte_specs) != 1:
                raise ValueError("compatibility must name one requested byte counter")
            if not compatibility.compatible:
                continue
            frame_population_ids = {
                spec.population_id
                for spec in self.requested_counters
                if spec.direction is compatibility.direction
                and spec.counter_kind is FrameCounterKind.FRAME_BIN
            }
            if frame_population_ids != {compatibility.frame_population_id}:
                raise ValueError("compatibility frame population does not match bins")
            if byte_specs[0].population_id != compatibility.byte_population_id:
                raise ValueError("compatibility byte population does not match counter")


def frame_bin_spec(
    counter_id: str,
    direction: FrameDirection,
    bin_id: str,
    population_id: str,
) -> FrameCounterSpec:
    return FrameCounterSpec(
        counter_id=counter_id,
        direction=direction,
        counter_kind=FrameCounterKind.FRAME_BIN,
        unit=FRAME_UNIT,
        population_id=population_id,
        bin_id=bin_id,
    )


def byte_count_spec(
    counter_id: str,
    direction: FrameDirection,
    population_id: str,
) -> FrameCounterSpec:
    return FrameCounterSpec(
        counter_id=counter_id,
        direction=direction,
        counter_kind=FrameCounterKind.BYTE_COUNT,
        unit=BYTE_UNIT,
        population_id=population_id,
    )


def _missing_result(
    spec: FrameCounterSpec,
    window_ns: tuple[int, int],
) -> FrameCounterResult:
    requested_duration_ns = window_ns[1] - window_ns[0]
    return FrameCounterResult(
        spec=spec,
        source_identity=None,
        status=FrameStatus.MISSING_DATA,
        quality_flags=frozenset({FrameQualityFlag.MISSING_COUNTER}),
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
        rate_per_second=None,
        zero_observed=False,
        persistent_zero=False,
        intervals=(),
    )


def _window_sample_count(series, window_ns):
    return sum(
        1
        for timestamp in series.timestamps_ns
        if window_ns[0] <= timestamp <= window_ns[1]
    )


def _unsupported_result(
    series: FrameCounterSeries,
    window_ns: tuple[int, int],
    quality_flags: set[FrameQualityFlag],
    reasons: list[str],
) -> FrameCounterResult:
    requested_duration_ns = window_ns[1] - window_ns[0]
    window_sample_count = _window_sample_count(series, window_ns)
    return FrameCounterResult(
        spec=series.spec,
        source_identity=series.identity,
        status=FrameStatus.UNSUPPORTED,
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
        rate_per_second=None,
        zero_observed=False,
        persistent_zero=False,
        intervals=(),
    )


def _analyze_counter(
    series: FrameCounterSeries,
    window_ns: tuple[int, int],
    bin_edge: FrameBinEdge | None,
) -> FrameCounterResult:
    window_sample_count = _window_sample_count(series, window_ns)
    quality_flags: set[FrameQualityFlag] = set(series.quality_flags)
    reasons: list[str] = []
    if series.value_kind is not FrameValueKind.CUMULATIVE:
        quality_flags.add(FrameQualityFlag.UNSUPPORTED_SERIES)
        reasons.append("Only cumulative counter series are supported.")
    if series.baseline_relative:
        quality_flags.add(FrameQualityFlag.BASELINE_RELATIVE)
        reasons.append(
            "Baseline-relative counters cannot support numeric frame totals."
        )
    if reasons:
        return _unsupported_result(series, window_ns, quality_flags, reasons)

    records = list(zip(series.timestamps_ns, series.values, strict=True))
    ordered = sorted(records, key=lambda record: record[0])
    if records != ordered:
        quality_flags.add(FrameQualityFlag.REORDERED)

    intervals: list[FrameCounterInterval] = []
    valid_intervals: list[FrameCounterInterval] = []
    invalid_interval_count = 0
    outside_window_interval_count = 0
    index = 0
    while index < len(ordered) - 1:
        start_timestamp, start_value = ordered[index]
        end_timestamp, end_value = ordered[index + 1]
        if start_timestamp == end_timestamp:
            if start_value == end_value:
                quality_flags.add(FrameQualityFlag.DUPLICATE)
                index += 1
                continue
            quality_flags.add(FrameQualityFlag.CONFLICTING_TIMESTAMP)
            interval = FrameCounterInterval(
                start_ns=start_timestamp,
                end_ns=end_timestamp,
                duration_ns=0,
                start_value=start_value,
                end_value=end_value,
                delta=0,
                valid=False,
                state=FrameIntervalState.CONFLICTING_TIMESTAMP,
                inside_window=window_ns[0] <= start_timestamp <= window_ns[1],
            )
            intervals.append(interval)
            if interval.inside_window:
                invalid_interval_count += 1
            else:
                outside_window_interval_count += 1
            index += 1
            continue

        delta = end_value - start_value
        state = (
            FrameIntervalState.VALID
            if delta >= 0
            else FrameIntervalState.DECREASE_OR_RESET
        )
        if delta < 0:
            quality_flags.add(FrameQualityFlag.DECREASE_OR_RESET)
        interval = FrameCounterInterval(
            start_ns=start_timestamp,
            end_ns=end_timestamp,
            duration_ns=end_timestamp - start_timestamp,
            start_value=start_value,
            end_value=end_value,
            delta=delta,
            valid=delta >= 0,
            state=state,
            inside_window=(
                window_ns[0] <= start_timestamp and end_timestamp <= window_ns[1]
            ),
        )
        intervals.append(interval)
        if not interval.inside_window:
            outside_window_interval_count += 1
            if (
                start_timestamp < window_ns[0] < end_timestamp
                or start_timestamp < window_ns[1] < end_timestamp
            ):
                quality_flags.add(FrameQualityFlag.BOUNDARY_INTERVAL_EXCLUDED)
        elif interval.valid:
            valid_intervals.append(interval)
        else:
            invalid_interval_count += 1
        index += 1

    selected_intervals = tuple(
        interval for interval in intervals if interval.inside_window
    )
    covered_duration_ns = sum(interval.duration_ns for interval in valid_intervals)
    total = sum(interval.delta for interval in valid_intervals)
    rate_per_second = (
        total / (covered_duration_ns / NANOSECONDS_PER_SECOND)
        if covered_duration_ns > 0
        else None
    )
    represented_window_ns = (
        (
            min(interval.start_ns for interval in valid_intervals),
            max(interval.end_ns for interval in valid_intervals),
        )
        if valid_intervals
        else None
    )
    if covered_duration_ns < window_ns[1] - window_ns[0]:
        quality_flags.add(FrameQualityFlag.PARTIAL_WINDOW)
    if not valid_intervals:
        status = FrameStatus.INSUFFICIENT_DATA
        reasons.append("The window contains no complete valid counter interval.")
    elif invalid_interval_count:
        status = FrameStatus.PARTIAL_INVALID_DATA
        reasons.append("One or more intervals were excluded for decrease or reset.")
    elif total == 0:
        status = FrameStatus.OBSERVED_ZERO
    else:
        status = FrameStatus.OBSERVED_NONZERO

    return FrameCounterResult(
        spec=series.spec,
        source_identity=series.identity,
        status=status,
        quality_flags=frozenset(quality_flags),
        status_reasons=tuple(reasons),
        requested_window_ns=window_ns,
        represented_window_ns=represented_window_ns,
        covered_duration_ns=covered_duration_ns,
        requested_duration_ns=window_ns[1] - window_ns[0],
        coverage_fraction=covered_duration_ns / (window_ns[1] - window_ns[0]),
        sample_count=window_sample_count,
        valid_interval_count=len(valid_intervals),
        invalid_interval_count=invalid_interval_count,
        outside_window_interval_count=outside_window_interval_count,
        total=total,
        unit=series.spec.unit,
        rate_per_second=rate_per_second,
        zero_observed=bool(valid_intervals) and total == 0,
        persistent_zero=(
            bool(valid_intervals)
            and all(interval.delta == 0 for interval in valid_intervals)
            and invalid_interval_count == 0
        ),
        intervals=selected_intervals,
        bin_edge=bin_edge,
    )


def _valid_boundaries(result: FrameCounterResult) -> tuple[tuple[int, int], ...]:
    return tuple(
        (interval.start_ns, interval.end_ns)
        for interval in result.intervals
        if interval.valid
    )


def _combine_status(results: tuple[FrameCounterResult, ...]) -> FrameStatus:
    if not results:
        return FrameStatus.UNSUPPORTED
    if all(result.status is FrameStatus.MISSING_DATA for result in results):
        return FrameStatus.MISSING_DATA
    if any(result.status is FrameStatus.MISSING_DATA for result in results):
        return FrameStatus.PARTIAL_MISSING_DATA
    if all(result.status is FrameStatus.UNSUPPORTED for result in results):
        return FrameStatus.UNSUPPORTED
    if any(result.status is FrameStatus.UNSUPPORTED for result in results):
        return FrameStatus.PARTIAL_UNSUPPORTED_DATA
    if not any(result.valid_interval_count for result in results):
        return FrameStatus.INSUFFICIENT_DATA
    if any(result.status is FrameStatus.PARTIAL_INVALID_DATA for result in results):
        return FrameStatus.PARTIAL_INVALID_DATA
    if any(result.total for result in results):
        return FrameStatus.OBSERVED_NONZERO
    return FrameStatus.OBSERVED_ZERO


def _quantile_results(
    bin_results: tuple[FrameCounterResult, ...],
    edges_complete: bool,
    distribution_supported: bool,
) -> tuple[FrameQuantileResult, ...]:
    if not edges_complete:
        return tuple(
            FrameQuantileResult(
                quantile=quantile,
                status=FrameQuantileStatus.UNSUPPORTED,
                lower_bytes=None,
                upper_bytes=None,
                status_reasons=("Sourced bin edges are required for quantile bounds.",),
            )
            for quantile in QUANTILES
        )
    if not distribution_supported:
        return tuple(
            FrameQuantileResult(
                quantile=quantile,
                status=FrameQuantileStatus.UNSUPPORTED,
                lower_bytes=None,
                upper_bytes=None,
                status_reasons=(
                    "All frame bins require identical valid intervals for a distribution.",
                ),
            )
            for quantile in QUANTILES
        )
    ordered_results = sorted(
        (result for result in bin_results if result.bin_edge is not None),
        key=lambda result: result.bin_edge.lower_bytes,
    )
    total = sum(result.total or 0 for result in ordered_results)
    if total <= 0:
        return tuple(
            FrameQuantileResult(
                quantile=quantile,
                status=FrameQuantileStatus.UNSUPPORTED,
                lower_bytes=None,
                upper_bytes=None,
                status_reasons=("A nonzero frame total is required.",),
            )
            for quantile in QUANTILES
        )
    quantile_results: list[FrameQuantileResult] = []
    for quantile in QUANTILES:
        threshold = quantile * total
        cumulative = 0
        selected = ordered_results[-1]
        for result in ordered_results:
            cumulative += result.total or 0
            if cumulative >= threshold:
                selected = result
                break
        edge = selected.bin_edge
        assert edge is not None
        if edge.open_ended:
            quantile_results.append(
                FrameQuantileResult(
                    quantile=quantile,
                    status=FrameQuantileStatus.OPEN_ENDED,
                    lower_bytes=edge.lower_bytes,
                    upper_bytes=None,
                    status_reasons=(
                        "The quantile falls in an open-ended tail; no upper bound is available.",
                    ),
                )
            )
        else:
            quantile_results.append(
                FrameQuantileResult(
                    quantile=quantile,
                    status=FrameQuantileStatus.BOUNDED,
                    lower_bytes=edge.lower_bytes,
                    upper_bytes=edge.upper_bytes,
                    status_reasons=(),
                )
            )
    return tuple(quantile_results)


def _direction_status_reasons(status: FrameStatus) -> tuple[str, ...]:
    return {
        FrameStatus.MISSING_DATA: ("All requested frame bins are missing.",),
        FrameStatus.PARTIAL_MISSING_DATA: (
            "At least one requested frame bin is missing.",
        ),
        FrameStatus.PARTIAL_INVALID_DATA: (
            "Valid frame-bin intervals are reported after excluding resets.",
        ),
        FrameStatus.PARTIAL_UNSUPPORTED_DATA: (
            "At least one frame-bin series has unsupported semantics.",
        ),
        FrameStatus.INSUFFICIENT_DATA: (
            "No frame bin has a complete valid interval in the window.",
        ),
        FrameStatus.UNSUPPORTED: ("No frame bins were requested.",),
    }.get(status, ())


def _analyze_direction(
    direction: FrameDirection,
    dataset: FrameDataset,
) -> FrameDirectionResult:
    requested_specs = [
        spec
        for spec in dataset.requested_counters
        if spec.direction is direction
        and spec.counter_kind is FrameCounterKind.FRAME_BIN
    ]
    series_by_id = {series.spec.counter_id: series for series in dataset.series}
    edge_set = next(
        (edges for edges in dataset.bin_edges if edges.direction is direction),
        None,
    )
    edges_by_bin = (
        {edge.bin_id: edge for edge in edge_set.edges} if edge_set is not None else {}
    )
    bin_results = tuple(
        _analyze_counter(
            series_by_id[spec.counter_id],
            dataset.window_ns,
            edges_by_bin.get(spec.bin_id),
        )
        if spec.counter_id in series_by_id
        else _missing_result(spec, dataset.window_ns)
        for spec in requested_specs
    )
    byte_specs = [
        spec
        for spec in dataset.requested_counters
        if spec.direction is direction
        and spec.counter_kind is FrameCounterKind.BYTE_COUNT
    ]
    byte_result = (
        _analyze_counter(series_by_id[byte_specs[0].counter_id], dataset.window_ns, None)
        if len(byte_specs) == 1 and byte_specs[0].counter_id in series_by_id
        else (
            _missing_result(byte_specs[0], dataset.window_ns)
            if len(byte_specs) == 1
            else None
        )
    )
    quality_flags: set[FrameQualityFlag] = set()
    status_reasons: list[str] = []
    for result in bin_results:
        quality_flags.update(result.quality_flags)
        status_reasons.extend(result.status_reasons)
    if byte_result is not None:
        quality_flags.update(byte_result.quality_flags)
        status_reasons.extend(byte_result.status_reasons)

    edges_complete = bool(requested_specs) and all(
        result.bin_edge is not None for result in bin_results
    )
    if requested_specs and not edges_by_bin:
        quality_flags.add(FrameQualityFlag.MISSING_BIN_EDGES)
        status_reasons.append(
            "Named-bin counts are reported without inferred bin edges or quantiles."
        )
    elif requested_specs and not edges_complete:
        quality_flags.add(FrameQualityFlag.MISSING_BIN_EDGE)
        status_reasons.append("At least one requested frame bin has no sourced edge.")
    open_ended_tail = any(
        result.bin_edge is not None and result.bin_edge.open_ended
        for result in bin_results
    )
    if open_ended_tail:
        quality_flags.add(FrameQualityFlag.OPEN_ENDED_BIN)

    supported_bins = tuple(
        result
        for result in bin_results
        if result.status in {FrameStatus.OBSERVED_ZERO, FrameStatus.OBSERVED_NONZERO}
    )
    common_boundaries: tuple[tuple[int, int], ...] | None = None
    boundaries_compatible = bool(supported_bins) and all(
        _valid_boundaries(result) == _valid_boundaries(supported_bins[0])
        for result in supported_bins
    )
    if boundaries_compatible and len(supported_bins) == len(bin_results):
        common_boundaries = _valid_boundaries(supported_bins[0])
    else:
        quality_flags.add(FrameQualityFlag.INCOMPATIBLE_COVERAGE)

    bin_total: int | None = None
    if common_boundaries is not None:
        bin_total = sum(result.total or 0 for result in bin_results)
        bin_results = tuple(
            replace(
                result,
                bin_share=(result.total or 0) / bin_total,
            )
            for result in bin_results
        )

    temporal_intervals: tuple[FrameTemporalInterval, ...] = ()
    if common_boundaries:
        temporal_intervals = tuple(
            FrameTemporalInterval(
                start_ns=start_ns,
                end_ns=end_ns,
                duration_ns=end_ns - start_ns,
                bin_deltas=tuple(
                    (result.spec.bin_id or "", interval.delta)
                    for result in bin_results
                    for interval in result.intervals
                    if interval.valid
                    and interval.start_ns == start_ns
                    and interval.end_ns == end_ns
                ),
                total_frames=sum(
                    interval.delta
                    for result in bin_results
                    for interval in result.intervals
                    if interval.valid
                    and interval.start_ns == start_ns
                    and interval.end_ns == end_ns
                ),
            )
            for start_ns, end_ns in common_boundaries
        )

    compatibility = next(
        (
            record
            for record in dataset.byte_count_compatibility
            if record.direction is direction
        ),
        None,
    )
    mean_reasons: list[str] = []
    mean_status = FrameMeanStatus.UNSUPPORTED
    mean_value: float | None = None
    mean_coverage: float | None = None
    if not requested_specs:
        mean_reasons.append("No frame-bin counters were requested.")
    elif not edges_complete:
        mean_reasons.append("Sourced bin edges are required for the frame mean.")
    elif byte_result is None:
        mean_reasons.append("No direction byte counter was requested.")
    elif compatibility is None:
        mean_reasons.append("No byte/frame population compatibility record was supplied.")
        quality_flags.add(FrameQualityFlag.INCOMPATIBLE_BYTE_COUNT_POPULATION)
    elif not compatibility.compatible:
        mean_reasons.append(
            "The supplied definitions do not permit byte/count population compatibility."
        )
        quality_flags.add(FrameQualityFlag.INCOMPATIBLE_BYTE_COUNT_POPULATION)
    elif common_boundaries is None or byte_result.total is None:
        mean_reasons.append("Byte and frame counters lack identical valid intervals.")
    elif _valid_boundaries(byte_result) != common_boundaries:
        mean_reasons.append("Byte and frame counters lack identical valid intervals.")
        quality_flags.add(FrameQualityFlag.INCOMPATIBLE_COVERAGE)
    elif bin_total == 0:
        mean_status = FrameMeanStatus.ZERO_DENOMINATOR
        mean_reasons.append("The compatible frame-count denominator is zero.")
        mean_coverage = (
            sum(end_ns - start_ns for start_ns, end_ns in common_boundaries)
            / (dataset.window_ns[1] - dataset.window_ns[0])
        )
    else:
        assert byte_result.total is not None
        mean_value = byte_result.total / bin_total
        mean_status = FrameMeanStatus.SUPPORTED
        mean_coverage = (
            sum(end_ns - start_ns for start_ns, end_ns in common_boundaries)
            / (dataset.window_ns[1] - dataset.window_ns[0])
        )

    approximate_bin_mean: float | None = None
    if (
        edges_complete
        and bin_total is not None
        and bin_total > 0
        and not open_ended_tail
    ):
        approximate_bin_mean = sum(
            (result.total or 0)
            * (
                (
                    result.bin_edge.lower_bytes
                    + (result.bin_edge.upper_bytes or result.bin_edge.lower_bytes)
                )
                / 2
            )
            for result in bin_results
            if result.bin_edge is not None
        ) / bin_total

    status = _combine_status(bin_results)
    direction_reasons = _direction_status_reasons(status)
    covered_duration_ns = (
        sum(end_ns - start_ns for start_ns, end_ns in common_boundaries)
        if common_boundaries is not None
        else None
    )
    represented_window_ns = (
        (common_boundaries[0][0], common_boundaries[-1][1])
        if common_boundaries
        else None
    )

    return FrameDirectionResult(
        direction=direction,
        status=status,
        quality_flags=frozenset(quality_flags),
        status_reasons=tuple([*direction_reasons, *status_reasons, *mean_reasons]),
        requested_window_ns=dataset.window_ns,
        represented_window_ns=represented_window_ns,
        covered_duration_ns=covered_duration_ns,
        requested_duration_ns=dataset.window_ns[1] - dataset.window_ns[0],
        coverage_fraction=(
            covered_duration_ns / (dataset.window_ns[1] - dataset.window_ns[0])
            if covered_duration_ns is not None
            else None
        ),
        bin_results=bin_results,
        byte_result=byte_result,
        bin_total=bin_total,
        byte_total=byte_result.total if byte_result is not None else None,
        temporal_intervals=temporal_intervals,
        bin_edges_source=edge_set.source if edge_set is not None else None,
        bin_edges_complete=edges_complete,
        open_ended_tail=open_ended_tail,
        mean_wire_frame_size_bytes=mean_value,
        mean_status=mean_status,
        mean_status_reasons=tuple(mean_reasons),
        mean_coverage_fraction=mean_coverage,
        approximate_bin_midpoint_mean_wire_frame_size_bytes=approximate_bin_mean,
        quantile_results=_quantile_results(
            bin_results,
            edges_complete,
            common_boundaries is not None,
        ),
    )


def analyze_frames(dataset: FrameDataset) -> FrameAnalysis:
    """Analyze directional wire-frame distributions without hidden assumptions."""

    directions = tuple(
        direction
        for direction in FrameDirection
        if any(spec.direction is direction for spec in dataset.requested_counters)
    )
    direction_results = tuple(
        _analyze_direction(direction, dataset) for direction in directions
    )
    quality_flags: set[FrameQualityFlag] = set()
    status_reasons: list[str] = []
    for result in direction_results:
        quality_flags.update(result.quality_flags)
        status_reasons.extend(result.status_reasons)
    status = (
        _combine_status(
            tuple(
                result
                for direction in direction_results
                for result in direction.bin_results
            )
        )
        if direction_results
        else FrameStatus.UNSUPPORTED
    )
    return FrameAnalysis(
        contract_version=FRAME_ANALYSIS_VERSION,
        population=dataset.population,
        status=status,
        quality_flags=frozenset(quality_flags),
        status_reasons=tuple(status_reasons),
        requested_window_ns=dataset.window_ns,
        direction_results=direction_results,
        interpretation=FRAME_INTERPRETATION,
        limitations=FRAME_LIMITATIONS,
    )
