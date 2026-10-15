"""Pure latency-histogram analysis for NIC cumulative counters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from traffic_analysis import MetricSeries, SeriesQualityFlag, SeriesIdentity, ValueKind


LATENCY_ANALYSIS_VERSION = 1
LATENCY_BIN_DEFINITION_VERSION = 1
DELTA_FORMULA = "end_value - start_value"
BOUNDARY_DECUMULATION_FORMULA = (
    "boundary delta[i] - boundary delta[i-1], with delta[-1] = 0"
)
NORMALIZED_FRACTION_FORMULA = "disjoint bin count / total observations"
QUANTILE_SELECTION_FORMULA = "smallest disjoint-bin cumulative count >= quantile * total"
MEAN_ESTIMATE_FORMULA = (
    "sum(disjoint bin count * bin midpoint) / total observations"
)

LATENCY_INTERPRETATION = (
    "Request/response and host-access latency are separate NIC histogram "
    "populations. Neither is an ATU-latency histogram, an MPI API duration, "
    "or a substitute for the other."
)
LATENCY_LIMITATIONS = (
    "Named-bin counts are unavailable for missing bins and are distinct from observed zero.",
    "Normalization and observation totals require verified disjoint bins after any boundary de-cumulation.",
    "Quantiles are reported as bin bounds, not exact latency values; a midpoint mean is only an estimate.",
    "An open-ended tail provides a lower bound but no upper bound or exact midpoint mean.",
    "Reset, nonmonotonic timestamp, and incompatible-encoding intervals are excluded from totals and coverage.",
    "Coverage is measured only over complete intervals between observed samples.",
    "NIC histograms and MPI API durations are distinct, and per-NIC quantiles must not be averaged into a job quantile.",
)


class LatencyHistogramGroup(str, Enum):
    REQUEST_RESPONSE = "req_rsp_latency"
    HOST_ACCESS = "host_access_latency"


class LatencyBinEncoding(str, Enum):
    DISJOINT = "disjoint"
    BOUNDARY_CUMULATIVE = "boundary_cumulative"
    UNKNOWN = "unknown"


class LatencyStatus(str, Enum):
    OBSERVED_ZERO = "observed_zero"
    OBSERVED_NONZERO = "observed_nonzero"
    ZERO_NAMED_BINS_ONLY = "zero_named_bins_only"
    NONZERO_NAMED_BINS_ONLY = "nonzero_named_bins_only"
    PARTIAL_MISSING_DATA = "partial_missing_data"
    PARTIAL_INVALID_DATA = "partial_invalid_data"
    PARTIAL_UNSUPPORTED_SERIES = "partial_unsupported_series"
    MISSING_DATA = "missing_data"
    INSUFFICIENT_DATA = "insufficient_data"
    INVALID_DATA = "invalid_data"
    UNSUPPORTED_SERIES = "unsupported_series"
    UNSUPPORTED_ENCODING = "unsupported_encoding"


class LatencyQualityFlag(str, Enum):
    MISSING_BIN = "missing_bin"
    UNSUPPORTED_SERIES = "unsupported_series"
    MISSING_BIN_EDGES = "missing_bin_edges"
    MISSING_EDGE_UNITS = "missing_edge_units"
    OPEN_TAIL = "open_tail"
    BASELINE_RELATIVE = "baseline_relative"
    DECREASE_OR_RESET = "decrease_or_reset"
    REORDERED = "reordered"
    DUPLICATE = "duplicate"
    CONFLICTING_TIMESTAMP = "conflicting_timestamp"
    INCOMPATIBLE_ENCODING = "incompatible_encoding"
    INCOMPATIBLE_TIMESTAMPS = "incompatible_timestamps"
    PARTIAL_WINDOW = "partial_window"
    BOUNDARY_INTERVAL_EXCLUDED = "boundary_interval_excluded"
    ATU_LATENCY_NOT_IDENTIFIED = "atu_latency_not_identified"


class LatencyIntervalState(str, Enum):
    VALID = "valid"
    DECREASE_OR_RESET = "decrease_or_reset"
    CONFLICTING_TIMESTAMP = "conflicting_timestamp"
    INCOMPATIBLE_ENCODING = "incompatible_encoding"


class LatencyCountKind(str, Enum):
    DISJOINT_BIN_COUNT = "disjoint_bin_count"
    NAMED_COUNTER_DELTA = "named_counter_delta"


class LatencyQuantileStatus(str, Enum):
    BOUNDED = "bounded"
    OPEN_TAIL = "open_tail"
    UNSUPPORTED = "unsupported"


class LatencyMeanStatus(str, Enum):
    ESTIMATED = "estimated"
    OPEN_TAIL = "open_tail"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class LatencyBinEdge:
    lower: float
    upper: float | None
    open_tailed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.lower, (int, float)) or not math.isfinite(self.lower):
            raise ValueError("bin lower edge must be a finite number")
        if self.lower < 0:
            raise ValueError("bin lower edge must be nonnegative")
        if self.open_tailed:
            if self.upper is not None:
                raise ValueError("an open-tailed bin cannot have an upper edge")
        elif (
            not isinstance(self.upper, (int, float))
            or not math.isfinite(self.upper)
            or self.upper <= self.lower
        ):
            raise ValueError("a closed bin upper edge must exceed its lower edge")


@dataclass(frozen=True)
class LatencyBinDefinition:
    definition_version: int
    group: LatencyHistogramGroup
    bin_names: tuple[str, ...]
    encoding: LatencyBinEncoding
    source: str
    bin_edges: tuple[LatencyBinEdge, ...] | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        if type(self.definition_version) is not int or self.definition_version <= 0:
            raise ValueError("definition_version must be a positive integer")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("bin definition source must be a nonempty string")
        if not self.bin_names or any(not name for name in self.bin_names):
            raise ValueError("bin_names must contain nonempty strings")
        if len(set(self.bin_names)) != len(self.bin_names):
            raise ValueError("bin_names must be unique")
        if self.encoding is LatencyBinEncoding.UNKNOWN:
            if self.bin_edges is not None or self.unit is not None:
                raise ValueError(
                    "unknown encoding cannot claim sourced bin edges or units"
                )
            return
        if self.bin_edges is None:
            return
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError("sourced bin edges require a nonempty unit")
        if len(self.bin_edges) != len(self.bin_names):
            raise ValueError("bin_edges and bin_names must have equal length")
        previous_upper: float | None = None
        open_tail_seen = False
        for edge in self.bin_edges:
            if open_tail_seen:
                raise ValueError("an open-tailed bin must be the final bin")
            if previous_upper is not None and edge.lower != previous_upper:
                raise ValueError("closed bins must be contiguous and nonoverlapping")
            if self.encoding is LatencyBinEncoding.BOUNDARY_CUMULATIVE:
                if edge.open_tailed or edge.upper is None:
                    raise ValueError("boundary-cumulative bins cannot be open-tailed")
            if edge.open_tailed:
                open_tail_seen = True
            previous_upper = edge.upper


@dataclass(frozen=True)
class LatencyCounterSpec:
    group: LatencyHistogramGroup
    bin_index: int


@dataclass(frozen=True)
class LatencyPopulation:
    run_id: str
    node_id: str
    nic_id: str
    collector_id: str
    direction: str

    @classmethod
    def from_identity(cls, identity: SeriesIdentity) -> "LatencyPopulation":
        direction = (
            "not_directional"
            if identity.counter_id.startswith(tuple(_COUNTER_PREFIXES.values()))
            else identity.direction.value
        )
        return cls(
            run_id=identity.run_id,
            node_id=identity.node_id,
            nic_id=identity.nic_id,
            collector_id=identity.collector_id,
            direction=direction,
        )


@dataclass(frozen=True)
class LatencyInterval:
    start_ns: int
    end_ns: int
    duration_ns: int
    start_values: tuple[int | float, ...]
    end_values: tuple[int | float, ...]
    bin_deltas: tuple[int | float, ...]
    disjoint_bin_counts: tuple[int | float, ...] | None
    valid: bool
    state: LatencyIntervalState


@dataclass(frozen=True)
class LatencyBinResult:
    bin_name: str
    bin_index: int
    source_counter: str | None
    count: int | float | None
    count_kind: LatencyCountKind | None
    normalized_fraction: float | None
    lower_edge: float | None
    upper_edge: float | None
    open_tailed: bool


@dataclass(frozen=True)
class LatencyQuantileResult:
    quantile: float
    status: LatencyQuantileStatus
    bin_name: str | None
    lower_edge: float | None
    upper_edge: float | None
    reason: str


@dataclass(frozen=True)
class LatencyMeanResult:
    status: LatencyMeanStatus
    value: float | None
    unit: str | None
    formula: str | None
    reason: str


@dataclass(frozen=True)
class LatencyHistogramResult:
    group: LatencyHistogramGroup
    population: LatencyPopulation
    status: LatencyStatus
    quality_flags: frozenset[LatencyQualityFlag]
    status_reasons: tuple[str, ...]
    requested_window_ns: tuple[int, int] | None
    represented_window_ns: tuple[int, int] | None
    bin_definition_version: int
    bin_definition_source: str
    encoding: LatencyBinEncoding
    bins_are_disjoint: bool
    bin_unit: str | None
    bin_results: tuple[LatencyBinResult, ...]
    total_observations: int | float | None
    normalized_distribution_available: bool
    quantile_results: tuple[LatencyQuantileResult, ...]
    mean_result: LatencyMeanResult
    sample_count: int
    interval_count: int
    valid_interval_count: int
    invalid_interval_count: int
    reset_excluded_interval_count: int
    boundary_excluded_interval_count: int
    covered_duration_ns: int
    requested_duration_ns: int
    uncovered_requested_duration_ns: int
    coverage_fraction: float | None
    intervals: tuple[LatencyInterval, ...]
    delta_formula: str
    boundary_decumulation_formula: str | None
    normalized_fraction_formula: str
    quantile_selection_formula: str
    mean_estimate_formula: str | None
    atu_latency_substitute: bool
    interpretation: str
    limitations: tuple[str, ...]


QUANTILES = (0.5, 0.99)
_COUNTER_PREFIXES: dict[LatencyHistogramGroup, str] = {
    LatencyHistogramGroup.REQUEST_RESPONSE: "pct_req_rsp_latency_",
    LatencyHistogramGroup.HOST_ACCESS: "pct_host_access_latency_",
}


def latency_counter_spec(counter_id: str) -> LatencyCounterSpec | None:
    for group, prefix in _COUNTER_PREFIXES.items():
        if not counter_id.startswith(prefix):
            continue
        suffix = counter_id[len(prefix) :]
        if not suffix.isdigit():
            return None
        return LatencyCounterSpec(group=group, bin_index=int(suffix))
    return None


def _window_sample_count(
    timestamps_ns: tuple[int, ...], requested_window_ns: tuple[int, int] | None
) -> int:
    if requested_window_ns is None:
        return len(timestamps_ns)
    return sum(
        1
        for timestamp in timestamps_ns
        if requested_window_ns[0] <= timestamp <= requested_window_ns[1]
    )


def analyze_latency_histograms(
    series: Sequence[MetricSeries],
    bin_definitions: Mapping[LatencyHistogramGroup, LatencyBinDefinition],
    *,
    requested_window_ns: tuple[int, int] | None = None,
    quantiles: Sequence[float] = QUANTILES,
) -> tuple[LatencyHistogramResult, ...]:
    """Analyze each run/node/NIC latency histogram without cross-population pooling."""
    if requested_window_ns is not None:
        window_start, window_end = requested_window_ns
        if type(window_start) is not int or type(window_end) is not int:
            raise ValueError("requested_window_ns must contain integers")
        if window_end <= window_start:
            raise ValueError("requested_window_ns end must exceed start")
    checked_quantiles = tuple(quantiles)
    if not checked_quantiles or any(
        type(quantile) is not float
        or not math.isfinite(quantile)
        or not 0 < quantile < 1
        for quantile in checked_quantiles
    ):
        raise ValueError("quantiles must contain finite floats strictly between zero and one")
    if len(set(checked_quantiles)) != len(checked_quantiles):
        raise ValueError("quantiles must be unique")

    definitions: dict[LatencyHistogramGroup, LatencyBinDefinition] = {}
    for group, definition in bin_definitions.items():
        if definition.group is not group:
            raise ValueError("bin definition group does not match its mapping key")
        definitions[group] = definition

    grouped: dict[tuple[LatencyHistogramGroup, LatencyPopulation], dict[int, MetricSeries]] = {}
    for metric_series in series:
        spec = latency_counter_spec(metric_series.identity.counter_id)
        if spec is None:
            raise ValueError(
                f"unsupported latency counter: {metric_series.identity.counter_id}"
            )
        if spec.group not in definitions:
            raise ValueError(f"no bin definition supplied for {spec.group.value}")
        if not 0 <= spec.bin_index < len(definitions[spec.group].bin_names):
            raise ValueError(
                f"bin index {spec.bin_index} is outside its supplied definition"
            )
        key = (spec.group, LatencyPopulation.from_identity(metric_series.identity))
        population_bins = grouped.setdefault(key, {})
        if spec.bin_index in population_bins:
            raise ValueError(
                f"duplicate latency bin series: {metric_series.identity.counter_id}"
            )
        population_bins[spec.bin_index] = metric_series

    return tuple(
        _population_result(
            group=group,
            population=population,
            definition=definition,
            series_by_bin=series_by_bin,
            requested_window_ns=requested_window_ns,
            quantiles=checked_quantiles,
        )
        for (group, population), series_by_bin in grouped.items()
        if (definition := definitions[group]) is not None
    )


def _population_result(
    *,
    group: LatencyHistogramGroup,
    population: LatencyPopulation,
    definition: LatencyBinDefinition,
    series_by_bin: Mapping[int, MetricSeries],
    requested_window_ns: tuple[int, int] | None,
    quantiles: tuple[float, ...],
) -> LatencyHistogramResult:
    quality_flags: set[LatencyQualityFlag] = {
        LatencyQualityFlag.ATU_LATENCY_NOT_IDENTIFIED
    }
    reasons: list[str] = []
    explicitly_missing_bins = tuple(
        sorted(
            index
            for index, metric_series in series_by_bin.items()
            if SeriesQualityFlag.MISSING in metric_series.quality_flags
        )
    )
    missing_bins = tuple(
        sorted(
            set(
                index
                for index in range(len(definition.bin_names))
                if index not in series_by_bin
            )
            | set(explicitly_missing_bins)
        )
    )
    unsupported_bins: list[int] = []
    valid_series_by_bin: dict[int, MetricSeries] = {}

    for bin_index, metric_series in series_by_bin.items():
        series_flags = set(metric_series.quality_flags)
        if SeriesQualityFlag.MISSING in series_flags:
            continue
        if SeriesQualityFlag.INVALID in series_flags:
            unsupported_bins.append(bin_index)
            quality_flags.add(LatencyQualityFlag.UNSUPPORTED_SERIES)
            continue
        if metric_series.value_kind is not ValueKind.CUMULATIVE:
            unsupported_bins.append(bin_index)
            quality_flags.add(LatencyQualityFlag.UNSUPPORTED_SERIES)
            continue
        valid_series_by_bin[bin_index] = metric_series
        if metric_series.baseline_relative:
            quality_flags.add(LatencyQualityFlag.BASELINE_RELATIVE)

    if missing_bins:
        quality_flags.add(LatencyQualityFlag.MISSING_BIN)
        reasons.append("Missing bins are unavailable and are not treated as zero.")
    if unsupported_bins:
        quality_flags.add(LatencyQualityFlag.UNSUPPORTED_SERIES)
        reasons.append("Unsupported series cannot contribute histogram counts.")

    intervals: tuple[LatencyInterval, ...] = ()
    incompatible_timestamps = False
    if valid_series_by_bin:
        ordered_series = [valid_series_by_bin[index] for index in sorted(valid_series_by_bin)]
        reference_timestamps = ordered_series[0].timestamps_ns
        incompatible_timestamps = any(
            not np.array_equal(
                metric_series.timestamps_ns, reference_timestamps
            )
            for metric_series in ordered_series[1:]
        )
        if incompatible_timestamps:
            quality_flags.update(
                {
                    LatencyQualityFlag.INCOMPATIBLE_TIMESTAMPS,
                    LatencyQualityFlag.CONFLICTING_TIMESTAMP,
                }
            )
            reasons.append("Bins must share an identical timestamp sequence.")
        elif _strictly_increasing(reference_timestamps):
            intervals = _build_intervals(
                ordered_series,
                definition,
                requested_window_ns,
                quality_flags,
            )
        else:
            quality_flags.update(
                {
                    LatencyQualityFlag.REORDERED,
                    LatencyQualityFlag.CONFLICTING_TIMESTAMP,
                }
            )
            reasons.append("Timestamps must be strictly increasing.")

    valid_intervals = [interval for interval in intervals if interval.valid]
    invalid_intervals = [interval for interval in intervals if not interval.valid]
    reset_count = sum(
        interval.state is LatencyIntervalState.DECREASE_OR_RESET
        for interval in intervals
    )
    boundary_excluded_count = 0
    if valid_series_by_bin and not incompatible_timestamps and _strictly_increasing(
        ordered_series[0].timestamps_ns
    ):
        boundary_excluded_count = sum(
            _overlaps_window(interval, requested_window_ns)
            and not _inside_window(interval, requested_window_ns)
            for interval in _all_intervals(ordered_series, definition)
        )

    if boundary_excluded_count:
        quality_flags.add(LatencyQualityFlag.BOUNDARY_INTERVAL_EXCLUDED)
    first_timestamp = ordered_series[0].timestamps_ns[0] if valid_series_by_bin else None
    last_timestamp = ordered_series[0].timestamps_ns[-1] if valid_series_by_bin else None
    if (
        requested_window_ns is not None
        and first_timestamp is not None
        and last_timestamp is not None
        and (
            first_timestamp < requested_window_ns[0]
            or last_timestamp > requested_window_ns[1]
        )
    ):
            quality_flags.add(LatencyQualityFlag.PARTIAL_WINDOW)

    if any(
        interval.state is LatencyIntervalState.DECREASE_OR_RESET
        for interval in intervals
    ):
        quality_flags.add(LatencyQualityFlag.DECREASE_OR_RESET)
    if any(
        interval.state is LatencyIntervalState.INCOMPATIBLE_ENCODING
        for interval in intervals
    ):
        quality_flags.add(LatencyQualityFlag.INCOMPATIBLE_ENCODING)

    covered_duration_ns = sum(interval.duration_ns for interval in valid_intervals)
    requested_duration_ns = (
        requested_window_ns[1] - requested_window_ns[0]
        if requested_window_ns is not None
        else 0
    )
    uncovered_requested_duration_ns = max(
        0, requested_duration_ns - covered_duration_ns
    )
    coverage_fraction = (
        covered_duration_ns / requested_duration_ns
        if requested_duration_ns > 0
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

    encoding_supported = (
        (
            definition.encoding is LatencyBinEncoding.DISJOINT
            or definition.encoding is LatencyBinEncoding.BOUNDARY_CUMULATIVE
        )
        and not missing_bins
        and not unsupported_bins
    )
    encoding_incompatible = any(
        interval.state is LatencyIntervalState.INCOMPATIBLE_ENCODING
        for interval in intervals
    )
    bins_are_disjoint = encoding_supported and not encoding_incompatible
    if definition.bin_edges is None:
        quality_flags.add(LatencyQualityFlag.MISSING_BIN_EDGES)
    if definition.unit is None or not definition.unit:
        quality_flags.add(LatencyQualityFlag.MISSING_EDGE_UNITS)
    if definition.bin_edges is not None and any(
        edge.open_tailed for edge in definition.bin_edges
    ):
        quality_flags.add(LatencyQualityFlag.OPEN_TAIL)

    counts_kind = (
        LatencyCountKind.DISJOINT_BIN_COUNT
        if bins_are_disjoint
        else LatencyCountKind.NAMED_COUNTER_DELTA
    )
    bin_results_list: list[LatencyBinResult] = []
    for bin_index, bin_name in enumerate(definition.bin_names):
        count = (
            _bin_count(
                bin_index=bin_index,
                intervals=intervals,
                bins_are_disjoint=bins_are_disjoint,
            )
            if bin_index in valid_series_by_bin
            else None
        )
        edge = definition.bin_edges[bin_index] if definition.bin_edges else None
        bin_results_list.append(
            LatencyBinResult(
                bin_name=bin_name,
                bin_index=bin_index,
                source_counter=(
                    series_by_bin[bin_index].identity.counter_id
                    if bin_index in series_by_bin
                    else None
                ),
                count=count,
                count_kind=counts_kind if count is not None else None,
                normalized_fraction=None,
                lower_edge=edge.lower if edge else None,
                upper_edge=edge.upper if edge else None,
                open_tailed=edge.open_tailed if edge else False,
            )
        )

    total_observations = (
        _sum_values(bin_result.count for bin_result in bin_results_list)
        if bins_are_disjoint
        and all(bin_result.count is not None for bin_result in bin_results_list)
        else None
    )
    normalized_available = (
        bins_are_disjoint
        and total_observations is not None
        and total_observations > 0
    )
    if normalized_available:
        for bin_index, bin_result in enumerate(bin_results_list):
            assert bin_result.count is not None
            bin_results_list[bin_index] = LatencyBinResult(
                **{
                    **bin_result.__dict__,
                    "normalized_fraction": float(bin_result.count)
                    / float(total_observations),
                }
            )

    quantile_results = _quantile_results(
        definition=definition,
        bin_results=tuple(bin_results_list),
        total_observations=total_observations,
        normalized_available=normalized_available,
        quantiles=quantiles,
    )
    mean_result = _mean_result(
        definition=definition,
        bin_results=tuple(bin_results_list),
        total_observations=total_observations,
        normalized_available=normalized_available,
    )

    status = _status(
        missing_bins=missing_bins,
        unsupported_bins=unsupported_bins,
        intervals=intervals,
        valid_intervals=valid_intervals,
        invalid_intervals=invalid_intervals,
        encoding_incompatible=encoding_incompatible,
        bins_are_disjoint=bins_are_disjoint,
        total_observations=total_observations,
        named_bin_count=_sum_values(
            bin_result.count
            for bin_result in bin_results_list
            if bin_result.count is not None
        ),
    )
    reasons.extend(_status_reasons(status, quality_flags))
    if LatencyQualityFlag.BASELINE_RELATIVE in quality_flags:
        reasons.append(
            "Baseline-relative values use only differences between observed samples; "
            "an omitted pre-first-sample interval is not inferred."
        )

    return LatencyHistogramResult(
        group=group,
        population=population,
        status=status,
        quality_flags=frozenset(quality_flags),
        status_reasons=tuple(dict.fromkeys(reasons)),
        requested_window_ns=requested_window_ns,
        represented_window_ns=represented_window_ns,
        bin_definition_version=definition.definition_version,
        bin_definition_source=definition.source,
        encoding=definition.encoding,
        bins_are_disjoint=bins_are_disjoint,
        bin_unit=definition.unit,
        bin_results=tuple(bin_results_list),
        total_observations=total_observations,
        normalized_distribution_available=normalized_available,
        quantile_results=quantile_results,
        mean_result=mean_result,
        sample_count=(
            _window_sample_count(
                ordered_series[0].timestamps_ns, requested_window_ns
            )
            if valid_series_by_bin
            else 0
        ),
        interval_count=len(intervals),
        valid_interval_count=len(valid_intervals),
        invalid_interval_count=len(invalid_intervals),
        reset_excluded_interval_count=reset_count,
        boundary_excluded_interval_count=boundary_excluded_count,
        covered_duration_ns=covered_duration_ns,
        requested_duration_ns=requested_duration_ns,
    uncovered_requested_duration_ns=uncovered_requested_duration_ns,
        coverage_fraction=coverage_fraction,
        intervals=intervals,
        delta_formula=DELTA_FORMULA,
        boundary_decumulation_formula=(
            BOUNDARY_DECUMULATION_FORMULA
            if definition.encoding is LatencyBinEncoding.BOUNDARY_CUMULATIVE
            and bins_are_disjoint
            else None
        ),
        normalized_fraction_formula=NORMALIZED_FRACTION_FORMULA,
        quantile_selection_formula=QUANTILE_SELECTION_FORMULA,
        mean_estimate_formula=(
            MEAN_ESTIMATE_FORMULA if mean_result.status is LatencyMeanStatus.ESTIMATED else None
        ),
        atu_latency_substitute=False,
        interpretation=LATENCY_INTERPRETATION,
        limitations=LATENCY_LIMITATIONS,
    )


def _build_intervals(
    ordered_series: Sequence[MetricSeries],
    definition: LatencyBinDefinition,
    requested_window_ns: tuple[int, int] | None,
    quality_flags: set[LatencyQualityFlag],
) -> tuple[LatencyInterval, ...]:
    intervals: list[LatencyInterval] = []
    timestamps = ordered_series[0].timestamps_ns
    for position in range(len(timestamps) - 1):
        start_ns = timestamps[position]
        end_ns = timestamps[position + 1]
        start_values = tuple(
            metric_series.values[position]
            for metric_series in ordered_series
        )
        end_values = tuple(
            metric_series.values[position + 1]
            for metric_series in ordered_series
        )
        interval = _make_interval(
            start_ns=start_ns,
            end_ns=end_ns,
            start_values=start_values,
            end_values=end_values,
            definition=definition,
        )
        if requested_window_ns is not None and not _inside_window(
            interval, requested_window_ns
        ):
            if _overlaps_window(interval, requested_window_ns):
                quality_flags.add(LatencyQualityFlag.BOUNDARY_INTERVAL_EXCLUDED)
            continue
        intervals.append(interval)
    return tuple(intervals)


def _make_interval(
    *,
    start_ns: int,
    end_ns: int,
    start_values: tuple[int | float, ...],
    end_values: tuple[int | float, ...],
    definition: LatencyBinDefinition,
) -> LatencyInterval:
    bin_deltas = tuple(
        _exact_numeric_value(end) - _exact_numeric_value(start)
        for start, end in zip(start_values, end_values)
    )
    state = LatencyIntervalState.VALID
    if any(delta < 0 for delta in bin_deltas):
        state = LatencyIntervalState.DECREASE_OR_RESET
    elif (
        definition.encoding is LatencyBinEncoding.BOUNDARY_CUMULATIVE
        and (
            any(left > right for left, right in zip(start_values, start_values[1:]))
            or any(left > right for left, right in zip(end_values, end_values[1:]))
        )
    ):
        state = LatencyIntervalState.INCOMPATIBLE_ENCODING

    disjoint_counts: tuple[int | float, ...] | None
    if state is not LatencyIntervalState.VALID:
        disjoint_counts = None
    elif definition.encoding is LatencyBinEncoding.DISJOINT:
        disjoint_counts = bin_deltas
    elif definition.encoding is LatencyBinEncoding.BOUNDARY_CUMULATIVE:
        disjoint_counts = tuple(
            bin_deltas[bin_index] - (
                bin_deltas[bin_index - 1] if bin_index > 0 else 0
            )
            for bin_index in range(len(bin_deltas))
        )
        if any(count < 0 for count in disjoint_counts):
            state = LatencyIntervalState.INCOMPATIBLE_ENCODING
            disjoint_counts = None
    else:
        disjoint_counts = None

    return LatencyInterval(
        start_ns=start_ns,
        end_ns=end_ns,
        duration_ns=end_ns - start_ns,
        start_values=start_values,
        end_values=end_values,
        bin_deltas=bin_deltas,
        disjoint_bin_counts=disjoint_counts,
        valid=state is LatencyIntervalState.VALID,
        state=state,
    )


def _exact_numeric_value(value: int | float) -> int | float:
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _all_intervals(
    ordered_series: Sequence[MetricSeries],
    definition: LatencyBinDefinition,
) -> tuple[LatencyInterval, ...]:
    if not ordered_series:
        return ()
    return _build_intervals(ordered_series, definition, None, set())


def _bin_count(
    *,
    bin_index: int,
    intervals: Sequence[LatencyInterval],
    bins_are_disjoint: bool,
) -> int | float | None:
    values: list[int | float] = []
    for interval in intervals:
        if not interval.valid:
            continue
        value = (
            interval.disjoint_bin_counts[bin_index]
            if bins_are_disjoint
            else interval.bin_deltas[bin_index]
        )
        if value is not None:
            values.append(value)
    if not values:
        return None
    if all(type(value) is int for value in values):
        return sum(values)
    return math.fsum(float(value) for value in values)


def _quantile_results(
    *,
    definition: LatencyBinDefinition,
    bin_results: Sequence[LatencyBinResult],
    total_observations: int | float | None,
    normalized_available: bool,
    quantiles: Sequence[float],
) -> tuple[LatencyQuantileResult, ...]:
    if not normalized_available or total_observations is None:
        reason = (
            "Quantiles require verified disjoint bins and a nonzero observation total."
            if definition.bin_edges is None
            else "Quantiles require a nonzero observation total."
        )
        return tuple(
            LatencyQuantileResult(
                quantile=quantile,
                status=LatencyQuantileStatus.UNSUPPORTED,
                bin_name=None,
                lower_edge=None,
                upper_edge=None,
                reason=reason,
            )
            for quantile in quantiles
        )
    if definition.bin_edges is None:
        return tuple(
            LatencyQuantileResult(
                quantile=quantile,
                status=LatencyQuantileStatus.UNSUPPORTED,
                bin_name=None,
                lower_edge=None,
                upper_edge=None,
                reason="Sourced bin edges and units are required for latency quantile bounds.",
            )
            for quantile in quantiles
        )

    results: list[LatencyQuantileResult] = []
    for quantile in quantiles:
        cumulative = 0
        rank = math.ceil(quantile * float(total_observations))
        selected_bin = next(
            (
                bin_result
                for bin_result in bin_results
                if (cumulative := cumulative + (bin_result.count or 0)) >= rank
            ),
            None,
        )
        if selected_bin is None:
            results.append(
                LatencyQuantileResult(
                    quantile=quantile,
                    status=LatencyQuantileStatus.UNSUPPORTED,
                    bin_name=None,
                    lower_edge=None,
                    upper_edge=None,
                    reason="The observation total was not reached while traversing bins.",
                )
            )
            continue
        results.append(
            LatencyQuantileResult(
                quantile=quantile,
                status=(
                    LatencyQuantileStatus.OPEN_TAIL
                    if selected_bin.open_tailed
                    else LatencyQuantileStatus.BOUNDED
                ),
                bin_name=selected_bin.bin_name,
                lower_edge=selected_bin.lower_edge,
                upper_edge=selected_bin.upper_edge,
                reason=(
                    "The open tail provides a lower bound but no upper bound."
                    if selected_bin.open_tailed
                    else "The quantile is bounded by the selected bin's edges."
                ),
            )
        )
    return tuple(results)


def _mean_result(
    *,
    definition: LatencyBinDefinition,
    bin_results: Sequence[LatencyBinResult],
    total_observations: int | float | None,
    normalized_available: bool,
) -> LatencyMeanResult:
    if not normalized_available or total_observations is None:
        return LatencyMeanResult(
            status=LatencyMeanStatus.UNSUPPORTED,
            value=None,
            unit=None,
            formula=None,
            reason="A mean requires verified disjoint bins and a nonzero observation total.",
        )
    if definition.bin_edges is None or definition.unit is None:
        return LatencyMeanResult(
            status=LatencyMeanStatus.UNSUPPORTED,
            value=None,
            unit=None,
            formula=None,
            reason="Sourced bin edges and units are required for a midpoint mean estimate.",
        )
    if any(bin_result.open_tailed for bin_result in bin_results):
        return LatencyMeanResult(
            status=LatencyMeanStatus.OPEN_TAIL,
            value=None,
            unit=None,
            formula=None,
            reason="An open-tailed bin has no midpoint or exact upper bound.",
        )
    weighted_sum = math.fsum(
        float(bin_result.count or 0)
        * ((bin_result.lower_edge or 0) + (bin_result.upper_edge or 0))
        / 2
        for bin_result in bin_results
    )
    return LatencyMeanResult(
        status=LatencyMeanStatus.ESTIMATED,
        value=weighted_sum / float(total_observations),
        unit=definition.unit,
        formula=MEAN_ESTIMATE_FORMULA,
        reason="This is a bin-midpoint estimate, not an exact latency mean.",
    )


def _status(
    *,
    missing_bins: Sequence[int],
    unsupported_bins: Sequence[int],
    intervals: Sequence[LatencyInterval],
    valid_intervals: Sequence[LatencyInterval],
    invalid_intervals: Sequence[LatencyInterval],
    encoding_incompatible: bool,
    bins_are_disjoint: bool,
    total_observations: int | float | None,
    named_bin_count: int | float | None,
) -> LatencyStatus:
    if encoding_incompatible:
        return LatencyStatus.UNSUPPORTED_ENCODING
    if not valid_intervals:
        if missing_bins and not unsupported_bins and not intervals:
            return LatencyStatus.MISSING_DATA
        if invalid_intervals:
            return LatencyStatus.INVALID_DATA
        return LatencyStatus.INSUFFICIENT_DATA
    if missing_bins:
        return LatencyStatus.PARTIAL_MISSING_DATA
    if unsupported_bins:
        return LatencyStatus.PARTIAL_UNSUPPORTED_SERIES
    if invalid_intervals:
        return LatencyStatus.PARTIAL_INVALID_DATA
    if not bins_are_disjoint:
        return (
            LatencyStatus.ZERO_NAMED_BINS_ONLY
            if named_bin_count == 0
            else LatencyStatus.NONZERO_NAMED_BINS_ONLY
        )
    return (
        LatencyStatus.OBSERVED_ZERO
        if total_observations == 0
        else LatencyStatus.OBSERVED_NONZERO
    )


def _status_reasons(
    status: LatencyStatus, quality_flags: set[LatencyQualityFlag]
) -> list[str]:
    reasons: list[str] = []
    if status in {
        LatencyStatus.PARTIAL_INVALID_DATA,
        LatencyStatus.INVALID_DATA,
    }:
        reasons.append("Reset, timestamp, and encoding-invalid intervals are excluded.")
    if LatencyQualityFlag.MISSING_BIN_EDGES in quality_flags:
        reasons.append("Bin edges are unsourced, so latency-unit quantiles and means are unsupported.")
    if LatencyQualityFlag.MISSING_EDGE_UNITS in quality_flags:
        reasons.append("Bin units are unsourced, so latency-unit summaries are unsupported.")
    if status in {
        LatencyStatus.ZERO_NAMED_BINS_ONLY,
        LatencyStatus.NONZERO_NAMED_BINS_ONLY,
    }:
        reasons.append("Bin encoding is unknown, so named counts are not normalized as a distribution.")
    return reasons


def _strictly_increasing(timestamps: Sequence[int]) -> bool:
    return all(left < right for left, right in zip(timestamps, timestamps[1:]))


def _inside_window(
    interval: LatencyInterval, requested_window_ns: tuple[int, int] | None
) -> bool:
    if requested_window_ns is None:
        return True
    return (
        requested_window_ns[0] <= interval.start_ns
        and interval.end_ns <= requested_window_ns[1]
    )


def _overlaps_window(
    interval: LatencyInterval, requested_window_ns: tuple[int, int] | None
) -> bool:
    if requested_window_ns is None:
        return False
    return (
        interval.end_ns > requested_window_ns[0]
        and interval.start_ns < requested_window_ns[1]
    )


def _sum_values(values) -> int | float | None:
    value_list = list(values)
    if not value_list:
        return None
    if all(type(value) is int for value in value_list):
        return sum(value_list)
    return math.fsum(float(value) for value in value_list)
