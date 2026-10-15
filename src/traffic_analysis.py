"""Numerically conservative traffic analysis for immutable metric series."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from enum import Enum
from itertools import groupby, pairwise
import math
from typing import Literal

import numpy as np

from counter_semantic_registry import CounterSemanticDefinition


UINT64_MAX = 2**64 - 1
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
NANOSECONDS_PER_SECOND = 1_000_000_000

RATE_FORMULA = "delta_bytes / duration_seconds"
MEAN_RATE_FORMULA = "sum(valid delta_bytes) / sum(valid duration_seconds)"
WEIGHTED_VARIANCE_FORMULA = (
    "sum(duration_seconds * (rate - mean_rate)^2) / sum(duration_seconds)"
)
QUANTILE_INTERPOLATION = (
    "duration-weighted upper order statistic; no interpolation between rates"
)


class ValueKind(str, Enum):
    CUMULATIVE = "cumulative"
    DELTA = "delta"
    GAUGE = "gauge"
    UNKNOWN = "unknown"


class TrafficDirection(str, Enum):
    TX = "TX"
    RX = "RX"


class SeriesQualityFlag(str, Enum):
    MISSING = "missing"
    INVALID = "invalid"
    INCOMPLETE = "incomplete"
    UNVALIDATED = "unvalidated"
    REORDERED = "reordered"
    DUPLICATE = "duplicate"
    CONFLICTING_TIMESTAMP = "conflicting_timestamp"
    DECREASE_OR_RESET = "decrease_or_reset"
    MULTIPLE_OBSERVATION_STREAMS = "multiple_observation_streams"
    PRECISION_LOSS_POSSIBLE = "precision_loss_possible"
    WRAP_SUSPECT = "wrap_suspect"
    RESET_EPOCH = "reset_epoch"
    GAP = "gap"
    LONG_GAP = "long_gap"
    BOUNDARY_INTERVAL_EXCLUDED = "boundary_interval_excluded"
    PARTIAL_WINDOW = "partial_window"
    BASELINE_RELATIVE = "baseline_relative"
    UNSUPPORTED_SERIES = "unsupported_series"
    NO_VALID_DISTRIBUTION_INTERVALS = "no_valid_distribution_intervals"


class IntervalState(str, Enum):
    VALID = "valid"
    DECREASE_OR_RESET = "decrease_or_reset"
    CONFLICTING_TIMESTAMP = "conflicting_timestamp"


class AnalysisStatus(str, Enum):
    OK = "ok"
    ZERO_TRAFFIC = "zero_traffic"
    PARTIAL_INVALID_DATA = "partial_invalid_data"
    MISSING_DATA = "missing_data"
    INSUFFICIENT_DATA = "insufficient_data"
    INVALID_DATA = "invalid_data"
    INVALID_WINDOW = "invalid_window"
    UNSUPPORTED_SERIES = "unsupported_series"


class GapPolicy(str, Enum):
    INCLUDE = "include"
    EXCLUDE_FROM_DISTRIBUTION = "exclude_from_distribution"


@dataclass(frozen=True)
class SeriesIdentity:
    """Identity that must remain constant across one analyzed series."""

    run_id: str
    node_id: str
    nic_id: str
    collector_id: str
    counter_id: str
    direction: TrafficDirection


@dataclass(frozen=True)
class MetricSeries:
    """Immutable, typed counter samples for one run/node/NIC/counter."""

    identity: SeriesIdentity
    timestamps_ns: Sequence[int]
    values: Sequence[int | float]
    value_dtype: Literal["uint64", "float64"]
    value_kind: ValueKind
    semantics: str
    unit: str
    baseline_relative: bool
    quality_flags: tuple[SeriesQualityFlag, ...] = ()
    counter_definition: CounterSemanticDefinition | None = None

    def __post_init__(self) -> None:
        if self.value_dtype not in ("uint64", "float64"):
            raise ValueError("value_dtype must be 'uint64' or 'float64'")
        if not isinstance(self.baseline_relative, bool):
            raise ValueError("baseline_relative must be boolean")

        if isinstance(self.timestamps_ns, np.ndarray) or isinstance(
            self.values, np.ndarray
        ):
            if not (
                isinstance(self.timestamps_ns, np.ndarray)
                and isinstance(self.values, np.ndarray)
            ):
                raise ValueError(
                    "timestamps_ns and values must both be sequences or NumPy arrays"
                )
            if self.timestamps_ns.dtype != np.dtype(np.int64):
                raise ValueError("timestamp array must have int64 dtype")
            if self.timestamps_ns.ndim != 1 or self.values.ndim != 1:
                raise ValueError("timestamp and value arrays must be one-dimensional")
            if self.timestamps_ns.shape != self.values.shape:
                raise ValueError("timestamps_ns and values must have equal length")
            if self.value_dtype == "uint64":
                if self.values.dtype != np.dtype(np.uint64):
                    raise ValueError("uint64 value array must have uint64 dtype")
            elif self.values.dtype != np.dtype(np.float64):
                raise ValueError("float64 value array must have float64 dtype")
            if self.value_dtype == "float64" and not np.all(np.isfinite(self.values)):
                raise ValueError("float64 values must be finite")
            self.timestamps_ns.setflags(write=False)
            self.values.setflags(write=False)
            return

        if len(self.timestamps_ns) != len(self.values):
            raise ValueError("timestamps_ns and values must have equal length")

        for timestamp in self.timestamps_ns:
            if type(timestamp) is not int or not INT64_MIN <= timestamp <= INT64_MAX:
                raise ValueError("every timestamp must be an int64 integer")

        for value in self.values:
            if self.value_dtype == "uint64":
                if type(value) is not int or not 0 <= value <= UINT64_MAX:
                    raise ValueError("uint64 values must be integers in [0, 2^64-1]")
            else:
                if type(value) is not float or not math.isfinite(value):
                    raise ValueError("float64 values must be finite Python floats")


@dataclass(frozen=True)
class TrafficInterval:
    """One adjacent-sample transition and its analysis disposition."""

    start_ns: int
    end_ns: int
    duration_ns: int
    start_value: int | float
    end_value: int | float
    valid: bool
    state: IntervalState
    delta_bytes: int | float | None = None
    rate_bytes_per_s: float | None = None
    long_gap: bool = False
    wrap_suspect: bool = False
    reset_epoch: bool = False


@dataclass(frozen=True)
class TrafficAnalysis:
    """Explicit results and limitations for one TX or RX series."""

    identity: SeriesIdentity
    status: AnalysisStatus
    quality_flags: frozenset[SeriesQualityFlag]
    status_reasons: tuple[str, ...]
    requested_window_ns: tuple[int, int] | None
    represented_window_ns: tuple[int, int] | None
    valid_interval_boundary_signature: Sequence[int]
    source_counter: str
    unit: str
    rate_unit: str
    value_kind: ValueKind
    semantics: str
    baseline_relative: bool
    total_bytes: int | float | None
    covered_duration_ns: int
    requested_duration_ns: int
    uncovered_requested_duration_ns: int
    coverage_fraction: float | None
    valid_interval_count: int
    invalid_interval_count: int
    distribution_interval_count: int
    distribution_excluded_duration_ns: int
    boundary_excluded_interval_count: int
    intervals: tuple[TrafficInterval, ...]
    boundary_excluded_intervals: tuple[TrafficInterval, ...]
    mean_rate_bytes_per_s: float | None
    p50_rate_bytes_per_s: float | None
    p99_rate_bytes_per_s: float | None
    weighted_variance: float | None
    weighted_standard_deviation: float | None
    coefficient_of_variation: float | None
    rate_formula: str
    mean_rate_formula: str
    weighted_variance_formula: str
    quantile_interpolation: str
    aggregation_population: str
    interval_detail_included: bool = True
    interval_count: int = 0
    sample_count: int = 0
    accumulation_epoch_count: int = 1
    wrap_suspect_interval_count: int = 0
    reset_epoch_boundary_count: int = 0
    counter_semantic_registry_version: str | None = None
    counter_semantics: str | None = None
    counter_quality: str | None = None
    counter_scope: str | None = None
    counter_width_bits: int | None = None
    counter_reset_behavior: str | None = None
    counter_caveats: tuple[str, ...] = ()


@dataclass(frozen=True)
class _SummaryArrays:
    sample_timestamps: np.ndarray
    sample_values: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    durations: np.ndarray
    valid: np.ndarray
    reset: np.ndarray
    conflict: np.ndarray
    delta_uint64: np.ndarray | None
    delta_float64: np.ndarray | None
    rates: np.ndarray
    long_gaps: np.ndarray
    wrap_suspects: np.ndarray
    reset_epochs: np.ndarray


class _BoundarySignature:
    """Immutable boundary sequence without materializing Python integers."""

    def __init__(self, values: np.ndarray):
        self._values = np.asarray(values, dtype=np.int64)
        self._values.setflags(write=False)

    def __len__(self) -> int:
        return int(self._values.size)

    def __iter__(self):
        return iter(self._values.tolist())

    def __getitem__(self, index: int | slice) -> int | list[int]:
        if isinstance(index, slice):
            return self._values[index].tolist()
        return int(self._values[index])

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _BoundarySignature):
            return bool(np.array_equal(self._values, other._values))
        if isinstance(other, (tuple, list, np.ndarray)):
            return bool(np.array_equal(self._values, other))
        return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return repr(self._values.tolist())


def _exact_timestamp_differences(ends: np.ndarray, starts: np.ndarray) -> np.ndarray:
    if ends.size == 0:
        return np.empty(0, dtype=np.int64)
    overflow = (
        ((starts < 0) & (ends > INT64_MAX + starts))
        | ((starts > 0) & (ends < INT64_MIN + starts))
    )
    if np.any(overflow):
        return np.asarray(
            [int(end_value) - int(start_value) for end_value, start_value in zip(ends, starts, strict=True)],
            dtype=object,
        )
    return ends - starts


def _ordered_samples(
    series: MetricSeries,
    gap_threshold_ns: int | None,
    include_intervals: bool,
) -> tuple[
    list[tuple[int, int | float]],
    tuple[TrafficInterval, ...] | list[_SummaryInterval],
    frozenset[SeriesQualityFlag],
]:
    def _transition(start_value, end_value):
        if series.counter_definition is None or series.value_dtype != "uint64":
            delta = end_value - start_value
            if delta < 0:
                return 0, False, IntervalState.DECREASE_OR_RESET, True, False
            return delta, True, IntervalState.VALID, False, False

        start_int = int(start_value)
        end_int = int(end_value)
        if start_int <= end_int:
            return end_int - start_int, True, IntervalState.VALID, False, False
        if end_int == 0:
            return 0, False, IntervalState.DECREASE_OR_RESET, False, True
        return 0, False, IntervalState.DECREASE_OR_RESET, True, False

    records = list(zip(series.timestamps_ns, series.values, strict=True))
    ordered = sorted(records, key=lambda record: record[0])
    flags: set[SeriesQualityFlag] = set(series.quality_flags)
    if [timestamp for timestamp, _ in records] != [
        timestamp for timestamp, _ in ordered
    ]:
        flags.add(SeriesQualityFlag.REORDERED)

    intervals: list[TrafficInterval] = []
    summary_intervals: list[_SummaryInterval] = []
    samples: list[tuple[int, int | float]] = []
    previous_sample: tuple[int, int | float] | None = None

    for timestamp, grouped_records in groupby(ordered, key=lambda record: record[0]):
        grouped_values = [value for _, value in grouped_records]
        if len(set(grouped_values)) != 1:
            first_value, second_value = grouped_values[:2]
            if include_intervals:
                intervals.append(
                    TrafficInterval(
                        start_ns=timestamp,
                        end_ns=timestamp,
                        duration_ns=0,
                        start_value=first_value,
                        end_value=second_value,
                        valid=False,
                        state=IntervalState.CONFLICTING_TIMESTAMP,
                    )
                )
            else:
                summary_intervals.append(
                    (
                        timestamp,
                        timestamp,
                        0,
                        False,
                        IntervalState.CONFLICTING_TIMESTAMP,
                        None,
                        None,
                        False,
                    )
                )
            flags.add(SeriesQualityFlag.CONFLICTING_TIMESTAMP)
            previous_sample = None
            continue

        sample = (timestamp, grouped_values[0])
        samples.append(sample)
        if len(grouped_values) > 1:
            flags.add(SeriesQualityFlag.DUPLICATE)
        if previous_sample is not None:
            start_ns, start_value = previous_sample
            end_ns, end_value = sample
            duration_ns = end_ns - start_ns
            long_gap = (
                gap_threshold_ns is not None and duration_ns > gap_threshold_ns
            )
            delta, valid, state, wrap_suspect, reset_epoch = _transition(
                start_value, end_value
            )
            if wrap_suspect:
                flags.add(SeriesQualityFlag.WRAP_SUSPECT)
            if reset_epoch:
                flags.add(SeriesQualityFlag.RESET_EPOCH)
            if not valid:
                if include_intervals:
                    intervals.append(
                        TrafficInterval(
                            start_ns=start_ns,
                            end_ns=end_ns,
                            duration_ns=duration_ns,
                            start_value=start_value,
                            end_value=end_value,
                            valid=False,
                            state=state,
                            long_gap=long_gap,
                            wrap_suspect=wrap_suspect,
                            reset_epoch=reset_epoch,
                        )
                    )
                else:
                    summary_intervals.append(
                        (
                            start_ns,
                            end_ns,
                            duration_ns,
                            False,
                            IntervalState.DECREASE_OR_RESET,
                            None,
                            None,
                            long_gap,
                        )
                    )
            else:
                rate = float(delta) / (duration_ns / NANOSECONDS_PER_SECOND)
                if not math.isfinite(rate):
                    raise ValueError("valid interval derived a non-finite rate")
                if include_intervals:
                    intervals.append(
                        TrafficInterval(
                            start_ns=start_ns,
                            end_ns=end_ns,
                            duration_ns=duration_ns,
                            start_value=start_value,
                            end_value=end_value,
                            valid=True,
                            state=state,
                            delta_bytes=delta,
                            rate_bytes_per_s=rate,
                            long_gap=long_gap,
                            wrap_suspect=wrap_suspect,
                            reset_epoch=reset_epoch,
                        )
                    )
                else:
                    summary_intervals.append(
                        (
                            start_ns,
                            end_ns,
                            duration_ns,
                            True,
                            IntervalState.VALID,
                            delta,
                            rate,
                            long_gap,
                        )
                    )
        previous_sample = sample

    interval_output = tuple(intervals) if include_intervals else summary_intervals
    return samples, interval_output, frozenset(flags)


def _summary_group_starts(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    starts = np.flatnonzero(np.concatenate(([True], values[1:] != values[:-1])))
    return starts.astype(np.int64, copy=False)


def _ordered_summary(
    series: MetricSeries, gap_threshold_ns: int | None
) -> tuple[_SummaryArrays, frozenset[SeriesQualityFlag]]:
    timestamps = np.asarray(series.timestamps_ns, dtype=np.int64)
    values = np.asarray(
        series.values,
        dtype=np.uint64 if series.value_dtype == "uint64" else np.float64,
    )
    sort_order = np.argsort(timestamps, kind="stable")
    sorted_timestamps = timestamps[sort_order]
    sorted_values = values[sort_order]
    flags: set[SeriesQualityFlag] = set(series.quality_flags)
    if not np.array_equal(timestamps, sorted_timestamps):
        flags.add(SeriesQualityFlag.REORDERED)

    group_starts = _summary_group_starts(sorted_timestamps)
    group_ends = np.concatenate((group_starts[1:], [timestamps.size]))
    group_sizes = group_ends - group_starts
    if timestamps.size > 1:
        same_timestamp = sorted_timestamps[1:] == sorted_timestamps[:-1]
        same_value = sorted_values[1:] == sorted_values[:-1]
        same_within_group = same_timestamp & same_value
        padded_same_within_group = np.concatenate((same_within_group, [False]))
        equal_pair_counts = np.add.reduceat(
            padded_same_within_group, group_starts
        ).astype(np.int64)
    else:
        equal_pair_counts = np.zeros(group_starts.size, dtype=np.int64)
    groups_equal = equal_pair_counts == group_sizes - 1

    if np.any(groups_equal & (group_sizes > 1)):
        flags.add(SeriesQualityFlag.DUPLICATE)

    sample_groups = np.flatnonzero(groups_equal)
    conflict_groups = np.flatnonzero(~groups_equal)
    sample_positions = group_starts[sample_groups]
    sample_timestamps = sorted_timestamps[sample_positions]
    sample_values = sorted_values[sample_positions]

    if sample_groups.size > 1:
        connected = sample_groups[1:] == sample_groups[:-1] + 1
        previous_positions = group_starts[sample_groups[:-1][connected]]
        current_positions = group_starts[sample_groups[1:][connected]]
    else:
        previous_positions = np.empty(0, dtype=np.int64)
        current_positions = np.empty(0, dtype=np.int64)

    transition_starts = sorted_timestamps[previous_positions]
    transition_ends = sorted_timestamps[current_positions]
    durations = _exact_timestamp_differences(transition_ends, transition_starts)
    long_gaps = (
        durations > gap_threshold_ns
        if gap_threshold_ns is not None
        else np.zeros(durations.size, dtype=bool)
    )
    start_values = sorted_values[previous_positions]
    end_values = sorted_values[current_positions]

    if series.value_dtype == "uint64":
        if series.counter_definition is not None:
            start_int64 = start_values.astype(np.int64)
            end_int64 = end_values.astype(np.int64)
            negative = end_int64 < start_int64
            reset_epochs = negative & (end_int64 == 0)
            wrap_suspects = negative & ~reset_epochs
            decreases = negative
            deltas = end_int64 - start_int64
            deltas = np.where(negative, 0, deltas)
            valid = ~negative
            if np.any(wrap_suspects):
                flags.add(SeriesQualityFlag.WRAP_SUSPECT)
            if np.any(reset_epochs):
                flags.add(SeriesQualityFlag.RESET_EPOCH)
        else:
            negative = end_values < start_values
            deltas = end_values - start_values
            decreases = negative
            reset_epochs = np.zeros(deltas.size, dtype=bool)
            wrap_suspects = negative
            valid = ~negative
        rates = np.full(durations.size, np.nan, dtype=np.float64)
        valid_rates = np.asarray(
            deltas[valid].astype(np.float64)
            / (durations[valid] / NANOSECONDS_PER_SECOND),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(valid_rates)):
            raise ValueError("valid interval derived a non-finite rate")
        rates[valid] = valid_rates
        delta_uint64 = deltas.astype(np.uint64, copy=False)
        delta_uint64[~valid] = 0
        delta_float64 = None
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            deltas = end_values - start_values
        valid = ~(deltas < 0)
        rates = np.full(durations.size, np.nan, dtype=np.float64)
        valid_rates = np.asarray(
            deltas[valid] / (durations[valid] / NANOSECONDS_PER_SECOND),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(valid_rates)):
            raise ValueError("valid interval derived a non-finite rate")
        rates[valid] = valid_rates
        delta_uint64 = None
        delta_float64 = deltas.copy()
        delta_float64[~valid] = 0
        decreases = np.asarray(deltas < 0, dtype=bool)
        reset_epochs = np.zeros(deltas.size, dtype=bool)
        wrap_suspects = decreases

    conflict_starts = sorted_timestamps[group_starts[conflict_groups]]
    all_starts = np.concatenate((transition_starts, conflict_starts))
    all_ends = np.concatenate((transition_ends, conflict_starts))
    all_durations = np.concatenate((durations, np.zeros(conflict_starts.size, dtype=np.int64)))
    all_valid = np.concatenate((valid, np.zeros(conflict_starts.size, dtype=bool)))
    all_reset = np.concatenate((decreases, np.zeros(conflict_starts.size, dtype=bool)))
    all_conflict = np.concatenate(
        (np.zeros(valid.size, dtype=bool), np.ones(conflict_starts.size, dtype=bool))
    )
    all_long_gaps = np.concatenate(
        (np.asarray(long_gaps, dtype=bool), np.zeros(conflict_starts.size, dtype=bool))
    )
    all_wrap_suspects = np.concatenate(
        (wrap_suspects, np.zeros(conflict_starts.size, dtype=bool))
    )
    all_reset_epochs = np.concatenate(
        (reset_epochs, np.zeros(conflict_starts.size, dtype=bool))
    )
    all_rates = np.concatenate((rates, np.full(conflict_starts.size, np.nan)))
    if delta_uint64 is None:
        all_delta_uint64 = None
        all_delta_float64 = np.concatenate(
            (delta_float64, np.zeros(conflict_starts.size, dtype=np.float64))
        )
    else:
        all_delta_uint64 = np.concatenate(
            (delta_uint64, np.zeros(conflict_starts.size, dtype=np.uint64))
        )
        all_delta_float64 = None

    interval_order = np.lexsort((all_ends, all_starts))
    summary = _SummaryArrays(
        sample_timestamps=sample_timestamps,
        sample_values=sample_values,
        starts=all_starts[interval_order],
        ends=all_ends[interval_order],
        durations=all_durations[interval_order],
        valid=all_valid[interval_order],
        reset=all_reset[interval_order],
        conflict=all_conflict[interval_order],
        delta_uint64=None if all_delta_uint64 is None else all_delta_uint64[interval_order],
        delta_float64=None if all_delta_float64 is None else all_delta_float64[interval_order],
        rates=all_rates[interval_order],
        long_gaps=all_long_gaps[interval_order],
        wrap_suspects=all_wrap_suspects[interval_order],
        reset_epochs=all_reset_epochs[interval_order],
    )
    if np.any(all_conflict):
        flags.add(SeriesQualityFlag.CONFLICTING_TIMESTAMP)
    return summary, frozenset(flags)


def _make_interval(
    start: tuple[int, int | float],
    end: tuple[int, int | float],
    value_dtype: Literal["uint64", "float64"],
    gap_threshold_ns: int | None,
) -> TrafficInterval:
    start_ns, start_value = start
    end_ns, end_value = end
    duration_ns = end_ns - start_ns
    long_gap = gap_threshold_ns is not None and duration_ns > gap_threshold_ns

    if value_dtype == "uint64":
        delta = end_value - start_value
    else:
        delta = end_value - start_value

    if delta < 0:
        return TrafficInterval(
            start_ns=start_ns,
            end_ns=end_ns,
            duration_ns=duration_ns,
            start_value=start_value,
            end_value=end_value,
            valid=False,
            state=IntervalState.DECREASE_OR_RESET,
            long_gap=long_gap,
        )

    rate = float(delta) / (duration_ns / NANOSECONDS_PER_SECOND)
    if not math.isfinite(rate):
        raise ValueError("valid interval derived a non-finite rate")
    return TrafficInterval(
        start_ns=start_ns,
        end_ns=end_ns,
        duration_ns=duration_ns,
        start_value=start_value,
        end_value=end_value,
        valid=True,
        state=IntervalState.VALID,
        delta_bytes=delta,
        rate_bytes_per_s=rate,
        long_gap=long_gap,
    )


def _duration_weighted_quantile(
    intervals: tuple[TrafficInterval, ...], quantile: float
) -> float | None:
    if not intervals:
        return None

    target_duration = quantile * sum(interval.duration_ns for interval in intervals)
    cumulative_duration = 0
    for interval in intervals:
        cumulative_duration += interval.duration_ns
        if cumulative_duration >= target_duration:
            rate = interval.rate_bytes_per_s
            if rate is None:
                raise ValueError("valid interval is missing its rate")
            return rate
    raise RuntimeError("weighted quantile failed to consume all interval duration")


def _weighted_moments(
    intervals: tuple[TrafficInterval, ...],
) -> tuple[float | None, float | None, float | None]:
    if not intervals:
        return None, None, None

    total_duration = sum(interval.duration_ns for interval in intervals)
    total_bytes = math.fsum(float(interval.delta_bytes) for interval in intervals)
    mean = total_bytes / (total_duration / NANOSECONDS_PER_SECOND)
    variance = math.fsum(
        interval.duration_ns * (interval.rate_bytes_per_s - mean) ** 2
        for interval in intervals
    ) / total_duration
    standard_deviation = math.sqrt(variance)
    coefficient_of_variation = (
        standard_deviation / mean if mean != 0 else None
    )
    return variance, standard_deviation, coefficient_of_variation


def _summary_weighted_quantile(
    durations: np.ndarray, rates: np.ndarray, quantile: float
) -> float | None:
    if durations.size == 0:
        return None

    total_duration = sum(map(int, durations))
    target_duration = quantile * total_duration
    if durations.dtype == object:
        cumulative_duration = 0
        for duration, rate in zip(durations, rates, strict=True):
            cumulative_duration += int(duration)
            if cumulative_duration >= target_duration:
                return float(rate)
        raise RuntimeError("weighted quantile failed to consume all interval duration")

    cumulative_duration = np.cumsum(durations, dtype=np.int64)
    selected = np.searchsorted(cumulative_duration, target_duration, side="left")
    if selected >= rates.size:
        raise RuntimeError("weighted quantile failed to consume all interval duration")
    return float(rates[selected])


def _summary_weighted_moments(
    durations: np.ndarray,
    deltas: np.ndarray,
    rates: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    if durations.size == 0:
        return None, None, None

    total_duration = sum(map(int, durations))
    total_bytes = math.fsum(map(float, deltas))
    mean = total_bytes / (total_duration / NANOSECONDS_PER_SECOND)
    variance = math.fsum(
        map(float, durations * (rates - mean) ** 2)
    ) / total_duration
    standard_deviation = math.sqrt(variance)
    coefficient_of_variation = (
        standard_deviation / mean if mean != 0 else None
    )
    return variance, standard_deviation, coefficient_of_variation


def _requested_window_sample_count(
    sample_timestamps, requested_window_ns
) -> int:
    timestamps = np.asarray(sample_timestamps, dtype=np.int64)
    if requested_window_ns is None:
        return int(timestamps.size)
    window_start, window_end = requested_window_ns
    inside = (timestamps >= window_start) & (timestamps <= window_end)
    return int(np.count_nonzero(inside))


def _summary_result(
    series: MetricSeries,
    summary: _SummaryArrays,
    requested_window_ns: tuple[int, int] | None,
    quality_flags: frozenset[SeriesQualityFlag],
    gap_threshold_ns: int | None,
    gap_policy: GapPolicy,
) -> TrafficAnalysis:
    quality = set(quality_flags)
    selected = np.ones(summary.starts.size, dtype=bool)
    boundary_excluded_count = 0
    if requested_window_ns is not None:
        window_start, window_end = requested_window_ns
        inside = (
            (window_start <= summary.starts)
            & (summary.ends <= window_end)
        )
        overlaps = (summary.ends > window_start) & (summary.starts < window_end)
        selected = inside
        boundary_excluded_count = int(np.count_nonzero(overlaps & ~inside))
        if boundary_excluded_count:
            quality.add(SeriesQualityFlag.BOUNDARY_INTERVAL_EXCLUDED)
        if summary.sample_timestamps.size and (
            summary.sample_timestamps[0] < window_start
            or summary.sample_timestamps[-1] > window_end
        ):
            quality.add(SeriesQualityFlag.PARTIAL_WINDOW)

    valid_intervals = selected & summary.valid
    invalid_count = int(np.count_nonzero(selected) - np.count_nonzero(valid_intervals))
    if np.any(summary.reset & selected):
        quality.add(SeriesQualityFlag.DECREASE_OR_RESET)
    if np.any(summary.wrap_suspects & selected):
        quality.add(SeriesQualityFlag.WRAP_SUSPECT)
    if np.any(summary.reset_epochs & selected):
        quality.add(SeriesQualityFlag.RESET_EPOCH)
    if np.any(summary.long_gaps & valid_intervals):
        quality.add(SeriesQualityFlag.LONG_GAP)

    distribution_mask = valid_intervals
    distribution_excluded_duration = 0
    if (
        gap_policy is GapPolicy.EXCLUDE_FROM_DISTRIBUTION
        and gap_threshold_ns is not None
    ):
        distribution_mask = valid_intervals & ~summary.long_gaps
        distribution_excluded_duration = sum(
            map(int, summary.durations[valid_intervals & summary.long_gaps])
        )
        if not np.any(distribution_mask):
            quality.add(SeriesQualityFlag.NO_VALID_DISTRIBUTION_INTERVALS)

    covered_duration_ns = sum(map(int, summary.durations[valid_intervals]))
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

    total_bytes: int | float | None
    if not np.any(valid_intervals):
        total_bytes = None
    elif series.value_dtype == "uint64":
        total_bytes = sum(map(int, summary.delta_uint64[valid_intervals]))
    else:
        total_bytes = math.fsum(map(float, summary.delta_float64[valid_intervals]))

    represented_window = (
        (
            int(np.min(summary.starts[valid_intervals])),
            int(np.max(summary.ends[valid_intervals])),
        )
        if np.any(valid_intervals)
        else None
    )

    distribution_indices = np.flatnonzero(distribution_mask)
    distribution_order = distribution_indices[
        np.lexsort(
            (
                summary.ends[distribution_indices],
                summary.starts[distribution_indices],
                summary.rates[distribution_indices],
            )
        )
    ]
    distribution_durations = summary.durations[distribution_order]
    distribution_rates = summary.rates[distribution_order]
    distribution_deltas = (
        summary.delta_uint64[distribution_order]
        if summary.delta_uint64 is not None
        else summary.delta_float64[distribution_order]
    )
    variance, standard_deviation, coefficient_of_variation = (
        _summary_weighted_moments(
            distribution_durations,
            distribution_deltas,
            distribution_rates,
        )
    )
    mean_rate = (
        total_bytes / (covered_duration_ns / NANOSECONDS_PER_SECOND)
        if total_bytes is not None
        else None
    )

    if not np.any(valid_intervals):
        if invalid_count:
            status = AnalysisStatus.INVALID_DATA
            reasons = ("No valid intervals remain after quality exclusions.",)
        elif summary.sample_timestamps.size == 0:
            status = AnalysisStatus.INSUFFICIENT_DATA
            reasons = ("At least two samples are required for an interval.",)
        else:
            status = AnalysisStatus.INSUFFICIENT_DATA
            reasons = (
                "The requested window contains no complete valid interval.",
            )
    elif total_bytes == 0:
        status = AnalysisStatus.ZERO_TRAFFIC
        reasons = ("All valid intervals have zero traffic.",)
    elif invalid_count:
        status = AnalysisStatus.PARTIAL_INVALID_DATA
        reasons = (
            "Numerical results include only valid intervals; invalid intervals are excluded.",
        )
    else:
        status = AnalysisStatus.OK
        reasons = ()

    status_reasons = list(reasons)
    if SeriesQualityFlag.MISSING in quality:
        status_reasons.append("Input quality flags include missing data.")
    if SeriesQualityFlag.CONFLICTING_TIMESTAMP in quality:
        status_reasons.append("Conflicting equal timestamps prevent safe differencing.")
    if SeriesQualityFlag.DECREASE_OR_RESET in quality:
        status_reasons.append(
            "A counter decrease or reset was excluded without bridging."
        )
    if SeriesQualityFlag.WRAP_SUSPECT in quality:
        status_reasons.append(
            "A counter decrease was conservatively excluded; 56-bit wrap remains unverified without an independent signal."
        )
    if SeriesQualityFlag.RESET_EPOCH in quality:
        status_reasons.append(
            "A hardware counter reset started a new accumulation epoch."
        )

    aggregation_population = (
        f"{np.count_nonzero(valid_intervals)} valid complete intervals; "
        f"{distribution_indices.size} intervals in the temporal distribution"
    )

    valid_indices = np.flatnonzero(valid_intervals)
    boundary_values = np.empty(valid_indices.size * 2, dtype=np.int64)
    boundary_values[0::2] = summary.starts[valid_indices]
    boundary_values[1::2] = summary.ends[valid_indices]

    return TrafficAnalysis(
        identity=series.identity,
        status=status,
        quality_flags=frozenset(quality),
        status_reasons=tuple(status_reasons),
        requested_window_ns=requested_window_ns,
        represented_window_ns=represented_window,
        valid_interval_boundary_signature=_BoundarySignature(boundary_values),
        source_counter=series.identity.counter_id,
        unit=series.unit,
        rate_unit=f"{series.unit}/s",
        value_kind=series.value_kind,
        semantics=series.semantics,
        baseline_relative=series.baseline_relative,
        total_bytes=total_bytes,
        covered_duration_ns=covered_duration_ns,
        requested_duration_ns=requested_duration_ns,
        uncovered_requested_duration_ns=uncovered_duration_ns,
        coverage_fraction=coverage_fraction,
        valid_interval_count=int(valid_indices.size),
        invalid_interval_count=invalid_count,
        distribution_interval_count=int(distribution_indices.size),
        distribution_excluded_duration_ns=distribution_excluded_duration,
        boundary_excluded_interval_count=boundary_excluded_count,
        intervals=(),
        boundary_excluded_intervals=(),
        mean_rate_bytes_per_s=mean_rate,
        p50_rate_bytes_per_s=_summary_weighted_quantile(
            distribution_durations, distribution_rates, 0.50
        ),
        p99_rate_bytes_per_s=_summary_weighted_quantile(
            distribution_durations, distribution_rates, 0.99
        ),
        weighted_variance=variance,
        weighted_standard_deviation=standard_deviation,
        coefficient_of_variation=coefficient_of_variation,
        rate_formula=RATE_FORMULA,
        mean_rate_formula=MEAN_RATE_FORMULA,
        weighted_variance_formula=WEIGHTED_VARIANCE_FORMULA,
        quantile_interpolation=QUANTILE_INTERPOLATION,
        aggregation_population=aggregation_population,
        interval_detail_included=False,
        interval_count=int(summary.starts.size),
        sample_count=_requested_window_sample_count(
            summary.sample_timestamps, requested_window_ns
        ),
        accumulation_epoch_count=1 + int(np.count_nonzero(summary.reset_epochs & selected)),
        wrap_suspect_interval_count=int(
            np.count_nonzero(summary.wrap_suspects & selected)
        ),
        reset_epoch_boundary_count=int(np.count_nonzero(summary.reset_epochs & selected)),
        counter_semantic_registry_version=(
            series.counter_definition.registry_version
            if series.counter_definition is not None
            else None
        ),
        counter_semantics=(
            series.counter_definition.measurement
            if series.counter_definition is not None
            else None
        ),
        counter_quality=(
            series.counter_definition.quality.value
            if series.counter_definition is not None
            else None
        ),
        counter_scope=(
            series.counter_definition.scope.value
            if series.counter_definition is not None
            else None
        ),
        counter_width_bits=(
            series.counter_definition.counter_width_bits
            if series.counter_definition is not None
            else None
        ),
        counter_reset_behavior=(
            series.counter_definition.reset_behavior.value
            if series.counter_definition is not None
            else None
        ),
        counter_caveats=(
            series.counter_definition.caveats
            if series.counter_definition is not None
            else ()
        ),
    )


def _empty_result(
    series: MetricSeries,
    status: AnalysisStatus,
    quality_flags: frozenset[SeriesQualityFlag],
    requested_window_ns: tuple[int, int] | None,
    reasons: tuple[str, ...],
    *,
    interval_detail_included: bool = True,
    sample_count: int = 0,
) -> TrafficAnalysis:
    requested_duration = (
        max(0, requested_window_ns[1] - requested_window_ns[0])
        if requested_window_ns is not None
        else 0
    )
    return TrafficAnalysis(
        identity=series.identity,
        status=status,
        quality_flags=quality_flags,
        status_reasons=reasons,
        requested_window_ns=requested_window_ns,
        represented_window_ns=None,
        valid_interval_boundary_signature=(),
        source_counter=series.identity.counter_id,
        unit=series.unit,
        rate_unit=f"{series.unit}/s",
        value_kind=series.value_kind,
        semantics=series.semantics,
        baseline_relative=series.baseline_relative,
        total_bytes=None,
        covered_duration_ns=0,
        requested_duration_ns=requested_duration,
        uncovered_requested_duration_ns=requested_duration,
        coverage_fraction=0.0 if requested_duration > 0 else None,
        valid_interval_count=0,
        invalid_interval_count=0,
        distribution_interval_count=0,
        distribution_excluded_duration_ns=0,
        boundary_excluded_interval_count=0,
        intervals=(),
        boundary_excluded_intervals=(),
        mean_rate_bytes_per_s=None,
        p50_rate_bytes_per_s=None,
        p99_rate_bytes_per_s=None,
        weighted_variance=None,
        weighted_standard_deviation=None,
        coefficient_of_variation=None,
        rate_formula=RATE_FORMULA,
        mean_rate_formula=MEAN_RATE_FORMULA,
        weighted_variance_formula=WEIGHTED_VARIANCE_FORMULA,
        quantile_interpolation=QUANTILE_INTERPOLATION,
        aggregation_population="0 valid complete intervals",
        interval_detail_included=interval_detail_included,
        sample_count=sample_count,
    )


def analyze_traffic_series(
    series: MetricSeries,
    window_start_ns: int | None = None,
    window_end_ns: int | None = None,
    gap_threshold_ns: int | None = None,
    gap_policy: GapPolicy = GapPolicy.INCLUDE,
    include_intervals: bool = True,
) -> TrafficAnalysis:
    """Analyze one immutable cumulative byte-counter series.

    The exact window policy includes only intervals whose endpoints are both
    inside the requested window. Straddling boundary intervals are reported but
    excluded. TX and RX are never combined because each call analyzes one
    direction-bearing :class:`MetricSeries`.

    Set ``include_intervals=False`` for a summary-only result. It omits
    per-interval detail while preserving exact arithmetic, exclusions,
    statistics, coverage, and quality state.
    """

    if type(include_intervals) is not bool:
        raise ValueError("include_intervals must be a boolean")
    if gap_threshold_ns is not None and (
        type(gap_threshold_ns) is not int or gap_threshold_ns <= 0
    ):
        raise ValueError("gap_threshold_ns must be a positive integer or None")
    if gap_policy is GapPolicy.EXCLUDE_FROM_DISTRIBUTION and gap_threshold_ns is None:
        raise ValueError(
            "GapPolicy.EXCLUDE_FROM_DISTRIBUTION requires a positive gap_threshold_ns"
        )

    if (window_start_ns is None) != (window_end_ns is None):
        raise ValueError("window_start_ns and window_end_ns must be supplied together")
    for boundary in (window_start_ns, window_end_ns):
        if boundary is not None and (
            type(boundary) is not int or not INT64_MIN <= boundary <= INT64_MAX
        ):
            raise ValueError("window boundaries must be int64 integers")

    if window_start_ns is not None and window_end_ns is not None:
        requested_window = (window_start_ns, window_end_ns)
        if window_start_ns >= window_end_ns:
            return _empty_result(
                series,
                AnalysisStatus.INVALID_WINDOW,
                frozenset(series.quality_flags) | {SeriesQualityFlag.INVALID},
                requested_window,
                ("The requested window must have a strictly positive duration.",),
                interval_detail_included=include_intervals,
                sample_count=len(series.timestamps_ns),
            )
    else:
        requested_window = None

    unsupported: set[SeriesQualityFlag] = set()
    if series.value_kind is not ValueKind.CUMULATIVE:
        unsupported.add(SeriesQualityFlag.UNSUPPORTED_SERIES)
    if (
        series.unit != "bytes"
        or series.semantics not in {"bytes", "network-frame octets"}
    ):
        unsupported.add(SeriesQualityFlag.UNSUPPORTED_SERIES)
    if unsupported:
        return _empty_result(
            series,
            AnalysisStatus.UNSUPPORTED_SERIES,
            frozenset(series.quality_flags) | unsupported,
            requested_window,
            ("Traffic analysis requires cumulative byte counters.",),
            interval_detail_included=include_intervals,
            sample_count=len(series.timestamps_ns),
        )

    if SeriesQualityFlag.MISSING in series.quality_flags:
        return _empty_result(
            series,
            AnalysisStatus.MISSING_DATA,
            frozenset(series.quality_flags),
            requested_window,
            ("The input series is explicitly marked missing.",),
            interval_detail_included=include_intervals,
            sample_count=len(series.timestamps_ns),
        )
    if SeriesQualityFlag.INVALID in series.quality_flags:
        return _empty_result(
            series,
            AnalysisStatus.INVALID_DATA,
            frozenset(series.quality_flags),
            requested_window,
            ("The input series is explicitly marked invalid.",),
            interval_detail_included=include_intervals,
            sample_count=len(series.timestamps_ns),
        )

    if not include_intervals:
        summary, quality_flags = _ordered_summary(series, gap_threshold_ns)
        quality_flags = set(quality_flags)
        if series.baseline_relative:
            quality_flags.add(SeriesQualityFlag.BASELINE_RELATIVE)
        if requested_window is None and summary.sample_timestamps.size:
            raw_timestamps = np.asarray(series.timestamps_ns, dtype=np.int64)
            requested_window = (
                int(np.min(raw_timestamps)),
                int(np.max(raw_timestamps)),
            )
        return _summary_result(
            series,
            summary,
            requested_window,
            quality_flags,
            gap_threshold_ns,
            gap_policy,
        )

    ordered_samples, intervals, quality_flags = _ordered_samples(
        series, gap_threshold_ns, include_intervals
    )
    quality_flags = set(quality_flags)
    if series.baseline_relative:
        quality_flags.add(SeriesQualityFlag.BASELINE_RELATIVE)

    if requested_window is None and ordered_samples:
        sorted_timestamps = sorted(series.timestamps_ns)
        requested_window = (sorted_timestamps[0], sorted_timestamps[-1])

    boundary_excluded: tuple[TrafficInterval, ...] = ()
    if requested_window is not None:
        window_start, window_end = requested_window
        selected_intervals = tuple(
            interval
            for interval in intervals
            if window_start <= interval.start_ns
            and interval.end_ns <= window_end
        )
        boundary_excluded = tuple(
            interval
            for interval in intervals
            if interval.end_ns > window_start
            and interval.start_ns < window_end
            and not (
                window_start <= interval.start_ns
                and interval.end_ns <= window_end
            )
        )
        if boundary_excluded:
            quality_flags.add(SeriesQualityFlag.BOUNDARY_INTERVAL_EXCLUDED)
        if ordered_samples and (
            ordered_samples[0][0] < window_start
            or ordered_samples[-1][0] > window_end
        ):
            quality_flags.add(SeriesQualityFlag.PARTIAL_WINDOW)
    else:
        selected_intervals = intervals

    valid_intervals = tuple(interval for interval in selected_intervals if interval.valid)
    invalid_intervals = tuple(
        interval for interval in selected_intervals if not interval.valid
    )
    if any(
        interval.state is IntervalState.DECREASE_OR_RESET
        for interval in selected_intervals
    ):
        quality_flags.add(SeriesQualityFlag.DECREASE_OR_RESET)
    if any(interval.long_gap for interval in valid_intervals):
        quality_flags.add(SeriesQualityFlag.LONG_GAP)

    distribution_intervals = valid_intervals
    distribution_excluded_duration = 0
    if (
        gap_policy is GapPolicy.EXCLUDE_FROM_DISTRIBUTION
        and gap_threshold_ns is not None
    ):
        distribution_intervals = tuple(
            interval for interval in valid_intervals if not interval.long_gap
        )
        distribution_excluded_duration = sum(
            interval.duration_ns
            for interval in valid_intervals
            if interval.long_gap
        )
        if not distribution_intervals:
            quality_flags.add(
                SeriesQualityFlag.NO_VALID_DISTRIBUTION_INTERVALS
            )

    covered_duration_ns = sum(interval.duration_ns for interval in valid_intervals)
    requested_duration_ns = (
        requested_window[1] - requested_window[0]
        if requested_window is not None
        else 0
    )
    uncovered_duration_ns = max(0, requested_duration_ns - covered_duration_ns)
    coverage_fraction = (
        covered_duration_ns / requested_duration_ns
        if requested_duration_ns > 0
        else None
    )

    total_bytes: int | float | None
    if not valid_intervals:
        total_bytes = None
    elif series.value_dtype == "uint64":
        total_bytes = sum(
            int(interval.delta_bytes)
            for interval in valid_intervals
            if interval.delta_bytes is not None
        )
    else:
        total_bytes = math.fsum(
            float(interval.delta_bytes)
            for interval in valid_intervals
            if interval.delta_bytes is not None
        )

    represented_window = (
        (
            min(interval.start_ns for interval in valid_intervals),
            max(interval.end_ns for interval in valid_intervals),
        )
        if valid_intervals
        else None
    )

    sorted_distribution = tuple(
        sorted(
            distribution_intervals,
            key=lambda interval: (
                interval.rate_bytes_per_s,
                interval.start_ns,
                interval.end_ns,
            ),
        )
    )
    variance, standard_deviation, coefficient_of_variation = _weighted_moments(
        sorted_distribution
    )
    mean_rate = (
        total_bytes / (covered_duration_ns / NANOSECONDS_PER_SECOND)
        if total_bytes is not None
        else None
    )

    if not valid_intervals:
        if invalid_intervals:
            status = AnalysisStatus.INVALID_DATA
            reasons = ("No valid intervals remain after quality exclusions.",)
        elif not ordered_samples:
            status = AnalysisStatus.INSUFFICIENT_DATA
            reasons = ("At least two samples are required for an interval.",)
        else:
            status = AnalysisStatus.INSUFFICIENT_DATA
            reasons = (
                "The requested window contains no complete valid interval.",
            )
    elif total_bytes == 0:
        status = AnalysisStatus.ZERO_TRAFFIC
        reasons = ("All valid intervals have zero traffic.",)
    elif invalid_intervals:
        status = AnalysisStatus.PARTIAL_INVALID_DATA
        reasons = (
            "Numerical results include only valid intervals; invalid intervals are excluded.",
        )
    else:
        status = AnalysisStatus.OK
        reasons = ()

    status_reasons = list(reasons)
    if SeriesQualityFlag.MISSING in quality_flags:
        status_reasons.append("Input quality flags include missing data.")
    if SeriesQualityFlag.CONFLICTING_TIMESTAMP in quality_flags:
        status_reasons.append("Conflicting equal timestamps prevent safe differencing.")
    if SeriesQualityFlag.DECREASE_OR_RESET in quality_flags:
        status_reasons.append(
            "A counter decrease or reset was excluded without bridging."
        )
    if SeriesQualityFlag.WRAP_SUSPECT in quality_flags:
        status_reasons.append(
            "A counter decrease was conservatively excluded; 56-bit wrap remains unverified without an independent signal."
        )
    if SeriesQualityFlag.RESET_EPOCH in quality_flags:
        status_reasons.append(
            "A hardware counter reset started a new accumulation epoch."
        )

    aggregation_population = (
        f"{len(valid_intervals)} valid complete intervals; "
        f"{len(sorted_distribution)} intervals in the temporal distribution"
    )

    return TrafficAnalysis(
        identity=series.identity,
        status=status,
        quality_flags=frozenset(quality_flags),
        status_reasons=tuple(status_reasons),
        requested_window_ns=requested_window,
        represented_window_ns=represented_window,
        valid_interval_boundary_signature=tuple(
            boundary
            for interval in valid_intervals
            for boundary in (interval.start_ns, interval.end_ns)
        ),
        source_counter=series.identity.counter_id,
        unit=series.unit,
        rate_unit=f"{series.unit}/s",
        value_kind=series.value_kind,
        semantics=series.semantics,
        baseline_relative=series.baseline_relative,
        total_bytes=total_bytes,
        covered_duration_ns=covered_duration_ns,
        requested_duration_ns=requested_duration_ns,
        uncovered_requested_duration_ns=uncovered_duration_ns,
        coverage_fraction=coverage_fraction,
        valid_interval_count=len(valid_intervals),
        invalid_interval_count=len(invalid_intervals),
        distribution_interval_count=len(sorted_distribution),
        distribution_excluded_duration_ns=distribution_excluded_duration,
        boundary_excluded_interval_count=len(boundary_excluded),
        intervals=selected_intervals,
        boundary_excluded_intervals=boundary_excluded,
        mean_rate_bytes_per_s=mean_rate,
        p50_rate_bytes_per_s=_duration_weighted_quantile(
            sorted_distribution, 0.50
        ),
        p99_rate_bytes_per_s=_duration_weighted_quantile(
            sorted_distribution, 0.99
        ),
        weighted_variance=variance,
        weighted_standard_deviation=standard_deviation,
        coefficient_of_variation=coefficient_of_variation,
        rate_formula=RATE_FORMULA,
        mean_rate_formula=MEAN_RATE_FORMULA,
        weighted_variance_formula=WEIGHTED_VARIANCE_FORMULA,
        quantile_interpolation=QUANTILE_INTERPOLATION,
        aggregation_population=aggregation_population,
        interval_detail_included=True,
        interval_count=len(intervals),
        sample_count=_requested_window_sample_count(
            [timestamp for timestamp, _ in ordered_samples], requested_window
        ),
        accumulation_epoch_count=1 + sum(
            interval.reset_epoch for interval in selected_intervals
        ),
        wrap_suspect_interval_count=sum(
            interval.wrap_suspect for interval in selected_intervals
        ),
        reset_epoch_boundary_count=sum(
            interval.reset_epoch for interval in selected_intervals
        ),
        counter_semantic_registry_version=(
            series.counter_definition.registry_version
            if series.counter_definition is not None
            else None
        ),
        counter_semantics=(
            series.counter_definition.measurement
            if series.counter_definition is not None
            else None
        ),
        counter_quality=(
            series.counter_definition.quality.value
            if series.counter_definition is not None
            else None
        ),
        counter_scope=(
            series.counter_definition.scope.value
            if series.counter_definition is not None
            else None
        ),
        counter_width_bits=(
            series.counter_definition.counter_width_bits
            if series.counter_definition is not None
            else None
        ),
        counter_reset_behavior=(
            series.counter_definition.reset_behavior.value
            if series.counter_definition is not None
            else None
        ),
        counter_caveats=(
            series.counter_definition.caveats
            if series.counter_definition is not None
            else ()
        ),
    )
