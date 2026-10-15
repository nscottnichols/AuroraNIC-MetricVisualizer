"""Pure host-path pressure analysis for cumulative NIC counters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


HOST_PRESSURE_ANALYSIS_VERSION = 1
CUMULATIVE_COUNTER_SEMANTICS = "cumulative counter"
PACKET_UNIT = "packets"
DELTA_FORMULA = "end_value - start_value"
RATE_FORMULA = "sum(valid interval deltas) / sum(valid interval durations)"
BLOCKED_PER_PACKET_FORMULA = "sum(blocked deltas) / sum(packet deltas)"
UINT64_MAX = 2**64 - 1
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
NANOSECONDS_PER_SECOND = 1_000_000_000

HOST_PRESSURE_LIMITATIONS = (
    "Posted and nonposted populations remain separate and are never merged.",
    "Blocked-counter units are unresolved until authoritative hardware metadata "
    "is supplied; no cycles/packet or count/packet interpretation is claimed.",
    "A zero packet denominator makes blocked-per-packet undefined rather than zero.",
    "Missing packet or blocked counters remain missing and are never treated as zero.",
    "Decreasing cumulative counters are treated as reset-or-wrap evidence and "
    "excluded from valid deltas, rates, and ratio denominators.",
    "Supporting host, dispatcher, and TX stall counters remain separate and are "
    "never summed into one stall population.",
    "Host-pressure evidence is correlation only and does not prove that PCIe is "
    "the limiting resource.",
)


class HostPressurePath(str, Enum):
    POSTED = "posted"
    NONPOSTED = "nonposted"


class HostPressureCounterKind(str, Enum):
    PACKET = "packet"
    BLOCKED = "blocked"
    STALL = "stall"


class HostPressureStallKind(str, Enum):
    HOST_DMA = "host_dma"
    DISPATCHER = "dispatcher"
    TX = "tx"


PATH_COUNTER_IDS = {
    HostPressurePath.POSTED: (
        "parbs_tarb_pi_posted_pkts",
        "parbs_tarb_pi_posted_blocked_cnt",
    ),
    HostPressurePath.NONPOSTED: (
        "parbs_tarb_pi_non_posted_pkts",
        "parbs_tarb_pi_non_posted_blocked_cnt",
    ),
}


class BlockedCounterUnit(str, Enum):
    EVENTS = "events"
    CYCLES = "cycles"


class HostPressureStatus(str, Enum):
    OBSERVED_ZERO = "observed_zero"
    OBSERVED_NONZERO = "observed_nonzero"
    MISSING_DATA = "missing_data"
    PARTIAL_MISSING_DATA = "partial_missing_data"
    INSUFFICIENT_DATA = "insufficient_data"
    INVALID_DATA = "invalid_data"
    PARTIAL_INVALID_DATA = "partial_invalid_data"
    UNSUPPORTED_UNITS = "unsupported_units"


class HostPressureQualityFlag(str, Enum):
    MISSING_PACKET = "missing_packet"
    MISSING_BLOCKED = "missing_blocked"
    BASELINE_RELATIVE = "baseline_relative"
    DECREASE_OR_RESET = "decrease_or_reset"
    REORDERED = "reordered"
    DUPLICATE = "duplicate"
    CONFLICTING_TIMESTAMP = "conflicting_timestamp"
    INCOMPATIBLE_UNITS = "incompatible_units"
    UNRESOLVED_BLOCKED_UNITS = "unresolved_blocked_units"
    UNSUPPORTED_SERIES = "unsupported_series"
    NO_COMMON_VALID_INTERVAL = "no_common_valid_interval"
    PARTIAL_WINDOW = "partial_window"
    BOUNDARY_INTERVAL_EXCLUDED = "boundary_interval_excluded"


class HostPressureDenominatorState(str, Enum):
    AVAILABLE = "available"
    MISSING_PACKET = "missing_packet"
    MISSING_BLOCKED = "missing_blocked"
    ZERO_PACKETS = "zero_packets"
    INVALID_PACKET_INTERVALS = "invalid_packet_intervals"
    NO_COMMON_VALID_INTERVAL = "no_common_valid_interval"
    INCOMPATIBLE_UNITS = "incompatible_units"
    UNRESOLVED_BLOCKED_UNITS = "unresolved_blocked_units"


class HostPressureValueKind(str, Enum):
    CUMULATIVE = "cumulative"
    DELTA = "delta"
    GAUGE = "gauge"
    UNKNOWN = "unknown"


class HostPressureIntervalState(str, Enum):
    VALID = "valid"
    DECREASE_OR_RESET = "decrease_or_reset"
    CONFLICTING_TIMESTAMP = "conflicting_timestamp"


@dataclass(frozen=True)
class BlockedUnitResolution:
    unit: BlockedCounterUnit
    evidence: str

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, str) or not self.evidence.strip():
            raise ValueError("blocked unit evidence must be a nonempty string")


@dataclass(frozen=True)
class HostPressurePopulation:
    run_id: str
    node_id: str
    nic_id: str
    collector_id: str


@dataclass(frozen=True)
class HostPressureSeriesSpec:
    counter_id: str
    counter_kind: HostPressureCounterKind
    unit: str
    path: HostPressurePath | None = None
    stall_kind: HostPressureStallKind | None = None

    def __post_init__(self) -> None:
        if not self.counter_id:
            raise ValueError("counter_id must be nonempty")
        if not self.unit:
            raise ValueError("counter unit must be nonempty")
        if self.counter_kind in {
            HostPressureCounterKind.PACKET,
            HostPressureCounterKind.BLOCKED,
        }:
            if self.path is None:
                raise ValueError("packet and blocked counters require a path")
            if self.stall_kind is not None:
                raise ValueError("packet and blocked counters cannot have a stall kind")
        elif self.counter_kind is HostPressureCounterKind.STALL:
            if self.path is not None:
                raise ValueError("stall counters cannot have a posted/nonposted path")
            if self.stall_kind is None:
                raise ValueError("stall counters require a stall kind")
        else:
            raise ValueError("unknown counter kind")


@dataclass(frozen=True)
class HostPressureSeriesIdentity:
    run_id: str
    node_id: str
    nic_id: str
    collector_id: str
    source_counter_id: str


@dataclass(frozen=True)
class HostPressureSeries:
    spec: HostPressureSeriesSpec
    identity: HostPressureSeriesIdentity
    timestamps_ns: tuple[int, ...]
    values: tuple[int, ...]
    value_kind: HostPressureValueKind
    semantics: str
    baseline_relative: bool
    quality_flags: tuple[HostPressureQualityFlag, ...] = ()

    def __post_init__(self) -> None:
        if len(self.timestamps_ns) != len(self.values):
            raise ValueError("timestamps_ns and values must have equal length")
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
class HostPressureDataset:
    population: HostPressurePopulation
    series: tuple[HostPressureSeries, ...]
    window_ns: tuple[int, int]
    blocked_unit_resolution: BlockedUnitResolution | None = None

    def __post_init__(self) -> None:
        counter_ids = [series.spec.counter_id for series in self.series]
        if len(counter_ids) != len(set(counter_ids)):
            raise ValueError("counter IDs must be unique")
        if len(self.window_ns) != 2:
            raise ValueError("window_ns must contain start and end")
        if type(self.window_ns[0]) is not int or type(self.window_ns[1]) is not int:
            raise ValueError("window bounds must be integers")
        if not self.window_ns[0] < self.window_ns[1]:
            raise ValueError("window end must be greater than window start")
        pair_keys = [
            (series.spec.path, series.spec.counter_kind)
            for series in self.series
            if series.spec.counter_kind
            in {HostPressureCounterKind.PACKET, HostPressureCounterKind.BLOCKED}
        ]
        if len(pair_keys) != len(set(pair_keys)):
            raise ValueError("each posted/nonposted packet or blocked counter is unique")
        for series in self.series:
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
class HostPressureInterval:
    start_ns: int
    end_ns: int
    duration_ns: int
    start_value: int
    end_value: int
    delta: int | None
    valid: bool
    state: HostPressureIntervalState
    inside_window: bool


@dataclass(frozen=True)
class HostPressureSeriesResult:
    spec: HostPressureSeriesSpec
    source_identity: HostPressureSeriesIdentity | None
    status: HostPressureStatus
    quality_flags: frozenset[HostPressureQualityFlag]
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
    rate_unit: str
    zero_observed: bool
    intervals: tuple[HostPressureInterval, ...]


@dataclass(frozen=True)
class HostPressurePathResult:
    path: HostPressurePath
    status: HostPressureStatus
    quality_flags: frozenset[HostPressureQualityFlag]
    status_reasons: tuple[str, ...]
    requested_window_ns: tuple[int, int]
    represented_window_ns: tuple[int, int] | None
    covered_duration_ns: int
    requested_duration_ns: int
    coverage_fraction: float | None
    packet_result: HostPressureSeriesResult
    blocked_result: HostPressureSeriesResult
    packet_total: int | None
    blocked_total: int | None
    packet_rate_per_second: float | None
    blocked_rate_per_second: float | None
    blocked_per_packet: float | None
    blocked_per_packet_unit: str | None
    denominator_state: HostPressureDenominatorState
    common_valid_interval_count: int
    reset_excluded_interval_count: int
    packet_delta_formula: str
    rate_formula: str
    blocked_per_packet_formula: str


@dataclass(frozen=True)
class HostPressureAnalysis:
    contract_version: int
    population: HostPressurePopulation
    status: HostPressureStatus
    quality_flags: frozenset[HostPressureQualityFlag]
    status_reasons: tuple[str, ...]
    requested_window_ns: tuple[int, int]
    represented_window_ns: tuple[int, int] | None
    coverage_fraction: float | None
    path_results: tuple[HostPressurePathResult, ...]
    stall_results: tuple[HostPressureSeriesResult, ...]
    missing_counter_ids: tuple[str, ...]
    blocked_unit: str
    blocked_unit_resolution: BlockedUnitResolution | None
    limitations: tuple[str, ...]


def _missing_series_result(
    spec: HostPressureSeriesSpec,
    window_ns: tuple[int, int],
    reason: str,
) -> HostPressureSeriesResult:
    quality_flags = {
        HostPressureQualityFlag.MISSING_PACKET
        if spec.counter_kind is HostPressureCounterKind.PACKET
        else HostPressureQualityFlag.MISSING_BLOCKED
    }
    return HostPressureSeriesResult(
        spec=spec,
        source_identity=None,
        status=HostPressureStatus.MISSING_DATA,
        quality_flags=frozenset(quality_flags),
        status_reasons=(reason,),
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
        rate_per_second=None,
        rate_unit=f"{spec.unit}/s",
        zero_observed=False,
        intervals=(),
    )


def _unsupported_series_result(
    series: HostPressureSeries,
    window_ns: tuple[int, int],
    quality_flags: set[HostPressureQualityFlag],
    reasons: list[str],
) -> HostPressureSeriesResult:
    window_sample_count = sum(
        1
        for timestamp in series.timestamps_ns
        if window_ns[0] <= timestamp <= window_ns[1]
    )
    return HostPressureSeriesResult(
        spec=series.spec,
        source_identity=series.identity,
        status=HostPressureStatus.UNSUPPORTED_UNITS,
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
        rate_per_second=None,
        rate_unit=f"{series.spec.unit}/s",
        zero_observed=False,
        intervals=(),
    )


def _series_intervals(
    series: HostPressureSeries,
    window_ns: tuple[int, int],
) -> tuple[tuple[HostPressureInterval, ...], frozenset[HostPressureQualityFlag]]:
    window_sample_count = sum(
        1
        for timestamp in series.timestamps_ns
        if window_ns[0] <= timestamp <= window_ns[1]
    )
    records = list(zip(series.timestamps_ns, series.values, strict=True))
    ordered = sorted(records, key=lambda record: record[0])
    quality_flags: set[HostPressureQualityFlag] = set(series.quality_flags)
    if records != ordered:
        quality_flags.add(HostPressureQualityFlag.REORDERED)

    intervals: list[HostPressureInterval] = []
    previous: tuple[int, int] | None = None
    index = 0
    while index < len(ordered):
        timestamp = ordered[index][0]
        same_timestamp = [record for record in ordered if record[0] == timestamp]
        values = [value for _, value in same_timestamp]
        if len(values) > 1:
            quality_flags.add(HostPressureQualityFlag.DUPLICATE)
        if len(set(values)) != 1:
            intervals.append(
                HostPressureInterval(
                    start_ns=timestamp,
                    end_ns=timestamp,
                    duration_ns=0,
                    start_value=values[0],
                    end_value=values[1],
                    delta=None,
                    valid=False,
                    state=HostPressureIntervalState.CONFLICTING_TIMESTAMP,
                    inside_window=window_ns[0] <= timestamp <= window_ns[1],
                )
            )
            quality_flags.add(HostPressureQualityFlag.CONFLICTING_TIMESTAMP)
            previous = None
            index += len(same_timestamp)
            continue

        value = values[0]
        if previous is not None:
            start_ns, start_value = previous
            duration_ns = timestamp - start_ns
            delta = value - start_value
            valid = duration_ns > 0 and delta >= 0
            state = (
                HostPressureIntervalState.VALID
                if valid
                else HostPressureIntervalState.DECREASE_OR_RESET
            )
            if not valid:
                quality_flags.add(HostPressureQualityFlag.DECREASE_OR_RESET)
            inside_window = (
                window_ns[0] <= start_ns
                and timestamp <= window_ns[1]
                and duration_ns > 0
            )
            intervals.append(
                HostPressureInterval(
                    start_ns=start_ns,
                    end_ns=timestamp,
                    duration_ns=duration_ns,
                    start_value=start_value,
                    end_value=value,
                    delta=delta,
                    valid=valid,
                    state=state,
                    inside_window=inside_window,
                )
            )
            if not inside_window and valid:
                quality_flags.add(HostPressureQualityFlag.BOUNDARY_INTERVAL_EXCLUDED)
        previous = (timestamp, value)
        index += len(same_timestamp)

    if intervals and not any(interval.inside_window for interval in intervals):
        quality_flags.add(HostPressureQualityFlag.PARTIAL_WINDOW)
    return tuple(intervals), frozenset(quality_flags)


def _analyze_counter(
    series: HostPressureSeries,
    window_ns: tuple[int, int],
) -> HostPressureSeriesResult:
    window_sample_count = sum(
        1
        for timestamp in series.timestamps_ns
        if window_ns[0] <= timestamp <= window_ns[1]
    )
    quality_flags: set[HostPressureQualityFlag] = set(series.quality_flags)
    reasons: list[str] = []
    if series.value_kind is not HostPressureValueKind.CUMULATIVE:
        quality_flags.add(HostPressureQualityFlag.UNSUPPORTED_SERIES)
        reasons.append(f"Unsupported value kind: {series.value_kind.value}.")
    if series.semantics != CUMULATIVE_COUNTER_SEMANTICS:
        quality_flags.add(HostPressureQualityFlag.UNSUPPORTED_SERIES)
        reasons.append(f"Unsupported counter semantics: {series.semantics!r}.")
    if series.baseline_relative:
        quality_flags.add(HostPressureQualityFlag.BASELINE_RELATIVE)
        reasons.append(
            "Baseline-relative samples cannot support exact cumulative interval deltas."
        )
    if series.spec.unit.lower() in {"unknown", "unresolved"}:
        quality_flags.add(HostPressureQualityFlag.UNSUPPORTED_SERIES)
        reasons.append("The counter unit is unresolved.")
    if reasons:
        return _unsupported_series_result(series, window_ns, quality_flags, reasons)

    intervals, interval_flags = _series_intervals(series, window_ns)
    quality_flags.update(interval_flags)
    valid_intervals = [
        interval for interval in intervals if interval.valid and interval.inside_window
    ]
    invalid_intervals = [interval for interval in intervals if not interval.valid]
    outside_count = sum(not interval.inside_window for interval in intervals)
    total = sum(interval.delta for interval in valid_intervals)
    covered_duration = sum(interval.duration_ns for interval in valid_intervals)
    represented_window = None
    if valid_intervals:
        represented_window = (
            min(interval.start_ns for interval in valid_intervals),
            max(interval.end_ns for interval in valid_intervals),
        )
    requested_duration = window_ns[1] - window_ns[0]
    coverage_fraction = (
        covered_duration / requested_duration if requested_duration else None
    )
    rate = total / (covered_duration / NANOSECONDS_PER_SECOND) if covered_duration else None

    if not valid_intervals:
        status = HostPressureStatus.INSUFFICIENT_DATA
        if invalid_intervals:
            status = HostPressureStatus.INVALID_DATA
        reasons.append("No complete valid interval is available inside the window.")
    elif invalid_intervals:
        status = HostPressureStatus.PARTIAL_INVALID_DATA
        reasons.append("One or more intervals were excluded for reset or conflict.")
    elif total == 0:
        status = HostPressureStatus.OBSERVED_ZERO
    else:
        status = HostPressureStatus.OBSERVED_NONZERO

    return HostPressureSeriesResult(
        spec=series.spec,
        source_identity=series.identity,
        status=status,
        quality_flags=frozenset(quality_flags),
        status_reasons=tuple(reasons),
        requested_window_ns=window_ns,
        represented_window_ns=represented_window,
        covered_duration_ns=covered_duration,
        requested_duration_ns=requested_duration,
        coverage_fraction=coverage_fraction,
        sample_count=window_sample_count,
        valid_interval_count=len(valid_intervals),
        invalid_interval_count=len(invalid_intervals),
        outside_window_interval_count=outside_count,
        total=total,
        unit=series.spec.unit,
        rate_per_second=rate,
        rate_unit=f"{series.spec.unit}/s",
        zero_observed=total == 0,
        intervals=intervals,
    )


def _series_by_key(
    series: tuple[HostPressureSeries, ...],
) -> dict[tuple[HostPressurePath | None, HostPressureCounterKind], HostPressureSeries]:
    return {
        (item.spec.path, item.spec.counter_kind): item
        for item in series
        if item.spec.counter_kind
        in {HostPressureCounterKind.PACKET, HostPressureCounterKind.BLOCKED}
    }


def _analyze_path(
    path: HostPressurePath,
    series: tuple[HostPressureSeries, ...],
    window_ns: tuple[int, int],
    blocked_unit_resolution: BlockedUnitResolution | None,
) -> HostPressurePathResult:
    series_map = _series_by_key(series)
    packet_series = series_map.get((path, HostPressureCounterKind.PACKET))
    blocked_series = series_map.get((path, HostPressureCounterKind.BLOCKED))
    packet_counter_id, blocked_counter_id = PATH_COUNTER_IDS[path]
    packet_spec = HostPressureSeriesSpec(
        counter_id=packet_counter_id,
        counter_kind=HostPressureCounterKind.PACKET,
        unit=PACKET_UNIT,
        path=path,
    )
    blocked_spec = HostPressureSeriesSpec(
        counter_id=blocked_counter_id,
        counter_kind=HostPressureCounterKind.BLOCKED,
        unit=blocked_unit_resolution.unit.value if blocked_unit_resolution else "unresolved",
        path=path,
    )
    packet_result = (
        _analyze_counter(packet_series, window_ns)
        if packet_series is not None
        else _missing_series_result(
            packet_spec,
            window_ns,
            f"The {path.value} packet counter is missing.",
        )
    )
    blocked_result = (
        _analyze_counter(blocked_series, window_ns)
        if blocked_series is not None
        else _missing_series_result(
            blocked_spec,
            window_ns,
            f"The {path.value} blocked counter is missing.",
        )
    )

    quality_flags: set[HostPressureQualityFlag] = set()
    quality_flags.update(packet_result.quality_flags)
    quality_flags.update(blocked_result.quality_flags)
    reasons: list[str] = []
    units_supported = True
    incompatible_units = False
    if packet_series is not None and packet_series.spec.unit != PACKET_UNIT:
        incompatible_units = True
        quality_flags.add(HostPressureQualityFlag.INCOMPATIBLE_UNITS)
        reasons.append(
            f"The packet counter unit must be {PACKET_UNIT!r}, not "
            f"{packet_series.spec.unit!r}."
        )
    if blocked_unit_resolution is None:
        units_supported = False
        quality_flags.add(HostPressureQualityFlag.UNRESOLVED_BLOCKED_UNITS)
        reasons.append("The blocked-counter unit has no authoritative resolution.")
    elif blocked_series is not None and (
        blocked_series.spec.unit != blocked_unit_resolution.unit.value
    ):
        incompatible_units = True
        units_supported = False
        quality_flags.add(HostPressureQualityFlag.INCOMPATIBLE_UNITS)
        reasons.append(
            f"The blocked counter unit must be {blocked_unit_resolution.unit.value!r}, "
            f"not {blocked_series.spec.unit!r}."
        )
    if incompatible_units:
        units_supported = False

    packet_intervals = {
        (interval.start_ns, interval.end_ns): interval
        for interval in packet_result.intervals
        if interval.valid and interval.inside_window
    }
    blocked_intervals = {
        (interval.start_ns, interval.end_ns): interval
        for interval in blocked_result.intervals
        if interval.valid and interval.inside_window
    }
    common_keys = set(packet_intervals).intersection(blocked_intervals)
    packet_total = (
        sum(packet_intervals[key].delta for key in common_keys)
        if packet_series is not None and common_keys
        else None
    )
    blocked_total = (
        sum(blocked_intervals[key].delta for key in common_keys)
        if blocked_series is not None and common_keys
        else None
    )
    covered_duration = sum(packet_intervals[key].duration_ns for key in common_keys)
    requested_duration = window_ns[1] - window_ns[0]
    represented_window = None
    if common_keys:
        starts = [key[0] for key in common_keys]
        ends = [key[1] for key in common_keys]
        represented_window = (min(starts), max(ends))
    coverage_fraction = (
        covered_duration / requested_duration if requested_duration else None
    )
    packet_rate = (
        packet_total / (covered_duration / NANOSECONDS_PER_SECOND)
        if packet_total is not None and covered_duration
        else None
    )
    blocked_rate = (
        blocked_total / (covered_duration / NANOSECONDS_PER_SECOND)
        if blocked_total is not None and covered_duration and units_supported
        else None
    )
    reset_excluded_count = sum(
        interval.state is HostPressureIntervalState.DECREASE_OR_RESET
        for interval in (*packet_result.intervals, *blocked_result.intervals)
    )

    if packet_series is None:
        denominator_state = HostPressureDenominatorState.MISSING_PACKET
    elif blocked_series is None:
        denominator_state = HostPressureDenominatorState.MISSING_BLOCKED
    elif not units_supported and (
        HostPressureQualityFlag.INCOMPATIBLE_UNITS in quality_flags
    ):
        denominator_state = HostPressureDenominatorState.INCOMPATIBLE_UNITS
    elif not units_supported:
        denominator_state = HostPressureDenominatorState.UNRESOLVED_BLOCKED_UNITS
    elif not common_keys:
        if packet_result.valid_interval_count == 0 and packet_result.invalid_interval_count:
            denominator_state = HostPressureDenominatorState.INVALID_PACKET_INTERVALS
        else:
            denominator_state = HostPressureDenominatorState.NO_COMMON_VALID_INTERVAL
            quality_flags.add(HostPressureQualityFlag.NO_COMMON_VALID_INTERVAL)
    elif packet_total == 0:
        denominator_state = HostPressureDenominatorState.ZERO_PACKETS
    else:
        denominator_state = HostPressureDenominatorState.AVAILABLE

    blocked_per_packet = None
    blocked_per_packet_unit = None
    if (
        denominator_state is HostPressureDenominatorState.AVAILABLE
        and packet_total is not None
        and packet_total > 0
        and blocked_total is not None
    ):
        blocked_per_packet = blocked_total / packet_total
        blocked_per_packet_unit = (
            f"{blocked_unit_resolution.unit.value}/packet"
            if blocked_unit_resolution is not None
            else None
        )

    if not units_supported:
        status = HostPressureStatus.UNSUPPORTED_UNITS
    elif packet_series is None and blocked_series is None:
        status = HostPressureStatus.MISSING_DATA
    elif packet_series is None or blocked_series is None:
        status = HostPressureStatus.PARTIAL_MISSING_DATA
    elif not common_keys:
        status = (
            HostPressureStatus.INVALID_DATA
            if denominator_state
            is HostPressureDenominatorState.INVALID_PACKET_INTERVALS
            else HostPressureStatus.INSUFFICIENT_DATA
        )
    elif packet_result.invalid_interval_count or blocked_result.invalid_interval_count:
        status = HostPressureStatus.PARTIAL_INVALID_DATA
    elif packet_total == 0 and blocked_total == 0:
        status = HostPressureStatus.OBSERVED_ZERO
    else:
        status = HostPressureStatus.OBSERVED_NONZERO

    if status is HostPressureStatus.UNSUPPORTED_UNITS and not reasons:
        reasons.append("The blocked-counter units are not authoritatively resolved.")
    if denominator_state is HostPressureDenominatorState.ZERO_PACKETS:
        reasons.append("The packet denominator is zero, so blocked-per-packet is undefined.")
    if denominator_state is HostPressureDenominatorState.NO_COMMON_VALID_INTERVAL:
        reasons.append("The packet and blocked counters have no common valid interval.")

    return HostPressurePathResult(
        path=path,
        status=status,
        quality_flags=frozenset(quality_flags),
        status_reasons=tuple(reasons),
        requested_window_ns=window_ns,
        represented_window_ns=represented_window,
        covered_duration_ns=covered_duration,
        requested_duration_ns=requested_duration,
        coverage_fraction=coverage_fraction,
        packet_result=packet_result,
        blocked_result=blocked_result,
        packet_total=packet_total,
        blocked_total=blocked_total,
        packet_rate_per_second=packet_rate,
        blocked_rate_per_second=blocked_rate,
        blocked_per_packet=blocked_per_packet,
        blocked_per_packet_unit=blocked_per_packet_unit,
        denominator_state=denominator_state,
        common_valid_interval_count=len(common_keys),
        reset_excluded_interval_count=reset_excluded_count,
        packet_delta_formula=DELTA_FORMULA,
        rate_formula=RATE_FORMULA,
        blocked_per_packet_formula=BLOCKED_PER_PACKET_FORMULA,
    )


def _overall_status(
    statuses: tuple[HostPressureStatus, ...],
) -> HostPressureStatus:
    if HostPressureStatus.UNSUPPORTED_UNITS in statuses:
        return HostPressureStatus.UNSUPPORTED_UNITS
    if HostPressureStatus.MISSING_DATA in statuses:
        return HostPressureStatus.MISSING_DATA
    if HostPressureStatus.PARTIAL_MISSING_DATA in statuses:
        return HostPressureStatus.PARTIAL_MISSING_DATA
    if HostPressureStatus.INVALID_DATA in statuses:
        return HostPressureStatus.INVALID_DATA
    if HostPressureStatus.PARTIAL_INVALID_DATA in statuses:
        return HostPressureStatus.PARTIAL_INVALID_DATA
    if HostPressureStatus.INSUFFICIENT_DATA in statuses:
        return HostPressureStatus.INSUFFICIENT_DATA
    if HostPressureStatus.OBSERVED_ZERO in statuses:
        return HostPressureStatus.OBSERVED_ZERO
    return HostPressureStatus.OBSERVED_NONZERO


def analyze_host_pressure(dataset: HostPressureDataset) -> HostPressureAnalysis:
    """Analyze posted/nonposted host pressure without unsupported causal claims."""
    path_results = tuple(
        _analyze_path(
            path,
            dataset.series,
            dataset.window_ns,
            dataset.blocked_unit_resolution,
        )
        for path in HostPressurePath
    )
    stall_results = tuple(
        _analyze_counter(series, dataset.window_ns)
        for series in dataset.series
        if series.spec.counter_kind is HostPressureCounterKind.STALL
    )
    quality_flags: set[HostPressureQualityFlag] = set()
    for result in (*path_results, *stall_results):
        quality_flags.update(result.quality_flags)
    statuses = tuple(result.status for result in path_results)
    status = _overall_status(statuses)
    reasons = tuple(
        reason
        for result in path_results
        for reason in result.status_reasons
        if result.status is status
    )
    coverage_fractions = [
        result.coverage_fraction
        for result in path_results
        if result.coverage_fraction is not None
    ]
    represented_windows = [
        result.represented_window_ns
        for result in path_results
        if result.represented_window_ns is not None
    ]
    missing_counter_ids = tuple(
        series_result.spec.counter_id
        for path_result in path_results
        for series_result in (
            path_result.packet_result,
            path_result.blocked_result,
        )
        if series_result.status is HostPressureStatus.MISSING_DATA
    ) + tuple(
        stall_result.spec.counter_id
        for stall_result in stall_results
        if stall_result.status is HostPressureStatus.MISSING_DATA
    )
    return HostPressureAnalysis(
        contract_version=HOST_PRESSURE_ANALYSIS_VERSION,
        population=dataset.population,
        status=status,
        quality_flags=frozenset(quality_flags),
        status_reasons=reasons,
        requested_window_ns=dataset.window_ns,
        represented_window_ns=(
            (min(window[0] for window in represented_windows), max(window[1] for window in represented_windows))
            if represented_windows
            else None
        ),
        coverage_fraction=min(coverage_fractions) if coverage_fractions else None,
        path_results=path_results,
        stall_results=stall_results,
        missing_counter_ids=missing_counter_ids,
        blocked_unit=(
            dataset.blocked_unit_resolution.unit.value
            if dataset.blocked_unit_resolution
            else "unresolved"
        ),
        blocked_unit_resolution=dataset.blocked_unit_resolution,
        limitations=HOST_PRESSURE_LIMITATIONS,
    )
