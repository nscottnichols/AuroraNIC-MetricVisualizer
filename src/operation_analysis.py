"""Pure command-queue operation analysis for cumulative NIC counters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import groupby


OPERATION_ANALYSIS_VERSION = 1
CUMULATIVE_COUNTER_SEMANTICS = "cumulative counter"
COMMAND_UNIT = "commands"
DMA_OPCODE_COUNTER_GROUP = "cq_dma_cmd_counts"
DMA_TOTAL_COUNTER_GROUP = "cq_num_dma_cmds"
IDC_TOTAL_COUNTER_GROUP = "cq_num_idc_cmds"
DMA_OPCODE_BIN_COUNT = 16
DMA_TOTAL_COUNTER_COUNT = 4
IDC_TOTAL_COUNTER_COUNT = 4
DELTA_FORMULA = "end_value - start_value"
RATE_FORMULA = "sum(valid interval deltas) / sum(valid interval durations)"
OPCODE_FRACTION_FORMULA = "opcode bin delta total / sum(DMA opcode bin delta totals)"
DMA_IDC_FRACTION_FORMULA = "family command total / sum(DMA and IDC command totals)"
UINT64_MAX = 2**64 - 1
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
NANOSECONDS_PER_SECOND = 1_000_000_000

OPERATION_LIMITATIONS = (
    "Command-queue counters are control-path command counts, not wire bytes.",
    "Command counts are not MPI payload or message sizes.",
    "This analysis does not infer eager or rendezvous behavior.",
    "Opcode labels require the supplied versioned, authoritative mapping.",
    "Unmapped opcode bins remain unknown and are never assigned guessed labels.",
    "Command counts may suggest temporal changes but do not prove causality.",
)


class OperationValueKind(str, Enum):
    CUMULATIVE = "cumulative"
    DELTA = "delta"
    GAUGE = "gauge"
    UNKNOWN = "unknown"


class OperationCounterKind(str, Enum):
    DMA_OPCODE_BIN = "dma_opcode_bin"
    DMA_TOTAL = "dma_total"
    IDC_TOTAL = "idc_total"


class OperationStatus(str, Enum):
    OBSERVED_ZERO = "observed_zero"
    OBSERVED_NONZERO = "observed_nonzero"
    MISSING_DATA = "missing_data"
    PARTIAL_MISSING_DATA = "partial_missing_data"
    INSUFFICIENT_DATA = "insufficient_data"
    PARTIAL_INVALID_DATA = "partial_invalid_data"
    UNSUPPORTED = "unsupported"


class OperationQualityFlag(str, Enum):
    MISSING_COUNTER = "missing_counter"
    BASELINE_RELATIVE = "baseline_relative"
    UNSUPPORTED_SERIES = "unsupported_series"
    DECREASE_OR_RESET = "decrease_or_reset"
    REORDERED = "reordered"
    DUPLICATE = "duplicate"
    CONFLICTING_TIMESTAMP = "conflicting_timestamp"
    PARTIAL_WINDOW = "partial_window"
    BOUNDARY_INTERVAL_EXCLUDED = "boundary_interval_excluded"
    INCOMPATIBLE_OPCODE_MAPPING = "incompatible_opcode_mapping"
    UNKNOWN_OPCODE = "unknown_opcode"
    INCOMPATIBLE_COVERAGE = "incompatible_coverage"


class OperationIntervalState(str, Enum):
    VALID = "valid"
    DECREASE_OR_RESET = "decrease_or_reset"
    CONFLICTING_TIMESTAMP = "conflicting_timestamp"


class OpcodeMappingStatus(str, Enum):
    MISSING_MAPPING = "missing_mapping"
    INCOMPATIBLE_MAPPING = "incompatible_mapping"
    PARTIAL_UNKNOWN_OPCODE = "partial_unknown_opcode"
    ALL_MAPPED = "all_mapped"
    NO_OPCODE_BINS = "no_opcode_bins"


@dataclass(frozen=True)
class OperationPopulation:
    run_id: str
    node_id: str
    nic_id: str
    collector_id: str


@dataclass(frozen=True)
class OperationSeriesIdentity:
    run_id: str
    node_id: str
    nic_id: str
    collector_id: str
    source_counter_id: str


@dataclass(frozen=True)
class OpcodeMappingEntry:
    bin_index: int
    label: str

    def __post_init__(self) -> None:
        if type(self.bin_index) is not int or not 0 <= self.bin_index < DMA_OPCODE_BIN_COUNT:
            raise ValueError("opcode bin_index must be an integer in [0, 15]")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("opcode label must be a nonempty string")


@dataclass(frozen=True)
class OpcodeMapping:
    mapping_version: str
    evidence: str
    counter_group: str
    entries: tuple[OpcodeMappingEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.mapping_version, str) or not self.mapping_version.strip():
            raise ValueError("mapping_version must be a nonempty string")
        if not isinstance(self.evidence, str) or not self.evidence.strip():
            raise ValueError("opcode mapping evidence must be a nonempty string")
        if not isinstance(self.counter_group, str) or not self.counter_group.strip():
            raise ValueError("opcode mapping counter_group must be nonempty")
        bin_indices = [entry.bin_index for entry in self.entries]
        if len(bin_indices) != len(set(bin_indices)):
            raise ValueError("opcode mapping entries must have unique bin indices")

    def label_for_bin(self, bin_index: int) -> str | None:
        return next(
            (entry.label for entry in self.entries if entry.bin_index == bin_index),
            None,
        )


@dataclass(frozen=True)
class OperationCounterSpec:
    counter_id: str
    counter_kind: OperationCounterKind
    unit: str
    bin_index: int | None = None
    queue_index: int | None = None

    def __post_init__(self) -> None:
        if not self.counter_id:
            raise ValueError("counter_id must be nonempty")
        if not self.unit:
            raise ValueError("counter unit must be nonempty")
        if self.counter_kind is OperationCounterKind.DMA_OPCODE_BIN:
            if type(self.bin_index) is not int or not 0 <= self.bin_index < DMA_OPCODE_BIN_COUNT:
                raise ValueError("DMA opcode bin counters require bin_index in [0, 15]")
            if self.counter_id != f"cq_dma_cmd_counts_{self.bin_index}":
                raise ValueError("DMA opcode bin counter_id does not match bin_index")
            if self.queue_index is not None:
                raise ValueError("DMA opcode bin counters cannot have queue_index")
        else:
            expected_count = (
                DMA_TOTAL_COUNTER_COUNT
                if self.counter_kind is OperationCounterKind.DMA_TOTAL
                else IDC_TOTAL_COUNTER_COUNT
            )
            expected_prefix = (
                "cq_num_dma_cmds_"
                if self.counter_kind is OperationCounterKind.DMA_TOTAL
                else "cq_num_idc_cmds_"
            )
            if (
                type(self.queue_index) is not int
                or not 0 <= self.queue_index < expected_count
            ):
                raise ValueError(f"{self.counter_kind.value} counters require a valid queue_index")
            if self.counter_id != f"{expected_prefix}{self.queue_index}":
                raise ValueError("command total counter_id does not match queue_index")
            if self.bin_index is not None:
                raise ValueError("command total counters cannot have bin_index")


def dma_opcode_spec(bin_index: int) -> OperationCounterSpec:
    return OperationCounterSpec(
        counter_id=f"cq_dma_cmd_counts_{bin_index}",
        counter_kind=OperationCounterKind.DMA_OPCODE_BIN,
        unit=COMMAND_UNIT,
        bin_index=bin_index,
    )


def dma_total_spec(queue_index: int) -> OperationCounterSpec:
    return OperationCounterSpec(
        counter_id=f"cq_num_dma_cmds_{queue_index}",
        counter_kind=OperationCounterKind.DMA_TOTAL,
        unit=COMMAND_UNIT,
        queue_index=queue_index,
    )


def idc_total_spec(queue_index: int) -> OperationCounterSpec:
    return OperationCounterSpec(
        counter_id=f"cq_num_idc_cmds_{queue_index}",
        counter_kind=OperationCounterKind.IDC_TOTAL,
        unit=COMMAND_UNIT,
        queue_index=queue_index,
    )


def standard_operation_specs() -> tuple[OperationCounterSpec, ...]:
    return (
        *(dma_opcode_spec(index) for index in range(DMA_OPCODE_BIN_COUNT)),
        *(dma_total_spec(index) for index in range(DMA_TOTAL_COUNTER_COUNT)),
        *(idc_total_spec(index) for index in range(IDC_TOTAL_COUNTER_COUNT)),
    )


@dataclass(frozen=True)
class OperationCounterSeries:
    spec: OperationCounterSpec
    identity: OperationSeriesIdentity
    timestamps_ns: tuple[int, ...]
    values: tuple[int, ...]
    value_kind: OperationValueKind
    semantics: str
    baseline_relative: bool
    quality_flags: tuple[OperationQualityFlag, ...] = ()

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
class OperationDataset:
    population: OperationPopulation
    requested_counters: tuple[OperationCounterSpec, ...]
    series: tuple[OperationCounterSeries, ...]
    window_ns: tuple[int, int]
    opcode_mapping: OpcodeMapping | None = None

    def __post_init__(self) -> None:
        counter_ids = [spec.counter_id for spec in self.requested_counters]
        if len(counter_ids) != len(set(counter_ids)):
            raise ValueError("requested counter IDs must be unique")
        series_ids = [series.spec.counter_id for series in self.series]
        if len(series_ids) != len(set(series_ids)):
            raise ValueError("counter IDs in series must be unique")
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


@dataclass(frozen=True)
class OperationInterval:
    start_ns: int
    end_ns: int
    duration_ns: int
    start_value: int
    end_value: int
    delta: int
    valid: bool
    state: OperationIntervalState
    inside_window: bool


@dataclass(frozen=True)
class OperationCounterResult:
    spec: OperationCounterSpec
    source_identity: OperationSeriesIdentity | None
    status: OperationStatus
    quality_flags: frozenset[OperationQualityFlag]
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
    intervals: tuple[OperationInterval, ...]


@dataclass(frozen=True)
class OperationAggregateResult:
    counter_kind: OperationCounterKind
    status: OperationStatus
    quality_flags: frozenset[OperationQualityFlag]
    status_reasons: tuple[str, ...]
    requested_window_ns: tuple[int, int]
    represented_window_ns: tuple[int, int] | None
    covered_duration_ns: int
    requested_duration_ns: int
    coverage_fraction: float | None
    source_counter_ids: tuple[str, ...]
    compatible_counter_ids: tuple[str, ...]
    incompatible_counter_ids: tuple[str, ...]
    missing_counter_ids: tuple[str, ...]
    common_valid_interval_count: int
    total: int | None
    unit: str
    rate_per_second: float | None
    rate_unit: str
    zero_observed: bool


@dataclass(frozen=True)
class OperationOpcodeResult:
    bin_index: int
    counter_id: str
    counter_result: OperationCounterResult
    opcode_label: str | None
    opcode_mapped: bool
    total: int | None
    rate_per_second: float | None
    fraction_of_dma_opcode_commands: float | None
    zero_observed: bool


@dataclass(frozen=True)
class OperationFamilyMix:
    status: OperationStatus
    quality_flags: frozenset[OperationQualityFlag]
    status_reasons: tuple[str, ...]
    dma_command_total: int | None
    idc_command_total: int | None
    total_commands: int | None
    dma_fraction: float | None
    idc_fraction: float | None
    covered_duration_ns: int
    coverage_fraction: float | None
    represented_window_ns: tuple[int, int] | None
    formula: str


@dataclass(frozen=True)
class OperationAnalysis:
    analysis_version: int
    population: OperationPopulation
    status: OperationStatus
    quality_flags: frozenset[OperationQualityFlag]
    status_reasons: tuple[str, ...]
    requested_window_ns: tuple[int, int]
    requested_duration_ns: int
    dma_opcode_bin_results: tuple[OperationCounterResult, ...]
    dma_total_results: tuple[OperationCounterResult, ...]
    idc_total_results: tuple[OperationCounterResult, ...]
    aggregate_dma_commands: OperationAggregateResult
    aggregate_idc_commands: OperationAggregateResult
    opcode_results: tuple[OperationOpcodeResult, ...]
    opcode_mapping_status: OpcodeMappingStatus
    opcode_mapping_version: str | None
    mapped_opcode_counter_ids: tuple[str, ...]
    unmapped_opcode_counter_ids: tuple[str, ...]
    missing_opcode_counter_ids: tuple[str, ...]
    mapped_opcode_total: int | None
    unknown_opcode_total: int | None
    dma_opcode_command_total: int | None
    dma_idc_mix: OperationFamilyMix
    delta_formula: str
    rate_formula: str
    opcode_fraction_formula: str
    dma_idc_fraction_formula: str
    limitations: tuple[str, ...]


def _missing_result(
    spec: OperationCounterSpec,
    window_ns: tuple[int, int],
) -> OperationCounterResult:
    return OperationCounterResult(
        spec=spec,
        source_identity=None,
        status=OperationStatus.MISSING_DATA,
        quality_flags=frozenset({OperationQualityFlag.MISSING_COUNTER}),
        status_reasons=("The requested command counter has no observation series.",),
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


def _window_sample_count(series, window_ns):
    return sum(
        1
        for timestamp in series.timestamps_ns
        if window_ns[0] <= timestamp <= window_ns[1]
    )


def _unsupported_result(
    series: OperationCounterSeries,
    window_ns: tuple[int, int],
    quality_flags: set[OperationQualityFlag],
    reasons: list[str],
) -> OperationCounterResult:
    window_sample_count = _window_sample_count(series, window_ns)
    return OperationCounterResult(
        spec=series.spec,
        source_identity=series.identity,
        status=OperationStatus.UNSUPPORTED,
        quality_flags=frozenset(quality_flags),
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


def _analyze_counter(
    series: OperationCounterSeries,
    window_ns: tuple[int, int],
) -> OperationCounterResult:
    window_sample_count = _window_sample_count(series, window_ns)
    quality_flags: set[OperationQualityFlag] = set(series.quality_flags)
    reasons: list[str] = []
    if series.baseline_relative:
        quality_flags.add(OperationQualityFlag.BASELINE_RELATIVE)
        reasons.append(
            "Baseline-relative pftrace samples cannot support exact cumulative interval deltas."
        )
    if series.value_kind is not OperationValueKind.CUMULATIVE:
        quality_flags.add(OperationQualityFlag.UNSUPPORTED_SERIES)
        reasons.append(f"Unsupported value kind: {series.value_kind.value}.")
    if series.semantics != CUMULATIVE_COUNTER_SEMANTICS:
        quality_flags.add(OperationQualityFlag.UNSUPPORTED_SERIES)
        reasons.append(f"Unsupported counter semantics: {series.semantics!r}.")
    if reasons:
        return _unsupported_result(series, window_ns, quality_flags, reasons)

    records = list(zip(series.timestamps_ns, series.values, strict=True))
    ordered = sorted(records, key=lambda record: record[0])
    if records != ordered:
        quality_flags.add(OperationQualityFlag.REORDERED)
    conflicting_timestamps = {
        timestamp
        for timestamp, _ in ordered
        if len({value for other_timestamp, value in ordered if other_timestamp == timestamp}) > 1
    }
    if conflicting_timestamps:
        quality_flags.add(OperationQualityFlag.DUPLICATE)
        quality_flags.add(OperationQualityFlag.CONFLICTING_TIMESTAMP)
        return _unsupported_result(
            series,
            window_ns,
            quality_flags,
            [
                "Conflicting counter values at duplicate timestamps make "
                "adjacent interval deltas unsupported."
            ],
        )

    intervals: list[OperationInterval] = []
    valid_intervals: list[OperationInterval] = []
    invalid_interval_count = 0
    outside_window_interval_count = 0
    index = 0
    while index < len(ordered):
        timestamp = ordered[index][0]
        same_timestamp = [record for record in ordered if record[0] == timestamp]
        values = [value for _, value in same_timestamp]
        value = values[0]
        if index > 0:
            start_ns, start_value = ordered[index - 1]
            duration_ns = timestamp - start_ns
            delta = value - start_value
            valid = duration_ns > 0 and delta >= 0
            state = (
                OperationIntervalState.VALID
                if valid
                else OperationIntervalState.DECREASE_OR_RESET
            )
            if not valid:
                quality_flags.add(OperationQualityFlag.DECREASE_OR_RESET)
            inside_window = (
                window_ns[0] <= start_ns
                and timestamp <= window_ns[1]
                and duration_ns > 0
            )
            interval = OperationInterval(
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
            intervals.append(interval)
            if not inside_window:
                outside_window_interval_count += 1
                if start_ns < window_ns[0] < timestamp or start_ns < window_ns[1] < timestamp:
                    quality_flags.add(OperationQualityFlag.BOUNDARY_INTERVAL_EXCLUDED)
            elif valid:
                valid_intervals.append(interval)
            else:
                invalid_interval_count += 1
        index += 1

    covered_duration_ns = sum(interval.duration_ns for interval in valid_intervals)
    requested_duration_ns = window_ns[1] - window_ns[0]
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
    if covered_duration_ns < requested_duration_ns:
        quality_flags.add(OperationQualityFlag.PARTIAL_WINDOW)

    if not valid_intervals:
        status = OperationStatus.INSUFFICIENT_DATA
        reasons.append("The window contains no complete valid command interval.")
    elif invalid_interval_count:
        status = OperationStatus.PARTIAL_INVALID_DATA
        reasons.append("One or more intervals were excluded for decrease or reset.")
    elif total == 0:
        status = OperationStatus.OBSERVED_ZERO
    else:
        status = OperationStatus.OBSERVED_NONZERO

    return OperationCounterResult(
        spec=series.spec,
        source_identity=series.identity,
        status=status,
        quality_flags=frozenset(quality_flags),
        status_reasons=tuple(reasons),
        requested_window_ns=window_ns,
        represented_window_ns=represented_window_ns,
        covered_duration_ns=covered_duration_ns,
        requested_duration_ns=requested_duration_ns,
        coverage_fraction=covered_duration_ns / requested_duration_ns,
        sample_count=window_sample_count,
        valid_interval_count=len(valid_intervals),
        invalid_interval_count=invalid_interval_count,
        outside_window_interval_count=outside_window_interval_count,
        total=total,
        unit=series.spec.unit,
        rate_per_second=rate_per_second,
        rate_unit=f"{series.spec.unit}/s",
        zero_observed=bool(valid_intervals) and total == 0,
        intervals=tuple(intervals),
    )


def _valid_interval_keys(result: OperationCounterResult) -> set[tuple[int, int]]:
    return {
        (interval.start_ns, interval.end_ns)
        for interval in result.intervals
        if interval.valid and interval.inside_window
    }


def _aggregate_results(
    counter_kind: OperationCounterKind,
    requested_specs: tuple[OperationCounterSpec, ...],
    results: tuple[OperationCounterResult, ...],
    window_ns: tuple[int, int],
) -> OperationAggregateResult:
    requested_ids = tuple(spec.counter_id for spec in requested_specs)
    present_results = [
        result for result in results if result.source_identity is not None
    ]
    missing_ids = tuple(
        result.spec.counter_id
        for result in results
        if result.source_identity is None
    )
    quality_flags: set[OperationQualityFlag] = set()
    for result in results:
        quality_flags.update(result.quality_flags)

    common_keys: set[tuple[int, int]] | None = None
    for result in present_results:
        keys = _valid_interval_keys(result)
        common_keys = keys if common_keys is None else common_keys & keys
    common_keys = common_keys or set()

    compatible_ids = tuple(
        result.spec.counter_id
        for result in present_results
        if _valid_interval_keys(result) == common_keys
        and common_keys
    )
    incompatible_ids = tuple(
        result.spec.counter_id
        for result in present_results
        if result.spec.counter_id not in compatible_ids
    )
    if incompatible_ids:
        quality_flags.add(OperationQualityFlag.INCOMPATIBLE_COVERAGE)

    covered_duration_ns = sum(end - start for start, end in common_keys)
    requested_duration_ns = window_ns[1] - window_ns[0]
    total = (
        sum(
            interval.delta
            for result in present_results
            for interval in result.intervals
            if interval.valid
            and interval.inside_window
            and (interval.start_ns, interval.end_ns) in common_keys
        )
        if present_results
        and not missing_ids
        and common_keys
        else None
    )
    rate_per_second = (
        total / (covered_duration_ns / NANOSECONDS_PER_SECOND)
        if total is not None and covered_duration_ns > 0
        else None
    )
    represented_window_ns = (
        (min(start for start, _ in common_keys), max(end for _, end in common_keys))
        if common_keys
        else None
    )
    reasons: list[str] = []
    if missing_ids:
        reasons.append("One or more requested command-total counters are missing.")
    if incompatible_ids:
        reasons.append("Command-total counters do not share complete valid interval coverage.")

    invalid_count = sum(result.invalid_interval_count for result in results)
    if missing_ids and not present_results:
        status = OperationStatus.MISSING_DATA
    elif missing_ids:
        status = OperationStatus.PARTIAL_MISSING_DATA
    elif not common_keys:
        status = OperationStatus.INSUFFICIENT_DATA
        reasons.append("No common complete valid interval is available.")
    elif invalid_count:
        status = OperationStatus.PARTIAL_INVALID_DATA
        reasons.append("One or more intervals were excluded for reset or conflict.")
    elif total == 0:
        status = OperationStatus.OBSERVED_ZERO
    else:
        status = OperationStatus.OBSERVED_NONZERO

    return OperationAggregateResult(
        counter_kind=counter_kind,
        status=status,
        quality_flags=frozenset(quality_flags),
        status_reasons=tuple(reasons),
        requested_window_ns=window_ns,
        represented_window_ns=represented_window_ns,
        covered_duration_ns=covered_duration_ns,
        requested_duration_ns=requested_duration_ns,
        coverage_fraction=covered_duration_ns / requested_duration_ns,
        source_counter_ids=requested_ids,
        compatible_counter_ids=compatible_ids,
        incompatible_counter_ids=incompatible_ids,
        missing_counter_ids=missing_ids,
        common_valid_interval_count=len(common_keys),
        total=total,
        unit=COMMAND_UNIT,
        rate_per_second=rate_per_second,
        rate_unit=f"{COMMAND_UNIT}/s",
        zero_observed=total == 0,
    )


def _overall_counter_status(
    results: tuple[OperationCounterResult, ...],
) -> OperationStatus:
    statuses = tuple(result.status for result in results)
    if not statuses:
        return OperationStatus.MISSING_DATA
    if OperationStatus.UNSUPPORTED in statuses:
        return OperationStatus.UNSUPPORTED
    if all(status is OperationStatus.MISSING_DATA for status in statuses):
        return OperationStatus.MISSING_DATA
    if OperationStatus.MISSING_DATA in statuses:
        return OperationStatus.PARTIAL_MISSING_DATA
    if OperationStatus.INSUFFICIENT_DATA in statuses:
        return OperationStatus.INSUFFICIENT_DATA
    if OperationStatus.PARTIAL_INVALID_DATA in statuses:
        return OperationStatus.PARTIAL_INVALID_DATA
    if OperationStatus.OBSERVED_ZERO in statuses:
        return OperationStatus.OBSERVED_ZERO
    return OperationStatus.OBSERVED_NONZERO


def _aggregate_interval_keys(
    aggregate: OperationAggregateResult,
    results: tuple[OperationCounterResult, ...],
) -> set[tuple[int, int]]:
    if aggregate.total is None:
        return set()
    present_results = [
        result for result in results if result.source_identity is not None
    ]
    common_keys: set[tuple[int, int]] | None = None
    for result in present_results:
        keys = _valid_interval_keys(result)
        common_keys = keys if common_keys is None else common_keys & keys
    return common_keys or set()


def _opcode_results(
    results: tuple[OperationCounterResult, ...],
    mapping: OpcodeMapping | None,
    window_ns: tuple[int, int],
) -> tuple[
    tuple[OperationOpcodeResult, ...],
    OpcodeMappingStatus,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    int | None,
    int | None,
    int | None,
    set[OperationQualityFlag],
]:
    present_results = [
        result for result in results if result.source_identity is not None
    ]
    common_keys: set[tuple[int, int]] | None = None
    for result in present_results:
        keys = _valid_interval_keys(result)
        common_keys = keys if common_keys is None else common_keys & keys
    common_keys = common_keys or set()
    missing = [result for result in results if result.source_identity is None]
    complete = not missing and present_results and common_keys

    totals: dict[int, int | None] = {}
    for result in present_results:
        totals[result.spec.bin_index] = (
            sum(
                interval.delta
                for interval in result.intervals
                if interval.valid
                and interval.inside_window
                and (interval.start_ns, interval.end_ns) in common_keys
            )
            if common_keys
            else None
        )
    denominator = (
        sum(total for total in totals.values() if total is not None)
        if complete
        else None
    )
    covered_duration_ns = sum(end - start for start, end in common_keys)

    mapping_compatible = mapping is not None and mapping.counter_group == DMA_OPCODE_COUNTER_GROUP
    opcode_results: list[OperationOpcodeResult] = []
    mapped_ids: list[str] = []
    unmapped_ids: list[str] = []
    missing_ids: list[str] = []
    mapped_total = 0
    unknown_total = 0
    quality_flags: set[OperationQualityFlag] = set()

    for result in results:
        bin_index = result.spec.bin_index
        assert bin_index is not None
        label = mapping.label_for_bin(bin_index) if mapping_compatible else None
        mapped = label is not None
        total = totals.get(bin_index)
        if result.source_identity is None:
            missing_ids.append(result.spec.counter_id)
        elif mapped:
            mapped_ids.append(result.spec.counter_id)
            if total is not None:
                mapped_total += total
        else:
            unmapped_ids.append(result.spec.counter_id)
            quality_flags.add(OperationQualityFlag.UNKNOWN_OPCODE)
            if total is not None:
                unknown_total += total

        opcode_results.append(
            OperationOpcodeResult(
                bin_index=bin_index,
                counter_id=result.spec.counter_id,
                counter_result=result,
                opcode_label=label,
                opcode_mapped=mapped,
                total=total,
                rate_per_second=(
                    total / (covered_duration_ns / NANOSECONDS_PER_SECOND)
                    if total is not None and covered_duration_ns > 0
                    else None
                ),
                fraction_of_dma_opcode_commands=(
                    total / denominator if total is not None and denominator else None
                ),
                zero_observed=total == 0,
            )
        )

    if mapping is None:
        mapping_status = OpcodeMappingStatus.MISSING_MAPPING
    elif not mapping_compatible:
        mapping_status = OpcodeMappingStatus.INCOMPATIBLE_MAPPING
        quality_flags.add(OperationQualityFlag.INCOMPATIBLE_OPCODE_MAPPING)
    elif not present_results:
        mapping_status = OpcodeMappingStatus.NO_OPCODE_BINS
    elif unmapped_ids:
        mapping_status = OpcodeMappingStatus.PARTIAL_UNKNOWN_OPCODE
    else:
        mapping_status = OpcodeMappingStatus.ALL_MAPPED

    return (
        tuple(opcode_results),
        mapping_status,
        tuple(mapped_ids),
        tuple(unmapped_ids),
        tuple(missing_ids),
        mapped_total if complete and mapping_compatible else None,
        unknown_total if complete and mapping_compatible else None,
        denominator,
        quality_flags,
    )


def _family_mix(
    dma: OperationAggregateResult,
    idc: OperationAggregateResult,
    dma_results: tuple[OperationCounterResult, ...],
    idc_results: tuple[OperationCounterResult, ...],
    window_ns: tuple[int, int],
) -> OperationFamilyMix:
    dma_keys = _aggregate_interval_keys(dma, dma_results)
    idc_keys = _aggregate_interval_keys(idc, idc_results)
    quality_flags: set[OperationQualityFlag] = set()
    quality_flags.update(dma.quality_flags)
    quality_flags.update(idc.quality_flags)
    reasons: list[str] = []

    if dma.total is None or idc.total is None:
        status = _overall_counter_status((*dma_results, *idc_results))
        if dma.total is None or idc.total is None:
            reasons.append("Both DMA and IDC totals are required for a family mix.")
        return OperationFamilyMix(
            status=status,
            quality_flags=frozenset(quality_flags),
            status_reasons=tuple(reasons),
            dma_command_total=dma.total,
            idc_command_total=idc.total,
            total_commands=None,
            dma_fraction=None,
            idc_fraction=None,
            covered_duration_ns=0,
            coverage_fraction=None,
            represented_window_ns=None,
            formula=DMA_IDC_FRACTION_FORMULA,
        )

    if dma_keys != idc_keys or not dma_keys:
        quality_flags.add(OperationQualityFlag.INCOMPATIBLE_COVERAGE)
        reasons.append("DMA and IDC totals do not share complete valid interval coverage.")
        return OperationFamilyMix(
            status=OperationStatus.INSUFFICIENT_DATA,
            quality_flags=frozenset(quality_flags),
            status_reasons=tuple(reasons),
            dma_command_total=dma.total,
            idc_command_total=idc.total,
            total_commands=None,
            dma_fraction=None,
            idc_fraction=None,
            covered_duration_ns=0,
            coverage_fraction=None,
            represented_window_ns=None,
            formula=DMA_IDC_FRACTION_FORMULA,
        )

    total_commands = dma.total + idc.total
    covered_duration_ns = sum(end - start for start, end in dma_keys)
    requested_duration_ns = window_ns[1] - window_ns[0]
    represented_window_ns = (
        min(start for start, _ in dma_keys),
        max(end for _, end in dma_keys),
    )
    if dma.status is OperationStatus.PARTIAL_INVALID_DATA or idc.status is OperationStatus.PARTIAL_INVALID_DATA:
        status = OperationStatus.PARTIAL_INVALID_DATA
    elif total_commands == 0:
        status = OperationStatus.OBSERVED_ZERO
    else:
        status = OperationStatus.OBSERVED_NONZERO

    return OperationFamilyMix(
        status=status,
        quality_flags=frozenset(quality_flags),
        status_reasons=tuple(reasons),
        dma_command_total=dma.total,
        idc_command_total=idc.total,
        total_commands=total_commands,
        dma_fraction=dma.total / total_commands,
        idc_fraction=idc.total / total_commands,
        covered_duration_ns=covered_duration_ns,
        coverage_fraction=covered_duration_ns / requested_duration_ns,
        represented_window_ns=represented_window_ns,
        formula=DMA_IDC_FRACTION_FORMULA,
    )


def analyze_operations(dataset: OperationDataset) -> OperationAnalysis:
    """Analyze command counts without unsupported data-path interpretations."""
    results = tuple(
        _analyze_counter(series, dataset.window_ns) if series is not None else None
        for series in dataset.series
    )
    result_by_spec = {
        series.spec: result
        for series, result in zip(dataset.series, results, strict=True)
    }
    ordered_results = tuple(
        result_by_spec.get(spec, _missing_result(spec, dataset.window_ns))
        for spec in dataset.requested_counters
    )
    dma_bin_results = tuple(
        result
        for result in ordered_results
        if result.spec.counter_kind is OperationCounterKind.DMA_OPCODE_BIN
    )
    dma_total_results = tuple(
        result
        for result in ordered_results
        if result.spec.counter_kind is OperationCounterKind.DMA_TOTAL
    )
    idc_total_results = tuple(
        result
        for result in ordered_results
        if result.spec.counter_kind is OperationCounterKind.IDC_TOTAL
    )

    aggregate_dma = _aggregate_results(
        OperationCounterKind.DMA_TOTAL,
        tuple(spec for spec in dataset.requested_counters if spec.counter_kind is OperationCounterKind.DMA_TOTAL),
        dma_total_results,
        dataset.window_ns,
    )
    aggregate_idc = _aggregate_results(
        OperationCounterKind.IDC_TOTAL,
        tuple(spec for spec in dataset.requested_counters if spec.counter_kind is OperationCounterKind.IDC_TOTAL),
        idc_total_results,
        dataset.window_ns,
    )
    (
        opcode_results,
        mapping_status,
        mapped_ids,
        unmapped_ids,
        missing_ids,
        mapped_opcode_total,
        unknown_opcode_total,
        dma_opcode_total,
        opcode_flags,
    ) = _opcode_results(
        dma_bin_results,
        dataset.opcode_mapping,
        dataset.window_ns,
    )
    family_mix = _family_mix(
        aggregate_dma,
        aggregate_idc,
        dma_total_results,
        idc_total_results,
        dataset.window_ns,
    )

    quality_flags: set[OperationQualityFlag] = set()
    for result in (*dma_bin_results, *dma_total_results, *idc_total_results):
        quality_flags.update(result.quality_flags)
    quality_flags.update(aggregate_dma.quality_flags)
    quality_flags.update(aggregate_idc.quality_flags)
    quality_flags.update(opcode_flags)
    quality_flags.update(family_mix.quality_flags)

    status = _overall_counter_status(ordered_results)
    reasons = tuple(
        reason
        for result in ordered_results
        for reason in result.status_reasons
        if result.status is status
    )

    return OperationAnalysis(
        analysis_version=OPERATION_ANALYSIS_VERSION,
        population=dataset.population,
        status=status,
        quality_flags=frozenset(quality_flags),
        status_reasons=reasons,
        requested_window_ns=dataset.window_ns,
        requested_duration_ns=dataset.window_ns[1] - dataset.window_ns[0],
        dma_opcode_bin_results=dma_bin_results,
        dma_total_results=dma_total_results,
        idc_total_results=idc_total_results,
        aggregate_dma_commands=aggregate_dma,
        aggregate_idc_commands=aggregate_idc,
        opcode_results=opcode_results,
        opcode_mapping_status=mapping_status,
        opcode_mapping_version=(
            dataset.opcode_mapping.mapping_version if dataset.opcode_mapping is not None else None
        ),
        mapped_opcode_counter_ids=mapped_ids,
        unmapped_opcode_counter_ids=unmapped_ids,
        missing_opcode_counter_ids=missing_ids,
        mapped_opcode_total=mapped_opcode_total,
        unknown_opcode_total=unknown_opcode_total,
        dma_opcode_command_total=dma_opcode_total,
        dma_idc_mix=family_mix,
        delta_formula=DELTA_FORMULA,
        rate_formula=RATE_FORMULA,
        opcode_fraction_formula=OPCODE_FRACTION_FORMULA,
        dma_idc_fraction_formula=DMA_IDC_FRACTION_FORMULA,
        limitations=OPERATION_LIMITATIONS,
    )
