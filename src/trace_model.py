"""Immutable, columnar normalized trace types.

The model deliberately keeps exact unsigned values in a separate NumPy column
from converted floating-point values.  A row's counter metadata determines
which value column is meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np


SCHEMA_VERSION = 1


class TraceValidationError(ValueError):
    """Raised when a normalized batch violates the trace contract."""


class ValueKind(str, Enum):
    EXACT_UINT64 = "exact_uint64"
    FLOAT64 = "float64"


class CounterSemantics(str, Enum):
    CUMULATIVE = "cumulative"
    DELTA = "delta"
    GAUGE = "gauge"
    UNKNOWN = "unknown"


class QualityFlag(int):
    NONE = 0
    REORDERED = 1 << 0
    NONMONOTONIC_TIMESTAMP = 1 << 1
    DUPLICATE_REMOVED = 1 << 2
    BASELINE_RELATIVE = 1 << 3
    PRECISION_LOSS_POSSIBLE = 1 << 4
    RESET_SUSPECT = 1 << 5
    WRAP_SUSPECT = 1 << 6
    GAP = 1 << 7
    UNKNOWN_METADATA = 1 << 8


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    source_fingerprint: str
    source_format: str
    reader_version: str
    conversion_lineage: str
    clock_domain: str
    synchronization_evidence: str
    time_origin_ns: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "source_fingerprint",
            "source_format",
            "reader_version",
            "conversion_lineage",
            "clock_domain",
            "synchronization_evidence",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise TraceValidationError(f"{name} must be a nonempty string")
        if self.time_origin_ns is not None and not isinstance(self.time_origin_ns, int):
            raise TraceValidationError("time_origin_ns must be an integer or None")


@dataclass(frozen=True)
class NodeIdentity:
    node_id: int
    label: str
    original_label: str
    stable_device_id: str | None = None

    def __post_init__(self) -> None:
        _validate_id(self.node_id, "node_id")
        _validate_nonempty_string(self.label, "node label")
        _validate_nonempty_string(self.original_label, "node original_label")


@dataclass(frozen=True)
class NicIdentity:
    nic_id: int
    node_id: int
    label: str
    original_label: str
    stable_device_id: str | None = None

    def __post_init__(self) -> None:
        _validate_id(self.nic_id, "nic_id")
        _validate_id(self.node_id, "nic node_id")
        _validate_nonempty_string(self.label, "NIC label")
        _validate_nonempty_string(self.original_label, "NIC original_label")


@dataclass(frozen=True)
class CollectorIdentity:
    collector_id: int
    name: str
    rank: int | None = None
    process: str | None = None
    collector_type: str | None = None

    def __post_init__(self) -> None:
        _validate_id(self.collector_id, "collector_id")
        _validate_nonempty_string(self.name, "collector name")
        if self.rank is not None:
            _validate_nonnegative_integer(self.rank, "collector rank")


@dataclass(frozen=True)
class CounterMetadata:
    counter_id: int
    name: str
    value_kind: ValueKind
    semantics: CounterSemantics
    unit: str
    baseline_relative: bool
    registry_version: str
    quality_flags: int = QualityFlag.NONE

    def __post_init__(self) -> None:
        _validate_id(self.counter_id, "counter_id")
        _validate_nonempty_string(self.name, "counter name")
        _validate_nonempty_string(self.unit, "counter unit")
        _validate_nonempty_string(self.registry_version, "counter registry_version")
        _validate_nonnegative_integer(self.quality_flags, "counter quality_flags")
        if self.baseline_relative and self.value_kind is not ValueKind.FLOAT64:
            raise TraceValidationError(
                "baseline_relative currently requires FLOAT64 values"
            )


@dataclass(frozen=True)
class Coverage:
    requested_counter_names: tuple[str, ...] = ()
    available_counter_names: tuple[str, ...] = ()
    readable_counter_names: tuple[str, ...] = ()
    observed_counter_names: tuple[str, ...] = ()
    unreadable_counter_names: tuple[str, ...] = ()
    acquisition_start_ns: int | None = None
    acquisition_end_ns: int | None = None
    lost_record_count: int = 0
    gap_count: int = 0
    notes: tuple[str, ...] = ()

    @property
    def missing_counter_names(self) -> tuple[str, ...]:
        observed = set(self.observed_counter_names)
        return tuple(name for name in self.requested_counter_names if name not in observed)

    def __post_init__(self) -> None:
        for names in (
            self.requested_counter_names,
            self.available_counter_names,
            self.readable_counter_names,
            self.observed_counter_names,
            self.unreadable_counter_names,
        ):
            _validate_unique_names(names)
        readable = set(self.readable_counter_names)
        available = set(self.available_counter_names)
        observed = set(self.observed_counter_names)
        unreadable = set(self.unreadable_counter_names)
        if not readable.issubset(available):
            raise TraceValidationError("readable counters must be a subset of available counters")
        if not observed.issubset(readable):
            raise TraceValidationError("observed counters must be a subset of readable counters")
        if readable & unreadable:
            raise TraceValidationError("counters cannot be both readable and unreadable")
        _validate_nonnegative_integer(self.lost_record_count, "lost_record_count")
        _validate_nonnegative_integer(self.gap_count, "gap_count")
        if (
            self.acquisition_start_ns is not None
            and self.acquisition_end_ns is not None
            and self.acquisition_end_ns < self.acquisition_start_ns
        ):
            raise TraceValidationError("acquisition_end_ns precedes acquisition_start_ns")

    def merged(self, other: Coverage, observed_counter_names: tuple[str, ...]) -> Coverage:
        def merge_names(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(dict.fromkeys((*left, *right)))

        starts = [value for value in (self.acquisition_start_ns, other.acquisition_start_ns) if value is not None]
        ends = [value for value in (self.acquisition_end_ns, other.acquisition_end_ns) if value is not None]
        return Coverage(
            requested_counter_names=merge_names(
                self.requested_counter_names, other.requested_counter_names
            ),
            available_counter_names=merge_names(
                self.available_counter_names, other.available_counter_names
            ),
            readable_counter_names=merge_names(
                self.readable_counter_names, other.readable_counter_names
            ),
            observed_counter_names=observed_counter_names,
            unreadable_counter_names=merge_names(
                self.unreadable_counter_names, other.unreadable_counter_names
            ),
            acquisition_start_ns=min(starts) if starts else None,
            acquisition_end_ns=max(ends) if ends else None,
            lost_record_count=self.lost_record_count + other.lost_record_count,
            gap_count=self.gap_count + other.gap_count,
            notes=tuple(dict.fromkeys((*self.notes, *other.notes))),
        )


@dataclass(frozen=True)
class NormalizedBatch:
    """One immutable, validated columnar batch belonging to a single run."""

    identity: RunIdentity
    nodes: tuple[NodeIdentity, ...]
    nics: tuple[NicIdentity, ...]
    counters: tuple[CounterMetadata, ...]
    collectors: tuple[CollectorIdentity, ...]
    coverage: Coverage
    node_ids: np.ndarray
    nic_ids: np.ndarray
    counter_ids: np.ndarray
    collector_ids: np.ndarray
    timestamps_ns: np.ndarray
    record_ids: np.ndarray
    uint64_values: np.ndarray
    float64_values: np.ndarray
    quality_flags: np.ndarray

    def __post_init__(self) -> None:
        node_registry = _identity_registry(self.nodes, "node", "node_id", "label", True)
        nic_registry = _identity_registry(self.nics, "NIC", "nic_id", "label", False)
        counter_registry = _identity_registry(
            self.counters, "counter", "counter_id", "name", True
        )
        collector_registry = _identity_registry(
            self.collectors, "collector", "collector_id", "name", True
        )
        if not node_registry or not nic_registry or not counter_registry or not collector_registry:
            raise TraceValidationError("dimension registries cannot be empty")

        for nic in self.nics:
            if nic.node_id not in node_registry:
                raise TraceValidationError(f"NIC {nic.label} references unknown node_id {nic.node_id}")

        row_count = _validate_row_columns(
            node_ids=self.node_ids,
            nic_ids=self.nic_ids,
            counter_ids=self.counter_ids,
            collector_ids=self.collector_ids,
            timestamps_ns=self.timestamps_ns,
            record_ids=self.record_ids,
            uint64_values=self.uint64_values,
            float64_values=self.float64_values,
            quality_flags=self.quality_flags,
            node_registry=node_registry,
            nic_registry=nic_registry,
            counter_registry=counter_registry,
            collector_registry=collector_registry,
        )
        if row_count == 0:
            raise TraceValidationError("a normalized batch cannot contain zero rows")

        for row_index in range(row_count):
            counter = counter_registry[int(self.counter_ids[row_index])]
            if counter.value_kind is ValueKind.EXACT_UINT64:
                value = int(self.uint64_values[row_index])
                if value < 0 or value > np.iinfo(np.uint64).max:
                    raise TraceValidationError("exact counter value is outside uint64 range")
                if not np.isnan(self.float64_values[row_index]):
                    raise TraceValidationError(
                        "EXACT_UINT64 rows must leave float64_values as NaN"
                    )
            else:
                value = float(self.float64_values[row_index])
                if not np.isfinite(value):
                    raise TraceValidationError("float counter value must be finite")
                if self.uint64_values[row_index] != 0:
                    raise TraceValidationError("FLOAT64 rows must leave uint64_values as zero")

        for array_name in (
            "node_ids",
            "nic_ids",
            "counter_ids",
            "collector_ids",
            "timestamps_ns",
            "record_ids",
            "uint64_values",
            "float64_values",
            "quality_flags",
        ):
            array = np.array(getattr(self, array_name), copy=True)
            array.setflags(write=False)
            object.__setattr__(self, array_name, array)
            array.setflags(write=False)


@dataclass(frozen=True)
class SeriesResult:
    node: NodeIdentity
    nic: NicIdentity
    counter: CounterMetadata
    collector: CollectorIdentity
    timestamps_ns: np.ndarray
    values: np.ndarray
    quality_flags: np.ndarray
    requested_start_ns: int | None
    requested_end_ns: int | None
    represented_start_ns: int | None
    represented_end_ns: int | None
    observation_stream_count: int
    run_identity: RunIdentity


@dataclass(frozen=True)
class SeriesQueryResult:
    results: tuple[SeriesResult, ...]
    duplicate_observation_streams: bool
    boundary_policy: str = "[start, end)"

    def unique(self) -> SeriesResult:
        if len(self.results) != 1:
            raise TraceValidationError(
                "series has multiple observation streams; select a collector_id"
            )
        return self.results[0]


def _identity_registry(
    identities: tuple[object, ...],
    label: str,
    id_attribute: str,
    label_attribute: str,
    require_unique_label: bool,
) -> Mapping[int, object]:
    registry: dict[int, object] = {}
    labels: set[str] = set()
    for identity in identities:
        identifier = getattr(identity, id_attribute)
        label_value = getattr(identity, label_attribute)
        if require_unique_label and label_value in labels:
            raise TraceValidationError(f"duplicate {label} label")
        if require_unique_label:
            labels.add(label_value)
        if identifier in registry:
            raise TraceValidationError(f"duplicate {label} id")
        registry[identifier] = identity
    return registry


def _validate_id(value: int, name: str) -> None:
    _validate_nonnegative_integer(value, name)


def _validate_nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TraceValidationError(f"{name} must be an integer")
    if value < 0:
        raise TraceValidationError(f"{name} cannot be negative")


def _validate_nonempty_string(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise TraceValidationError(f"{name} must be a nonempty string")


def _validate_unique_names(names: tuple[str, ...]) -> None:
    if len(names) != len(set(names)):
        raise TraceValidationError("coverage counter names must be unique")
    for name in names:
        _validate_nonempty_string(name, "coverage counter name")


def _validate_row_columns(
    *,
    node_ids: np.ndarray,
    nic_ids: np.ndarray,
    counter_ids: np.ndarray,
    collector_ids: np.ndarray,
    timestamps_ns: np.ndarray,
    record_ids: np.ndarray,
    uint64_values: np.ndarray,
    float64_values: np.ndarray,
    quality_flags: np.ndarray,
    node_registry: Mapping[int, NodeIdentity],
    nic_registry: Mapping[int, NicIdentity],
    counter_registry: Mapping[int, CounterMetadata],
    collector_registry: Mapping[int, CollectorIdentity],
) -> int:
    arrays = (
        node_ids,
        nic_ids,
        counter_ids,
        collector_ids,
        timestamps_ns,
        record_ids,
        uint64_values,
        float64_values,
        quality_flags,
    )
    row_count = arrays[0].shape[0] if arrays[0].ndim else 0
    if any(array.ndim != 1 for array in arrays):
        raise TraceValidationError("all normalized columns must be one-dimensional")
    if any(array.shape[0] != row_count for array in arrays):
        raise TraceValidationError("all normalized columns must have equal row counts")

    for array, name, dtype in (
        (node_ids, "node_ids", np.int64),
        (nic_ids, "nic_ids", np.int64),
        (counter_ids, "counter_ids", np.int64),
        (collector_ids, "collector_ids", np.int64),
        (timestamps_ns, "timestamps_ns", np.int64),
        (record_ids, "record_ids", None),
        (uint64_values, "uint64_values", np.uint64),
        (float64_values, "float64_values", np.float64),
        (quality_flags, "quality_flags", np.int64),
    ):
        if not isinstance(array, np.ndarray):
            raise TraceValidationError(f"{name} must be a NumPy array")
        if dtype is not None and array.dtype != dtype:
            raise TraceValidationError(f"{name} must have dtype {np.dtype(dtype).name}")
        if name in ("node_ids", "nic_ids", "counter_ids", "collector_ids", "quality_flags"):
            if np.any(array < 0):
                raise TraceValidationError(f"{name} cannot contain negative IDs or flags")

        if row_count:
            for column, registry, label in (
                (node_ids, node_registry, "node"),
                (nic_ids, nic_registry, "NIC"),
                (counter_ids, counter_registry, "counter"),
                (collector_ids, collector_registry, "collector"),
            ):
                for identifier in np.unique(column):
                    if int(identifier) not in registry:
                        raise TraceValidationError(f"{label} column contains unknown ID {int(identifier)}")
            if np.any(record_ids == ""):
                raise TraceValidationError("record_ids cannot be empty")
            if record_ids.dtype.kind not in ("U", "S") and not all(
                isinstance(record_id, str) for record_id in record_ids.tolist()
            ):
                raise TraceValidationError("record_ids must contain strings")

    return row_count
