"""Indexed, immutable storage for a normalized trace run."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

from trace_model import (
    SCHEMA_VERSION,
    CollectorIdentity,
    CounterMetadata,
    Coverage,
    NodeIdentity,
    NormalizedBatch,
    NicIdentity,
    QualityFlag,
    RunIdentity,
    SeriesQueryResult,
    SeriesResult,
    TraceValidationError,
    ValueKind,
)


class RunIdentityMismatchError(TraceValidationError):
    """Raised when batches from different runs or formats are combined."""


class UnknownSeriesError(TraceValidationError):
    """Raised for an unknown node, NIC, counter, or collector."""


class AmbiguousCollectorError(TraceValidationError):
    """Raised when a unique series request has multiple collector streams."""


@dataclass(frozen=True)
class TraceStore:
    """A flat, indexed columnar trace for exactly one run.

    Series are indexed by ``(node_id, nic_id, collector_id, counter_id)``.  The
    public ``(node, NIC, counter)`` query returns one result per collector so
    duplicate rank observations are never silently averaged or concatenated.
    """

    schema_version: int
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
    _index: Mapping[tuple[int, int, int], Mapping[int, np.ndarray]]

    @classmethod
    def from_batches(cls, batches: Iterable[NormalizedBatch]) -> TraceStore:
        batch_list = tuple(batches)
        if not batch_list:
            raise TraceValidationError("at least one normalized batch is required")

        identity = batch_list[0].identity
        nodes: list[NodeIdentity] = list(batch_list[0].nodes)
        nics: list[NicIdentity] = list(batch_list[0].nics)
        counters: list[CounterMetadata] = list(batch_list[0].counters)
        collectors: list[CollectorIdentity] = list(batch_list[0].collectors)
        coverage = batch_list[0].coverage

        arrays: dict[str, list[np.ndarray]] = {name: [] for name in (
            "node_ids",
            "nic_ids",
            "counter_ids",
            "collector_ids",
            "timestamps_ns",
            "record_ids",
            "uint64_values",
            "float64_values",
            "quality_flags",
        )}
        for batch in batch_list:
            if batch.identity != identity:
                raise RunIdentityMismatchError(
                    "all batches must have identical run, source, and clock identity"
                )
            _merge_dimensions(nodes, batch.nodes, "node", "node_id", "label", True)
            _merge_dimensions(nics, batch.nics, "NIC", "nic_id", "label", False)
            _merge_dimensions(
                counters, batch.counters, "counter", "counter_id", "name", True
            )
            _merge_dimensions(
                collectors, batch.collectors, "collector", "collector_id", "name", True
            )
            coverage = coverage.merged(batch.coverage, ())
            for name in arrays:
                arrays[name].append(getattr(batch, name))

        columns = {
            name: values[0] if len(values) == 1 else np.concatenate(values)
            for name, values in arrays.items()
        }
        row_count = columns["timestamps_ns"].shape[0]
        if row_count == 0:
            raise TraceValidationError("a trace store cannot contain zero rows")

        unique_rows = _remove_demonstrable_duplicates(columns)
        order = np.lexsort(
            (
                unique_rows["record_ids"],
                unique_rows["timestamps_ns"],
                unique_rows["collector_ids"],
                unique_rows["counter_ids"],
                unique_rows["nic_ids"],
                unique_rows["node_ids"],
            )
        )
        _validate_timestamps_and_flag_reordering(unique_rows, order)

        node_registry = {node.node_id: node for node in nodes}
        nic_registry = {nic.nic_id: nic for nic in nics}
        counter_registry = {counter.counter_id: counter for counter in counters}
        collector_registry = {collector.collector_id: collector for collector in collectors}
        observed_counter_names = tuple(
            counter_registry[int(counter_id)].name
            for counter_id in np.unique(unique_rows["counter_ids"])
        )
        coverage = coverage.merged(Coverage(), observed_counter_names)
        _apply_counter_quality(unique_rows, counter_registry)

        for array in unique_rows.values():
            array.setflags(write=False)

        stream_starts = _sorted_group_starts(
            order,
            unique_rows,
            ("node_ids", "nic_ids", "counter_ids", "collector_ids"),
        )
        public_index: dict[tuple[int, int, int], dict[int, np.ndarray]] = {}
        for start, end in zip(stream_starts, (*stream_starts[1:], order.size), strict=True):
            row_indices = np.asarray(order[start:end], dtype=np.int64)
            row_indices.setflags(write=False)
            public_index.setdefault(
                (
                    int(unique_rows["node_ids"][order[start]]),
                    int(unique_rows["nic_ids"][order[start]]),
                    int(unique_rows["counter_ids"][order[start]]),
                )
                , {}
            )[int(unique_rows["collector_ids"][order[start]])] = row_indices
        immutable_index = {
            key: MappingProxyType(streams) for key, streams in public_index.items()
        }
        return cls(
            schema_version=SCHEMA_VERSION,
            identity=identity,
            nodes=tuple(nodes),
            nics=tuple(nics),
            counters=tuple(counters),
            collectors=tuple(collectors),
            coverage=coverage,
            _index=MappingProxyType(immutable_index),
            **unique_rows,
        )

    @property
    def row_count(self) -> int:
        return int(self.timestamps_ns.shape[0])

    def column(self, name: str) -> np.ndarray:
        allowed = {
            "node_ids",
            "nic_ids",
            "counter_ids",
            "collector_ids",
            "timestamps_ns",
            "record_ids",
            "uint64_values",
            "float64_values",
            "quality_flags",
        }
        if name not in allowed:
            raise TraceValidationError(f"unknown trace column {name!r}")
        return getattr(self, name)

    def query_series(
        self,
        node: NodeIdentity | int | str,
        nic: NicIdentity | int | str,
        counter: CounterMetadata | int | str,
        *,
        collector_id: int | None = None,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> SeriesQueryResult:
        node_identity = self._resolve_node(node)
        nic_identity = self._resolve_nic(nic)
        counter_metadata = self._resolve_counter(counter)
        _validate_window(start_ns, end_ns)

        key = (node_identity.node_id, nic_identity.nic_id, counter_metadata.counter_id)
        streams = self._index.get(key)
        if streams is None:
            raise UnknownSeriesError("unknown node, NIC, counter series")
        if collector_id is not None:
            if collector_id not in streams:
                raise UnknownSeriesError(f"unknown collector_id {collector_id}")
            selected_streams = {collector_id: streams[collector_id]}
        else:
            selected_streams = dict(streams)

        results: list[SeriesResult] = []
        for current_collector_id, row_indices in selected_streams.items():
            timestamps = self.timestamps_ns[row_indices]
            boundary = np.ones(row_indices.size, dtype=bool)
            if start_ns is not None:
                boundary &= timestamps >= start_ns
            if end_ns is not None:
                boundary &= timestamps < end_ns
            selected_positions = np.flatnonzero(boundary)
            selected_rows = row_indices[selected_positions]
            values = (
                self.uint64_values[selected_rows]
                if counter_metadata.value_kind is ValueKind.EXACT_UINT64
                else self.float64_values[selected_rows]
            )
            values.setflags(write=False)
            result_timestamps = timestamps[selected_positions]
            result_timestamps.setflags(write=False)
            quality = self.quality_flags[selected_rows]
            quality.setflags(write=False)
            results.append(
                SeriesResult(
                    node=node_identity,
                    nic=nic_identity,
                    counter=counter_metadata,
                    collector=self._collector_registry[current_collector_id],
                    timestamps_ns=result_timestamps,
                    values=values,
                    quality_flags=quality,
                    requested_start_ns=start_ns,
                    requested_end_ns=end_ns,
                    represented_start_ns=(
                        int(result_timestamps[0]) if result_timestamps.size else None
                    ),
                    represented_end_ns=(
                        int(result_timestamps[-1]) if result_timestamps.size else None
                    ),
                    observation_stream_count=len(streams),
                    run_identity=self.identity,
                )
            )
        return SeriesQueryResult(
            results=tuple(results),
            duplicate_observation_streams=len(streams) > 1,
        )

    def query_unique_series(
        self,
        node: NodeIdentity | int | str,
        nic: NicIdentity | int | str,
        counter: CounterMetadata | int | str,
        *,
        collector_id: int | None = None,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ):
        result = self.query_series(
            node,
            nic,
            counter,
            collector_id=collector_id,
            start_ns=start_ns,
            end_ns=end_ns,
        )
        if len(result.results) != 1:
            raise AmbiguousCollectorError(
                "multiple collector streams observe this node, NIC, counter"
            )
        return result.results[0]

    def list_nodes(self) -> tuple[NodeIdentity, ...]:
        return self.nodes

    def list_nics(self) -> tuple[NicIdentity, ...]:
        return self.nics

    def list_counters(self) -> tuple[CounterMetadata, ...]:
        return self.counters

    def list_collectors(self) -> tuple[CollectorIdentity, ...]:
        return self.collectors

    @property
    def _collector_registry(self) -> Mapping[int, CollectorIdentity]:
        return {collector.collector_id: collector for collector in self.collectors}

    def _resolve_node(self, node: NodeIdentity | int | str) -> NodeIdentity:
        if isinstance(node, NodeIdentity):
            if node not in self.nodes:
                raise UnknownSeriesError("node identity does not belong to this run")
            return node
        if isinstance(node, int) and not isinstance(node, bool):
            try:
                return next(identity for identity in self.nodes if identity.node_id == node)
            except StopIteration as error:
                raise UnknownSeriesError(f"unknown node_id {node}") from error
        if isinstance(node, str):
            try:
                return next(identity for identity in self.nodes if identity.label == node)
            except StopIteration as error:
                raise UnknownSeriesError(f"unknown node label {node!r}") from error
        raise TraceValidationError("node must be a NodeIdentity, integer ID, or label")

    def _resolve_nic(self, nic: NicIdentity | int | str) -> NicIdentity:
        if isinstance(nic, NicIdentity):
            if nic not in self.nics:
                raise UnknownSeriesError("NIC identity does not belong to this run")
            return nic
        if isinstance(nic, int) and not isinstance(nic, bool):
            try:
                return next(identity for identity in self.nics if identity.nic_id == nic)
            except StopIteration as error:
                raise UnknownSeriesError(f"unknown nic_id {nic}") from error
        if isinstance(nic, str):
            matches = [identity for identity in self.nics if identity.label == nic]
            if not matches:
                raise UnknownSeriesError(f"unknown NIC label {nic!r}") from error
            if len(matches) > 1:
                raise TraceValidationError(f"NIC label {nic!r} is ambiguous; use nic_id")
            return matches[0]
        raise TraceValidationError("NIC must be a NicIdentity, integer ID, or label")

    def _resolve_counter(self, counter: CounterMetadata | int | str) -> CounterMetadata:
        if isinstance(counter, CounterMetadata):
            if counter not in self.counters:
                raise UnknownSeriesError("counter metadata does not belong to this run")
            return counter
        if isinstance(counter, int) and not isinstance(counter, bool):
            try:
                return next(identity for identity in self.counters if identity.counter_id == counter)
            except StopIteration as error:
                raise UnknownSeriesError(f"unknown counter_id {counter}") from error
        if isinstance(counter, str):
            try:
                return next(identity for identity in self.counters if identity.name == counter)
            except StopIteration as error:
                raise UnknownSeriesError(f"unknown counter name {counter!r}") from error
        raise TraceValidationError("counter must be CounterMetadata, integer ID, or name")


def _merge_dimensions(
    existing: list[object],
    additions: tuple[object, ...],
    label: str,
    id_attribute: str,
    label_attribute: str,
    require_unique_label: bool,
) -> None:
    by_id = {getattr(identity, id_attribute): identity for identity in existing}
    by_label = {getattr(identity, label_attribute): identity for identity in existing}
    for identity in additions:
        identifier = getattr(identity, id_attribute)
        identity_label = getattr(identity, label_attribute)
        if identifier in by_id and by_id[identifier] != identity:
            raise TraceValidationError(f"conflicting {label} metadata for ID {identifier}")
        if (
            require_unique_label
            and identity_label in by_label
            and by_label[identity_label] != identity
        ):
            raise TraceValidationError(f"conflicting {label} metadata for label {identity_label!r}")
        if identifier not in by_id:
            by_id[identifier] = identity
            by_label[identity_label] = identity
            existing.append(identity)


def _group_starts(*columns: np.ndarray) -> np.ndarray:
    row_count = columns[0].shape[0]
    if row_count == 0:
        return np.empty(0, dtype=np.int64)
    changed = np.zeros(row_count, dtype=bool)
    changed[0] = True
    for column in columns:
        changed[1:] |= column[1:] != column[:-1]
    return np.flatnonzero(changed)


def _sorted_group_starts(
    order: np.ndarray,
    columns: Mapping[str, np.ndarray],
    names: tuple[str, ...],
) -> np.ndarray:
    changed = np.zeros(order.size, dtype=bool)
    changed[0] = True
    for name in names:
        sorted_values = columns[name][order]
        changed[1:] |= sorted_values[1:] != sorted_values[:-1]
        del sorted_values
    return np.flatnonzero(changed)


def _remove_demonstrable_duplicates(
    columns: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    row_count = columns["timestamps_ns"].shape[0]
    duplicate_order = np.lexsort(
        (
            columns["record_ids"],
            columns["timestamps_ns"],
            columns["collector_ids"],
            columns["counter_ids"],
            columns["nic_ids"],
            columns["node_ids"],
        )
    )
    group_starts = _sorted_group_starts(
        duplicate_order,
        columns,
        (
            "node_ids",
            "nic_ids",
            "counter_ids",
            "collector_ids",
            "timestamps_ns",
            "record_ids",
        ),
    )
    group_ends = (*group_starts[1:], row_count)
    duplicate_group_starts: list[int] = []

    for start, end in zip(group_starts, group_ends, strict=True):
        if end - start == 1:
            continue
        group_indices = duplicate_order[start:end]
        first_index = group_indices[0]
        float_first = columns["float64_values"][first_index]
        float_group = columns["float64_values"][group_indices]
        floats_equal = np.logical_or(
            np.isnan(float_first) & np.isnan(float_group),
            float_first == float_group,
        )
        if (
            not np.all(
                columns["uint64_values"][group_indices]
                == columns["uint64_values"][first_index]
            )
            or not np.all(floats_equal)
            or not np.all(
                columns["quality_flags"][group_indices]
                == columns["quality_flags"][first_index]
            )
        ):
            raise TraceValidationError(
                "duplicate records contain conflicting values or quality flags"
            )
        duplicate_group_starts.append(start)

    keep = np.sort(duplicate_order[group_starts], kind="stable")
    if np.array_equal(keep, np.arange(row_count, dtype=np.int64)):
        result = dict(columns)
        result["quality_flags"] = columns["quality_flags"].copy()
    else:
        result = {name: values[keep] for name, values in columns.items()}
        duplicate_representatives = duplicate_order[
            np.asarray(duplicate_group_starts, dtype=np.int64)
        ]
        result_positions = np.searchsorted(keep, duplicate_representatives)
        result["quality_flags"][result_positions] |= QualityFlag.DUPLICATE_REMOVED

    record_reuse_order = np.lexsort(
        (
            columns["timestamps_ns"],
            columns["collector_ids"],
            columns["counter_ids"],
            columns["nic_ids"],
            columns["node_ids"],
            columns["record_ids"],
        )
    )
    sorted_record_ids = columns["record_ids"][record_reuse_order]
    record_starts = _group_starts(sorted_record_ids)
    record_ends = (*record_starts[1:], row_count)
    for start, end in zip(record_starts, record_ends, strict=True):
        if end - start == 1:
            continue
        group_indices = record_reuse_order[start:end]
        for name in (
            "node_ids",
            "nic_ids",
            "counter_ids",
            "collector_ids",
            "timestamps_ns",
        ):
            values = columns[name][group_indices]
            if np.any(values[1:] != values[0]):
                raise TraceValidationError(
                    "record_id is reused for different trace records"
                )

    return result


def _validate_timestamps_and_flag_reordering(
    columns: Mapping[str, np.ndarray], order: np.ndarray
) -> None:
    stream_starts = _sorted_group_starts(
        order,
        columns,
        ("node_ids", "nic_ids", "counter_ids", "collector_ids"),
    )
    stream_ends = (*stream_starts[1:], order.size)
    for start, end in zip(stream_starts, stream_ends, strict=True):
        row_indices = order[start:end]
        timestamp_values = columns["timestamps_ns"][row_indices]
        if np.any(timestamp_values[1:] < timestamp_values[:-1]):
            raise TraceValidationError("sorted series contains decreasing timestamps")
        if np.any(np.diff(row_indices) < 0):
            columns["quality_flags"][row_indices] |= (
                QualityFlag.REORDERED | QualityFlag.NONMONOTONIC_TIMESTAMP
            )
        if np.any(timestamp_values[1:] == timestamp_values[:-1]):
            raise TraceValidationError("a collector stream has duplicate timestamps")


def _apply_counter_quality(
    columns: Mapping[str, np.ndarray],
    counter_registry: Mapping[int, CounterMetadata],
) -> None:
    quality_flags = columns["quality_flags"]
    counter_ids = columns["counter_ids"]
    for counter in counter_registry.values():
        selected_rows = counter_ids == counter.counter_id
        quality_flags[selected_rows] |= counter.quality_flags
        if counter.baseline_relative:
            quality_flags[selected_rows] |= QualityFlag.BASELINE_RELATIVE
        if counter.value_kind is ValueKind.FLOAT64:
            quality_flags[selected_rows] |= QualityFlag.PRECISION_LOSS_POSSIBLE


def _validate_window(start_ns: int | None, end_ns: int | None) -> None:
    if start_ns is not None and (isinstance(start_ns, bool) or not isinstance(start_ns, (int, np.integer))):
        raise TraceValidationError("start_ns must be an integer")
    if end_ns is not None and (isinstance(end_ns, bool) or not isinstance(end_ns, (int, np.integer))):
        raise TraceValidationError("end_ns must be an integer")
    if start_ns is not None and end_ns is not None and end_ns < start_ns:
        raise TraceValidationError("end_ns cannot precede start_ns")
