"""Strict structured adapter for Perfetto rows extracted from pftrace.

The public row schema is intentionally small.  A caller must first extract
structured track and counter rows with a maintained Perfetto reader.  This
module does not parse binary pftrace, Chrome JSON, or formatted text.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from cassini_metrics import counter_definition, REGISTRY_VERSION

from trace_model import (
    CollectorIdentity,
    CounterMetadata,
    CounterSemantics,
    Coverage,
    NicIdentity,
    NodeIdentity,
    NormalizedBatch,
    QualityFlag,
    RunIdentity,
    TraceValidationError,
    ValueKind,
)


PFTRACE_SOURCE_FORMAT = "perfetto-structured-pftrace"
PFTRACE_CONVERSION_LINEAGE = "baseline-relative change-only double conversion"
_TOP_LEVEL_FIELDS = frozenset({"tracks", "counter_rows"})
_COMMON_TRACK_FIELDS = frozenset({"id", "parent_id", "name", "type"})
_COUNTER_TRACK_FIELDS = _COMMON_TRACK_FIELDS | {"collector"}
_COLLECTOR_FIELDS = frozenset({"name", "rank", "process"})
_COUNTER_ROW_FIELDS = frozenset({"ts", "track_id", "value"})
_TRACK_TYPES = frozenset({"node", "nic", "counter", "other"})
_INT64_MIN = np.iinfo(np.int64).min
_INT64_MAX = np.iinfo(np.int64).max
_SAFE_INTEGER_LIMIT = 2**53


class PftraceAdapterError(TraceValidationError):
    """Raised when pftrace rows or adapter configuration violate the schema."""


class PerfettoTraceProcessorUnavailableError(PftraceAdapterError, ImportError):
    """Raised when the optional Perfetto Trace Processor package is missing."""


@dataclass(frozen=True)
class PftraceAdapterConfig:
    run_id: str
    source_fingerprint: str
    reader_version: str
    clock_domain: str
    synchronization_evidence: str
    requested_counter_names: tuple[str, ...] = ()
    time_origin_ns: int | None = None
    group_collectors_by_source: bool = False
    source_format: str = PFTRACE_SOURCE_FORMAT
    conversion_lineage: str = PFTRACE_CONVERSION_LINEAGE

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "source_fingerprint",
            "reader_version",
            "clock_domain",
            "synchronization_evidence",
            "source_format",
            "conversion_lineage",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise PftraceAdapterError(f"{name} must be a nonempty string")
        if "pftrace" not in self.source_format.lower():
            raise PftraceAdapterError("source_format must identify pftrace")
        if "perfetto" not in self.source_format.lower():
            raise PftraceAdapterError(
                "source_format must identify its Perfetto reader"
            )
        lineage = self.conversion_lineage.lower()
        if (
            "raw ctf" in lineage
            or "interval delta" in lineage
            or "independent delta" in lineage
        ):
            raise PftraceAdapterError(
                "pftrace rows cannot be declared raw CTF or independent interval deltas"
            )
        if not isinstance(self.requested_counter_names, tuple):
            raise PftraceAdapterError("requested_counter_names must be a tuple")
        if any(
            not isinstance(name, str) or not name
            for name in self.requested_counter_names
        ):
            raise PftraceAdapterError("requested counter names must be nonempty strings")
        if len(self.requested_counter_names) != len(set(self.requested_counter_names)):
            raise PftraceAdapterError("requested counter names must be unique")
        if self.time_origin_ns is not None and (
            isinstance(self.time_origin_ns, bool)
            or not isinstance(self.time_origin_ns, int)
            or not _INT64_MIN <= self.time_origin_ns <= _INT64_MAX
        ):
            raise PftraceAdapterError("time_origin_ns must be an int64 integer or None")


@dataclass(frozen=True)
class _Track:
    track_id: int
    parent_id: int | None
    name: str
    track_type: str
    collector: tuple[str, int | None, str | None] | None = None


@dataclass
class _ParsedRow:
    source_index: int
    timestamp_ns: int
    value: float
    precision_loss: bool
    node_track: _Track
    nic_track: _Track
    counter_track: _Track
    collector: tuple[str, int | None, str | None]
    duplicate: bool = False
    duplicate_source_indices: tuple[int, ...] = ()
    reordered: bool = False

    @property
    def series_key(self) -> tuple[Any, ...]:
        return (
            self.node_track.track_id,
            self.nic_track.track_id,
            self.counter_track.track_id,
            self.collector,
        )


def pftrace_rows_to_normalized_batch(
    rows: Mapping[str, Any], config: PftraceAdapterConfig
) -> NormalizedBatch:
    """Normalize strict, structured track and counter rows from pftrace."""

    if isinstance(rows, (str, bytes)):
        raise PftraceAdapterError(
            "pftrace input must be structured mappings, not a serialized trace or JSON"
        )
    if not isinstance(rows, Mapping):
        raise PftraceAdapterError("pftrace rows must be a structured mapping")
    if "traceEvents" in rows:
        raise PftraceAdapterError(
            "Chrome JSON is not binary pftrace; use a maintained Perfetto reader"
        )

    fields = set(rows.keys())
    missing_fields = _TOP_LEVEL_FIELDS - fields
    unknown_fields = fields - _TOP_LEVEL_FIELDS
    if missing_fields or unknown_fields:
        details = []
        if missing_fields:
            details.append(f"missing fields: {sorted(missing_fields)}")
        if unknown_fields:
            details.append(f"undocumented fields: {sorted(unknown_fields)}")
        raise PftraceAdapterError(
            f"pftrace rows violate schema ({'; '.join(details)})"
        )

    tracks = _parse_tracks(rows["tracks"])
    parsed_rows = tuple(
        _parse_counter_row(row, index, tracks)
        for index, row in enumerate(rows["counter_rows"])
    )
    if not parsed_rows:
        raise PftraceAdapterError("at least one pftrace counter row is required")
    _reject_duplicate_series_tracks(parsed_rows)
    return _build_batch(_remove_exact_duplicates(parsed_rows), config)


def adapt_perfetto_trace_processor_rows(
    rows: Mapping[str, Any], config: PftraceAdapterConfig
) -> NormalizedBatch:
    """Adapt rows already extracted with Perfetto Trace Processor.

    This entry point deliberately does not install or invoke an arbitrary
    parser.  It verifies that the optional Trace Processor package is available
    and then applies the same strict row contract.
    """

    try:
        import perfetto.trace_processor
    except ImportError as error:
        raise PerfettoTraceProcessorUnavailableError(
            "The optional Perfetto Trace Processor Python package is required "
            "for adapt_perfetto_trace_processor_rows. Install perfetto or pass "
            "already-extracted rows to pftrace_rows_to_normalized_batch."
        ) from error
    return pftrace_rows_to_normalized_batch(rows, config)


def extract_perfetto_trace_processor_rows(
    source: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Extract strict pftrace rows with Perfetto Trace Processor.

    ``source`` may be a ``perfetto.trace_processor.TraceProcessor`` instance or
    a path to a binary pftrace.  Only the standard ``track``,
    ``counter_track``, and ``counter`` tables are queried.  Each observed
    counter's root ancestor becomes the node, its direct parent becomes the
    NIC, and the counter track itself retains the collector identity.  All
    other real ancestors are retained as ``other`` tracks.
    """

    trace_processor, close_trace_processor = _get_trace_processor(source)
    try:
        track_rows = _query_trace_processor(
            trace_processor,
            "SELECT id, parent_id, name, type AS source_type FROM track ORDER BY id",
            ("id", "parent_id", "name", "source_type"),
        )
        counter_track_rows = _query_trace_processor(
            trace_processor,
            "SELECT id FROM counter_track ORDER BY id",
            ("id",),
        )
        counter_rows = _query_trace_processor(
            trace_processor,
            "SELECT ts, track_id, value FROM counter ORDER BY ts, track_id",
            ("ts", "track_id", "value"),
        )
        counter_track_ids = {
            _parse_nonnegative_integer(row["id"], "counter_track id")
            for row in counter_track_rows
        }
        if len(counter_track_ids) != len(counter_track_rows):
            raise PftraceAdapterError("counter_track returned duplicate track IDs")

        tracks_by_id = _build_extracted_tracks(
            track_rows, counter_track_ids, counter_rows
        )
        return {
            "tracks": [
                {
                    "id": track.track_id,
                    "parent_id": track.parent_id,
                    "name": track.name,
                    "type": track.track_type,
                    **(
                        {"collector": _collector_from_track_id(track.track_id)}
                        if track.track_type == "counter"
                        else {}
                    ),
                }
                for track in tracks_by_id.values()
            ],
            "counter_rows": counter_rows,
        }
    finally:
        if close_trace_processor:
            trace_processor.close()


def adapt_perfetto_trace_processor(
    source: Any, config: PftraceAdapterConfig
) -> NormalizedBatch:
    """Extract and normalize a pftrace through Trace Processor."""

    return pftrace_rows_to_normalized_batch(
        extract_perfetto_trace_processor_rows(source), config
    )


def _get_trace_processor(source: Any) -> tuple[Any, bool]:
    if hasattr(source, "query") and callable(source.query):
        return source, False
    if isinstance(source, (str, os.PathLike)):
        try:
            import perfetto.trace_processor
        except ImportError as error:
            raise PerfettoTraceProcessorUnavailableError(
                "The optional Perfetto Trace Processor Python package is required "
                "for pftrace path extraction. Install perfetto or pass a "
                "TraceProcessor instance."
            ) from error
        path = Path(source)
        if not path.is_file():
            raise PftraceAdapterError(
                f"pftrace path must be an existing file: {path}"
            )
        try:
            return perfetto.trace_processor.TraceProcessor(trace=str(path)), True
        except Exception as error:
            raise PftraceAdapterError(
                f"Perfetto Trace Processor failed to open {path}: {error}"
            ) from error
    raise PftraceAdapterError(
        "source must be a TraceProcessor instance or pftrace path"
    )


def _query_trace_processor(
    trace_processor: Any, sql: str, expected_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    try:
        result = trace_processor.query(sql)
        rows = list(result)
    except Exception as error:
        raise PftraceAdapterError(
            f"Perfetto Trace Processor query failed: {error}"
        ) from error

    column_names = tuple(getattr(result, "column_names", ()))
    if column_names != expected_fields:
        raise PftraceAdapterError(
            "Perfetto Trace Processor returned unexpected query columns: "
            f"{column_names!r}"
        )
    extracted_rows: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            extracted_rows.append({name: row[name] for name in expected_fields})
        else:
            try:
                extracted_rows.append(
                    {name: getattr(row, name) for name in expected_fields}
                )
            except AttributeError as error:
                raise PftraceAdapterError(
                    "Perfetto Trace Processor row lacks an expected column"
                ) from error
    return extracted_rows


def _build_extracted_tracks(
    track_rows: list[dict[str, Any]],
    counter_track_ids: set[int],
    counter_rows: list[dict[str, Any]],
) -> dict[int, _Track]:
    source_tracks: dict[int, _Track] = {}
    for row in track_rows:
        track_id = _parse_nonnegative_integer(row["id"], "track id")
        parent_id = row["parent_id"]
        if parent_id is not None:
            parent_id = _parse_nonnegative_integer(
                parent_id, f"track {track_id} parent_id"
            )
        name = row["name"]
        if not isinstance(name, str) or not name:
            raise PftraceAdapterError(f"track {track_id} name must be nonempty")
        if track_id in source_tracks:
            raise PftraceAdapterError(f"duplicate track id {track_id}")
        source_tracks[track_id] = _Track(
            track_id=track_id,
            parent_id=parent_id,
            name=name,
            track_type="other",
        )

    missing_counter_tracks = counter_track_ids - set(source_tracks)
    if missing_counter_tracks:
        raise PftraceAdapterError(
            "counter_track rows missing from track table: "
            f"{sorted(missing_counter_tracks)}"
        )

    tracks = {
        track_id: track
        for track_id, track in source_tracks.items()
        if track_id not in counter_track_ids
    }
    unknown_parents = {
        track.parent_id
        for track in source_tracks.values()
        if track.parent_id is not None and track.parent_id not in source_tracks
    }
    if unknown_parents:
        raise PftraceAdapterError(
            f"track table references unknown parents: {sorted(unknown_parents)}"
        )

    node_ids: set[int] = set()
    nic_ids: set[int] = set()
    for row in counter_rows:
        track_id = _parse_nonnegative_integer(row["track_id"], "counter track_id")
        if track_id not in counter_track_ids:
            raise PftraceAdapterError(
                f"counter row references non-counter_track track {track_id}"
            )
        parent_id = source_tracks[track_id].parent_id
        if parent_id is None or parent_id in counter_track_ids:
            raise PftraceAdapterError(
                f"counter track {track_id} lacks a non-counter parent track"
            )
        chain = _ancestor_ids(parent_id, source_tracks)
        if not chain:
            raise PftraceAdapterError(
                f"counter track {track_id} parent {parent_id} is missing"
            )
        node_id = chain[-1]
        if node_id == parent_id:
            raise PftraceAdapterError(
                f"counter track {track_id} lacks distinct node and NIC ancestors"
            )
        node_ids.add(node_id)
        nic_ids.add(parent_id)

    for track_id in counter_track_ids:
        source_track = source_tracks[track_id]
        tracks[track_id] = _Track(
            track_id=track_id,
            parent_id=source_track.parent_id,
            name=source_track.name,
            track_type="counter",
            collector=_collector_from_track_id(track_id),
        )
    for track_id in node_ids:
        tracks[track_id] = _replace_track_type(tracks[track_id], "node")
    for track_id in nic_ids:
        tracks[track_id] = _replace_track_type(tracks[track_id], "nic")

    if node_ids & nic_ids:
        raise PftraceAdapterError(
            "the same track cannot represent both node and NIC identity"
        )
    return tracks


def _ancestor_ids(track_id: int, tracks: Mapping[int, _Track]) -> list[int]:
    chain: list[int] = []
    visited: set[int] = set()
    current_id: int | None = track_id
    while current_id is not None:
        if current_id in visited:
            raise PftraceAdapterError(
                f"track ancestry contains cycle at {current_id}"
            )
        visited.add(current_id)
        chain.append(current_id)
        current_id = tracks[current_id].parent_id
    return chain


def _collector_from_track_id(track_id: int) -> dict[str, int | str | None]:
    return {
        "name": f"perfetto-track:{track_id}",
        "rank": None,
        "process": None,
    }


def _replace_track_type(track: _Track, track_type: str) -> _Track:
    return _Track(
        track_id=track.track_id,
        parent_id=track.parent_id,
        name=track.name,
        track_type=track_type,
        collector=track.collector,
    )


def _parse_tracks(value: Iterable[Any]) -> dict[int, _Track]:
    if isinstance(value, (str, bytes)):
        raise PftraceAdapterError("tracks must be an iterable of structured mappings")

    tracks: dict[int, _Track] = {}
    for index, track in enumerate(value):
        if not isinstance(track, Mapping):
            raise PftraceAdapterError(f"track {index} must be a structured mapping")

        expected_fields = (
            _COUNTER_TRACK_FIELDS
            if track.get("type") == "counter"
            else _COMMON_TRACK_FIELDS
        )
        fields = set(track.keys())
        missing_fields = expected_fields - fields
        unknown_fields = fields - expected_fields
        if missing_fields or unknown_fields:
            details = []
            if missing_fields:
                details.append(f"missing fields: {sorted(missing_fields)}")
            if unknown_fields:
                details.append(f"undocumented fields: {sorted(unknown_fields)}")
            raise PftraceAdapterError(
                f"track {index} violates schema ({'; '.join(details)})"
            )

        track_id = _parse_nonnegative_integer(track["id"], f"track {index} id")
        if track_id in tracks:
            raise PftraceAdapterError(f"duplicate track id {track_id}")
        parent_id = track["parent_id"]
        if parent_id is not None:
            parent_id = _parse_nonnegative_integer(
                parent_id, f"track {index} parent_id"
            )
            if parent_id == track_id:
                raise PftraceAdapterError(f"track {track_id} cannot be its own parent")
        if not isinstance(track["name"], str) or not track["name"]:
            raise PftraceAdapterError(f"track {track_id} name must be nonempty")
        if track["type"] not in _TRACK_TYPES:
            raise PftraceAdapterError(
                f"track {track_id} has unsupported type {track['type']!r}"
            )
        if track["type"] == "node" and parent_id is not None:
            raise PftraceAdapterError(f"node track {track_id} must be a root track")

        collector = None
        if track["type"] == "counter":
            collector = _parse_collector(track["collector"], track_id)

        tracks[track_id] = _Track(
            track_id=track_id,
            parent_id=parent_id,
            name=track["name"],
            track_type=track["type"],
            collector=collector,
        )

    for track in tracks.values():
        if track.parent_id is not None and track.parent_id not in tracks:
            raise PftraceAdapterError(
                f"track {track.track_id} references unknown parent {track.parent_id}"
            )
        _ensure_track_acyclic(track, tracks)
    node_names = [track.name for track in tracks.values() if track.track_type == "node"]
    if len(node_names) != len(set(node_names)):
        raise PftraceAdapterError("node track names must be unique")
    if not tracks:
        raise PftraceAdapterError("at least one track descriptor is required")
    return tracks


def _parse_counter_row(
    row: Any, source_index: int, tracks: Mapping[int, _Track]
) -> _ParsedRow:
    if not isinstance(row, Mapping):
        raise PftraceAdapterError(
            f"counter row {source_index} must be a structured mapping"
        )
    fields = set(row.keys())
    missing_fields = _COUNTER_ROW_FIELDS - fields
    unknown_fields = fields - _COUNTER_ROW_FIELDS
    if missing_fields or unknown_fields:
        details = []
        if missing_fields:
            details.append(f"missing fields: {sorted(missing_fields)}")
        if unknown_fields:
            details.append(f"undocumented fields: {sorted(unknown_fields)}")
        raise PftraceAdapterError(
            f"counter row {source_index} violates schema ({'; '.join(details)})"
        )

    timestamp = row["ts"]
    if isinstance(timestamp, bool) or not isinstance(timestamp, int):
        raise PftraceAdapterError(
            f"counter row {source_index} ts must be an integer"
        )
    if not _INT64_MIN <= timestamp <= _INT64_MAX:
        raise PftraceAdapterError(
            f"counter row {source_index} ts is outside int64 range"
        )

    track_id = _parse_nonnegative_integer(
        row["track_id"], f"counter row {source_index} track_id"
    )
    track = tracks.get(track_id)
    if track is None:
        raise PftraceAdapterError(
            f"counter row {source_index} references unknown track {track_id}"
        )
    if track.track_type != "counter" or track.collector is None:
        raise PftraceAdapterError(
            f"counter row {source_index} references non-counter track {track_id}"
        )

    value = row["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PftraceAdapterError(
            f"counter row {source_index} value must be a number"
        )
    if not math.isfinite(value) or value < 0:
        raise PftraceAdapterError(
            f"counter row {source_index} value must be finite and nonnegative"
        )
    converted_value = float(value)
    if not math.isfinite(converted_value):
        raise PftraceAdapterError(
            f"counter row {source_index} value is outside FLOAT64 range"
        )

    ancestors = _resolve_counter_ancestors(track, tracks)
    return _ParsedRow(
        source_index=source_index,
        timestamp_ns=timestamp,
        value=converted_value,
        precision_loss=abs(value) > _SAFE_INTEGER_LIMIT,
        node_track=ancestors["node"],
        nic_track=ancestors["nic"],
        counter_track=track,
        collector=track.collector,
    )


def _parse_collector(
    value: Any, track_id: int
) -> tuple[str, int | None, str | None]:
    if not isinstance(value, Mapping):
        raise PftraceAdapterError(
            f"counter track {track_id} collector must be a structured mapping"
        )
    fields = set(value.keys())
    missing_fields = _COLLECTOR_FIELDS - fields
    unknown_fields = fields - _COLLECTOR_FIELDS
    if missing_fields or unknown_fields:
        details = []
        if missing_fields:
            details.append(f"missing fields: {sorted(missing_fields)}")
        if unknown_fields:
            details.append(f"undocumented fields: {sorted(unknown_fields)}")
        raise PftraceAdapterError(
            f"counter track {track_id} collector violates schema ({'; '.join(details)})"
        )
    if not isinstance(value["name"], str) or not value["name"]:
        raise PftraceAdapterError(
            f"counter track {track_id} collector name must be nonempty"
        )
    rank = value["rank"]
    if rank is not None and (
        isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
    ):
        raise PftraceAdapterError(
            f"counter track {track_id} collector rank must be a nonnegative integer or None"
        )
    process = value["process"]
    if process is not None and (not isinstance(process, str) or not process):
        raise PftraceAdapterError(
            f"counter track {track_id} collector process must be a nonempty string or None"
        )
    return value["name"], rank, process


def _resolve_counter_ancestors(
    track: _Track, tracks: Mapping[int, _Track]
) -> dict[str, _Track]:
    node_track: _Track | None = None
    nic_track: _Track | None = None
    current: _Track | None = track
    visited: set[int] = set()
    while current is not None:
        if current.track_id in visited:
            raise PftraceAdapterError(
                f"track ancestry contains cycle at {current.track_id}"
            )
        visited.add(current.track_id)
        if current.track_type == "node":
            if node_track is not None:
                raise PftraceAdapterError(
                    f"counter track {track.track_id} has multiple node ancestors"
                )
            node_track = current
        elif current.track_type == "nic":
            if nic_track is not None:
                raise PftraceAdapterError(
                    f"counter track {track.track_id} has multiple NIC ancestors"
                )
            nic_track = current
        elif current.track_type == "counter" and current is not track:
            raise PftraceAdapterError(
                f"counter track {track.track_id} cannot have a counter ancestor"
            )
        current = tracks.get(current.parent_id) if current.parent_id is not None else None
    if node_track is None or nic_track is None:
        raise PftraceAdapterError(
            f"counter track {track.track_id} lacks a node or NIC ancestor"
        )
    return {"node": node_track, "nic": nic_track}


def _ensure_track_acyclic(track: _Track, tracks: Mapping[int, _Track]) -> None:
    visited: set[int] = set()
    current: _Track | None = track
    while current is not None:
        if current.track_id in visited:
            raise PftraceAdapterError(
                f"track ancestry contains cycle at {current.track_id}"
            )
        visited.add(current.track_id)
        current = tracks.get(current.parent_id) if current.parent_id is not None else None


def _reject_duplicate_series_tracks(rows: tuple[_ParsedRow, ...]) -> None:
    track_ids_by_series: dict[tuple[Any, ...], int] = {}
    for row in rows:
        logical_key = (
            row.node_track.track_id,
            row.nic_track.track_id,
            row.counter_track.name,
            row.collector,
        )
        previous_track_id = track_ids_by_series.get(logical_key)
        if (
            previous_track_id is not None
            and previous_track_id != row.counter_track.track_id
        ):
            raise PftraceAdapterError(
                "conflicting duplicate counter tracks for the same "
                "node/NIC/counter/collector identity"
            )
        track_ids_by_series[logical_key] = row.counter_track.track_id


def _remove_exact_duplicates(rows: tuple[_ParsedRow, ...]) -> list[_ParsedRow]:
    seen: dict[tuple[Any, ...], _ParsedRow] = {}
    for row in rows:
        key = (*row.series_key, row.timestamp_ns)
        previous = seen.get(key)
        if previous is None:
            seen[key] = row
            continue
        if previous.value != row.value:
            raise PftraceAdapterError(
                f"conflicting records for track {row.counter_track.track_id} at "
                f"timestamp {row.timestamp_ns}: {previous.value} and {row.value}"
            )
        previous.duplicate = True
        previous.duplicate_source_indices = (
            *previous.duplicate_source_indices,
            row.source_index,
        )
    return list(seen.values())


def _mark_reordering(rows: list[_ParsedRow]) -> None:
    streams: dict[tuple[Any, ...], list[_ParsedRow]] = {}
    for row in rows:
        streams.setdefault(row.series_key, []).append(row)
    for stream in streams.values():
        timestamps = [row.timestamp_ns for row in stream]
        if any(
            timestamps[index] < timestamps[index - 1]
            for index in range(1, len(timestamps))
        ):
            for row in stream:
                row.reordered = True


def _build_batch(
    rows: list[_ParsedRow], config: PftraceAdapterConfig
) -> NormalizedBatch:
    _mark_reordering(rows)

    node_tracks = tuple(
        sorted({row.node_track for row in rows}, key=lambda track: track.name)
    )
    node_labels = [track.name for track in node_tracks]
    if len(node_labels) != len(set(node_labels)):
        raise PftraceAdapterError("node track names must be unique")
    nodes = tuple(
        NodeIdentity(
            node_id=index,
            label=track.name,
            original_label=track.name,
            stable_device_id=f"perfetto-track:{track.track_id}",
        )
        for index, track in enumerate(node_tracks)
    )
    node_ids = {node.label: node.node_id for node in nodes}

    nic_keys = tuple(
        sorted(
            {
                (row.node_track.track_id, row.nic_track.track_id, row.nic_track.name)
                for row in rows
            }
        )
    )
    nics_list: list[NicIdentity] = []
    for index, (node_track_id, nic_track_id, nic_name) in enumerate(nic_keys):
        node_track_name = next(
            row.node_track.name
            for row in rows
            if row.node_track.track_id == node_track_id
        )
        nic_parent_id = next(
            row.nic_track.parent_id
            for row in rows
            if row.nic_track.track_id == nic_track_id
        )
        nics_list.append(
            NicIdentity(
                nic_id=index,
                node_id=node_ids[node_track_name],
                label=nic_name,
                original_label=nic_name,
                stable_device_id=(
                    f"perfetto-track:{nic_track_id}:"
                    f"parent-track:{nic_parent_id}"
                ),
            )
        )
    nics = tuple(nics_list)
    nic_ids = {
        (node_track_id, nic_track_id, nic_name): nic.nic_id
        for nic, (node_track_id, nic_track_id, nic_name) in zip(
            nics, nic_keys, strict=True
        )
    }

    counter_names = tuple(sorted({row.counter_track.name for row in rows}))
    precision_by_counter = {
        name: any(row.precision_loss for row in rows if row.counter_track.name == name)
        for name in counter_names
    }
    counters = tuple(
        CounterMetadata(
            counter_id=index,
            name=name,
            value_kind=ValueKind.FLOAT64,
            semantics=CounterSemantics.CUMULATIVE if counter_definition(name) else CounterSemantics.UNKNOWN,
            unit=counter_definition(name).unit if counter_definition(name) else "unknown",
            baseline_relative=True,
            registry_version=REGISTRY_VERSION if counter_definition(name) else "unvalidated-pftrace-v1",
            quality_flags=int(
                QualityFlag.BASELINE_RELATIVE
                | (QualityFlag.NONE if counter_definition(name) else QualityFlag.UNKNOWN_METADATA)
                | (
                    QualityFlag.PRECISION_LOSS_POSSIBLE
                    if precision_by_counter[name]
                    else 0
                )
            ),
        )
        for index, name in enumerate(counter_names)
    )
    counter_ids = {counter.name: counter.counter_id for counter in counters}

    collector_key = lambda row: (*row.collector, -1 if config.group_collectors_by_source else row.counter_track.track_id)
    collector_keys = tuple(sorted({collector_key(row) for row in rows}))
    collectors = tuple(
        CollectorIdentity(
            collector_id=index,
            name=f"{key[0]} ({key[2]})" if key[3] == -1 else f"{key[0]}@track-{key[3]}",
            rank=key[1],
            process=key[2],
            collector_type=f"perfetto-track:{key[3]}",
        )
        for index, key in enumerate(collector_keys)
    )
    collector_ids = {
        key: collector.collector_id
        for key, collector in zip(collector_keys, collectors, strict=True)
    }

    rows.sort(
        key=lambda row: (
            row.node_track.name,
            row.nic_track.name,
            row.counter_track.name,
            row.collector,
            row.counter_track.track_id,
            row.timestamp_ns,
            row.source_index,
        )
    )
    quality_flags = np.asarray(
        [
            int(
                QualityFlag.BASELINE_RELATIVE
                | (QualityFlag.NONE if counter_definition(row.counter_track.name) else QualityFlag.UNKNOWN_METADATA)
                | (QualityFlag.PRECISION_LOSS_POSSIBLE if row.precision_loss else 0)
                | (QualityFlag.DUPLICATE_REMOVED if row.duplicate else 0)
                | (
                    QualityFlag.REORDERED | QualityFlag.NONMONOTONIC_TIMESTAMP
                    if row.reordered
                    else 0
                )
            )
            for row in rows
        ],
        dtype=np.int64,
    )

    observed_names = tuple(sorted({row.counter_track.name for row in rows}))
    requested_names = tuple(dict.fromkeys(config.requested_counter_names))
    notes = [
        "Counter semantics and units are sourced only for explicit Cassini registry entries.",
        "Values are baseline-relative offsets from an unknown hidden initial sample; the baseline is not reconstructed.",
        "The change-only conversion omits the initial sample and unchanged samples; the omitted-sample count is unknown.",
        "Acquisition boundaries, sampling cadence, gaps, and lost records are not inferred from these rows.",
        "Absent tracks or events do not establish idle, all-zero counters, complete collection, or a final idle tail.",
        "Native-cadence P50/P99/CV and activity-duration statistics are unsupported from change-only pftrace rows.",
        "The zero-valued gap and lost-record coverage counts are placeholders because their values are unknown.",
    ]
    notes.extend(
        f"Observed counter {name!r} was not in the requested list."
        for name in observed_names
        if name not in requested_names
    )
    if any(row.duplicate for row in rows):
        duplicate_indices = sorted(
            index
            for row in rows
            for index in row.duplicate_source_indices
        )
        notes.append(
            "Exact duplicate records were removed; duplicate source indices: "
            f"{duplicate_indices}."
        )

    return NormalizedBatch(
        identity=RunIdentity(
            run_id=config.run_id,
            source_fingerprint=config.source_fingerprint,
            source_format=config.source_format,
            reader_version=config.reader_version,
            conversion_lineage=config.conversion_lineage,
            clock_domain=config.clock_domain,
            synchronization_evidence=config.synchronization_evidence,
            time_origin_ns=config.time_origin_ns,
        ),
        nodes=nodes,
        nics=nics,
        counters=counters,
        collectors=collectors,
        coverage=Coverage(
            requested_counter_names=requested_names,
            available_counter_names=observed_names,
            readable_counter_names=observed_names,
            observed_counter_names=observed_names,
            notes=tuple(notes),
        ),
        node_ids=np.asarray(
            [node_ids[row.node_track.name] for row in rows], dtype=np.int64
        ),
        nic_ids=np.asarray(
            [
                nic_ids[
                    (
                        row.node_track.track_id,
                        row.nic_track.track_id,
                        row.nic_track.name,
                    )
                ]
                for row in rows
            ],
            dtype=np.int64,
        ),
        counter_ids=np.asarray(
            [counter_ids[row.counter_track.name] for row in rows], dtype=np.int64
        ),
        collector_ids=np.asarray(
            [
                collector_ids[collector_key(row)]
                for row in rows
            ],
            dtype=np.int64,
        ),
        timestamps_ns=np.asarray(
            [row.timestamp_ns for row in rows], dtype=np.int64
        ),
        record_ids=np.asarray(
            [
                f"pftrace-row-{row.source_index}:track-{row.counter_track.track_id}"
                for row in rows
            ]
        ),
        uint64_values=np.zeros(len(rows), dtype=np.uint64),
        float64_values=np.asarray(
            [row.value for row in rows], dtype=np.float64
        ),
        quality_flags=quality_flags,
    )


def _parse_nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PftraceAdapterError(f"{name} must be an integer")
    if value < 0:
        raise PftraceAdapterError(f"{name} cannot be negative")
    return value
