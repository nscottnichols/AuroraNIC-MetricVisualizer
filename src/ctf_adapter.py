"""Schema-aware adapter for structured raw THAPI CTF events."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from counter_semantic_registry import counter_semantic_definition
from cassini_metrics import counter_definition as diagnostic_counter_definition
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


RAW_CTF_EVENT_NAME = "cxi"
RAW_CTF_PROVIDER = "lttng_ust_cxi_sampling"
RAW_CTF_KIND = "raw_sampling"
RAW_CTF_QUALIFIED_EVENT_NAME = f"{RAW_CTF_PROVIDER}:{RAW_CTF_EVENT_NAME}"
RAW_CTF_SOURCE_FORMAT = "babeltrace2-structured-raw-ctf"
_EVENT_FIELDS = frozenset(
    {"name", "provider", "kind", "timestamp_ns", "hostname", "collector", "payload"}
)
_COLLECTOR_FIELDS = frozenset({"name", "rank", "process"})
_PAYLOAD_FIELDS = frozenset({"interface_name", "counter", "value"})
_INT64_MIN = np.iinfo(np.int64).min
_INT64_MAX = np.iinfo(np.int64).max
_UINT64_MAX = np.iinfo(np.uint64).max


class CtfAdapterError(TraceValidationError):
    """Raised when an event or adapter configuration violates the schema."""


class Bt2UnavailableError(CtfAdapterError, ImportError):
    """Raised when Babeltrace 2 Python bindings are missing."""


def _counter_metadata(counter_id: int, name: str) -> CounterMetadata:
    definition = counter_semantic_definition(name) or diagnostic_counter_definition(name)
    if definition is None:
        return CounterMetadata(
            counter_id,
            name,
            ValueKind.EXACT_UINT64,
            CounterSemantics.UNKNOWN,
            "unknown",
            baseline_relative=False,
            registry_version="unvalidated-raw-ctf-v1",
            quality_flags=QualityFlag.UNKNOWN_METADATA,
        )
    return CounterMetadata(
        counter_id,
        name,
        ValueKind.EXACT_UINT64,
        CounterSemantics.CUMULATIVE,
        definition.unit,
        baseline_relative=False,
        registry_version=definition.registry_version,
        quality_flags=QualityFlag.NONE,
    )


@dataclass(frozen=True)
class CtfAdapterConfig:
    run_id: str
    source_fingerprint: str
    reader_version: str
    clock_domain: str
    synchronization_evidence: str
    requested_counter_names: tuple[str, ...] = ()
    time_origin_ns: int | None = None
    source_format: str = RAW_CTF_SOURCE_FORMAT
    conversion_lineage: str = "direct structured event mapping"

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
                raise CtfAdapterError(f"{name} must be a nonempty string")
        if "raw-ctf" not in self.source_format:
            raise CtfAdapterError("source_format must identify raw CTF")
        if "baseline" in self.conversion_lineage.lower():
            raise CtfAdapterError(
                "raw CTF events cannot be declared baseline-relative without evidence"
            )
        if not isinstance(self.requested_counter_names, tuple):
            raise CtfAdapterError("requested_counter_names must be a tuple")
        if any(
            not isinstance(name, str) or not name
            for name in self.requested_counter_names
        ):
            raise CtfAdapterError("requested counter names must be nonempty strings")
        if len(self.requested_counter_names) != len(set(self.requested_counter_names)):
            raise CtfAdapterError("requested counter names must be unique")
        if self.time_origin_ns is not None and (
            isinstance(self.time_origin_ns, bool)
            or not isinstance(self.time_origin_ns, int)
            or not _INT64_MIN <= self.time_origin_ns <= _INT64_MAX
        ):
            raise CtfAdapterError("time_origin_ns must be an int64 integer or None")


@dataclass(frozen=True)
class CtfExtractionStats:
    cxi_event_count: int
    filtered_event_count: int


@dataclass
class _ParsedEvent:
    source_index: int
    hostname: str
    interface_name: str
    counter_name: str
    collector_name: str
    collector_rank: int | None
    collector_process: str | None
    timestamp_ns: int
    value: int
    duplicate: bool = False
    duplicate_source_indices: tuple[int, ...] = ()
    reordered: bool = False

    @property
    def collector_key(self) -> tuple[str, str, int | None, str | None]:
        return (
            self.hostname,
            self.collector_name,
            self.collector_rank,
            self.collector_process,
        )


def ctf_events_to_normalized_batch(
    events: Iterable[Mapping[str, Any]], config: CtfAdapterConfig
) -> NormalizedBatch:
    """Normalize strict, structured raw CXI CTF events."""

    if isinstance(events, (str, bytes)):
        raise CtfAdapterError("events must be an iterable of structured mappings")
    parsed = tuple(
        _parse_event(event, index) for index, event in enumerate(events)
    )
    if not parsed:
        raise CtfAdapterError("at least one raw CTF event is required")
    return _build_batch(_remove_exact_duplicates(parsed), config)


def adapt_babeltrace2_events(
    events: Iterable[Mapping[str, Any]], config: CtfAdapterConfig
) -> NormalizedBatch:
    """Adapt structured events already extracted with Babeltrace 2."""

    try:
        import bt2
    except ImportError as error:
        raise Bt2UnavailableError(
            "The optional Babeltrace 2 Python bindings (bt2) are required for "
            "adapt_babeltrace2_events. Use the site-provided bt2 bindings, or the guarded portable reader; "
            "or use ctf_events_to_normalized_batch with already-extracted events."
        ) from error
    return ctf_events_to_normalized_batch(events, config)


def extract_babeltrace2_events(
    path: str | os.PathLike[str],
) -> list[dict[str, Any]]:
    events, _ = extract_babeltrace2_events_with_stats(path)
    return events


def extract_babeltrace2_events_with_stats(
    path: str | os.PathLike[str],
) -> tuple[list[dict[str, Any]], CtfExtractionStats]:
    """Extract raw THAPI CXI events from a CTF path with Babeltrace 2."""

    try:
        import bt2
    except ImportError as error:
        raise Bt2UnavailableError(
            "The optional Babeltrace 2 Python bindings (bt2) are required for "
            "extract_babeltrace2_events. Use the site-provided bt2 bindings, or the guarded portable reader; "
            "before reading a CTF path."
        ) from error

    try:
        trace_path = os.fspath(path)
    except TypeError as error:
        raise CtfAdapterError(
            "path must be a string or path-like object"
        ) from error
    if not isinstance(trace_path, str):
        raise CtfAdapterError("path must be a string or path-like object")

    event_message_types = tuple(
        message_type
        for message_type in (
            getattr(bt2, "_EventMessage", None),
            getattr(bt2, "_EventMessageConst", None),
        )
        if message_type is not None
    )
    try:
        message_iterator = bt2.TraceCollectionMessageIterator(trace_path)
    except Exception as error:
        raise CtfAdapterError(
            f"unable to create the Babeltrace 2 trace-collection iterator for "
            f"{trace_path!r}: {error}"
        ) from error

    events: list[dict[str, Any]] = []
    filtered_event_count = 0
    try:
        for message in message_iterator:
            loss_types = tuple(t for t in (getattr(bt2, '_DiscardedEventsMessageConst', None),
                getattr(bt2, '_DiscardedPacketsMessageConst', None)) if t is not None)
            if loss_types and isinstance(message, loss_types):
                raise CtfAdapterError('CTF reports discarded events/packets; a complete counter trace is required. Lost samples are not converted into measured zeros.')
            if not isinstance(message, event_message_types):
                continue

            event = message.event
            qualified_name = _qualified_event_name(event)
            if qualified_name != RAW_CTF_QUALIFIED_EVENT_NAME:
                filtered_event_count += 1
                continue

            stream = event.stream
            hostname = _trace_or_stream_hostname(stream)
            collector_process = _event_collector_process(event, stream)
            timestamp_ns = _clock_timestamp_ns(message)
            payload = _event_payload(event)

            events.append(
                {
                    "name": RAW_CTF_EVENT_NAME,
                    "provider": RAW_CTF_PROVIDER,
                    "kind": RAW_CTF_KIND,
                    "timestamp_ns": timestamp_ns,
                    "hostname": hostname,
                    "collector": {
                        "name": "babeltrace2",
                        "rank": None,
                        "process": collector_process,
                    },
                    "payload": payload,
                }
            )
    except CtfAdapterError:
        raise
    except Exception as error:
        raise CtfAdapterError(
            f"error while reading Babeltrace 2 messages from {trace_path!r}: {error}"
        ) from error

    return events, CtfExtractionStats(
        cxi_event_count=len(events),
        filtered_event_count=filtered_event_count,
    )


def _qualified_event_name(event: Any) -> str:
    event_name = event.name
    if not isinstance(event_name, str) or not event_name:
        raise CtfAdapterError("Babeltrace 2 event name must be a nonempty string")
    if ":" in event_name:
        return event_name

    event_class = getattr(event, "event_class", None)
    if event_class is None:
        event_class = getattr(event, "cls", None)
    namespace = getattr(event_class, "namespace", None)
    if not isinstance(namespace, str) or not namespace:
        raise CtfAdapterError(
            "Babeltrace 2 event namespace is required to identify raw CTF events"
        )
    return f"{namespace}:{event_name}"


def _trace_or_stream_hostname(stream: Any) -> str:
    try:
        trace = stream.trace
    except AttributeError as error:
        raise CtfAdapterError("Babeltrace 2 event stream has no trace context") from error

    hostnames = []
    for context_name, context in (("trace", trace), ("stream", stream)):
        environment = getattr(context, "environment", None)
        if environment is None or "hostname" not in environment:
            continue
        hostname = environment["hostname"]
        hostname = _normalize_environment_string(hostname)
        if not hostname:
            raise CtfAdapterError(
                f"{context_name} environment hostname must be a nonempty string"
            )
        hostnames.append(hostname)

    if not hostnames:
        raise CtfAdapterError(
            "Babeltrace 2 trace or stream environment has no hostname; "
            "hostname will not be inferred"
        )
    if len(set(hostnames)) != 1:
        raise CtfAdapterError(
            "Babeltrace 2 trace and stream hostname contexts conflict: "
            f"{hostnames!r}"
        )
    return hostnames[0]


def _normalize_environment_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if type(value).__module__.startswith("bt2"):
        return str(value)
    raise CtfAdapterError(
        "Babeltrace 2 environment hostname must be a string value"
    )


def _event_collector_process(event, stream):
    """Per-CPU buffers are transport, not sampling-process identity."""
    context = getattr(event, 'common_context_field', None)
    if context is None:
        context = getattr(event, 'context_field', None)
    if context is not None:
        for pid_name, tid_name in (('vpid','vtid'),('_vpid','_vtid')):
            try:
                pid, tid = int(context[pid_name]), int(context[tid_name])
                return f"trace={_trace_identity(stream)};pid={pid};tid={tid}"
            except (KeyError, TypeError, ValueError):
                continue
    # Older context-free captures cannot be joined on a guessed process ID.
    return _collector_process(stream)


def _collector_process(stream: Any) -> str:
    trace_identity = _trace_identity(stream)
    stream_identity = _stream_identity(stream)
    return f"trace={trace_identity};stream={stream_identity}"


def _trace_identity(stream: Any) -> str:
    trace = stream.trace
    trace_uuid = None
    try:
        trace_uuid = trace.uuid
    except (AttributeError, ValueError):
        trace_uuid = None
    trace_name = getattr(trace, "name", None)
    if trace_uuid is None and (
        not isinstance(trace_name, str) or not trace_name
    ):
        raise CtfAdapterError(
            "Babeltrace 2 trace identity is missing; collector provenance "
            "will not be inferred"
        )

    identity = []
    if trace_uuid is not None:
        identity.append(f"uuid={trace_uuid!s}")
    if isinstance(trace_name, str) and trace_name:
        identity.append(f"name={trace_name!r}")
    return ";".join(identity)


def _stream_identity(stream: Any) -> str:
    stream_name = getattr(stream, "name", None)
    stream_id = getattr(stream, "id", None)
    if stream_name is None and stream_id is None:
        raise CtfAdapterError(
            "Babeltrace 2 stream identity is missing; collector provenance "
            "will not be inferred"
        )

    identity = []
    if stream_name is not None:
        identity.append(f"name={stream_name!r}")
    if stream_id is not None:
        identity.append(f"id={stream_id!r}")
    return ";".join(identity)


def _clock_timestamp_ns(message: Any) -> int:
    try:
        return message.default_clock_snapshot.ns_from_origin
    except AttributeError as error:
        raise CtfAdapterError(
            "Babeltrace 2 event message has no default clock snapshot with "
            "nanoseconds from origin"
        ) from error


def _event_payload(event: Any) -> dict[str, Any]:
    payload = getattr(event, "payload", None)
    if payload is None:
        payload = getattr(event, "payload_field", None)
    if payload is None:
        raise CtfAdapterError(
            "Babeltrace 2 event payload is unavailable"
        )
    extracted = {}
    for field_name in ("interface_name", "counter", "value"):
        if field_name not in payload:
            raise CtfAdapterError(
                f"Babeltrace 2 raw CXI event payload is missing {field_name!r}"
            )
        value = payload[field_name]
        if field_name == "value":
            extracted[field_name] = _field_integer(value)
        else:
            extracted[field_name] = _field_string(value)
    return extracted


def _field_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if type(value).__module__.startswith("bt2"):
        return str(value)
    raise CtfAdapterError("Babeltrace 2 string field is not a string value")


def _field_integer(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if type(value).__module__.startswith("bt2"):
        return int(value)
    raise CtfAdapterError("Babeltrace 2 integer field is not an integer value")


def _build_batch(
    events: list[_ParsedEvent], config: CtfAdapterConfig
) -> NormalizedBatch:
    _mark_reordering(events)
    node_labels = tuple(sorted({event.hostname for event in events}))
    nic_keys = tuple(
        sorted({(event.hostname, event.interface_name) for event in events})
    )
    counter_names = tuple(sorted({event.counter_name for event in events}))
    collector_keys = tuple(sorted({event.collector_key for event in events}))

    nodes = tuple(
        NodeIdentity(node_id, label, label)
        for node_id, label in enumerate(node_labels)
    )
    node_ids = {node.label: node.node_id for node in nodes}
    nics = tuple(
        NicIdentity(nic_id, node_ids[hostname], interface, interface)
        for nic_id, (hostname, interface) in enumerate(nic_keys)
    )
    nic_ids = {
        (nic.node_id, nic.original_label): nic.nic_id for nic in nics
    }
    counters = tuple(
        _counter_metadata(counter_id, name)
        for counter_id, name in enumerate(counter_names)
    )
    counter_ids = {counter.name: counter.counter_id for counter in counters}
    collectors = tuple(
        CollectorIdentity(
            collector_id,
            _collector_display_name(key),
            rank=key[2],
            process=key[3],
        )
        for collector_id, key in enumerate(collector_keys)
    )
    collector_ids = {collector.name: collector.collector_id for collector in collectors}

    events.sort(
        key=lambda event: (
            event.hostname,
            event.interface_name,
            event.counter_name,
            event.collector_key,
            event.timestamp_ns,
            event.source_index,
        )
    )
    quality_flags = np.asarray(
        [
            int(
                (
                    QualityFlag.NONE
                    if (counter_semantic_definition(event.counter_name) or diagnostic_counter_definition(event.counter_name)) is not None
                    else QualityFlag.UNKNOWN_METADATA
                )
                | (QualityFlag.DUPLICATE_REMOVED if event.duplicate else 0)
                | (
                    QualityFlag.REORDERED | QualityFlag.NONMONOTONIC_TIMESTAMP
                    if event.reordered
                    else 0
                )
            )
            for event in events
        ],
        dtype=np.int64,
    )

    observed_names = tuple(sorted({event.counter_name for event in events}))
    requested_names = tuple(dict.fromkeys(config.requested_counter_names))
    notes = [
        "Counter semantics and units are unvalidated raw CTF metadata except for explicitly sourced Cassini registry entries.",
        "Acquisition boundaries, cadence, and lost-record coverage are not inferred from events.",
    ]
    notes.extend(
        f"Observed counter {name!r} was not in the requested list."
        for name in observed_names
        if name not in requested_names
    )
    if any(event.duplicate for event in events):
        duplicate_indices = sorted(
            index
            for event in events
            for index in event.duplicate_source_indices
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
            [node_ids[event.hostname] for event in events], dtype=np.int64
        ),
        nic_ids=np.asarray(
            [nic_ids[(node_ids[event.hostname], event.interface_name)] for event in events],
            dtype=np.int64,
        ),
        counter_ids=np.asarray(
            [counter_ids[event.counter_name] for event in events], dtype=np.int64
        ),
        collector_ids=np.asarray(
            [
                collector_ids[_collector_display_name(event.collector_key)]
                for event in events
            ],
            dtype=np.int64,
        ),
        timestamps_ns=np.asarray(
            [event.timestamp_ns for event in events], dtype=np.int64
        ),
        record_ids=np.asarray(
            [f"ctf-event-{event.source_index}" for event in events]
        ),
        uint64_values=np.asarray(
            [event.value for event in events], dtype=np.uint64
        ),
        float64_values=np.full(len(events), np.nan, dtype=np.float64),
        quality_flags=quality_flags,
    )


def _parse_event(event: Mapping[str, Any], source_index: int) -> _ParsedEvent:
    if not isinstance(event, Mapping):
        raise CtfAdapterError(f"event {source_index} must be a structured mapping")

    event_fields = set(event.keys())
    missing_fields = _EVENT_FIELDS - event_fields
    unknown_fields = event_fields - _EVENT_FIELDS
    if missing_fields or unknown_fields:
        details = []
        if missing_fields:
            details.append(f"missing fields: {sorted(missing_fields)}")
        if unknown_fields:
            details.append(f"undocumented fields: {sorted(unknown_fields)}")
        raise CtfAdapterError(
            f"event {source_index} violates the raw CTF schema ({'; '.join(details)})"
        )

    if event["kind"] == "analysis_interval":
        raise CtfAdapterError(
            "analysis/interval CTF events are not raw sampling events; "
            "their observed schema and baseline semantics must be handled separately"
        )
    if event["name"] != RAW_CTF_EVENT_NAME:
        if event["name"] == "sampling_nic":
            raise CtfAdapterError(
                "analysis/interval CTF events are not raw sampling events; "
                "their observed schema and baseline semantics must be handled separately"
            )
        raise CtfAdapterError(
            f"event {source_index} has unsupported event name {event['name']!r}"
        )
    if event["provider"] != RAW_CTF_PROVIDER:
        raise CtfAdapterError(
            f"event {source_index} has unsupported provider {event['provider']!r}"
        )
    if event["kind"] != RAW_CTF_KIND:
        raise CtfAdapterError(f"event {source_index} must have kind {RAW_CTF_KIND!r}")
    if not isinstance(event["hostname"], str) or not event["hostname"]:
        raise CtfAdapterError(f"event {source_index} hostname must be nonempty")

    timestamp = event["timestamp_ns"]
    if isinstance(timestamp, bool) or not isinstance(timestamp, int):
        raise CtfAdapterError(f"event {source_index} timestamp_ns must be an integer")
    if not _INT64_MIN <= timestamp <= _INT64_MAX:
        raise CtfAdapterError(f"event {source_index} timestamp_ns is outside int64 range")

    collector = event["collector"]
    if not isinstance(collector, Mapping):
        raise CtfAdapterError(f"event {source_index} collector must be a mapping")
    collector_fields = set(collector.keys())
    missing_collector = _COLLECTOR_FIELDS - collector_fields
    unknown_collector = collector_fields - _COLLECTOR_FIELDS
    if missing_collector or unknown_collector:
        details = []
        if missing_collector:
            details.append(f"missing fields: {sorted(missing_collector)}")
        if unknown_collector:
            details.append(f"undocumented fields: {sorted(unknown_collector)}")
        raise CtfAdapterError(
            f"event {source_index} collector violates schema ({'; '.join(details)})"
        )
    if not isinstance(collector["name"], str) or not collector["name"]:
        raise CtfAdapterError(f"event {source_index} collector name must be nonempty")
    rank = collector["rank"]
    if rank is not None and (
        isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
    ):
        raise CtfAdapterError(
            f"event {source_index} collector rank must be a nonnegative integer or None"
        )
    process = collector["process"]
    if process is not None and (not isinstance(process, str) or not process):
        raise CtfAdapterError(
            f"event {source_index} collector process must be a nonempty string or None"
        )

    payload = event["payload"]
    if not isinstance(payload, Mapping):
        raise CtfAdapterError(f"event {source_index} payload must be a mapping")
    payload_fields = set(payload.keys())
    missing_payload = _PAYLOAD_FIELDS - payload_fields
    unknown_payload = payload_fields - _PAYLOAD_FIELDS
    if missing_payload or unknown_payload:
        details = []
        if missing_payload:
            details.append(f"missing fields: {sorted(missing_payload)}")
        if unknown_payload:
            details.append(f"undocumented fields: {sorted(unknown_payload)}")
        raise CtfAdapterError(
            f"event {source_index} payload violates schema ({'; '.join(details)})"
        )
    if not isinstance(payload["interface_name"], str) or not payload["interface_name"]:
        raise CtfAdapterError(f"event {source_index} interface_name must be nonempty")
    if not isinstance(payload["counter"], str) or not payload["counter"]:
        raise CtfAdapterError(f"event {source_index} counter must be nonempty")
    value = payload["value"]
    if isinstance(value, bool) or not isinstance(value, int):
        raise CtfAdapterError(f"event {source_index} value must be an integer")
    if not 0 <= value <= _UINT64_MAX:
        raise CtfAdapterError(f"event {source_index} value is outside uint64 range")

    return _ParsedEvent(
        source_index=source_index,
        hostname=event["hostname"],
        interface_name=payload["interface_name"],
        counter_name=payload["counter"],
        collector_name=collector["name"],
        collector_rank=rank,
        collector_process=process,
        timestamp_ns=timestamp,
        value=value,
    )


def _remove_exact_duplicates(
    events: tuple[_ParsedEvent, ...]
) -> list[_ParsedEvent]:
    seen: dict[tuple[Any, ...], _ParsedEvent] = {}
    for event in events:
        key = (
            event.hostname,
            event.interface_name,
            event.counter_name,
            event.collector_key,
            event.timestamp_ns,
        )
        previous = seen.get(key)
        if previous is None:
            seen[key] = event
            continue
        if previous.value != event.value:
            raise CtfAdapterError(
                f"conflicting records for {event.hostname}/{event.interface_name}/"
                f"{event.counter_name} at timestamp {event.timestamp_ns}: "
                f"{previous.value} and {event.value}"
            )
        previous.duplicate = True
        previous.duplicate_source_indices = (
            *previous.duplicate_source_indices,
            event.source_index,
        )
    return list(seen.values())


def _mark_reordering(events: list[_ParsedEvent]) -> None:
    streams: dict[tuple[Any, ...], list[_ParsedEvent]] = {}
    for event in events:
        key = (
            event.hostname,
            event.interface_name,
            event.counter_name,
            event.collector_key,
        )
        streams.setdefault(key, []).append(event)
    for stream in streams.values():
        timestamps = [event.timestamp_ns for event in stream]
        if any(
            timestamps[index] < timestamps[index - 1]
            for index in range(1, len(timestamps))
        ):
            for event in stream:
                event.reordered = True


def _collector_display_name(collector_key: tuple[Any, ...]) -> str:
    return repr(collector_key)
