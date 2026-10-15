"""Safe, versioned cache persistence for one immutable :class:`TraceStore`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, is_dataclass
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any
import uuid
import zipfile

import numpy as np

from trace_model import (
    SCHEMA_VERSION,
    CollectorIdentity,
    CounterMetadata,
    CounterSemantics,
    Coverage,
    NicIdentity,
    NodeIdentity,
    NormalizedBatch,
    RunIdentity,
    TraceValidationError,
    ValueKind,
)
from trace_store import TraceStore


CACHE_FORMAT = "aurora-nic-metric-visualizer/trace-store-cache"
CACHE_SCHEMA_VERSION = 1
MAX_TRACE_ROWS = 10_000_000
MAX_CACHE_FILE_BYTES = 2 * 1024**3
MAX_TOTAL_UNCOMPRESSED_BYTES = 2 * 1024**3
MAX_METADATA_BYTES = 16 * 1024**2
MAX_ENTITY_COUNT = 100_000
MAX_RECORD_ID_LENGTH_BYTES = 1024
MAX_RECORD_ID_TOTAL_BYTES = 512 * 1024**2
MAX_NPY_HEADER_BYTES = 16 * 1024

COLUMN_NAMES = (
    "node_ids",
    "nic_ids",
    "counter_ids",
    "collector_ids",
    "timestamps_ns",
    "record_ids",
    "uint64_values",
    "float64_values",
    "quality_flags",
)
COLUMN_DTYPES = {
    "node_ids": np.dtype(np.int64),
    "nic_ids": np.dtype(np.int64),
    "counter_ids": np.dtype(np.int64),
    "collector_ids": np.dtype(np.int64),
    "timestamps_ns": np.dtype(np.int64),
    "uint64_values": np.dtype(np.uint64),
    "float64_values": np.dtype(np.float64),
    "quality_flags": np.dtype(np.int64),
}
ARCHIVE_ENTRIES = frozenset(("metadata_json", *COLUMN_NAMES))
ZIP_ENTRY_NAMES = frozenset(f"{name}.npy" for name in ARCHIVE_ENTRIES)
TOP_LEVEL_METADATA_FIELDS = frozenset(
    (
        "cache_format",
        "cache_schema_version",
        "trace_schema_version",
        "source_fingerprint",
        "row_count",
        "batch",
    )
)
BATCH_METADATA_FIELDS = frozenset(
    (
        "identity",
        "nodes",
        "nics",
        "counters",
        "collectors",
        "coverage",
    )
)


class TraceCacheError(RuntimeError):
    """Raised when a trace cache operation cannot complete safely."""


class TraceCacheValidationError(TraceCacheError, ValueError):
    """Raised when cache metadata or arrays violate the cache contract."""


class TraceCacheCorruptError(TraceCacheValidationError):
    """Raised when a cache file is truncated or otherwise unreadable."""


class TraceCacheStaleError(TraceCacheValidationError):
    """Raised when a cache belongs to a different source fingerprint."""


class TraceCacheCancelled(TraceCacheError):
    """Raised when the caller cancels a cache operation."""


def write_trace_store_cache(
    path: str | os.PathLike[str],
    store: TraceStore,
    source_fingerprint: str,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    """Atomically publish one trace store as a non-executable NPZ cache.

    Cancellation is cooperative: checks occur only at safe boundaries and do
    not interrupt NumPy once an array operation has started.
    """

    _validate_cancel_callback(should_cancel)
    _check_cancellation(should_cancel)
    cache_path = Path(path)
    _validate_source_fingerprint(source_fingerprint)
    if not isinstance(store, TraceStore):
        raise TraceCacheValidationError("store must be a TraceStore")
    if source_fingerprint != store.identity.source_fingerprint:
        raise TraceCacheValidationError(
            "source_fingerprint does not match the store run identity"
        )
    if store.schema_version != SCHEMA_VERSION:
        raise TraceCacheValidationError(
            f"unsupported trace schema version {store.schema_version}"
        )
    row_count = store.row_count
    _validate_row_count(row_count)
    columns = {name: store.column(name) for name in COLUMN_NAMES}
    _validate_columns(columns, row_count)
    metadata = _metadata_document(store, source_fingerprint)
    counters = _decode_counters(metadata)
    _validate_value_semantics(columns, counters)
    _validate_record_id_storage(columns)
    _validate_metadata_list_counts(metadata)
    _check_cancellation(should_cancel)

    document = _metadata_array(metadata)
    _validate_metadata_array(document)
    temporary_path = cache_path.with_name(
        f".{cache_path.name}.{uuid.uuid4().hex}.tmp.npz"
    )
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with temporary_path.open("wb") as cache_file:
            np.savez_compressed(cache_file, metadata_json=document, **columns)
            cache_file.flush()
            os.fsync(cache_file.fileno())
        os.chmod(temporary_path, 0o600)
        _check_cancellation(should_cancel)
        os.replace(temporary_path, cache_path)
    except TraceCacheCancelled:
        raise
    except (OSError, ValueError) as error:
        raise TraceCacheError(f"unable to write trace cache {cache_path}: {error}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def read_trace_store_cache(
    path: str | os.PathLike[str],
    source_fingerprint: str,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> TraceStore:
    """Read and strictly validate a trace cache without pickle execution.

    Cancellation is cooperative: checks occur only at safe boundaries and do
    not interrupt NumPy once an array operation has started.
    """

    _validate_cancel_callback(should_cancel)
    _check_cancellation(should_cancel)
    cache_path = Path(path)
    _validate_source_fingerprint(source_fingerprint)
    if not cache_path.is_file():
        raise TraceCacheCorruptError(f"trace cache is missing: {cache_path}")

    try:
        with cache_path.open("rb") as cache_stream:
            with zipfile.ZipFile(cache_stream, mode="r") as zip_file:
                _validate_zip_layout(cache_path, zip_file)
            cache_stream.seek(0)
            with np.load(cache_stream, allow_pickle=False) as archive:
                if set(archive.files) != ARCHIVE_ENTRIES:
                    raise TraceCacheValidationError(
                        "trace cache arrays do not match the required schema"
                    )
                _validate_metadata_header(archive.zip)
                metadata_array = archive["metadata_json"]
                _validate_metadata_array(metadata_array)
                _check_cancellation(should_cancel)
                document = _decode_metadata(metadata_array)
                _validate_metadata(document, source_fingerprint)
                _validate_metadata_list_counts(document)
                _check_cancellation(should_cancel)
                _validate_array_headers(archive.zip, document["row_count"])
                columns = {}
                for name in COLUMN_NAMES:
                    _check_cancellation(should_cancel)
                    columns[name] = _load_column(archive, name)
                    _check_cancellation(should_cancel)
    except TraceCacheError:
        raise
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        raise TraceCacheCorruptError(
            f"trace cache is corrupt or unreadable: {cache_path}: {error}"
        ) from error

    row_count = document["row_count"]
    _validate_columns(columns, row_count)
    counters = _decode_counters(document)
    _validate_value_semantics(columns, counters)
    try:
        batch = _batch_from_document(document, columns)
        return TraceStore.from_batches((batch,))
    except (TraceValidationError, TypeError, ValueError) as error:
        raise TraceCacheValidationError(
            f"trace cache contains invalid normalized data: {error}"
        ) from error
    finally:
        _check_cancellation(should_cancel)


def _validate_cancel_callback(
    should_cancel: Callable[[], bool] | None,
) -> None:
    if should_cancel is not None and not callable(should_cancel):
        raise TraceCacheValidationError("should_cancel must be callable")


def _check_cancellation(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise TraceCacheCancelled("trace cache operation was cancelled")


def _validate_zip_layout(cache_path: Path, zip_file: zipfile.ZipFile) -> None:
    cache_file_size = cache_path.stat().st_size
    if cache_file_size > MAX_CACHE_FILE_BYTES:
        raise TraceCacheValidationError(
            f"trace cache file exceeds {MAX_CACHE_FILE_BYTES} bytes"
        )

    entries = zip_file.infolist()
    names = [entry.filename for entry in entries]
    total_uncompressed_bytes = sum(entry.file_size for entry in entries)
    if total_uncompressed_bytes > MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise TraceCacheValidationError(
            f"trace cache expands to {total_uncompressed_bytes} bytes, "
            f"exceeding {MAX_TOTAL_UNCOMPRESSED_BYTES} bytes"
        )
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise TraceCacheValidationError(
            f"trace cache contains duplicate entries: {duplicate_names}"
        )
    unexpected_names = sorted(set(names) - ZIP_ENTRY_NAMES)
    missing_names = sorted(ZIP_ENTRY_NAMES - set(names))
    if unexpected_names or missing_names:
        details = []
        if unexpected_names:
            details.append(f"unexpected: {unexpected_names}")
        if missing_names:
            details.append(f"missing: {missing_names}")
        raise TraceCacheValidationError(
            "trace cache entries do not match the required schema "
            f"({'; '.join(details)})"
        )

    metadata_entry = zip_file.getinfo("metadata_json.npy")
    if metadata_entry.file_size > MAX_METADATA_BYTES + MAX_NPY_HEADER_BYTES:
        raise TraceCacheValidationError("trace cache metadata exceeds its byte budget")


def _read_npy_header(
    zip_file: zipfile.ZipFile, entry_name: str
) -> tuple[np.dtype[Any], tuple[int, ...], bool, int]:
    try:
        with zip_file.open(entry_name, mode="r") as entry:
            major_version, minor_version = np.lib.format.read_magic(entry)
            if major_version == 1:
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                    entry, max_header_size=MAX_NPY_HEADER_BYTES
                )
            elif major_version == 2:
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                    entry, max_header_size=MAX_NPY_HEADER_BYTES
                )
            else:
                raise TraceCacheValidationError(
                    f"unsupported NPY version {major_version}.{minor_version}"
                )
            data_offset = entry.tell()
    except TraceCacheError:
        raise
    except (OSError, ValueError, EOFError) as error:
        raise TraceCacheCorruptError(
            f"trace cache NPY header is malformed for {entry_name}: {error}"
        ) from error
    return np.dtype(dtype), tuple(shape), bool(fortran_order), data_offset


def _validate_metadata_header(zip_file: zipfile.ZipFile) -> None:
    dtype, shape, fortran_order, data_offset = _read_npy_header(
        zip_file, "metadata_json.npy"
    )
    if dtype != np.dtype(np.uint8) or len(shape) != 1 or fortran_order:
        raise TraceCacheValidationError("metadata_json has an invalid NPY header")
    metadata_bytes = shape[0]
    if not 0 < metadata_bytes <= MAX_METADATA_BYTES:
        raise TraceCacheValidationError("trace cache metadata exceeds its byte budget")
    entry_size = zip_file.getinfo("metadata_json.npy").file_size
    if entry_size - data_offset != metadata_bytes:
        raise TraceCacheValidationError("metadata_json NPY size does not match its header")


def _validate_array_headers(
    zip_file: zipfile.ZipFile, row_count: int
) -> None:
    _validate_metadata_header(zip_file)
    for name in COLUMN_NAMES:
        dtype, shape, fortran_order, data_offset = _read_npy_header(
            zip_file, f"{name}.npy"
        )
        if fortran_order or shape != (row_count,):
            raise TraceCacheValidationError(
                f"{name} NPY header must describe one dimension with {row_count} rows"
            )
        if name == "record_ids":
            if dtype.kind not in ("U", "S"):
                raise TraceCacheValidationError("record_ids must contain strings")
            bytes_per_record_id = dtype.itemsize
            if bytes_per_record_id > MAX_RECORD_ID_LENGTH_BYTES:
                raise TraceCacheValidationError(
                    "record_ids exceed the per-record byte budget"
                )
            if row_count * bytes_per_record_id > MAX_RECORD_ID_TOTAL_BYTES:
                raise TraceCacheValidationError(
                    "record_ids exceed the total record-ID byte budget"
                )
        elif dtype != COLUMN_DTYPES[name]:
            raise TraceCacheValidationError(
                f"{name} must have dtype {COLUMN_DTYPES[name].name}"
            )

        expected_bytes = row_count * dtype.itemsize
        entry_size = zip_file.getinfo(f"{name}.npy").file_size
        if entry_size - data_offset != expected_bytes:
            raise TraceCacheValidationError(
                f"{name} NPY size does not match row_count and dtype"
            )


def _load_column(archive: Any, name: str) -> np.ndarray:
    return archive[name]


def _validate_metadata_array(metadata_array: np.ndarray) -> None:
    if (
        not isinstance(metadata_array, np.ndarray)
        or metadata_array.dtype != np.dtype(np.uint8)
        or metadata_array.ndim != 1
        or not 0 < metadata_array.size <= MAX_METADATA_BYTES
    ):
        raise TraceCacheValidationError("metadata_json exceeds its byte budget")


def _validate_metadata_list_counts(value: Any, label: str = "metadata") -> None:
    if isinstance(value, list):
        if len(value) > MAX_ENTITY_COUNT:
            raise TraceCacheValidationError(
                f"{label} exceeds the {MAX_ENTITY_COUNT}-entry budget"
            )
        for index, item in enumerate(value):
            _validate_metadata_list_counts(item, f"{label}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_metadata_list_counts(item, f"{label}.{key}")


def _validate_record_id_storage(columns: dict[str, np.ndarray]) -> None:
    record_ids = columns["record_ids"]
    bytes_per_record_id = record_ids.dtype.itemsize
    if bytes_per_record_id > MAX_RECORD_ID_LENGTH_BYTES:
        raise TraceCacheValidationError(
            "record_ids exceed the per-record byte budget"
        )
    if record_ids.shape[0] * bytes_per_record_id > MAX_RECORD_ID_TOTAL_BYTES:
        raise TraceCacheValidationError(
            "record_ids exceed the total record-ID byte budget"
        )


def _validate_source_fingerprint(source_fingerprint: str) -> None:
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        raise TraceCacheValidationError("source_fingerprint must be a nonempty string")


def _validate_row_count(row_count: int) -> None:
    if type(row_count) is not int or not 0 < row_count <= MAX_TRACE_ROWS:
        raise TraceCacheValidationError(
            f"trace row count must be in [1, {MAX_TRACE_ROWS}]"
        )


def _json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise TraceCacheValidationError("cache metadata cannot contain NaN or infinity")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    raise TraceCacheValidationError(f"unsupported cache metadata value {type(value)!r}")


def _metadata_document(store: TraceStore, source_fingerprint: str) -> dict[str, Any]:
    return {
        "cache_format": CACHE_FORMAT,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "trace_schema_version": store.schema_version,
        "source_fingerprint": source_fingerprint,
        "row_count": store.row_count,
        "batch": {
            "identity": _json_ready(store.identity),
            "nodes": _json_ready(store.nodes),
            "nics": _json_ready(store.nics),
            "counters": _json_ready(store.counters),
            "collectors": _json_ready(store.collectors),
            "coverage": _json_ready(store.coverage),
        },
    }


def _metadata_array(document: dict[str, Any]) -> np.ndarray:
    try:
        serialized = json.dumps(
            document,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise TraceCacheValidationError(
            f"cache metadata cannot be encoded as JSON: {error}"
        ) from error
    return np.frombuffer(serialized.encode("utf-8"), dtype=np.uint8).copy()


def _decode_metadata(metadata_array: np.ndarray) -> dict[str, Any]:
    if (
        not isinstance(metadata_array, np.ndarray)
        or metadata_array.dtype != np.dtype(np.uint8)
        or metadata_array.ndim != 1
        or metadata_array.size == 0
    ):
        raise TraceCacheValidationError("metadata_json must be a nonempty uint8 array")
    try:
        document = json.loads(
            metadata_array.tobytes().decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, ValueError) as error:
        raise TraceCacheCorruptError(f"cache metadata is malformed: {error}") from error
    if not isinstance(document, dict):
        raise TraceCacheValidationError("cache metadata must be a JSON object")
    return document


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _validate_metadata(
    document: dict[str, Any], expected_source_fingerprint: str
) -> None:
    _require_exact_keys(document, TOP_LEVEL_METADATA_FIELDS, "cache metadata")
    if document["cache_format"] != CACHE_FORMAT:
        raise TraceCacheValidationError("unsupported trace cache format")
    if type(document["cache_schema_version"]) is not int:
        raise TraceCacheValidationError("cache schema version must be an integer")
    if document["cache_schema_version"] != CACHE_SCHEMA_VERSION:
        raise TraceCacheValidationError(
            f"unsupported trace cache schema version {document['cache_schema_version']}"
        )
    if type(document["trace_schema_version"]) is not int:
        raise TraceCacheValidationError("trace schema version must be an integer")
    if document["trace_schema_version"] != SCHEMA_VERSION:
        raise TraceCacheValidationError(
            f"unsupported trace schema version {document['trace_schema_version']}"
        )
    if not isinstance(document["source_fingerprint"], str) or not document["source_fingerprint"]:
        raise TraceCacheValidationError("cache source_fingerprint must be a nonempty string")
    if document["source_fingerprint"] != expected_source_fingerprint:
        raise TraceCacheStaleError("trace cache source fingerprint does not match")
    _validate_row_count(document["row_count"])
    if not isinstance(document["batch"], dict):
        raise TraceCacheValidationError("cache batch metadata must be an object")
    _require_exact_keys(document["batch"], BATCH_METADATA_FIELDS, "cache batch")


def _require_exact_keys(
    value: dict[str, Any], expected: frozenset[str], label: str
) -> None:
    if set(value) != expected:
        raise TraceCacheValidationError(f"{label} fields do not match the schema")


def _validate_columns(
    columns: dict[str, np.ndarray], row_count: int
) -> None:
    for name, array in columns.items():
        if not isinstance(array, np.ndarray):
            raise TraceCacheValidationError(f"{name} must be a NumPy array")
        if array.ndim != 1:
            raise TraceCacheValidationError(f"{name} must be one-dimensional")
        if array.shape[0] != row_count:
            raise TraceCacheValidationError(
                f"{name} has {array.shape[0]} rows; expected {row_count}"
            )
        if name == "record_ids":
            if array.dtype.kind not in ("U", "S"):
                raise TraceCacheValidationError("record_ids must contain strings")
            if row_count and np.any(array == ""):
                raise TraceCacheValidationError("record_ids cannot be empty")
            continue
        if array.dtype != COLUMN_DTYPES[name]:
            raise TraceCacheValidationError(
                f"{name} must have dtype {COLUMN_DTYPES[name].name}"
            )
        if name in {
            "node_ids",
            "nic_ids",
            "counter_ids",
            "collector_ids",
            "quality_flags",
        } and np.any(array < 0):
            raise TraceCacheValidationError(f"{name} cannot contain negative values")


def _decode_counters(document: dict[str, Any]) -> tuple[CounterMetadata, ...]:
    batch = document["batch"]
    counters = []
    try:
        for value in _metadata_list(batch["counters"], "counter metadata"):
            metadata = _dataclass_mapping(value, CounterMetadata, "counter metadata")
            metadata["value_kind"] = ValueKind(metadata["value_kind"])
            metadata["semantics"] = CounterSemantics(metadata["semantics"])
            counters.append(CounterMetadata(**metadata))
    except TraceCacheError:
        raise
    except (TypeError, ValueError) as error:
        raise TraceCacheValidationError(
            f"invalid counter metadata: {error}"
        ) from error
    return tuple(counters)


def _validate_value_semantics(
    columns: dict[str, np.ndarray], counters: tuple[CounterMetadata, ...]
) -> None:
    counter_ids = columns["counter_ids"]
    uint64_values = columns["uint64_values"]
    float64_values = columns["float64_values"]
    float_counter_ids = tuple(
        counter.counter_id
        for counter in counters
        if counter.value_kind is ValueKind.FLOAT64
    )
    exact_counter_ids = tuple(
        counter.counter_id
        for counter in counters
        if counter.value_kind is ValueKind.EXACT_UINT64
    )
    float_rows = np.isin(counter_ids, float_counter_ids)
    exact_rows = np.isin(counter_ids, exact_counter_ids)
    if np.any(~float_rows & ~exact_rows):
        raise TraceCacheValidationError("counter_ids contains an unknown counter kind")
    if np.any(~np.isnan(float64_values[exact_rows])):
        raise TraceCacheValidationError(
            "EXACT_UINT64 rows must leave float64_values as NaN"
        )
    if np.any(~np.isfinite(float64_values[float_rows])):
        raise TraceCacheValidationError("FLOAT64 values must be finite")
    if np.any(uint64_values[float_rows] != 0):
        raise TraceCacheValidationError("FLOAT64 rows must leave uint64_values as zero")


def _dataclass_mapping(
    value: Any, dataclass_type: type, label: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TraceCacheValidationError(f"{label} must be an object")
    expected = {field.name for field in fields(dataclass_type)}
    if set(value) != expected:
        raise TraceCacheValidationError(f"{label} fields do not match the schema")
    return dict(value)


def _metadata_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TraceCacheValidationError(f"{label} must be an array")
    return value


def _batch_from_document(
    document: dict[str, Any], columns: dict[str, np.ndarray]
) -> NormalizedBatch:
    batch = document["batch"]
    return NormalizedBatch(
        identity=RunIdentity(
            **_dataclass_mapping(batch["identity"], RunIdentity, "run identity")
        ),
        nodes=tuple(
            NodeIdentity(
                **_dataclass_mapping(value, NodeIdentity, "node identity")
            )
            for value in _metadata_list(batch["nodes"], "node metadata")
        ),
        nics=tuple(
            NicIdentity(**_dataclass_mapping(value, NicIdentity, "NIC identity"))
            for value in _metadata_list(batch["nics"], "NIC metadata")
        ),
        counters=_decode_counters(document),
        collectors=tuple(
            CollectorIdentity(
                **_dataclass_mapping(
                    value, CollectorIdentity, "collector identity"
                )
            )
            for value in _metadata_list(batch["collectors"], "collector metadata")
        ),
        coverage=Coverage(
            **_dataclass_mapping(batch["coverage"], Coverage, "coverage")
        ),
        **columns,
    )
