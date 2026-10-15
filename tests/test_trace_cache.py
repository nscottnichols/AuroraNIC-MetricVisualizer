import numpy as np
import pytest
import zipfile

import trace_cache
from trace_cache import (
    MAX_TRACE_ROWS,
    TraceCacheCancelled,
    TraceCacheCorruptError,
    TraceCacheError,
    TraceCacheStaleError,
    TraceCacheValidationError,
    read_trace_store_cache,
    write_trace_store_cache,
)
from trace_model import (
    CollectorIdentity,
    CounterMetadata,
    CounterSemantics,
    Coverage,
    NicIdentity,
    NodeIdentity,
    NormalizedBatch,
    RunIdentity,
    ValueKind,
)
from trace_store import TraceStore


RUN = RunIdentity(
    run_id="run-cache",
    source_fingerprint="sha256:cache-test",
    source_format="raw-ctf",
    reader_version="reader-1",
    conversion_lineage="direct",
    clock_domain="monotonic-ns",
    synchronization_evidence="single source clock",
    time_origin_ns=0,
)
NODE = NodeIdentity(1, "node-a", "node-a-original")
NIC = NicIdentity(10, NODE.node_id, "cxi0", "cxi0-original")
UINT_COUNTER = CounterMetadata(
    100,
    "rx_ok_octets",
    ValueKind.EXACT_UINT64,
    CounterSemantics.CUMULATIVE,
    "bytes",
    baseline_relative=False,
    registry_version="registry-1",
)
FLOAT_COUNTER = CounterMetadata(
    101,
    "converted_rx_octets",
    ValueKind.FLOAT64,
    CounterSemantics.CUMULATIVE,
    "bytes",
    baseline_relative=True,
    registry_version="registry-1",
)
COLLECTOR = CollectorIdentity(1, "collector-rank-0", rank=0)


def make_store():
    batch = NormalizedBatch(
        identity=RUN,
        nodes=(NODE,),
        nics=(NIC,),
        counters=(UINT_COUNTER, FLOAT_COUNTER),
        collectors=(COLLECTOR,),
        coverage=Coverage(
            requested_counter_names=(UINT_COUNTER.name, FLOAT_COUNTER.name),
            available_counter_names=(UINT_COUNTER.name, FLOAT_COUNTER.name),
            readable_counter_names=(UINT_COUNTER.name, FLOAT_COUNTER.name),
            observed_counter_names=(UINT_COUNTER.name, FLOAT_COUNTER.name),
            acquisition_start_ns=1,
            acquisition_end_ns=3,
        ),
        node_ids=np.asarray([NODE.node_id] * 3, dtype=np.int64),
        nic_ids=np.asarray([NIC.nic_id] * 3, dtype=np.int64),
        counter_ids=np.asarray(
            [UINT_COUNTER.counter_id, UINT_COUNTER.counter_id, FLOAT_COUNTER.counter_id],
            dtype=np.int64,
        ),
        collector_ids=np.asarray([COLLECTOR.collector_id] * 3, dtype=np.int64),
        timestamps_ns=np.asarray([1, 2, 3], dtype=np.int64),
        record_ids=np.asarray(["record-a", "record-b", "record-c"]),
        uint64_values=np.asarray([0, 2**64 - 1, 0], dtype=np.uint64),
        float64_values=np.asarray([np.nan, np.nan, 1.5], dtype=np.float64),
        quality_flags=np.zeros(3, dtype=np.int64),
    )
    return TraceStore.from_batches((batch,))


def write_valid_cache(path):
    store = make_store()
    write_trace_store_cache(path, store, RUN.source_fingerprint)
    return store


def test_round_trip_preserves_metadata_columns_and_exact_uint64(tmp_path):
    path = tmp_path / "trace.npz"
    store = make_store()

    write_trace_store_cache(path, store, RUN.source_fingerprint)
    loaded = read_trace_store_cache(path, RUN.source_fingerprint)

    assert isinstance(loaded, TraceStore)
    assert loaded.identity == store.identity
    assert loaded.nodes == store.nodes
    assert loaded.nics == store.nics
    assert loaded.counters == store.counters
    assert loaded.collectors == store.collectors
    assert loaded.coverage == store.coverage
    assert loaded.row_count == store.row_count == 3
    for name in trace_cache.COLUMN_NAMES:
        np.testing.assert_array_equal(loaded.column(name), store.column(name))
    exact = loaded.query_unique_series(NODE, NIC, UINT_COUNTER)
    assert exact.values.dtype == np.dtype(np.uint64)
    assert exact.values[1] == 2**64 - 1
    converted = loaded.query_unique_series(NODE, NIC, FLOAT_COUNTER)
    assert converted.values.tolist() == [1.5]


def test_read_reconstructs_through_trace_store_from_batches(monkeypatch, tmp_path):
    path = tmp_path / "trace.npz"
    store = make_store()
    write_trace_store_cache(path, store, RUN.source_fingerprint)
    original = TraceStore.from_batches
    calls = []

    def wrapped(cls, batches):
        batch_tuple = tuple(batches)
        calls.append(batch_tuple)
        return original(batch_tuple)

    monkeypatch.setattr(TraceStore, "from_batches", classmethod(wrapped))
    loaded = read_trace_store_cache(path, RUN.source_fingerprint)

    assert len(calls) == 1
    assert isinstance(calls[0][0], NormalizedBatch)
    assert loaded.identity == RUN


def test_atomic_replacement_preserves_existing_cache_on_failure(monkeypatch, tmp_path):
    path = tmp_path / "trace.npz"
    store = make_store()
    write_trace_store_cache(path, store, RUN.source_fingerprint)
    original_bytes = path.read_bytes()

    def failing_replace(source, destination):
        raise OSError("publication blocked")

    monkeypatch.setattr(trace_cache.os, "replace", failing_replace)
    with pytest.raises(TraceCacheError, match="unable to write trace cache"):
        write_trace_store_cache(path, store, RUN.source_fingerprint)

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [path]


def test_stale_source_fingerprint_is_rejected(tmp_path):
    path = tmp_path / "trace.npz"
    write_valid_cache(path)

    with pytest.raises(TraceCacheStaleError, match="source fingerprint"):
        read_trace_store_cache(path, "sha256:different")


def test_truncated_cache_is_rejected_as_corrupt(tmp_path):
    path = tmp_path / "trace.npz"
    write_valid_cache(path)
    original = path.read_bytes()
    path.write_bytes(original[: len(original) // 2])

    with pytest.raises(TraceCacheCorruptError, match="corrupt or unreadable"):
        read_trace_store_cache(path, RUN.source_fingerprint)


def test_non_standard_json_constants_are_rejected(tmp_path):
    path = tmp_path / "trace.npz"
    store = make_store()
    columns = {name: store.column(name) for name in trace_cache.COLUMN_NAMES}
    metadata = np.frombuffer(b"NaN", dtype=np.uint8)
    with path.open("wb") as cache_file:
        np.savez(cache_file, metadata_json=metadata, **columns)

    with pytest.raises(TraceCacheCorruptError, match="cache metadata is malformed"):
        read_trace_store_cache(path, RUN.source_fingerprint)


def test_wrong_cache_schema_version_is_rejected(monkeypatch, tmp_path):
    path = tmp_path / "trace.npz"
    store = make_store()
    monkeypatch.setattr(trace_cache, "CACHE_SCHEMA_VERSION", 999)
    write_trace_store_cache(path, store, RUN.source_fingerprint)
    monkeypatch.setattr(trace_cache, "CACHE_SCHEMA_VERSION", 1)

    with pytest.raises(TraceCacheValidationError, match="unsupported trace cache schema"):
        read_trace_store_cache(path, RUN.source_fingerprint)


def test_wrong_array_dtype_is_rejected(monkeypatch, tmp_path):
    path = tmp_path / "trace.npz"
    store = make_store()
    original_savez = trace_cache.np.savez_compressed

    def wrong_dtype_savez(file, **arrays):
        arrays["uint64_values"] = arrays["uint64_values"].astype(np.int64)
        original_savez(file, **arrays)

    monkeypatch.setattr(trace_cache.np, "savez_compressed", wrong_dtype_savez)
    write_trace_store_cache(path, store, RUN.source_fingerprint)

    with pytest.raises(TraceCacheValidationError, match="uint64_values must have dtype"):
        read_trace_store_cache(path, RUN.source_fingerprint)


def test_wrong_array_shape_is_rejected(monkeypatch, tmp_path):
    path = tmp_path / "trace.npz"
    store = make_store()
    original_savez = trace_cache.np.savez_compressed

    def wrong_shape_savez(file, **arrays):
        arrays["timestamps_ns"] = arrays["timestamps_ns"][:-1]
        original_savez(file, **arrays)

    monkeypatch.setattr(trace_cache.np, "savez_compressed", wrong_shape_savez)
    write_trace_store_cache(path, store, RUN.source_fingerprint)

    def fail_column_load(archive, name):
        pytest.fail(f"column {name} was loaded before header validation")

    monkeypatch.setattr(trace_cache, "_load_column", fail_column_load)
    with pytest.raises(TraceCacheValidationError, match="one dimension with 3 rows"):
        read_trace_store_cache(path, RUN.source_fingerprint)


def test_unexpected_archive_array_is_rejected(monkeypatch, tmp_path):
    path = tmp_path / "trace.npz"
    store = make_store()
    original_savez = trace_cache.np.savez_compressed

    def extra_array_savez(file, **arrays):
        arrays["unexpected"] = np.zeros(1, dtype=np.int64)
        original_savez(file, **arrays)

    monkeypatch.setattr(trace_cache.np, "savez_compressed", extra_array_savez)
    write_trace_store_cache(path, store, RUN.source_fingerprint)

    with pytest.raises(TraceCacheValidationError, match="entries do not match"):
        read_trace_store_cache(path, RUN.source_fingerprint)


def test_nonfinite_float_rows_are_rejected(monkeypatch, tmp_path):
    path = tmp_path / "trace.npz"
    store = make_store()
    original_savez = trace_cache.np.savez_compressed

    def nonfinite_savez(file, **arrays):
        values = arrays["float64_values"].copy()
        values[2] = np.nan
        arrays["float64_values"] = values
        original_savez(file, **arrays)

    monkeypatch.setattr(trace_cache.np, "savez_compressed", nonfinite_savez)
    write_trace_store_cache(path, store, RUN.source_fingerprint)

    with pytest.raises(TraceCacheValidationError, match="FLOAT64 values must be finite"):
        read_trace_store_cache(path, RUN.source_fingerprint)


def test_row_count_is_bounded_for_ten_million_target(monkeypatch, tmp_path):
    assert MAX_TRACE_ROWS == 10_000_000
    path = tmp_path / "trace.npz"
    store = make_store()
    monkeypatch.setattr(trace_cache, "MAX_TRACE_ROWS", 2)

    with pytest.raises(TraceCacheValidationError, match="row count must be in"):
        write_trace_store_cache(path, store, RUN.source_fingerprint)
    assert not path.exists()


def test_cache_file_size_is_bounded(monkeypatch, tmp_path):
    path = tmp_path / "trace.npz"
    write_valid_cache(path)
    monkeypatch.setattr(trace_cache, "MAX_CACHE_FILE_BYTES", 1)

    with pytest.raises(TraceCacheValidationError, match="file exceeds"):
        read_trace_store_cache(path, RUN.source_fingerprint)


def test_total_uncompressed_compressed_entries_are_bounded(monkeypatch, tmp_path):
    path = tmp_path / "compressed-bomb.npz"
    np.savez_compressed(path, metadata_json=np.zeros(128 * 1024, dtype=np.uint8))
    assert path.stat().st_size < 64 * 1024
    monkeypatch.setattr(trace_cache, "MAX_TOTAL_UNCOMPRESSED_BYTES", 64 * 1024)

    with pytest.raises(TraceCacheValidationError, match="expands to"):
        read_trace_store_cache(path, RUN.source_fingerprint)


def test_oversized_metadata_is_rejected_before_publication(monkeypatch, tmp_path):
    path = tmp_path / "trace.npz"
    store = make_store()
    monkeypatch.setattr(trace_cache, "MAX_METADATA_BYTES", 16)
    monkeypatch.setattr(
        trace_cache,
        "_metadata_array",
        lambda document: np.zeros(32, dtype=np.uint8),
    )

    with pytest.raises(TraceCacheValidationError, match="metadata_json exceeds"):
        write_trace_store_cache(path, store, RUN.source_fingerprint)
    assert not path.exists()


def test_metadata_entity_lists_are_bounded(monkeypatch, tmp_path):
    path = tmp_path / "trace.npz"
    store = make_store()
    monkeypatch.setattr(trace_cache, "MAX_ENTITY_COUNT", 1)

    with pytest.raises(TraceCacheValidationError, match="entry budget"):
        write_trace_store_cache(path, store, RUN.source_fingerprint)
    assert not path.exists()


def test_oversized_record_id_storage_is_rejected(monkeypatch, tmp_path):
    path = tmp_path / "trace.npz"
    store = make_store()
    monkeypatch.setattr(trace_cache, "MAX_RECORD_ID_LENGTH_BYTES", 16)

    with pytest.raises(TraceCacheValidationError, match="per-record byte budget"):
        write_trace_store_cache(path, store, RUN.source_fingerprint)
    assert not path.exists()


def test_total_record_id_bytes_are_bounded(monkeypatch, tmp_path):
    path = tmp_path / "trace.npz"
    store = make_store()
    monkeypatch.setattr(trace_cache, "MAX_RECORD_ID_TOTAL_BYTES", 16)

    with pytest.raises(TraceCacheValidationError, match="total record-ID byte budget"):
        write_trace_store_cache(path, store, RUN.source_fingerprint)
    assert not path.exists()


def test_duplicate_zip_entries_are_rejected(tmp_path):
    path = tmp_path / "trace.npz"
    write_valid_cache(path)
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(path, mode="a") as archive:
            archive.writestr("metadata_json.npy", b"")

    with pytest.raises(TraceCacheValidationError, match="duplicate entries"):
        read_trace_store_cache(path, RUN.source_fingerprint)


def test_writer_uses_compressed_npz_entries(tmp_path):
    path = tmp_path / "trace.npz"
    write_valid_cache(path)

    with zipfile.ZipFile(path, mode="r") as archive:
        assert all(
            entry.compress_type != zipfile.ZIP_STORED
            for entry in archive.infolist()
        )


def test_ten_million_row_target_fits_declared_budgets():
    numeric_bytes = MAX_TRACE_ROWS * sum(
        trace_cache.COLUMN_DTYPES[name].itemsize for name in trace_cache.COLUMN_DTYPES
    )
    assert (
        numeric_bytes
        + trace_cache.MAX_RECORD_ID_TOTAL_BYTES
        + trace_cache.MAX_METADATA_BYTES
        < trace_cache.MAX_TOTAL_UNCOMPRESSED_BYTES
    )
    assert trace_cache.MAX_RECORD_ID_TOTAL_BYTES // MAX_TRACE_ROWS >= 16


def test_cancellation_occurs_between_column_loads(monkeypatch, tmp_path):
    path = tmp_path / "trace.npz"
    write_valid_cache(path)
    loaded_columns = []
    original_load = trace_cache._load_column

    def record_column_load(archive, name):
        loaded_columns.append(name)
        return original_load(archive, name)

    monkeypatch.setattr(trace_cache, "_load_column", record_column_load)

    with pytest.raises(TraceCacheCancelled, match="cancelled"):
        read_trace_store_cache(
            path,
            RUN.source_fingerprint,
            should_cancel=lambda: bool(loaded_columns),
        )

    assert loaded_columns == ["node_ids"]


def test_cancellation_has_dedicated_exception_and_leaves_no_cache(tmp_path):
    path = tmp_path / "trace.npz"
    store = make_store()

    with pytest.raises(TraceCacheCancelled, match="cancelled"):
        write_trace_store_cache(
            path,
            store,
            RUN.source_fingerprint,
            should_cancel=lambda: True,
        )
    assert not path.exists()

    write_valid_cache(path)
    calls = {"count": 0}

    def cancel_after_metadata():
        calls["count"] += 1
        return calls["count"] > 1

    with pytest.raises(TraceCacheCancelled, match="cancelled"):
        read_trace_store_cache(
            path,
            RUN.source_fingerprint,
            should_cancel=cancel_after_metadata,
        )
