import numpy as np
import pytest
from dataclasses import replace

from trace_model import (
    Coverage,
    NicIdentity,
    NodeIdentity,
    NormalizedBatch,
    QualityFlag,
    RunIdentity,
    CollectorIdentity,
    CounterMetadata,
    CounterSemantics,
    ValueKind,
)
from trace_store import (
    AmbiguousCollectorError,
    RunIdentityMismatchError,
    TraceStore,
    TraceValidationError,
)


RUN_A = RunIdentity(
    run_id="run-a",
    source_fingerprint="sha256:a",
    source_format="synthetic-ctf",
    reader_version="test-reader-1",
    conversion_lineage="direct",
    clock_domain="monotonic-test",
    synchronization_evidence="synthetic-common-clock",
    time_origin_ns=0,
)
RUN_B = RunIdentity(
    run_id="run-b",
    source_fingerprint="sha256:b",
    source_format="synthetic-ctf",
    reader_version="test-reader-1",
    conversion_lineage="direct",
    clock_domain="monotonic-test",
    synchronization_evidence="synthetic-common-clock",
    time_origin_ns=0,
)
NODE_A = NodeIdentity(10, "node-a", "node-a-original")
NODE_B = NodeIdentity(11, "node-b", "node-b-original")
NIC_A = NicIdentity(100, NODE_A.node_id, "cxi0", "cxi0-original")
NIC_B = NicIdentity(101, NODE_B.node_id, "cxi0", "cxi0-original")
UINT_COUNTER = CounterMetadata(
    1000,
    "rx_bytes",
    ValueKind.EXACT_UINT64,
    CounterSemantics.CUMULATIVE,
    "bytes",
    baseline_relative=False,
    registry_version="test-registry-1",
)
FLOAT_COUNTER = CounterMetadata(
    1001,
    "converted_rx_bytes",
    ValueKind.FLOAT64,
    CounterSemantics.CUMULATIVE,
    "bytes",
    baseline_relative=True,
    registry_version="test-registry-1",
)
RANK_0 = CollectorIdentity(1, "collector-rank-0", rank=0)
RANK_1 = CollectorIdentity(2, "collector-rank-1", rank=1)


def make_batch(
    *,
    node_ids,
    nic_ids,
    counter_ids,
    collector_ids,
    timestamps_ns,
    uint64_values=None,
    float64_values=None,
    record_ids=None,
    identity=RUN_A,
    nodes=(NODE_A, NODE_B),
    nics=(NIC_A, NIC_B),
    counters=(UINT_COUNTER, FLOAT_COUNTER),
    collectors=(RANK_0, RANK_1),
    coverage=None,
    quality_flags=None,
):
    row_count = len(timestamps_ns)
    if uint64_values is None:
        uint64_values = np.zeros(row_count, dtype=np.uint64)
    if float64_values is None:
        float64_values = np.full(row_count, np.nan, dtype=np.float64)
    if record_ids is None:
        record_ids = np.asarray([f"record-{index}" for index in range(row_count)])
    if coverage is None:
        acquisition_bounds = (
            (int(min(timestamps_ns)), int(max(timestamps_ns)))
            if row_count
            else (None, None)
        )
        coverage = Coverage(
            requested_counter_names=("rx_bytes", "converted_rx_bytes", "missing_counter"),
            available_counter_names=("rx_bytes", "converted_rx_bytes"),
            readable_counter_names=("rx_bytes", "converted_rx_bytes"),
            acquisition_start_ns=acquisition_bounds[0],
            acquisition_end_ns=acquisition_bounds[1],
        )
    if quality_flags is None:
        quality_flags = np.zeros(row_count, dtype=np.int64)
    return NormalizedBatch(
        identity=identity,
        nodes=nodes,
        nics=nics,
        counters=counters,
        collectors=collectors,
        coverage=coverage,
        node_ids=np.asarray(node_ids, dtype=np.int64),
        nic_ids=np.asarray(nic_ids, dtype=np.int64),
        counter_ids=np.asarray(counter_ids, dtype=np.int64),
        collector_ids=np.asarray(collector_ids, dtype=np.int64),
        timestamps_ns=np.asarray(timestamps_ns, dtype=np.int64),
        record_ids=np.asarray(record_ids),
        uint64_values=np.asarray(uint64_values, dtype=np.uint64),
        float64_values=np.asarray(float64_values, dtype=np.float64),
        quality_flags=np.asarray(quality_flags, dtype=np.int64),
    )


def uint_rows(values, timestamps=None, node=NODE_A, nic=NIC_A, collector=RANK_0, identity=RUN_A):
    timestamps = [10 + 10 * index for index in range(len(values))] if timestamps is None else timestamps
    return make_batch(
        identity=identity,
        node_ids=[node.node_id] * len(values),
        nic_ids=[nic.nic_id] * len(values),
        counter_ids=[UINT_COUNTER.counter_id] * len(values),
        collector_ids=[collector.collector_id] * len(values),
        timestamps_ns=timestamps,
        uint64_values=values,
        float64_values=np.full(len(values), np.nan),
    )


def test_multi_run_node_and_nic_series_remain_isolated():
    batch = make_batch(
        node_ids=[NODE_A.node_id, NODE_A.node_id, NODE_B.node_id, NODE_B.node_id],
        nic_ids=[NIC_A.nic_id, NIC_A.nic_id, NIC_B.nic_id, NIC_B.nic_id],
        counter_ids=[UINT_COUNTER.counter_id] * 4,
        collector_ids=[RANK_0.collector_id] * 4,
        timestamps_ns=[10, 20, 10, 20],
        uint64_values=[1, 2, 3, 4],
    )
    run_a = TraceStore.from_batches([batch])
    run_b = TraceStore.from_batches(
        [uint_rows([5, 6], identity=RUN_B)]
    )

    node_a = run_a.query_unique_series(NODE_A, NIC_A, UINT_COUNTER).values.tolist()
    node_b = run_a.query_unique_series(NODE_B, NIC_B, UINT_COUNTER).values.tolist()
    run_b_values = run_b.query_unique_series(NODE_A, NIC_A, UINT_COUNTER).values.tolist()

    assert node_a == [1, 2]
    assert node_b == [3, 4]
    assert run_b_values == [5, 6]
    with pytest.raises(RunIdentityMismatchError):
        TraceStore.from_batches([batch, uint_rows([9], identity=RUN_B)])


def test_duplicate_rank_observations_stay_separate():
    batch = make_batch(
        node_ids=[NODE_A.node_id] * 4,
        nic_ids=[NIC_A.nic_id] * 4,
        counter_ids=[UINT_COUNTER.counter_id] * 4,
        collector_ids=[RANK_0.collector_id, RANK_0.collector_id, RANK_1.collector_id, RANK_1.collector_id],
        timestamps_ns=[10, 20, 10, 20],
        uint64_values=[10, 20, 30, 40],
        record_ids=["rank0-a", "rank0-b", "rank1-a", "rank1-b"],
    )
    store = TraceStore.from_batches([batch])
    grouped = store.query_series(NODE_A, NIC_A, UINT_COUNTER)

    assert grouped.duplicate_observation_streams is True
    assert len(grouped.results) == 2
    assert [result.values.tolist() for result in grouped.results] == [[10, 20], [30, 40]]
    assert all(result.observation_stream_count == 2 for result in grouped.results)
    with pytest.raises(AmbiguousCollectorError):
        store.query_unique_series(NODE_A, NIC_A, UINT_COUNTER)

    rank_zero = store.query_unique_series(
        NODE_A, NIC_A, UINT_COUNTER, collector_id=RANK_0.collector_id
    )
    assert rank_zero.values.tolist() == [10, 20]


def test_exact_uint64_value_is_preserved():
    exact = 2**64 - 1
    store = TraceStore.from_batches([uint_rows([exact, exact - 1])])
    result = store.query_unique_series(NODE_A, NIC_A, UINT_COUNTER)

    assert result.values.dtype == np.uint64
    assert result.values[0] == exact
    assert result.values[1] == exact - 1
    assert not result.values.flags.writeable
    assert store.column("uint64_values").dtype == np.uint64


def test_baseline_relative_float_metadata_and_quality_flags():
    store = TraceStore.from_batches(
        [
            make_batch(
                node_ids=[NODE_A.node_id, NODE_A.node_id],
                nic_ids=[NIC_A.nic_id, NIC_A.nic_id],
                counter_ids=[FLOAT_COUNTER.counter_id, FLOAT_COUNTER.counter_id],
                collector_ids=[RANK_0.collector_id, RANK_0.collector_id],
                timestamps_ns=[10, 20],
                uint64_values=[0, 0],
                float64_values=[1.5, 2.25],
            )
        ]
    )
    result = store.query_unique_series(NODE_A, NIC_A, FLOAT_COUNTER)

    assert result.counter.baseline_relative is True
    assert result.values.dtype == np.float64
    assert np.all(result.quality_flags & QualityFlag.BASELINE_RELATIVE)
    assert np.all(result.quality_flags & QualityFlag.PRECISION_LOSS_POSSIBLE)


def test_query_window_is_inclusive_start_and_exclusive_end():
    store = TraceStore.from_batches([uint_rows([1, 2, 3])])
    result = store.query_unique_series(
        NODE_A, NIC_A, UINT_COUNTER, start_ns=10, end_ns=30
    )
    empty = store.query_unique_series(
        NODE_A, NIC_A, UINT_COUNTER, start_ns=20, end_ns=20
    )

    assert result.timestamps_ns.tolist() == [10, 20]
    assert result.represented_start_ns == 10
    assert result.represented_end_ns == 20
    assert empty.timestamps_ns.tolist() == []
    assert empty.represented_start_ns is None
    assert empty.represented_end_ns is None
    with pytest.raises(TraceValidationError):
        store.query_unique_series(NODE_A, NIC_A, UINT_COUNTER, start_ns=20, end_ns=10)


def test_reordering_is_sorted_and_quality_flagged():
    store = TraceStore.from_batches([uint_rows([2, 1], timestamps=[20, 10])])
    result = store.query_unique_series(NODE_A, NIC_A, UINT_COUNTER)

    assert result.timestamps_ns.tolist() == [10, 20]
    assert result.values.tolist() == [1, 2]
    assert np.all(result.quality_flags & QualityFlag.REORDERED)
    assert np.all(result.quality_flags & QualityFlag.NONMONOTONIC_TIMESTAMP)


def test_exact_duplicate_is_removed_with_provenance_flag():
    store = TraceStore.from_batches(
        [
            make_batch(
                node_ids=[NODE_A.node_id] * 2,
                nic_ids=[NIC_A.nic_id] * 2,
                counter_ids=[UINT_COUNTER.counter_id] * 2,
                collector_ids=[RANK_0.collector_id] * 2,
                timestamps_ns=[10, 10],
                uint64_values=[7, 7],
                record_ids=["same-record", "same-record"],
            )
        ]
    )
    result = store.query_unique_series(NODE_A, NIC_A, UINT_COUNTER)

    assert store.row_count == 1
    assert result.values.tolist() == [7]
    assert np.all(result.quality_flags & QualityFlag.DUPLICATE_REMOVED)


def test_duplicate_with_equal_values_but_conflicting_quality_is_rejected():
    with pytest.raises(
        TraceValidationError,
        match="duplicate records contain conflicting values or quality flags",
    ):
        TraceStore.from_batches(
            [
                make_batch(
                    node_ids=[NODE_A.node_id] * 2,
                    nic_ids=[NIC_A.nic_id] * 2,
                    counter_ids=[UINT_COUNTER.counter_id] * 2,
                    collector_ids=[RANK_0.collector_id] * 2,
                    timestamps_ns=[10, 10],
                    uint64_values=[7, 7],
                    record_ids=["same-record", "same-record"],
                    quality_flags=[QualityFlag.NONE, QualityFlag.REORDERED],
                )
            ]
        )


def test_vectorized_duplicate_groups_remove_one_record_per_group():
    store = TraceStore.from_batches(
        [
            make_batch(
                node_ids=[
                    NODE_A.node_id,
                    NODE_A.node_id,
                    NODE_B.node_id,
                    NODE_B.node_id,
                    NODE_A.node_id,
                ],
                nic_ids=[
                    NIC_A.nic_id,
                    NIC_A.nic_id,
                    NIC_B.nic_id,
                    NIC_B.nic_id,
                    NIC_A.nic_id,
                ],
                counter_ids=[UINT_COUNTER.counter_id] * 5,
                collector_ids=[RANK_0.collector_id] * 5,
                timestamps_ns=[10, 10, 20, 20, 30],
                uint64_values=[7, 7, 9, 9, 11],
                record_ids=["dup-a", "dup-a", "dup-b", "dup-b", "unique"],
            )
        ]
    )

    grouped_a = store.query_series(NODE_A, NIC_A, UINT_COUNTER)
    grouped_b = store.query_series(NODE_B, NIC_B, UINT_COUNTER)

    assert store.row_count == 3
    assert grouped_a.results[0].values.tolist() == [7, 11]
    assert grouped_b.results[0].values.tolist() == [9]
    assert grouped_a.results[0].quality_flags[0] & QualityFlag.DUPLICATE_REMOVED
    assert grouped_b.results[0].quality_flags[0] & QualityFlag.DUPLICATE_REMOVED


def test_vectorized_reorder_flags_are_limited_to_reordered_stream():
    store = TraceStore.from_batches(
        [
            make_batch(
                node_ids=[NODE_A.node_id] * 4,
                nic_ids=[NIC_A.nic_id] * 4,
                counter_ids=[UINT_COUNTER.counter_id] * 4,
                collector_ids=[
                    RANK_0.collector_id,
                    RANK_0.collector_id,
                    RANK_1.collector_id,
                    RANK_1.collector_id,
                ],
                timestamps_ns=[20, 10, 30, 40],
                uint64_values=[2, 1, 3, 4],
                record_ids=["rank0-b", "rank0-a", "rank1-a", "rank1-b"],
            )
        ]
    )

    grouped = store.query_series(NODE_A, NIC_A, UINT_COUNTER)
    rank_zero = next(result for result in grouped.results if result.collector is RANK_0)
    rank_one = next(result for result in grouped.results if result.collector is RANK_1)

    assert np.all(rank_zero.quality_flags & QualityFlag.REORDERED)
    assert np.all(rank_zero.quality_flags & QualityFlag.NONMONOTONIC_TIMESTAMP)
    assert not np.any(rank_one.quality_flags & QualityFlag.REORDERED)
    assert not np.any(rank_one.quality_flags & QualityFlag.NONMONOTONIC_TIMESTAMP)


def test_vectorized_record_id_reuse_is_rejected_across_trace_records():
    with pytest.raises(
        TraceValidationError,
        match="record_id is reused for different trace records",
    ):
        TraceStore.from_batches(
            [
                make_batch(
                    node_ids=[NODE_A.node_id, NODE_B.node_id],
                    nic_ids=[NIC_A.nic_id, NIC_B.nic_id],
                    counter_ids=[UINT_COUNTER.counter_id] * 2,
                    collector_ids=[RANK_0.collector_id] * 2,
                    timestamps_ns=[10, 20],
                    uint64_values=[1, 2],
                    record_ids=["same-record", "same-record"],
                )
            ]
        )


def test_conflicting_duplicate_timestamp_is_rejected():
    batch = make_batch(
        node_ids=[NODE_A.node_id] * 2,
        nic_ids=[NIC_A.nic_id] * 2,
        counter_ids=[UINT_COUNTER.counter_id] * 2,
        collector_ids=[RANK_0.collector_id] * 2,
        timestamps_ns=[10, 10],
        uint64_values=[7, 8],
        record_ids=["record-a", "record-b"],
    )
    with pytest.raises(TraceValidationError, match="duplicate timestamps"):
        TraceStore.from_batches([batch])


def test_empty_malformed_and_incompatible_batches_are_rejected():
    with pytest.raises(TraceValidationError, match="zero rows"):
        TraceStore.from_batches(
            [
                make_batch(
                    node_ids=np.array([], dtype=np.int64),
                    nic_ids=np.array([], dtype=np.int64),
                    counter_ids=np.array([], dtype=np.int64),
                    collector_ids=np.array([], dtype=np.int64),
                    timestamps_ns=np.array([], dtype=np.int64),
                    uint64_values=np.array([], dtype=np.uint64),
                    float64_values=np.array([], dtype=np.float64),
                    record_ids=np.array([], dtype=np.str_),
                )
            ]
        )

    with pytest.raises(TraceValidationError, match="equal row counts"):
        make_batch(
            node_ids=[NODE_A.node_id],
            nic_ids=[NIC_A.nic_id],
            counter_ids=[UINT_COUNTER.counter_id],
            collector_ids=[RANK_0.collector_id],
            timestamps_ns=[10, 20],
            uint64_values=[1, 2],
        )

    with pytest.raises(TraceValidationError, match="unknown ID"):
        make_batch(
            node_ids=[999],
            nic_ids=[NIC_A.nic_id],
            counter_ids=[UINT_COUNTER.counter_id],
            collector_ids=[RANK_0.collector_id],
            timestamps_ns=[10],
            uint64_values=[1],
        )

    with pytest.raises(TraceValidationError, match="dtype"):
        replace(
            make_batch(
            node_ids=[NODE_A.node_id],
            nic_ids=[NIC_A.nic_id],
            counter_ids=[UINT_COUNTER.counter_id],
            collector_ids=[RANK_0.collector_id],
                timestamps_ns=[10],
                uint64_values=[1],
            ),
            timestamps_ns=np.asarray([10], dtype=np.float64),
        )

    with pytest.raises(TraceValidationError, match="float64_values"):
        make_batch(
            node_ids=[NODE_A.node_id],
            nic_ids=[NIC_A.nic_id],
            counter_ids=[UINT_COUNTER.counter_id],
            collector_ids=[RANK_0.collector_id],
            timestamps_ns=[10],
            uint64_values=[1],
            float64_values=[0.0],
        )

    with pytest.raises(TraceValidationError, match="uint64_values"):
        make_batch(
            node_ids=[NODE_A.node_id],
            nic_ids=[NIC_A.nic_id],
            counter_ids=[FLOAT_COUNTER.counter_id],
            collector_ids=[RANK_0.collector_id],
            timestamps_ns=[10],
            uint64_values=[1],
            float64_values=[1.5],
        )


def test_flat_columns_do_not_allocate_dense_cartesian_product():
    store = TraceStore.from_batches(
        [
            make_batch(
                node_ids=[NODE_A.node_id, NODE_B.node_id, NODE_A.node_id],
                nic_ids=[NIC_A.nic_id, NIC_B.nic_id, NIC_A.nic_id],
                counter_ids=[
                    UINT_COUNTER.counter_id,
                    FLOAT_COUNTER.counter_id,
                    UINT_COUNTER.counter_id,
                ],
                collector_ids=[RANK_0.collector_id] * 3,
                timestamps_ns=[10, 20, 30],
                uint64_values=[1, 0, 3],
                float64_values=[np.nan, 2.5, np.nan],
            )
        ]
    )

    assert store.row_count == 3
    assert all(store.column(name).shape == (3,) for name in (
        "node_ids",
        "nic_ids",
        "counter_ids",
        "collector_ids",
        "timestamps_ns",
        "uint64_values",
        "float64_values",
        "quality_flags",
    ))


def test_coverage_exposes_requested_and_missing_counters():
    store = TraceStore.from_batches([uint_rows([1, 2])])

    assert store.coverage.observed_counter_names == ("rx_bytes",)
    assert store.coverage.requested_counter_names == (
        "rx_bytes",
        "converted_rx_bytes",
        "missing_counter",
    )
    assert store.coverage.missing_counter_names == (
        "converted_rx_bytes",
        "missing_counter",
    )
