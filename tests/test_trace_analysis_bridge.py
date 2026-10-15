import numpy as np
import pytest

from trace_analysis_bridge import (
    infer_traffic_direction,
    metric_series_from_series_result,
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
from traffic_analysis import SeriesQualityFlag, TrafficDirection


RUN = RunIdentity(
    run_id="run-1",
    source_fingerprint="sha256:test",
    source_format="synthetic",
    reader_version="test",
    conversion_lineage="direct",
    clock_domain="test-clock",
    synchronization_evidence="test",
)
NODE = NodeIdentity(1, "node-1", "node-1")
NIC = NicIdentity(2, NODE.node_id, "cxi0", "cxi0")
COLLECTOR = CollectorIdentity(3, "collector", rank=0)


def make_store(counter: CounterMetadata, values, *, float_values=None):
    row_count = len(values)
    quality = np.zeros(row_count, dtype=np.int64)
    if counter.baseline_relative:
        quality |= 1 << 3
    batch = NormalizedBatch(
        identity=RUN,
        nodes=(NODE,),
        nics=(NIC,),
        counters=(counter,),
        collectors=(COLLECTOR,),
        coverage=Coverage(
            requested_counter_names=(counter.name,),
            available_counter_names=(counter.name,),
            readable_counter_names=(counter.name,),
        ),
        node_ids=np.full(row_count, NODE.node_id, dtype=np.int64),
        nic_ids=np.full(row_count, NIC.nic_id, dtype=np.int64),
        counter_ids=np.full(row_count, counter.counter_id, dtype=np.int64),
        collector_ids=np.full(row_count, COLLECTOR.collector_id, dtype=np.int64),
        timestamps_ns=np.arange(row_count, dtype=np.int64) * 1_000_000_000,
        record_ids=np.asarray([f"record-{index}" for index in range(row_count)]),
        uint64_values=(
            np.asarray(values, dtype=np.uint64)
            if counter.value_kind is ValueKind.EXACT_UINT64
            else np.zeros(row_count, dtype=np.uint64)
        ),
        float64_values=(
            np.asarray(values, dtype=np.float64)
            if counter.value_kind is ValueKind.FLOAT64
            else np.full(row_count, np.nan, dtype=np.float64)
        ),
        quality_flags=quality,
    )
    return TraceStore.from_batches([batch])


def test_bridge_preserves_identity_direction_and_exact_values():
    counter = CounterMetadata(
        10,
        "tx_bytes",
        ValueKind.EXACT_UINT64,
        CounterSemantics.CUMULATIVE,
        "bytes",
        baseline_relative=False,
        registry_version="test",
    )
    result = make_store(counter, [0, 100, 300]).query_unique_series(
        NODE, NIC, counter
    )
    series = metric_series_from_series_result(result, run_id=RUN.run_id)

    assert series.identity.direction is TrafficDirection.TX
    assert series.identity.run_id == RUN.run_id
    assert series.identity.node_id == NODE.label
    assert series.identity.nic_id == NIC.label
    assert series.identity.collector_id == COLLECTOR.name
    np.testing.assert_array_equal(series.values, np.asarray([0, 100, 300], dtype=np.uint64))
    assert series.values.dtype == np.dtype(np.uint64)
    assert not series.values.flags.writeable


def test_bridge_uses_result_run_and_rejects_mismatched_caller_run():
    counter = CounterMetadata(
        12,
        "tx_bytes",
        ValueKind.EXACT_UINT64,
        CounterSemantics.CUMULATIVE,
        "bytes",
        baseline_relative=False,
        registry_version="test",
    )
    result = make_store(counter, [0, 10]).query_unique_series(
        NODE, NIC, counter
    )

    assert result.run_identity == RUN
    series = metric_series_from_series_result(result)
    assert series.identity.run_id == RUN.run_id

    with pytest.raises(ValueError, match="does not match"):
        metric_series_from_series_result(result, run_id="not-this-run")


def test_bridge_preserves_baseline_and_precision_quality():
    counter = CounterMetadata(
        11,
        "rx_bytes",
        ValueKind.FLOAT64,
        CounterSemantics.CUMULATIVE,
        "bytes",
        baseline_relative=True,
        registry_version="test",
        quality_flags=1 << 4,
    )
    result = make_store(counter, [0.0, 10.0]).query_unique_series(
        NODE, NIC, counter
    )
    series = metric_series_from_series_result(result, run_id=RUN.run_id)

    assert series.identity.direction is TrafficDirection.RX
    assert series.baseline_relative is True
    assert SeriesQualityFlag.BASELINE_RELATIVE in series.quality_flags
    assert SeriesQualityFlag.PRECISION_LOSS_POSSIBLE in series.quality_flags


def test_direction_must_be_inferrable_or_explicit():
    with pytest.raises(ValueError, match="cannot infer"):
        infer_traffic_direction("good_codewords")
