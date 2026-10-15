"""Deterministic synthetic data for analysis application tests and browser capture."""

from __future__ import annotations

import numpy as np

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
from counter_semantic_registry import COUNTER_SEMANTIC_REGISTRY_VERSION
from trace_store import TraceStore


RUN = RunIdentity(
    run_id="run-synthetic",
    source_fingerprint="sha256:synthetic",
    source_format="synthetic-ctf",
    reader_version="test-reader",
    conversion_lineage="direct",
    clock_domain="monotonic-test",
    synchronization_evidence="single-clock",
)
NODE_A = NodeIdentity(1, "node-a", "node-a")
NODE_B = NodeIdentity(2, "node-b", "node-b")
NIC_A = NicIdentity(10, NODE_A.node_id, "cxi-0", "cxi-0")
NIC_B = NicIdentity(11, NODE_B.node_id, "cxi-0", "cxi-0")
COLLECTOR = CollectorIdentity(20, "collector", rank=0)
TX_COUNTER = CounterMetadata(
    30,
    "hni_sts_tx_ok_octets",
    ValueKind.EXACT_UINT64,
    CounterSemantics.CUMULATIVE,
    "bytes",
    baseline_relative=False,
    registry_version=COUNTER_SEMANTIC_REGISTRY_VERSION,
)
RX_COUNTER = CounterMetadata(
    31,
    "hni_sts_rx_ok_octets",
    ValueKind.EXACT_UINT64,
    CounterSemantics.CUMULATIVE,
    "bytes",
    baseline_relative=False,
    registry_version=COUNTER_SEMANTIC_REGISTRY_VERSION,
)


def make_store(*, include_rx: bool = True, sample_count: int = 600) -> TraceStore:
    """Build a small two-node, two-NIC store with exact cumulative TX/RX samples."""

    counters = (TX_COUNTER, RX_COUNTER) if include_rx else (TX_COUNTER,)
    rows = []
    node_nics = ((NODE_A, NIC_A), (NODE_B, NIC_B))
    for node, nic in node_nics:
        for index in range(sample_count):
            timestamp_ns = index * 100_000_000
            rows.append((node, nic, TX_COUNTER, timestamp_ns, index * 1000))
            if include_rx:
                rows.append((node, nic, RX_COUNTER, timestamp_ns, 0))
    row_count = len(rows)
    requested = ("hni_sts_tx_ok_octets", "hni_sts_rx_ok_octets")
    available = requested if include_rx else requested[:1]
    batch = NormalizedBatch(
        identity=RUN,
        nodes=(NODE_A, NODE_B),
        nics=(NIC_A, NIC_B),
        counters=counters,
        collectors=(COLLECTOR,),
        coverage=Coverage(
            requested_counter_names=requested,
            available_counter_names=available,
            readable_counter_names=available,
            observed_counter_names=tuple(counter.name for counter in counters),
            acquisition_start_ns=0,
            acquisition_end_ns=(sample_count - 1) * 100_000_000,
            lost_record_count=0,
            gap_count=0,
        ),
        node_ids=np.asarray([row[0].node_id for row in rows], dtype=np.int64),
        nic_ids=np.asarray([row[1].nic_id for row in rows], dtype=np.int64),
        counter_ids=np.asarray([row[2].counter_id for row in rows], dtype=np.int64),
        collector_ids=np.full(row_count, COLLECTOR.collector_id, dtype=np.int64),
        timestamps_ns=np.asarray([row[3] for row in rows], dtype=np.int64),
        record_ids=np.asarray([f"record-{index}" for index in range(row_count)]),
        uint64_values=np.asarray([row[4] for row in rows], dtype=np.uint64),
        float64_values=np.full(row_count, np.nan, dtype=np.float64),
        quality_flags=np.zeros(row_count, dtype=np.int64),
    )
    return TraceStore.from_batches([batch])
