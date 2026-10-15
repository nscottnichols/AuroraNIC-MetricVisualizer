#!/usr/bin/env python3
"""Bounded legacy-dense versus columnar TraceStore benchmark.

The benchmark uses one deterministic logical dataset for both paths.  The query
and analysis stages therefore compare data-model cost on the same series; the
load stages use different input representations and are not a pure parser speedup.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import resource
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = APP_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metrics_processing import prepare_metric_array, process_all_jobs  # noqa: E402
from trace_model import (  # noqa: E402
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
from trace_store import TraceStore  # noqa: E402


NICS_PER_NODE = 8
COUNTER_COUNT = 2
SAMPLE_INTERVAL_NS = 1_000_000_000
RUN_ID = "legacy-columnar-benchmark"
TX_COUNTER = CounterMetadata(
    1,
    "hni_sts_tx_ok_octets",
    ValueKind.EXACT_UINT64,
    CounterSemantics.CUMULATIVE,
    "bytes",
    baseline_relative=False,
    registry_version="benchmark-v1",
)
RX_COUNTER = CounterMetadata(
    2,
    "hni_sts_rx_ok_octets",
    ValueKind.EXACT_UINT64,
    CounterSemantics.CUMULATIVE,
    "bytes",
    baseline_relative=False,
    registry_version="benchmark-v1",
)
COUNTERS = (TX_COUNTER, RX_COUNTER)
COLLECTOR = CollectorIdentity(1, "benchmark-collector", rank=0)
RUN_IDENTITY = RunIdentity(
    run_id=RUN_ID,
    source_fingerprint="sha256:legacy-columnar-benchmark-v1",
    source_format="synthetic-equivalent-dataset",
    reader_version="legacy-columnar-benchmark-1",
    conversion_lineage="same logical synthetic dataset",
    clock_domain="synthetic-monotonic-ns",
    synchronization_evidence="single synthetic clock",
    time_origin_ns=0,
)


class BenchmarkError(RuntimeError):
    """Raised when benchmark invariants or resource guards fail."""


T = TypeVar("T")


@dataclass(frozen=True)
class BenchmarkConfig:
    node_count: int
    repetitions: int
    warmup_repetitions: int
    max_runtime_seconds: float
    max_rss_mib: int


@dataclass(frozen=True)
class LogicalFixture:
    node_count: int
    series_count: int
    before_values: np.ndarray
    after_values: np.ndarray
    increments: np.ndarray
    nodes: tuple[NodeIdentity, ...]
    nics: tuple[NicIdentity, ...]
    batch: NormalizedBatch


@dataclass(frozen=True)
class LegacyResult:
    dense_array: np.ndarray
    node_counts: list[int]
    element_counts: list[int]


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare legacy text/dense normalization and query with an equivalent "
            "columnar TraceStore path."
        )
    )
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-runtime-seconds", type=float, default=120.0)
    parser.add_argument("--max-rss-mib", type=int, default=4096)
    args = parser.parse_args(argv)
    if args.nodes < 4:
        parser.error("--nodes must be at least 4")
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.max_runtime_seconds <= 0:
        parser.error("--max-runtime-seconds must be positive")
    if args.max_rss_mib <= 0:
        parser.error("--max-rss-mib must be positive")
    return BenchmarkConfig(
        node_count=args.nodes,
        repetitions=args.repetitions,
        warmup_repetitions=args.warmup,
        max_runtime_seconds=args.max_runtime_seconds,
        max_rss_mib=args.max_rss_mib,
    )


def build_logical_fixture(node_count: int) -> LogicalFixture:
    series_count = node_count * NICS_PER_NODE * COUNTER_COUNT
    node_label_width = max(3, len(str(node_count - 1)))
    before_values = np.arange(series_count, dtype=np.uint64).reshape(
        node_count, NICS_PER_NODE, COUNTER_COUNT
    )
    increments = np.arange(1, series_count + 1, dtype=np.uint64).reshape(
        node_count, NICS_PER_NODE, COUNTER_COUNT
    )
    after_values = before_values + increments

    nodes = tuple(
        NodeIdentity(
            index,
            f"node-{index:0{node_label_width}d}",
            f"node-{index:0{node_label_width}d}",
        )
        for index in range(node_count)
    )
    nics = tuple(
        NicIdentity(
            nic_index,
            nodes[node_index].node_id,
            f"cxi{nic_offset}",
            f"cxi{nic_offset}",
        )
        for node_index in range(node_count)
        for nic_offset in range(NICS_PER_NODE)
        for nic_index in [node_index * NICS_PER_NODE + nic_offset]
    )

    row_count = series_count * 2
    node_ids = np.empty(row_count, dtype=np.int64)
    nic_ids = np.empty(row_count, dtype=np.int64)
    counter_ids = np.empty(row_count, dtype=np.int64)
    timestamps_ns = np.empty(row_count, dtype=np.int64)
    uint64_values = np.empty(row_count, dtype=np.uint64)
    record_ids = np.empty(row_count, dtype=object)
    row_index = 0
    for node_index in range(node_count):
        for nic_offset in range(NICS_PER_NODE):
            nic_index = node_index * NICS_PER_NODE + nic_offset
            for counter_index, counter in enumerate(COUNTERS):
                for sample_index, timestamp in enumerate((0, SAMPLE_INTERVAL_NS)):
                    node_ids[row_index] = node_index
                    nic_ids[row_index] = nic_index
                    counter_ids[row_index] = counter.counter_id
                    timestamps_ns[row_index] = timestamp
                    uint64_values[row_index] = (
                        before_values[node_index, nic_offset, counter_index]
                        if sample_index == 0
                        else after_values[node_index, nic_offset, counter_index]
                    )
                    record_ids[row_index] = f"row-{row_index:08d}"
                    row_index += 1

    batch = NormalizedBatch(
        identity=RUN_IDENTITY,
        nodes=nodes,
        nics=nics,
        counters=COUNTERS,
        collectors=(COLLECTOR,),
        coverage=Coverage(
            requested_counter_names=(TX_COUNTER.name, RX_COUNTER.name),
            available_counter_names=(TX_COUNTER.name, RX_COUNTER.name),
            readable_counter_names=(TX_COUNTER.name, RX_COUNTER.name),
            observed_counter_names=(TX_COUNTER.name, RX_COUNTER.name),
            acquisition_start_ns=0,
            acquisition_end_ns=SAMPLE_INTERVAL_NS,
        ),
        node_ids=node_ids,
        nic_ids=nic_ids,
        counter_ids=counter_ids,
        collector_ids=np.full(row_count, COLLECTOR.collector_id, dtype=np.int64),
        timestamps_ns=timestamps_ns,
        record_ids=record_ids,
        uint64_values=uint64_values,
        float64_values=np.full(row_count, np.nan, dtype=np.float64),
        quality_flags=np.zeros(row_count, dtype=np.int64),
    )
    return LogicalFixture(
        node_count=node_count,
        series_count=series_count,
        before_values=before_values,
        after_values=after_values,
        increments=increments,
        nodes=nodes,
        nics=nics,
        batch=batch,
    )


def write_legacy_fixture(root: Path, fixture: LogicalFixture) -> None:
    job_dir = root / RUN_ID
    metrics_dir = job_dir / f"metrics_{RUN_ID}"
    pair_dir = metrics_dir / "1_1"
    pair_dir.mkdir(parents=True, exist_ok=True)

    for node_index in range(fixture.node_count):
        node_label = fixture.nodes[node_index].label
        before_path = pair_dir / f"metric_before.{node_label}"
        after_path = pair_dir / f"metric_after.{node_label}"
        before_lines = []
        after_lines = []
        for nic_offset in range(NICS_PER_NODE):
            for counter_index, counter in enumerate(COUNTERS):
                label = (
                    f"{RUN_ID}.{node_label}.cxi{nic_offset}.{counter.name}"
                )
                before_lines.append(
                    f"{fixture.before_values[node_index, nic_offset, counter_index]}@{label}"
                )
                after_lines.append(
                    f"{fixture.after_values[node_index, nic_offset, counter_index]}@{label}"
                )
        before_path.write_text("\n".join(before_lines) + "\n", encoding="utf-8")
        after_path.write_text("\n".join(after_lines) + "\n", encoding="utf-8")


def load_legacy_fixture(root: Path, fixture: LogicalFixture) -> LegacyResult:
    node_map, results, results_nodes = process_all_jobs(root)
    dense_array, node_counts, element_counts = prepare_metric_array(
        results,
        NICS_PER_NODE,
        fixture.node_count,
        results_nodes,
    )
    return LegacyResult(
        dense_array=dense_array,
        node_counts=node_counts,
        element_counts=element_counts,
    )


def build_columnar_store(fixture: LogicalFixture) -> TraceStore:
    return TraceStore.from_batches([fixture.batch])


def expected_interface_aggregate(fixture: LogicalFixture) -> dict[str, list[int | float]]:
    totals = fixture.increments.sum(axis=(0, 2), dtype=np.int64)
    counts = np.full(
        NICS_PER_NODE, fixture.node_count * COUNTER_COUNT, dtype=np.int64
    )
    minimums = fixture.increments.min(axis=(0, 2))
    maximums = fixture.increments.max(axis=(0, 2))
    means = totals / counts
    return {
        "totals": [int(value) for value in totals],
        "counts": [int(value) for value in counts],
        "minimums": [int(value) for value in minimums],
        "maximums": [int(value) for value in maximums],
        "means": [float(value) for value in means],
    }


def legacy_whole_dataset_aggregation(legacy: LegacyResult) -> dict[str, list[int | float]]:
    totals = []
    counts = []
    minimums = []
    maximums = []
    means = []

    for interface_index in range(NICS_PER_NODE):
        values = []
        for counter_index in range(COUNTER_COUNT):
            for node_index in range(legacy.dense_array.shape[-1]):
                values.append(
                    int(
                        legacy.dense_array[
                            interface_index,
                            counter_index,
                            0,
                            0,
                            node_index,
                        ]
                    )
                )
        total = sum(values)
        totals.append(total)
        counts.append(len(values))
        minimums.append(min(values))
        maximums.append(max(values))
        means.append(total / len(values))

    return {
        "totals": totals,
        "counts": counts,
        "minimums": minimums,
        "maximums": maximums,
        "means": means,
    }


def columnar_whole_dataset_aggregation(store: TraceStore) -> dict[str, list[int | float]]:
    if store.row_count % 2 != 0:
        raise BenchmarkError("columnar store does not contain paired samples")

    values = store.column("uint64_values").reshape(-1, 2)
    deltas = values[:, 1].astype(np.int64) - values[:, 0].astype(np.int64)
    interface_ids = (
        store.column("nic_ids").reshape(-1, 2)[:, 0] % NICS_PER_NODE
    ).astype(np.int64)

    totals = np.zeros(NICS_PER_NODE, dtype=np.int64)
    counts = np.zeros(NICS_PER_NODE, dtype=np.int64)
    minimums = np.full(NICS_PER_NODE, np.iinfo(np.int64).max, dtype=np.int64)
    maximums = np.full(NICS_PER_NODE, np.iinfo(np.int64).min, dtype=np.int64)

    np.add.at(totals, interface_ids, deltas)
    np.add.at(counts, interface_ids, 1)
    np.minimum.at(minimums, interface_ids, deltas)
    np.maximum.at(maximums, interface_ids, deltas)
    means = totals / counts

    return {
        "totals": [int(value) for value in totals],
        "counts": [int(value) for value in counts],
        "minimums": [int(value) for value in minimums],
        "maximums": [int(value) for value in maximums],
        "means": [float(value) for value in means],
    }


def aggregates_match(
    left: dict[str, list[int | float]], right: dict[str, list[int | float]]
) -> bool:
    return all(left[key] == right[key] for key in left)


def _require_matching_aggregate(
    result: dict[str, list[int | float]],
    expected: dict[str, list[int | float]],
) -> None:
    if not aggregates_match(result, expected):
        raise BenchmarkError("aggregate result does not match the logical dataset")


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def measure_stage(
    action: Callable[[], T],
    config: BenchmarkConfig,
    validate: Callable[[T], None] | None = None,
) -> dict[str, int | float]:
    for _ in range(config.warmup_repetitions):
        result = action()
        if validate is not None:
            validate(result)

    samples_ms = []
    for _ in range(config.repetitions):
        started_ns = time.perf_counter_ns()
        result = action()
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        if validate is not None:
            validate(result)
        samples_ms.append(elapsed_ms)

    return {
        "repetitions": config.repetitions,
        "median_ms": round(statistics.median(samples_ms), 3),
        "p95_ms": round(percentile(samples_ms, 0.95), 3),
        "min_ms": round(min(samples_ms), 3),
        "max_ms": round(max(samples_ms), 3),
    }


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_benchmark(config: BenchmarkConfig) -> dict[str, object]:
    benchmark_started = time.perf_counter()
    fixture = build_logical_fixture(config.node_count)
    expected_aggregate = expected_interface_aggregate(fixture)

    with tempfile.TemporaryDirectory(prefix="legacy-columnar-benchmark-") as temporary:
        root = Path(temporary)
        write_legacy_fixture(root, fixture)
        legacy = load_legacy_fixture(root, fixture)
        store = build_columnar_store(fixture)

        legacy_load_summary = measure_stage(
            lambda: load_legacy_fixture(root, fixture),
            config,
            lambda result: None,
        )
        columnar_store_summary = measure_stage(
            lambda: build_columnar_store(fixture),
            config,
            lambda result: None,
        )
        legacy_aggregate_summary = measure_stage(
            lambda: legacy_whole_dataset_aggregation(legacy),
            config,
            lambda result: _require_matching_aggregate(result, expected_aggregate),
        )
        columnar_aggregate_summary = measure_stage(
            lambda: columnar_whole_dataset_aggregation(store),
            config,
            lambda result: _require_matching_aggregate(result, expected_aggregate),
        )

        legacy_aggregate = legacy_whole_dataset_aggregation(legacy)
        columnar_aggregate = columnar_whole_dataset_aggregation(store)

        correctness = {
            "legacy_dense_shape_matches": legacy.dense_array.shape
            == (NICS_PER_NODE, COUNTER_COUNT, 1, 1, config.node_count),
            "legacy_node_axis_matches": legacy.node_counts == [1]
            and legacy.element_counts == [1],
            "columnar_row_count_matches": store.row_count
            == fixture.series_count * 2,
            "legacy_aggregate_matches_expected": aggregates_match(
                legacy_aggregate, expected_aggregate
            ),
            "columnar_aggregate_matches_expected": aggregates_match(
                columnar_aggregate, expected_aggregate
            ),
            "legacy_and_columnar_aggregates_match": aggregates_match(
                legacy_aggregate, columnar_aggregate
            ),
        }

        if not all(correctness.values()):
            failed = [name for name, passed in correctness.items() if not passed]
            raise BenchmarkError(f"benchmark correctness failed: {failed}")

        elapsed_seconds = time.perf_counter() - benchmark_started
        peak = peak_rss_mib()
        if elapsed_seconds > config.max_runtime_seconds:
            raise BenchmarkError(
                f"benchmark exceeded runtime guard: {elapsed_seconds:.3f}s > "
                f"{config.max_runtime_seconds:.3f}s"
            )
        if peak > config.max_rss_mib:
            raise BenchmarkError(
                f"benchmark exceeded RSS guard: {peak:.3f} MiB > "
                f"{config.max_rss_mib} MiB"
            )

        gc.collect()
        return {
            "schema_version": 1,
            "benchmark": "legacy_dense_vs_columnar_trace_store",
            "status": "ok",
            "configuration": {
                "node_count": config.node_count,
                "repetitions": config.repetitions,
                "warmup_repetitions": config.warmup_repetitions,
                "max_runtime_seconds": config.max_runtime_seconds,
                "max_rss_mib": config.max_rss_mib,
            },
            "dataset": {
                "logical_series_count": fixture.series_count,
                "legacy_dense_shape": list(legacy.dense_array.shape),
                "legacy_dense_nbytes": int(legacy.dense_array.nbytes),
                "columnar_row_count": store.row_count,
                "columnar_sample_count_per_series": 2,
                "same_logical_dataset": True,
                "comparison_scope": (
                    "Aggregation stages use the same whole logical dataset; "
                    "load stages use different input representations and are not "
                    "a pure text-parsing comparison."
                ),
            },
            "measurements": {
                "legacy_text_load_and_dense_normalization": legacy_load_summary,
                "legacy_dense_whole_dataset_aggregation": legacy_aggregate_summary,
                "columnar_store_construction": columnar_store_summary,
                "columnar_whole_dataset_aggregation": columnar_aggregate_summary,
            },
            "aggregate": {
                **columnar_aggregate,
                "legacy_matches_columnar": True,
            },
            "correctness": correctness,
            "resources": {
                "peak_rss_mib": round(peak, 3),
                "elapsed_seconds": round(elapsed_seconds, 3),
            },
        }


def main(argv=None):
    config = parse_arguments(argv)
    result = run_benchmark(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
