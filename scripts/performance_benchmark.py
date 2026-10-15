#!/usr/bin/env python3
"""Reproducible local synthetic benchmark for the columnar analysis path.

This benchmark intentionally measures the existing normalized batch, indexed
store, bridge/traffic analysis, and bounded view-model path.  It does not
start the Dash server and does not measure real-trace or browser performance.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = APP_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis_app import AnalysisFilterState, build_view_model  # noqa: E402
from trace_analysis_bridge import metric_series_from_series_result  # noqa: E402
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
from trace_cache import read_trace_store_cache, write_trace_store_cache  # noqa: E402
from traffic_analysis import AnalysisStatus, analyze_traffic_series  # noqa: E402


NODE_COUNT = 4
NICS_PER_NODE = 8
COUNTER_COUNT = 2
SERIES_COUNT = NODE_COUNT * NICS_PER_NODE * COUNTER_COUNT
SAMPLE_INTERVAL_NS = 1_000_000
TX_INCREMENT_BYTES = 1_000
RX_INCREMENT_BYTES = 600
CONCURRENT_VIEW_MODEL_WORKERS = 4
CONCURRENT_VIEW_MODEL_REPETITIONS = 4

RUN_IDENTITY = RunIdentity(
    run_id="synthetic-local",
    source_fingerprint="sha256:synthetic-local-v1",
    source_format="synthetic-normalized",
    reader_version="performance-benchmark-1",
    conversion_lineage="deterministic-in-memory-fixture",
    clock_domain="synthetic-monotonic-ns",
    synchronization_evidence="single synthetic clock",
    time_origin_ns=0,
)
COLLECTOR = CollectorIdentity(1, "synthetic-collector", rank=0)
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


class BenchmarkError(RuntimeError):
    """Raised with the stage and remediation for a benchmark failure."""


T = TypeVar("T")


@dataclass(frozen=True)
class BenchmarkConfig:
    requested_samples: int
    repetitions: int
    warmup_repetitions: int
    max_runtime_seconds: float
    max_rss_mib: int


@dataclass(frozen=True)
class SyntheticFixture:
    nodes: tuple[NodeIdentity, ...]
    nics: tuple[NicIdentity, ...]
    counters: tuple[CounterMetadata, ...]
    samples_per_series: int
    row_count: int
    node_ids: np.ndarray
    nic_ids: np.ndarray
    counter_ids: np.ndarray
    collector_ids: np.ndarray
    timestamps_ns: np.ndarray
    record_ids: np.ndarray
    uint64_values: np.ndarray
    float64_values: np.ndarray
    quality_flags: np.ndarray
    acquisition_end_ns: int

    def batch(self) -> NormalizedBatch:
        return NormalizedBatch(
            identity=RUN_IDENTITY,
            nodes=self.nodes,
            nics=self.nics,
            counters=self.counters,
            collectors=(COLLECTOR,),
            coverage=Coverage(
                requested_counter_names=(TX_COUNTER.name, RX_COUNTER.name),
                available_counter_names=(TX_COUNTER.name, RX_COUNTER.name),
                readable_counter_names=(TX_COUNTER.name, RX_COUNTER.name),
                observed_counter_names=(TX_COUNTER.name, RX_COUNTER.name),
                acquisition_start_ns=0,
                acquisition_end_ns=self.acquisition_end_ns,
                lost_record_count=0,
                gap_count=0,
                notes=("Deterministic local synthetic fixture",),
            ),
            node_ids=self.node_ids,
            nic_ids=self.nic_ids,
            counter_ids=self.counter_ids,
            collector_ids=self.collector_ids,
            timestamps_ns=self.timestamps_ns,
            record_ids=self.record_ids,
            uint64_values=self.uint64_values,
            float64_values=self.float64_values,
            quality_flags=self.quality_flags,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the local synthetic columnar store/query/analysis/UI path."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100_000,
        help="target counter observations; rounded down to a whole fixed query mix (default: 100000)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="timed repetitions for simple stages; query/view mixes cycle within each (default: 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="warm-up repetitions or full query/view-mix cycles (default: 2)",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=30.0,
        help="wall-clock guard; checked before and after each operation (default: 30)",
    )
    parser.add_argument(
        "--max-rss-mib",
        type=int,
        default=4096,
        help="process peak-RSS guard in MiB (default: 4096)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON result path; parent directory must already exist",
    )
    args = parser.parse_args()
    if args.samples < SERIES_COUNT:
        parser.error(f"--samples must be at least {SERIES_COUNT} for this fixture")
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.max_runtime_seconds <= 0:
        parser.error("--max-runtime-seconds must be positive")
    if args.max_rss_mib <= 0:
        parser.error("--max-rss-mib must be positive")
    return args


def build_fixture(requested_samples: int) -> SyntheticFixture:
    samples_per_series = requested_samples // SERIES_COUNT
    if samples_per_series < 1:
        raise BenchmarkError(
            f"requested sample count {requested_samples} leaves fewer than one sample "
            f"per fixed series ({SERIES_COUNT} series)"
        )

    nodes = tuple(
        NodeIdentity(node_id, f"node-{node_id}", f"node-{node_id}-original")
        for node_id in range(NODE_COUNT)
    )
    nics = tuple(
        NicIdentity(
            node_id * NICS_PER_NODE + nic_index,
            node_id,
            f"node-{node_id}/cxi-{nic_index}",
            f"node-{node_id}/cxi-{nic_index}-original",
        )
        for node_id in range(NODE_COUNT)
        for nic_index in range(NICS_PER_NODE)
    )
    counters = (TX_COUNTER, RX_COUNTER)

    row_count = SERIES_COUNT * samples_per_series
    series_ids = np.repeat(np.arange(SERIES_COUNT, dtype=np.int64), samples_per_series)
    node_index = series_ids // (NICS_PER_NODE * COUNTER_COUNT)
    nic_index = (series_ids // COUNTER_COUNT) % (NODE_COUNT * NICS_PER_NODE)
    counter_index = series_ids % COUNTER_COUNT
    sample_index = np.tile(np.arange(samples_per_series, dtype=np.int64), SERIES_COUNT)

    timestamps_ns = sample_index * SAMPLE_INTERVAL_NS
    increments = np.where(
        counter_index == 0,
        TX_INCREMENT_BYTES + nic_index * 17 + node_index * 101,
        RX_INCREMENT_BYTES + nic_index * 23 + node_index * 103,
    ).astype(np.uint64)
    cumulative_values = increments * (sample_index + 1).astype(np.uint64)
    record_ids = np.asarray(
        [
            f"r{int(series_id):02x}{int(sample_id):05x}"
            for series_id, sample_id in zip(series_ids, sample_index, strict=True)
        ],
        dtype="S8",
    )

    return SyntheticFixture(
        nodes=nodes,
        nics=nics,
        counters=counters,
        samples_per_series=samples_per_series,
        row_count=row_count,
        node_ids=node_index.astype(np.int64),
        nic_ids=nic_index.astype(np.int64),
        counter_ids=(counter_index + 1).astype(np.int64),
        collector_ids=np.full(row_count, COLLECTOR.collector_id, dtype=np.int64),
        timestamps_ns=timestamps_ns,
        record_ids=record_ids,
        uint64_values=cumulative_values,
        float64_values=np.full(row_count, np.nan, dtype=np.float64),
        quality_flags=np.zeros(row_count, dtype=np.int64),
        acquisition_end_ns=(samples_per_series - 1) * SAMPLE_INTERVAL_NS,
    )


def _percentile_95(values: list[float]) -> float:
    if not values:
        raise BenchmarkError("cannot summarize an empty timing sample")
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95, method="linear"))


def _timing_summary(values: list[float]) -> dict[str, object]:
    return {
        "raw_ms": [round(value, 6) for value in values],
        "median_ms": round(statistics.median(values), 6),
        "p95_ms": round(_percentile_95(values), 6),
        "timed_operations": len(values),
    }


def _run_concurrent_view_model_stage(
    store: TraceStore,
    view_filters: tuple[tuple[str, AnalysisFilterState], ...],
    fixture: SyntheticFixture,
) -> dict[str, object]:
    requests = tuple(
        (name, filters)
        for _ in range(CONCURRENT_VIEW_MODEL_REPETITIONS)
        for name, filters in view_filters
    )
    futures: list[concurrent.futures.Future[dict[str, object]]] = []
    submitted_at_ns: list[int] = []

    def build_request(
        request_index: int, filters: AnalysisFilterState
    ) -> tuple[int, int, dict[str, object]]:
        view = build_view_model(store, filters)
        worker_completed_at_ns = time.perf_counter_ns()
        return request_index, worker_completed_at_ns, view

    stage_started_ns = time.perf_counter_ns()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=CONCURRENT_VIEW_MODEL_WORKERS,
        thread_name_prefix="benchmark-view-model",
    ) as executor:
        for request_index, (name, filters) in enumerate(requests):
            submitted_at_ns.append(time.perf_counter_ns())
            futures.append(executor.submit(build_request, request_index, filters))
        results: list[dict[str, object] | None] = [None] * len(requests)
        errors: list[dict[str, object]] = []
        for future in concurrent.futures.as_completed(futures):
            request_index = futures.index(future)
            try:
                request_index, worker_completed_at_ns, view = future.result()
            except Exception as error:
                errors.append(
                    {
                        "request_index": request_index,
                        "request_name": requests[request_index][0],
                        "error": str(error),
                    }
                )
                continue
            name, filters = requests[request_index]
            expected_tx, expected_rx = _expected_totals(
                fixture, filters.node_id, filters.nic_id
            )
            if view["overview"]["traffic_state"] != "ready":
                raise BenchmarkError(
                    f"concurrent {name} view state was "
                    f"{view['overview']['traffic_state']!r}"
                )
            if view["overview"]["tx_total_bytes"] != expected_tx:
                raise BenchmarkError(
                    f"concurrent {name} TX total was "
                    f"{view['overview']['tx_total_bytes']}; expected {expected_tx}"
                )
            if view["overview"]["rx_total_bytes"] != expected_rx:
                raise BenchmarkError(
                    f"concurrent {name} RX total was "
                    f"{view['overview']['rx_total_bytes']}; expected {expected_rx}"
                )
            results[request_index] = {
                "request_index": request_index,
                "request_name": requests[request_index][0],
                "submitted_at_ns": submitted_at_ns[request_index],
                "worker_completed_at_ns": worker_completed_at_ns,
                "main_observed_at_ns": time.perf_counter_ns(),
                "latency_ms": round(
                    (worker_completed_at_ns - submitted_at_ns[request_index])
                    / 1_000_000.0,
                    6,
                ),
                "status": "ok",
            }
    stage_elapsed_seconds = (time.perf_counter_ns() - stage_started_ns) / 1e9

    request_records = [record for record in results if record is not None]
    if errors:
        raise BenchmarkError(
            "concurrent warm view-model stage failed: "
            + "; ".join(error["error"] for error in errors)
        )

    latencies_ms = [float(record["latency_ms"]) for record in request_records]
    summary = _timing_summary(latencies_ms)
    summary.update(
        {
            "worker_count": CONCURRENT_VIEW_MODEL_WORKERS,
            "request_count": len(requests),
            "successful_request_count": len(request_records),
            "error_count": len(errors),
            "errors": errors,
            "aggregate_throughput_requests_per_second": round(
                len(request_records) / stage_elapsed_seconds, 6
            ),
            "cache_reused": True,
            "requests": request_records,
        }
    )
    return summary


def _peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _current_rss_mib() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return _peak_rss_mib()


def _check_limits(
    config: BenchmarkConfig, started: float, stage: str
) -> None:
    elapsed = time.perf_counter() - started
    if elapsed > config.max_runtime_seconds:
        raise BenchmarkError(
            f"benchmark exceeded runtime limit during {stage}: "
            f"{elapsed:.3f}s > {config.max_runtime_seconds:.3f}s; "
            "reduce --samples or --repetitions, or raise --max-runtime-seconds deliberately"
        )
    peak_rss = _peak_rss_mib()
    if peak_rss > config.max_rss_mib:
        raise BenchmarkError(
            f"benchmark exceeded RSS limit during {stage}: "
            f"{peak_rss:.1f} MiB > {config.max_rss_mib} MiB; "
            "reduce --samples or raise --max-rss-mib on a machine with sufficient memory"
        )


def _measure(
    *,
    label: str,
    action: Callable[[int], T],
    validate: Callable[[T, int], None],
    config: BenchmarkConfig,
    started: float,
    measurements: dict[str, dict[str, object]],
) -> None:
    values_ms: list[float] = []
    try:
        for warmup_index in range(config.warmup_repetitions):
            _check_limits(config, started, f"{label} warm-up")
            validate(action(warmup_index), warmup_index)
        for repetition in range(config.repetitions):
            _check_limits(config, started, label)
            gc.collect()
            operation_start = time.perf_counter_ns()
            result = action(repetition)
            elapsed_ms = (time.perf_counter_ns() - operation_start) / 1_000_000.0
            values_ms.append(elapsed_ms)
            validate(result, repetition)
            _check_limits(config, started, label)
    except Exception as error:
        if isinstance(error, BenchmarkError):
            raise
        raise BenchmarkError(f"{label} failed: {error}") from error
    measurements[label] = _timing_summary(values_ms)


def _git_details() -> dict[str, object]:
    def run_git(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=APP_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()

    try:
        commit = run_git("rev-parse", "HEAD")
        short_commit = run_git("rev-parse", "--short", "HEAD")
        status_lines = [
            line
            for line in run_git("status", "--short").splitlines()
            if line.strip()
        ]
        return {
            "commit": commit,
            "short_commit": short_commit,
            "dirty_tree": bool(status_lines),
            "dirty_file_count": len(status_lines),
        }
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "commit": None,
            "short_commit": None,
            "dirty_tree": None,
            "error": f"git metadata unavailable: {error}",
        }


def _dependency_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as error:
        return f"unavailable: {error}"


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _total_memory_gib() -> float | None:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / (1024.0 * 1024.0), 3)
    except (OSError, ValueError):
        pass
    return None


def _environment() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "numpy_version": _dependency_version("numpy"),
        "dash_version": _dependency_version("dash"),
        "plotly_version": _dependency_version("plotly"),
        "cpu_model": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "total_memory_gib": _total_memory_gib(),
        "working_directory": str(APP_ROOT),
    }


def _expected_totals(fixture: SyntheticFixture, node_id: int | None = None, nic_id: int | None = None) -> tuple[int, int]:
    tx_total = 0
    rx_total = 0
    for node in fixture.nodes:
        if node_id is not None and node.node_id != node_id:
            continue
        for nic in fixture.nics:
            if nic.node_id != node.node_id:
                continue
            if nic_id is not None and nic.nic_id != nic_id:
                continue
            tx_increment = TX_INCREMENT_BYTES + nic.nic_id * 17 + node.node_id * 101
            rx_increment = RX_INCREMENT_BYTES + nic.nic_id * 23 + node.node_id * 103
            tx_total += tx_increment * (fixture.samples_per_series - 1)
            rx_total += rx_increment * (fixture.samples_per_series - 1)
    return tx_total, rx_total


def _view_payload(view: dict[str, object], label: str) -> int:
    try:
        serialized = json.dumps(
            view, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise BenchmarkError(f"view model {label} is not JSON-serializable: {error}") from error
    return len(serialized.encode("utf-8"))


def _cache_sizes(cache_path: Path) -> tuple[int, int]:
    with zipfile.ZipFile(cache_path, mode="r") as archive:
        uncompressed_bytes = sum(entry.file_size for entry in archive.infolist())
    return cache_path.stat().st_size, uncompressed_bytes


def _at_most_target(actual: float, limit: float, unit: str) -> dict[str, object]:
    return {
        "actual": round(actual, 6),
        "limit": limit,
        "unit": unit,
        "passed": actual <= limit,
    }


def _query_mix(fixture: SyntheticFixture) -> tuple[dict[str, object], ...]:
    narrow_start = fixture.acquisition_end_ns // 3
    narrow_end = (fixture.acquisition_end_ns * 2) // 3
    narrow_count = (
        narrow_end // SAMPLE_INTERVAL_NS - narrow_start // SAMPLE_INTERVAL_NS
    )
    return (
        {
            "name": "narrow_tx",
            "node": 1,
            "nic": NICS_PER_NODE + 3,
            "counter": TX_COUNTER,
            "start_ns": narrow_start,
            "end_ns": narrow_end,
            "expected_points": narrow_count,
        },
        {
            "name": "narrow_rx",
            "node": 2,
            "nic": 2 * NICS_PER_NODE + 5,
            "counter": RX_COUNTER,
            "start_ns": narrow_start,
            "end_ns": narrow_end,
            "expected_points": narrow_count,
        },
        {
            "name": "wide_tx",
            "node": 3,
            "nic": 3 * NICS_PER_NODE + 7,
            "counter": TX_COUNTER,
            "start_ns": 0,
            "end_ns": fixture.acquisition_end_ns + SAMPLE_INTERVAL_NS,
            "expected_points": fixture.samples_per_series,
        },
        {
            "name": "wide_rx",
            "node": 0,
            "nic": 0,
            "counter": RX_COUNTER,
            "start_ns": 0,
            "end_ns": fixture.acquisition_end_ns + SAMPLE_INTERVAL_NS,
            "expected_points": fixture.samples_per_series,
        },
    )


def run_benchmark(config: BenchmarkConfig) -> dict[str, object]:
    benchmark_started = time.perf_counter()
    fixture = build_fixture(config.requested_samples)
    if fixture.row_count != fixture.samples_per_series * SERIES_COUNT:
        raise BenchmarkError(
            f"fixture row invariant failed: {fixture.row_count} != "
            f"{fixture.samples_per_series * SERIES_COUNT}"
        )
    heavy_stage_config = (
        replace(config, repetitions=1, warmup_repetitions=0)
        if fixture.row_count >= 10_000_000
        else config
    )

    baseline_batch = fixture.batch()
    store = TraceStore.from_batches([baseline_batch])
    if store.row_count != fixture.row_count:
        raise BenchmarkError(
            f"store row invariant failed: {store.row_count} != {fixture.row_count}"
        )

    measurements: dict[str, dict[str, object]] = {}
    _measure(
        label="normalized_batch_construction",
        action=lambda _: fixture.batch(),
        validate=lambda batch, _: (
            BenchmarkError("timed batch has wrong row count")
            if batch.node_ids.shape[0] != fixture.row_count
            else None
        ),
        config=config,
        started=benchmark_started,
        measurements=measurements,
    )
    _measure(
        label="columnar_store_index_build",
        action=lambda _: TraceStore.from_batches([baseline_batch]),
        validate=lambda candidate, _: (
            BenchmarkError("timed store has wrong row count")
            if candidate.row_count != fixture.row_count
            else None
        ),
        config=heavy_stage_config,
        started=benchmark_started,
        measurements=measurements,
    )
    def run_scaled_ingest(_: int):
        ingest_batch = fixture.batch()
        return TraceStore.from_batches([ingest_batch])

    _measure(
        label="scaled_ingest_path",
        action=run_scaled_ingest,
        validate=lambda candidate, _: (
            BenchmarkError("timed ingest store has wrong row count")
            if candidate.row_count != fixture.row_count
            else None
        ),
        config=heavy_stage_config,
        started=benchmark_started,
        measurements=measurements,
    )
    del baseline_batch
    ingestion_peak_rss_mib = _peak_rss_mib()

    query_mix = _query_mix(fixture)
    def make_query_mix_runner(query_store: TraceStore):
        def run_query_mix(_: int):
            results = []
            for query in query_mix:
                results.append(
                    query_store.query_unique_series(
                        int(query["node"]),
                        int(query["nic"]),
                        query["counter"],
                        start_ns=int(query["start_ns"]),
                        end_ns=int(query["end_ns"]),
                    )
                )
            return tuple(results)

        return run_query_mix

    run_query_mix = make_query_mix_runner(store)

    def validate_query_mix(results, _repetition: int) -> None:
        for query, result in zip(query_mix, results, strict=True):
            expected = int(query["expected_points"])
            if result.values.shape[0] != expected:
                raise BenchmarkError(
                    f"query {query['name']} returned {result.values.shape[0]} points; "
                    f"expected {expected}"
                )

    _measure(
        label="indexed_series_query_mix",
        action=run_query_mix,
        validate=validate_query_mix,
        config=config,
        started=benchmark_started,
        measurements=measurements,
    )

    analysis_inputs = (
        store.query_unique_series(
            fixture.nodes[1], fixture.nics[NICS_PER_NODE + 3], TX_COUNTER,
            start_ns=0, end_ns=fixture.acquisition_end_ns + SAMPLE_INTERVAL_NS,
        ),
        store.query_unique_series(
            fixture.nodes[2], fixture.nics[2 * NICS_PER_NODE + 5], RX_COUNTER,
            start_ns=0, end_ns=fixture.acquisition_end_ns + SAMPLE_INTERVAL_NS,
        ),
    )
    def run_analysis_mix(_: int):
        analyses = []
        for result in analysis_inputs:
            series = metric_series_from_series_result(
                result, run_id=store.identity.run_id
            )
            analyses.append(
                analyze_traffic_series(
                    series,
                    window_start_ns=0,
                    window_end_ns=fixture.acquisition_end_ns + SAMPLE_INTERVAL_NS,
                )
            )
        return tuple(analyses)
        series = metric_series_from_series_result(
            result, run_id=store.identity.run_id
        )
        analysis = analyze_traffic_series(
            series,
            window_start_ns=0,
            window_end_ns=fixture.acquisition_end_ns + SAMPLE_INTERVAL_NS,
        )
        return analysis, expected_index

    expected_analysis_totals = (
        (TX_INCREMENT_BYTES + (NICS_PER_NODE + 3) * 17 + 1 * 101)
        * (fixture.samples_per_series - 1),
        (RX_INCREMENT_BYTES + (2 * NICS_PER_NODE + 5) * 23 + 2 * 103)
        * (fixture.samples_per_series - 1),
    )

    def validate_analysis_mix(analyses, _repetition: int) -> None:
        for analysis, expected in zip(analyses, expected_analysis_totals, strict=True):
            if analysis.status is not AnalysisStatus.OK:
                raise BenchmarkError(f"traffic analysis status was {analysis.status}")
            if analysis.total_bytes != expected:
                raise BenchmarkError(
                    f"traffic analysis total was {analysis.total_bytes}; expected {expected}"
                )

    _measure(
        label="bridge_and_traffic_analysis",
        action=run_analysis_mix,
        validate=validate_analysis_mix,
        config=config,
        started=benchmark_started,
        measurements=measurements,
    )

    view_filters = (
        ("job_overview", AnalysisFilterState(direction="all", start_ns=0, end_ns=fixture.acquisition_end_ns + SAMPLE_INTERVAL_NS)),
        ("node_comparison", AnalysisFilterState(node_id=1, direction="all", start_ns=0, end_ns=fixture.acquisition_end_ns + SAMPLE_INTERVAL_NS)),
        ("nic_detail", AnalysisFilterState(node_id=2, nic_id=2 * NICS_PER_NODE + 5, direction="all", start_ns=0, end_ns=fixture.acquisition_end_ns + SAMPLE_INTERVAL_NS)),
    )
    def run_view_mix(_: int):
        views = []
        for _, filters in view_filters:
            views.append(build_view_model(store, filters))
        return tuple(views)

    def validate_view_mix(views, _repetition: int) -> None:
        for view, (name, filters) in zip(views, view_filters, strict=True):
            expected_tx, expected_rx = _expected_totals(
                fixture, filters.node_id, filters.nic_id
            )
            if view["overview"]["traffic_state"] != "ready":
                raise BenchmarkError(
                    f"{name} view state was {view['overview']['traffic_state']!r}"
                )
            if view["overview"]["tx_total_bytes"] != expected_tx:
                raise BenchmarkError(
                    f"{name} TX total was {view['overview']['tx_total_bytes']}; "
                    f"expected {expected_tx}"
                )
            if view["overview"]["rx_total_bytes"] != expected_rx:
                raise BenchmarkError(
                    f"{name} RX total was {view['overview']['rx_total_bytes']}; "
                    f"expected {expected_rx}"
                )
            display_points = sum(
                int(record.get("display_point_count", 0))
                for record in view["nic_detail"]["series"]
            )
            if display_points > 2 * 500:
                raise BenchmarkError(
                    f"{name} view exceeded display bound with {display_points} points"
                )

    _measure(
        label="view_model_build_mix",
        action=run_view_mix,
        validate=validate_view_mix,
        config=heavy_stage_config,
        started=benchmark_started,
        measurements=measurements,
    )

    overview_filters = view_filters[0][1]

    def run_warm_overview(_: int):
        return build_view_model(store, overview_filters)

    def validate_warm_overview(view, _repetition: int) -> None:
        expected_tx, expected_rx = _expected_totals(fixture)
        if view["overview"]["traffic_state"] != "ready":
            raise BenchmarkError(
                f"warm overview state was {view['overview']['traffic_state']!r}"
            )
        if view["overview"]["tx_total_bytes"] != expected_tx:
            raise BenchmarkError(
                f"warm overview TX total was {view['overview']['tx_total_bytes']}; "
                f"expected {expected_tx}"
            )
        if view["overview"]["rx_total_bytes"] != expected_rx:
            raise BenchmarkError(
                f"warm overview RX total was {view['overview']['rx_total_bytes']}; "
                f"expected {expected_rx}"
            )

    _measure(
        label="warm_overview_build",
        action=run_warm_overview,
        validate=validate_warm_overview,
        config=heavy_stage_config,
        started=benchmark_started,
        measurements=measurements,
    )
    measurements["concurrent_warm_view_model_requests"] = (
        _run_concurrent_view_model_stage(store, view_filters, fixture)
    )

    payload_bytes = {}
    max_display_points_per_series = 0
    for name, filters in view_filters:
        view = build_view_model(store, filters)
        payload_bytes[name] = _view_payload(view, name)
        max_display_points_per_series = max(
            max_display_points_per_series,
            *(
                int(record.get("display_point_count", 0))
                for record in view["nic_detail"]["series"]
            ),
        )

    del view
    del run_warm_overview
    del run_view_mix
    del run_analysis_mix
    del run_query_mix
    del analysis_inputs
    gc.collect()

    cache_file_bytes = 0
    cache_uncompressed_bytes = 0
    with tempfile.TemporaryDirectory(prefix="benchmark-trace-cache-") as cache_directory:
        cache_path = Path(cache_directory) / "trace-store.npz"

        def cold_write_trace_cache(_: int):
            cache_path.unlink(missing_ok=True)
            write_trace_store_cache(
                cache_path,
                store,
                RUN_IDENTITY.source_fingerprint,
            )
            return cache_path.stat().st_size

        _measure(
            label="trace_cache_cold_write",
            action=cold_write_trace_cache,
            validate=lambda size, _: (
                BenchmarkError("trace cache was not published")
                if size <= 0
                else None
            ),
            config=heavy_stage_config,
            started=benchmark_started,
            measurements=measurements,
        )
        original_identity = store.identity
        del cold_write_trace_cache
        del store
        gc.collect()

        def warm_read_trace_cache(_: int):
            return read_trace_store_cache(
                cache_path,
                RUN_IDENTITY.source_fingerprint,
            )

        def validate_reconstructed_store(candidate, _repetition: int) -> None:
            if candidate.row_count != fixture.row_count:
                raise BenchmarkError(
                    f"reconstructed cache row count was {candidate.row_count}; "
                    f"expected {fixture.row_count}"
                )
            if candidate.identity != original_identity:
                raise BenchmarkError("reconstructed cache run identity changed")

        _measure(
            label="trace_cache_warm_read",
            action=warm_read_trace_cache,
            validate=validate_reconstructed_store,
            config=heavy_stage_config,
            started=benchmark_started,
            measurements=measurements,
        )

        reconstructed_store = read_trace_store_cache(
            cache_path,
            RUN_IDENTITY.source_fingerprint,
        )
        store = reconstructed_store
        reconstructed_query_mix = make_query_mix_runner(reconstructed_store)
        _measure(
            label="reconstructed_store_query_mix",
            action=reconstructed_query_mix,
            validate=validate_query_mix,
            config=config,
            started=benchmark_started,
            measurements=measurements,
        )
        cache_file_bytes, cache_uncompressed_bytes = _cache_sizes(cache_path)

    dense_shape = (
        NODE_COUNT,
        NICS_PER_NODE,
        COUNTER_COUNT,
        fixture.samples_per_series,
    )

    def allocate_dense(_: int):
        array = np.full(dense_shape, np.nan, dtype=np.float64)
        if not np.isnan(array).all():
            raise BenchmarkError("dense baseline failed to initialize every element")
        return array.size

    _measure(
        label="dense_array_allocation_baseline",
        action=allocate_dense,
        validate=lambda size, _: (
            BenchmarkError("dense baseline size mismatch")
            if size != fixture.row_count
            else None
        ),
        config=config,
        started=benchmark_started,
        measurements=measurements,
    )

    column_names = (
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
    column_bytes = sum(int(store.column(name).nbytes) for name in column_names)
    total_elapsed = time.perf_counter() - benchmark_started
    _check_limits(config, benchmark_started, "final result assembly")

    m7_targets = {
        "scaled_ingest": _at_most_target(
            float(measurements["scaled_ingest_path"]["median_ms"]) / 1000.0,
            60.0,
            "seconds",
        ),
        "peak_ingestion_rss": _at_most_target(
            ingestion_peak_rss_mib,
            4096.0,
            "MiB",
        ),
        "warm_overview": _at_most_target(
            float(measurements["warm_overview_build"]["median_ms"]) / 1000.0,
            2.0,
            "seconds",
        ),
        "backend_query_p95": _at_most_target(
            float(measurements["reconstructed_store_query_mix"]["p95_ms"]),
            500.0,
            "ms",
        ),
        "chart_payload": _at_most_target(
            max(payload_bytes.values()),
            2 * 1024 * 1024,
            "bytes",
        ),
        "display_points_per_series": _at_most_target(
            max_display_points_per_series,
            2000,
            "points",
        ),
        "concurrent_warm_view_model_p95": _at_most_target(
            float(
                measurements["concurrent_warm_view_model_requests"]["p95_ms"]
            ),
            500.0,
            "ms",
        ),
        "concurrent_warm_view_model_errors": _at_most_target(
            int(
                measurements["concurrent_warm_view_model_requests"]["error_count"]
            ),
            0,
            "errors",
        ),
        "trace_cache_file_size": _at_most_target(
            cache_file_bytes,
            2 * 1024**3,
            "bytes",
        ),
        "trace_cache_uncompressed_size": _at_most_target(
            cache_uncompressed_bytes,
            2 * 1024**3,
            "bytes",
        ),
    }
    all_targets_passed = all(
        bool(target["passed"]) for target in m7_targets.values()
    )

    return {
        "schema_version": 1,
        "benchmark": "synthetic-local-columnar-path",
        "status": "ok" if all_targets_passed else "completed_with_target_failures",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": _git_details(),
        "environment": _environment(),
        "configuration": {
            "requested_samples": config.requested_samples,
            "repetitions": config.repetitions,
            "warmup_repetitions": config.warmup_repetitions,
            "max_runtime_seconds": config.max_runtime_seconds,
            "max_rss_mib": config.max_rss_mib,
            "worker_count": 1,
            "concurrent_view_model_worker_count": CONCURRENT_VIEW_MODEL_WORKERS,
            "concurrent_view_model_request_count": (
                CONCURRENT_VIEW_MODEL_REPETITIONS * len(view_filters)
            ),
            "dash_server_started": False,
            "heavy_stage_repetitions": heavy_stage_config.repetitions,
            "heavy_stage_warmup_repetitions": heavy_stage_config.warmup_repetitions,
            "heavy_stage_policy": (
                "one timed operation with no warm-up at >=10,000,000 rows; "
                "otherwise the requested repetition policy"
            ),
        },
        "fixture": {
            "seed_or_formula": "fixed arithmetic formula; no RNG",
            "node_count": NODE_COUNT,
            "nic_count": len(fixture.nics),
            "nics_per_node": NICS_PER_NODE,
            "counter_count": COUNTER_COUNT,
            "counter_names": (TX_COUNTER.name, RX_COUNTER.name),
            "series_count": SERIES_COUNT,
            "samples_per_series": fixture.samples_per_series,
            "counter_observation_count": fixture.row_count,
            "sample_interval_ns": SAMPLE_INTERVAL_NS,
            "acquisition_end_ns": fixture.acquisition_end_ns,
            "value_kind": "exact_uint64_cumulative_bytes",
            "record_id_encoding": (
                "fixed-width 8-byte ASCII: r + 2 hexadecimal series digits "
                "+ 5 hexadecimal sample digits"
            ),
        },
        "query_mix": [
            {
                key: value
                for key, value in query.items()
                if key not in {"counter"}
            }
            for query in query_mix
        ],
        "view_mix": [
            {
                "name": name,
                "node_id": filters.node_id,
                "nic_id": filters.nic_id,
                "direction": filters.direction,
                "start_ns": filters.start_ns,
                "end_ns": filters.end_ns,
            }
            for name, filters in view_filters
        ],
        "measurements": measurements,
        "concurrency": {
            "stage": "concurrent warm view-model requests",
            "worker_bound": CONCURRENT_VIEW_MODEL_WORKERS,
            "request_mix": "fixed view mix repeated in deterministic order",
            "cache": "existing bounded process-local view-model cache",
            "per_request_latency_definition": (
                "submission to future completion, including worker queue wait"
            ),
        },
        "payload": {
            "definition": "UTF-8 bytes of compact JSON for each bounded view model",
            "bytes_by_view": payload_bytes,
            "max_view_model_payload_bytes": max(payload_bytes.values()),
        },
        "cache": {
            "format": "compressed non-pickle NPZ with strict JSON metadata",
            "file_bytes": cache_file_bytes,
            "uncompressed_bytes": cache_uncompressed_bytes,
            "source_fingerprint": RUN_IDENTITY.source_fingerprint,
            "row_count": fixture.row_count,
            "temporary_path_cleanup": True,
        },
        "memory": {
            "process_peak_rss_mib": round(_peak_rss_mib(), 3),
            "process_current_rss_mib_at_end": round(_current_rss_mib(), 3),
            "ingestion_peak_rss_mib": round(ingestion_peak_rss_mib, 3),
            "columnar_array_nbytes": column_bytes,
            "dense_baseline_shape": dense_shape,
            "dense_baseline_nbytes": fixture.row_count * 8,
        },
        "verification": {
            "store_row_count_matches_fixture": True,
            "all_query_mix_points_verified": True,
            "traffic_totals_match_arithmetic_reference": True,
            "view_totals_match_arithmetic_reference": True,
            "display_points_at_most_1000": True,
            "concurrent_view_model_requests_succeeded": True,
            "trace_cache_round_trip_row_count_and_identity": True,
            "reconstructed_store_query_points_verified": True,
        },
        "m7_targets": {
            "all_passed": all_targets_passed,
            "results": m7_targets,
            "scope": (
                "synthetic normalized 10-million-counter-sample targets when "
                "--samples 10000000 is used; browser/tunnel and real-trace "
                "targets are not claimed"
            ),
        },
        "limits_and_scope": {
            "runtime_guard_seconds": config.max_runtime_seconds,
            "peak_rss_guard_mib": config.max_rss_mib,
            "scope": "local synthetic normalized data; excludes source decoding and network/browser rendering",
            "dense_baseline_scope": "allocation and initialization only; not legacy text parsing or callback execution",
            "milestone_target_claims": "none",
        },
        "total_wall_time_seconds": round(total_elapsed, 6),
    }


def main() -> int:
    args = parse_args()
    config = BenchmarkConfig(
        requested_samples=args.samples,
        repetitions=args.repetitions,
        warmup_repetitions=args.warmup,
        max_runtime_seconds=args.max_runtime_seconds,
        max_rss_mib=args.max_rss_mib,
    )
    try:
        result = run_benchmark(config)
    except BenchmarkError as error:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "benchmark": "synthetic-local-columnar-path",
                    "status": "failed",
                    "error": str(error),
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
        )
        return 2

    rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output is None:
        print(rendered)
        return 0
    try:
        args.output.parent.mkdir(parents=False, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    except OSError as error:
        print(
            f"Benchmark succeeded but could not write {args.output}: {error}",
            file=sys.stderr,
        )
        return 3
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
