from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import time
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = APP_ROOT / "scripts" / "performance_benchmark.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "performance_benchmark_under_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_synthetic_benchmark_small_scale_and_invariants():
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--samples",
            "640",
            "--repetitions",
            "2",
            "--warmup",
            "1",
            "--max-runtime-seconds",
            "15",
            "--max-rss-mib",
            "1024",
        ],
        cwd=APP_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)

    assert result["status"] == "ok"
    assert result["fixture"]["counter_observation_count"] == 640
    assert result["fixture"]["samples_per_series"] == 10
    assert result["configuration"]["dash_server_started"] is False
    assert result["memory"]["dense_baseline_nbytes"] == 640 * 8
    assert result["payload"]["max_view_model_payload_bytes"] > 0
    assert result["cache"]["file_bytes"] > 0
    assert result["cache"]["uncompressed_bytes"] > 0
    assert result["cache"]["row_count"] == 640
    assert result["m7_targets"]["all_passed"] is True
    assert result["fixture"]["record_id_encoding"].startswith("fixed-width")
    assert (
        result["configuration"]["concurrent_view_model_worker_count"] == 4
    )
    assert result["configuration"]["concurrent_view_model_request_count"] == 12
    assert all(
        summary["timed_operations"] == 2
        for label, summary in result["measurements"].items()
        if label != "concurrent_warm_view_model_requests"
    )
    concurrent_summary = result["measurements"]["concurrent_warm_view_model_requests"]
    assert all(
        request["worker_completed_at_ns"] <= request["main_observed_at_ns"]
        for request in concurrent_summary["requests"]
    )
    assert concurrent_summary["worker_count"] == 4
    assert concurrent_summary["request_count"] == 12
    assert concurrent_summary["successful_request_count"] == 12
    assert concurrent_summary["error_count"] == 0
    assert concurrent_summary["timed_operations"] == 12
    assert concurrent_summary["aggregate_throughput_requests_per_second"] > 0
    assert concurrent_summary["p95_ms"] <= 500
    assert [request["request_name"] for request in concurrent_summary["requests"]] == [
        "job_overview",
        "node_comparison",
        "nic_detail",
    ] * 4
    assert all(
        request["status"] == "ok" and request["latency_ms"] >= 0
        for request in concurrent_summary["requests"]
    )
    assert result["verification"]["concurrent_view_model_requests_succeeded"] is True
    assert result["m7_targets"]["results"]["concurrent_warm_view_model_p95"]["passed"] is True
    assert result["m7_targets"]["results"]["concurrent_warm_view_model_errors"]["passed"] is True
    assert set(result["measurements"]) == {
        "normalized_batch_construction",
        "columnar_store_index_build",
        "scaled_ingest_path",
        "indexed_series_query_mix",
        "bridge_and_traffic_analysis",
        "view_model_build_mix",
        "warm_overview_build",
        "concurrent_warm_view_model_requests",
        "trace_cache_cold_write",
        "trace_cache_warm_read",
        "reconstructed_store_query_mix",
        "dense_array_allocation_baseline",
    }


def test_concurrent_latency_uses_worker_completion_and_excludes_validation(monkeypatch):
    benchmark = _load_benchmark_module()
    fixture = benchmark.build_fixture(640)
    view_filters = (
        (
            "job_overview",
            benchmark.AnalysisFilterState(
                direction="all",
                start_ns=0,
                end_ns=fixture.acquisition_end_ns + benchmark.SAMPLE_INTERVAL_NS,
            ),
        ),
        (
            "node_comparison",
            benchmark.AnalysisFilterState(
                node_id=1,
                direction="all",
                start_ns=0,
                end_ns=fixture.acquisition_end_ns + benchmark.SAMPLE_INTERVAL_NS,
            ),
        ),
        (
            "nic_detail",
            benchmark.AnalysisFilterState(
                node_id=2,
                nic_id=2 * benchmark.NICS_PER_NODE + 5,
                direction="all",
                start_ns=0,
                end_ns=fixture.acquisition_end_ns + benchmark.SAMPLE_INTERVAL_NS,
            ),
        ),
    )
    expected_totals = {
        filters: benchmark._expected_totals(fixture, filters.node_id, filters.nic_id)
        for _, filters in view_filters
    }

    def fake_build_view_model(_store, filters):
        tx_total, rx_total = expected_totals[filters]
        return {
            "overview": {
                "traffic_state": "ready",
                "tx_total_bytes": tx_total,
                "rx_total_bytes": rx_total,
            }
        }

    original_expected_totals = benchmark._expected_totals
    validation_delay_seconds = 0.05

    def delayed_expected_totals(fixture, node_id=None, nic_id=None):
        time.sleep(validation_delay_seconds)
        return original_expected_totals(fixture, node_id, nic_id)

    monkeypatch.setattr(benchmark, "build_view_model", fake_build_view_model)
    monkeypatch.setattr(benchmark, "_expected_totals", delayed_expected_totals)

    summary = benchmark._run_concurrent_view_model_stage(
        None, view_filters, fixture
    )

    assert summary["request_count"] == 12
    assert summary["error_count"] == 0
    for request in summary["requests"]:
        latency_ns = (
            request["worker_completed_at_ns"] - request["submitted_at_ns"]
        )
        observed_delay_ns = (
            request["main_observed_at_ns"] - request["worker_completed_at_ns"]
        )
        assert request["latency_ms"] == round(latency_ns / 1_000_000.0, 6)
        assert request["worker_completed_at_ns"] <= request["main_observed_at_ns"]
        assert observed_delay_ns >= validation_delay_seconds * 1_000_000_000 * 0.9
        assert latency_ns < observed_delay_ns
