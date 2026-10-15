"""Focused tests for performance evidence correctness."""

from __future__ import annotations

import json
import threading

import pytest

from performance_metrics import (
    PERFORMANCE_STAGES,
    PERCENTILE_MINIMUM_SAMPLES,
    MetricIdentity,
    PerformanceMetricsError,
    PerformanceMetricsRecorder,
)


def identity(request_token: str = "request-1") -> MetricIdentity:
    return MetricIdentity(
        dataset_token="dataset-token",
        request_token=request_token,
        selection_token="selection-token",
    )


def record_duration(
    recorder: PerformanceMetricsRecorder,
    stage: str,
    request_token: str,
    duration_ns: int,
) -> None:
    operation = recorder.begin(
        stage,
        dataset_token="dataset-token",
        request_token=request_token,
        selection_token="selection-token",
    )
    operation = recorder.finish(operation, payload_bytes=100)
    # Durations are measured with monotonic_ns; replace only for deterministic tests.
    object.__setattr__(operation, "duration_ns", duration_ns)
    recorder._samples[stage][-1] = operation


def test_required_stages_and_identity_tokens_are_recorded():
    recorder = PerformanceMetricsRecorder()
    assert PERFORMANCE_STAGES == (
        "cold_source_load",
        "cache_reopen",
        "app_readiness",
        "first_meaningful_render",
        "warm_new_query",
        "warm_repeat",
    )

    for stage in PERFORMANCE_STAGES:
        operation = recorder.begin(
            stage,
            dataset_token="dataset-token",
            request_token=f"request-{stage}",
            selection_token="selection-token",
        )
        sample = recorder.finish(operation, payload_bytes=256)
        assert sample.success is True
        assert sample.expected_identity["dataset_token"] == "dataset-token"
        assert sample.expected_identity["request_token"] == f"request-{stage}"
        assert sample.expected_identity["selection_token"] == "selection-token"
        assert sample.result_identity == sample.expected_identity
        assert sample.payload_bytes == 256
        assert recorder.stage_summary(stage).success_count == 1


def test_percentiles_require_thirty_successful_samples():
    recorder = PerformanceMetricsRecorder()
    for index in range(PERCENTILE_MINIMUM_SAMPLES - 1):
        record_duration(recorder, "warm_repeat", f"request-{index}", index + 1)
    summary = recorder.stage_summary("warm_repeat")
    assert summary.success_count == 29
    assert summary.percentile_available is False
    assert summary.p50_ms is None
    assert summary.p95_ms is None

    record_duration(
        recorder, "warm_repeat", "request-final", PERCENTILE_MINIMUM_SAMPLES
    )
    summary = recorder.stage_summary("warm_repeat")
    assert summary.percentile_available is True
    assert summary.p50_ms == 0.000015
    assert summary.p95_ms == 0.000029


def test_stale_and_wrong_identity_results_never_succeed():
    recorder = PerformanceMetricsRecorder()
    operation = recorder.begin(
        "warm_new_query",
        dataset_token="dataset-token",
        request_token="request-stale",
        selection_token="selection-token",
    )
    wrong_identity = MetricIdentity(
        dataset_token="other-dataset",
        request_token="request-stale",
        selection_token="selection-token",
    )
    sample = recorder.finish(operation, result_identity=wrong_identity, success=True)
    assert sample.success is False
    assert sample.stale_result_rejected is True
    assert "stale or wrong-identity result rejected" in sample.error

    operation = recorder.begin(
        "first_meaningful_render",
        dataset_token="dataset-token",
        request_token="request-error",
        selection_token="selection-token",
    )
    sample = recorder.record_exception(operation, RuntimeError("render failed"))
    assert sample.success is False
    assert sample.error == "RuntimeError: render failed"
    assert recorder.stage_summary("first_meaningful_render").error_count == 1
    assert (
        recorder.stage_summary("first_meaningful_render").stale_result_rejection_count == 0
    )


def test_concurrent_operations_are_thread_safe():
    recorder = PerformanceMetricsRecorder()
    barrier = threading.Barrier(40)

    def worker(index: int) -> None:
        operation = recorder.begin(
            "cold_source_load",
            dataset_token="dataset-token",
            request_token=f"request-{index}",
            selection_token="selection-token",
        )
        barrier.wait()
        recorder.finish(operation, payload_bytes=index)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    summary = recorder.stage_summary("cold_source_load")
    assert summary.sample_count == 40
    assert summary.success_count == 40
    assert summary.error_count == 0


def test_evidence_serializes_to_strict_json_with_context():
    recorder = PerformanceMetricsRecorder()
    operation = recorder.begin(
        "cache_reopen",
        dataset_token="dataset-token",
        request_token="request-json",
        selection_token="selection-token",
    )
    recorder.finish(operation, payload_bytes=1234)
    evidence_json = recorder.serialize_evidence(
        context={
            "dataset_size_bytes": 4096,
            "hardware": "test-hardware",
            "tunnel": "none",
        }
    )
    document = json.loads(evidence_json)
    assert document["schema_version"] == 1
    assert document["context"] == {
        "dataset_size_bytes": 4096,
        "hardware": "test-hardware",
        "tunnel": "none",
    }
    assert document["stages"]["cache_reopen"]["success_count"] == 1
    assert document["samples"]["cache_reopen"][0]["payload_bytes"] == 1234
    assert document["samples"]["cache_reopen"][0]["duration_ms"] >= 0


def test_duplicate_active_requests_and_double_finish_are_rejected():
    recorder = PerformanceMetricsRecorder()
    operation = recorder.begin(
        "app_readiness",
        dataset_token="dataset-token",
        request_token="request-active",
        selection_token="selection-token",
    )
    with pytest.raises(PerformanceMetricsError):
        recorder.begin(
            "app_readiness",
            dataset_token="dataset-token",
            request_token="request-active",
            selection_token="selection-token",
        )
    recorder.finish(operation)
    with pytest.raises(PerformanceMetricsError):
        recorder.finish(operation)
