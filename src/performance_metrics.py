"""Thread-safe, identity-aware performance evidence for UI recovery.

The recorder deliberately separates lifecycle stages and rejects stale or
wrong-identity results before they can be counted as successful interactions.
It uses only the Python standard library and serializes bounded evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from threading import RLock
import time
from typing import Any, Literal


PerformanceStage = Literal[
    "cold_source_load",
    "cache_reopen",
    "app_readiness",
    "first_meaningful_render",
    "warm_new_query",
    "warm_repeat",
]

PERFORMANCE_STAGES: tuple[PerformanceStage, ...] = (
    "cold_source_load",
    "cache_reopen",
    "app_readiness",
    "first_meaningful_render",
    "warm_new_query",
    "warm_repeat",
)

PERCENTILE_MINIMUM_SAMPLES = 30
DEFAULT_MAX_SAMPLES_PER_STAGE = 10_000
DEFAULT_MAX_ACTIVE_OPERATIONS = 4_096
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MetricIdentity:
    """Opaque dataset, request, and selection identity expected for a result."""

    dataset_token: str
    request_token: str
    selection_token: str

    def to_dict(self) -> dict[str, str]:
        return {
            "dataset_token": self.dataset_token,
            "request_token": self.request_token,
            "selection_token": self.selection_token,
        }


@dataclass(frozen=True)
class StageOperation:
    """One in-flight operation owned by a unique request token."""

    stage: PerformanceStage
    identity: MetricIdentity
    started_at_ns: int


@dataclass(frozen=True)
class TimingSample:
    stage: PerformanceStage
    duration_ns: int
    expected_identity: dict[str, str]
    result_identity: dict[str, str]
    payload_bytes: int
    success: bool
    error: str | None
    stale_result_rejected: bool

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000


@dataclass(frozen=True)
class StageSummary:
    stage: PerformanceStage
    sample_count: int
    success_count: int
    error_count: int
    stale_result_rejection_count: int
    minimum_samples_for_percentile: int
    percentile_available: bool
    p50_ms: float | None
    p95_ms: float | None
    payload_bytes_min: int | None
    payload_bytes_max: int | None
    payload_bytes_mean: float | None
    dropped_sample_count: int


class PerformanceMetricsError(RuntimeError):
    """Raised for invalid recorder operations or capacity exhaustion."""


def _validate_token(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _validate_stage(stage: PerformanceStage) -> None:
    if stage not in PERFORMANCE_STAGES:
        raise ValueError(f"unknown performance stage {stage!r}")


def _nearest_rank_percentile(sorted_values: list[int], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    rank = max(1, math.ceil(percentile * len(sorted_values)))
    return sorted_values[rank - 1] / 1_000_000


class PerformanceMetricsRecorder:
    """Record bounded lifecycle timings without counting stale results as success."""

    def __init__(
        self,
        *,
        max_samples_per_stage: int = DEFAULT_MAX_SAMPLES_PER_STAGE,
        max_active_operations: int = DEFAULT_MAX_ACTIVE_OPERATIONS,
    ) -> None:
        if max_samples_per_stage < 1:
            raise ValueError("max_samples_per_stage must be positive")
        if max_active_operations < 1:
            raise ValueError("max_active_operations must be positive")
        self._max_samples_per_stage = max_samples_per_stage
        self._max_active_operations = max_active_operations
        self._lock = RLock()
        self._samples: dict[PerformanceStage, list[TimingSample]] = {
            stage: [] for stage in PERFORMANCE_STAGES
        }
        self._dropped_samples: dict[PerformanceStage, int] = {
            stage: 0 for stage in PERFORMANCE_STAGES
        }
        self._active: dict[str, StageOperation] = {}

    def begin(
        self,
        stage: PerformanceStage,
        *,
        dataset_token: str,
        request_token: str,
        selection_token: str,
    ) -> StageOperation:
        """Begin a stage and return its operation handle."""

        _validate_stage(stage)
        _validate_token("dataset_token", dataset_token)
        _validate_token("request_token", request_token)
        _validate_token("selection_token", selection_token)
        identity = MetricIdentity(dataset_token, request_token, selection_token)
        with self._lock:
            if request_token in self._active:
                raise PerformanceMetricsError(f"request token already active: {request_token!r}")
            if len(self._active) >= self._max_active_operations:
                raise PerformanceMetricsError("active performance operation limit reached")
            operation = StageOperation(
                stage=stage,
                identity=identity,
                started_at_ns=time.monotonic_ns(),
            )
            self._active[request_token] = operation
            return operation

    def finish(
        self,
        operation: StageOperation,
        *,
        result_identity: MetricIdentity | None = None,
        payload_bytes: int = 0,
        success: bool = True,
        error: str | None = None,
    ) -> TimingSample:
        """Finish an operation; mismatched identity is forced to a stale error."""

        if not isinstance(operation, StageOperation):
            raise TypeError("operation must come from PerformanceMetricsRecorder.begin")
        if isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int) or payload_bytes < 0:
            raise ValueError("payload_bytes must be a nonnegative integer")
        with self._lock:
            active = self._active.get(operation.identity.request_token)
            if active is not operation:
                raise PerformanceMetricsError("operation is not active or was already finished")
            del self._active[operation.identity.request_token]

        resolved_identity = result_identity or operation.identity
        stale = resolved_identity != operation.identity
        resolved_success = bool(success) and not stale
        resolved_error = error
        if stale:
            resolved_error = (
                f"stale or wrong-identity result rejected; expected {operation.identity}; "
                f"received {resolved_identity}"
            )
        elif not resolved_success and resolved_error is None:
            resolved_error = "unspecified stage error"

        sample = TimingSample(
            stage=operation.stage,
            duration_ns=max(0, time.monotonic_ns() - operation.started_at_ns),
            expected_identity=operation.identity.to_dict(),
            result_identity=resolved_identity.to_dict(),
            payload_bytes=payload_bytes,
            success=resolved_success,
            error=resolved_error,
            stale_result_rejected=stale,
        )
        self._append_sample(sample)
        return sample

    def record_exception(self, operation: StageOperation, error: BaseException) -> TimingSample:
        """Finish an operation as an error while preserving its identity."""

        return self.finish(
            operation,
            result_identity=operation.identity,
            success=False,
            error=f"{type(error).__name__}: {error}",
        )

    def reject_stale(
        self,
        operation: StageOperation,
        *,
        result_identity: MetricIdentity,
        reason: str = "stale result",
    ) -> TimingSample:
        """Explicitly finish an operation as a stale-result rejection."""

        return self.finish(
            operation,
            result_identity=result_identity,
            success=False,
            error=reason,
        )

    def _append_sample(self, sample: TimingSample) -> None:
        with self._lock:
            samples = self._samples[sample.stage]
            if len(samples) >= self._max_samples_per_stage:
                self._dropped_samples[sample.stage] += 1
                return
            samples.append(sample)

    def stage_summary(
        self,
        stage: PerformanceStage,
        *,
        minimum_samples: int = PERCENTILE_MINIMUM_SAMPLES,
    ) -> StageSummary:
        """Summarize a stage; percentiles remain None until enough samples exist."""

        _validate_stage(stage)
        if isinstance(minimum_samples, bool) or minimum_samples < 1:
            raise ValueError("minimum_samples must be a positive integer")
        with self._lock:
            samples = tuple(self._samples[stage])
            dropped = self._dropped_samples[stage]
        successful = [sample.duration_ns for sample in samples if sample.success]
        payloads = [sample.payload_bytes for sample in samples if sample.success]
        percentile_available = len(successful) >= minimum_samples
        return StageSummary(
            stage=stage,
            sample_count=len(samples),
            success_count=len(successful),
            error_count=sum(not sample.success for sample in samples),
            stale_result_rejection_count=sum(sample.stale_result_rejected for sample in samples),
            minimum_samples_for_percentile=minimum_samples,
            percentile_available=percentile_available,
            p50_ms=_nearest_rank_percentile(sorted(successful), 0.50) if percentile_available else None,
            p95_ms=_nearest_rank_percentile(sorted(successful), 0.95) if percentile_available else None,
            payload_bytes_min=min(payloads) if payloads else None,
            payload_bytes_max=max(payloads) if payloads else None,
            payload_bytes_mean=(sum(payloads) / len(payloads)) if payloads else None,
            dropped_sample_count=dropped,
        )

    def evidence_document(
        self,
        *,
        context: dict[str, Any] | None = None,
        minimum_samples: int = PERCENTILE_MINIMUM_SAMPLES,
    ) -> dict[str, Any]:
        """Return a JSON-safe evidence document with summaries and raw samples."""

        with self._lock:
            samples = {
                stage: tuple(self._samples[stage]) for stage in PERFORMANCE_STAGES
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "context": dict(context or {}),
            "percentile_policy": {
                "method": "nearest-rank",
                "minimum_samples": minimum_samples,
                "successful_samples_only": True,
            },
            "stages": {
                stage: asdict(self.stage_summary(stage, minimum_samples=minimum_samples))
                for stage in PERFORMANCE_STAGES
            },
            "samples": {
                stage: [
                    {
                        **asdict(sample),
                        "duration_ms": sample.duration_ms,
                    }
                    for sample in stage_samples
                ]
                for stage, stage_samples in samples.items()
            },
        }

    def serialize_evidence(
        self,
        *,
        context: dict[str, Any] | None = None,
        minimum_samples: int = PERCENTILE_MINIMUM_SAMPLES,
    ) -> str:
        """Serialize evidence as deterministic, strict JSON."""

        return json.dumps(
            self.evidence_document(
                context=context,
                minimum_samples=minimum_samples,
            ),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
