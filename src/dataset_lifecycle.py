"""Thread-safe lifecycle management for approved server-side datasets."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
import threading
import time
from typing import Any
import uuid

from performance_metrics import MetricIdentity, PerformanceMetricsRecorder


class DatasetLifecycleError(RuntimeError):
    """Base class for dataset lifecycle errors."""


class UnknownDatasetError(DatasetLifecycleError, ValueError):
    """Raised when a dataset ID was not explicitly approved."""


class DatasetPathError(DatasetLifecycleError, ValueError):
    """Raised when an approved descriptor escapes the approved root."""


class DatasetValidationError(DatasetLifecycleError, ValueError):
    """Raised when a descriptor or request violates the lifecycle contract."""


class DatasetManagerClosedError(DatasetLifecycleError, RuntimeError):
    """Raised when a closed manager receives a new request."""


class DatasetOperationCancelled(DatasetLifecycleError):
    """Raised by cooperative cancellation before a load completes."""


class DatasetState(str, Enum):
    IDLE = "idle"
    QUEUED = "queued"
    LOADING = "loading"
    CANCELLING = "cancelling"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class DatasetDescriptor:
    """One explicitly approved dataset and its production loader arguments."""

    dataset_id: str
    trace_path: Path
    cache_path: Path | None
    args: Any


@dataclass(frozen=True)
class DatasetRequestToken:
    """Opaque identity for one open or switch request."""

    request_id: str
    dataset_id: str
    generation: int


@dataclass(frozen=True)
class DatasetProgress:
    stage: str
    fraction: float | None = None
    message: str | None = None


@dataclass(frozen=True)
class DatasetStatusSnapshot:
    token: DatasetRequestToken | None
    state: DatasetState
    progress: DatasetProgress
    error: str | None = None
    current_dataset_id: str | None = None
    last_good_dataset_id: str | None = None
    created_at: float | None = None
    started_at: float | None = None
    ended_at: float | None = None
    cancelled: bool = False
    superseded: bool = False


@dataclass(frozen=True)
class DatasetHandle:
    """A successfully loaded dataset and its approved descriptor."""

    token: DatasetRequestToken
    descriptor: DatasetDescriptor
    store: Any


@dataclass
class DatasetCancellation:
    """Cooperative cancellation object compatible with load_trace_store."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._signal_number: int | None = None

    def request(self, signal_number: int | None = None) -> None:
        self._signal_number = signal_number
        self._event.set()

    def should_cancel(self) -> bool:
        return self._event.is_set()

    def check(self, stage: str) -> None:
        if self._event.is_set():
            raise DatasetOperationCancelled(
                f"Dataset operation cancelled before {stage} completed."
            )


@dataclass
class _LifecycleRequest:
    token: DatasetRequestToken
    descriptor: DatasetDescriptor
    cancellation: DatasetCancellation
    created_at: float
    started_at: float | None = None
    ended_at: float | None = None
    state: DatasetState = DatasetState.QUEUED
    error: str | None = None
    cancelled: bool = False
    superseded: bool = False


_TERMINAL_STATES = frozenset(
    {
        DatasetState.READY,
        DatasetState.FAILED,
        DatasetState.CANCELLED,
        DatasetState.SUPERSEDED,
    }
)
_ACTIVE_STATES = frozenset(
    {
        DatasetState.QUEUED,
        DatasetState.LOADING,
        DatasetState.CANCELLING,
    }
)


def _production_loader(args: Any, cancellation: DatasetCancellation) -> Any:
    """Load a production trace store through the existing CLI loader."""

    from main import load_trace_store

    return load_trace_store(args, cancellation=cancellation)


class DatasetLifecycleManager:
    """Manage open, switch, cancel, and status for approved datasets.

    A single worker thread serializes loads. New requests cooperatively cancel
    and supersede older in-flight work. Only the newest non-superseded request
    can become the active and last-good dataset.
    """

    def __init__(
        self,
        approved_root: Path | str,
        descriptors: Iterable[DatasetDescriptor],
        *,
        loader: Callable[[Any, DatasetCancellation], Any] | None = None,
        performance_recorder: PerformanceMetricsRecorder | None = None,
    ) -> None:
        self._root = Path(approved_root).resolve(strict=True)
        if not self._root.is_dir():
            raise DatasetPathError(f"approved root is not a directory: {self._root}")

        self._descriptors: dict[str, DatasetDescriptor] = {}
        for descriptor in descriptors:
            self._register_descriptor(descriptor)

        self._loader = loader if loader is not None else _production_loader
        self._performance_recorder = performance_recorder
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="aurora-dataset-lifecycle",
            daemon=True,
        )
        self._closed = False
        self._generation = 0
        self._pending_request: _LifecycleRequest | None = None
        self._active_request: _LifecycleRequest | None = None
        self._latest_request: _LifecycleRequest | None = None
        self._requests: dict[str, _LifecycleRequest] = {}
        self._current_handle: DatasetHandle | None = None
        self._last_good_handle: DatasetHandle | None = None
        self._worker.start()

    @property
    def approved_root(self) -> Path:
        return self._root

    def register(self, descriptor: DatasetDescriptor) -> str:
        """Approve a root-contained path selected through the local UI."""
        with self._condition:
            existing = self._descriptors.get(descriptor.dataset_id)
            if existing is not None:
                if existing.trace_path != Path(descriptor.trace_path).resolve():
                    raise DatasetValidationError("dataset ID collision")
                return descriptor.dataset_id
            if len(self._descriptors) >= 64:
                protected = {h.descriptor.dataset_id for h in (self._current_handle, self._last_good_handle) if h}
                protected.update(r.descriptor.dataset_id for r in self._requests.values() if r.state in _ACTIVE_STATES)
                for key in list(self._descriptors):
                    if key not in protected:
                        del self._descriptors[key]
                        break
                else:
                    raise DatasetValidationError("Too many active datasets; cancel pending work first.")
            self._register_descriptor(descriptor)
            return descriptor.dataset_id

    def open(self, dataset_id: str) -> DatasetRequestToken:
        """Open an explicitly approved dataset."""

        return self._submit_request(dataset_id)

    def switch(self, dataset_id: str) -> DatasetRequestToken:
        """Switch to an explicitly approved dataset."""

        return self._submit_request(dataset_id)

    def cancel(self, token: DatasetRequestToken) -> None:
        """Request cooperative cancellation for one request."""

        with self._condition:
            request = self._requests.get(token.request_id)
            if request is None:
                raise DatasetValidationError(f"unknown dataset request token: {token}")
            if request.state not in _TERMINAL_STATES:
                request.cancelled = True
                request.state = DatasetState.CANCELLING
                request.cancellation.request()
                self._condition.notify_all()

    def status(
        self, token: DatasetRequestToken | None = None
    ) -> DatasetStatusSnapshot:
        """Return status for a request, or the latest request when omitted."""

        with self._condition:
            if token is not None:
                request = self._requests.get(token.request_id)
                if request is None:
                    raise DatasetValidationError(
                        f"unknown dataset request token: {token}"
                    )
                return self._snapshot(request)
            if self._latest_request is None:
                return DatasetStatusSnapshot(
                    token=None,
                    state=DatasetState.IDLE,
                    progress=DatasetProgress(stage="idle"),
                    current_dataset_id=self._current_handle.descriptor.dataset_id
                    if self._current_handle is not None
                    else None,
                    last_good_dataset_id=self._last_good_handle.descriptor.dataset_id
                    if self._last_good_handle is not None
                    else None,
                )
            return self._snapshot(self._latest_request)

    def current(self) -> DatasetHandle | None:
        """Return the currently usable dataset handle, if any."""

        with self._lock:
            return self._current_handle

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        """Return the explicitly approved dataset IDs."""

        return tuple(self._descriptors)

    def last_good(self) -> DatasetHandle | None:
        """Return the last successful dataset handle, distinct from active."""

        with self._lock:
            return self._last_good_handle

    def wait(
        self, token: DatasetRequestToken, timeout: float | None = None
    ) -> DatasetStatusSnapshot:
        """Wait until a request reaches a terminal state."""

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            request = self._requests.get(token.request_id)
            if request is None:
                raise DatasetValidationError(
                    f"unknown dataset request token: {token}"
                )
            while request.state not in _TERMINAL_STATES:
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"dataset request {token.request_id} timed out"
                        )
                self._condition.wait(remaining)
            return self._snapshot(request)

    def close(self) -> None:
        """Cancel in-flight work and stop the worker."""

        with self._condition:
            if self._closed:
                return
            self._closed = True
            if self._pending_request is not None:
                self._supersede_request(self._pending_request)
            if (
                self._active_request is not None
                and self._active_request.state in _ACTIVE_STATES
            ):
                self._supersede_request(self._active_request)
            self._condition.notify_all()
        self._worker.join(timeout=5.0)

    def __enter__(self) -> DatasetLifecycleManager:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _register_descriptor(self, descriptor: DatasetDescriptor) -> None:
        if not isinstance(descriptor.dataset_id, str) or not descriptor.dataset_id:
            raise DatasetValidationError("dataset_id must be a non-empty string")
        if descriptor.dataset_id in self._descriptors:
            raise DatasetValidationError(
                f"duplicate dataset_id: {descriptor.dataset_id}"
            )

        trace_path = Path(descriptor.trace_path).resolve(strict=False)
        cache_path = (
            Path(descriptor.cache_path).resolve(strict=False)
            if descriptor.cache_path is not None
            else None
        )
        if not self._is_within_root(trace_path):
            raise DatasetPathError(
                f"trace path is outside approved root: {trace_path}"
            )
        if cache_path is not None and not self._is_within_root(cache_path):
            raise DatasetPathError(
                f"cache path is outside approved root: {cache_path}"
            )

        args = descriptor.args
        if not hasattr(args, "trace"):
            raise DatasetValidationError("descriptor args must define trace")
        if Path(args.trace).resolve(strict=False) != trace_path:
            raise DatasetValidationError(
                "descriptor trace path does not match args.trace"
            )
        args_cache = getattr(args, "trace_cache", None)
        if cache_path is None:
            if args_cache is not None:
                raise DatasetValidationError(
                    "descriptor has no cache path but args.trace_cache is set"
                )
        else:
            if args_cache is None:
                raise DatasetValidationError(
                    "descriptor cache path is missing from args.trace_cache"
                )
            if Path(args_cache).resolve(strict=False) != cache_path:
                raise DatasetValidationError(
                    "descriptor cache path does not match args.trace_cache"
                )

        self._descriptors[descriptor.dataset_id] = replace(
            descriptor,
            trace_path=trace_path,
            cache_path=cache_path,
        )

    def _is_within_root(self, path: Path) -> bool:
        try:
            path.relative_to(self._root)
            return True
        except ValueError:
            return False

    def _submit_request(self, dataset_id: str) -> DatasetRequestToken:
        with self._condition:
            if self._closed:
                raise DatasetManagerClosedError("dataset lifecycle manager is closed")
            descriptor = self._descriptors.get(dataset_id)
            if descriptor is None:
                raise UnknownDatasetError(f"dataset was not approved: {dataset_id}")
            resolved = descriptor.trace_path.resolve(strict=True)
            if not self._is_within_root(resolved):
                raise DatasetPathError("trace path now resolves outside the approved root")
            is_ctf = getattr(descriptor.args, "trace_format", None) == "ctf"
            if not (resolved.is_file() or (is_ctf and resolved.is_dir())):
                raise DatasetPathError(f"Select a trace file or CTF directory: {resolved}")
            if descriptor.cache_path is not None and not self._is_within_root(descriptor.cache_path.resolve(strict=False)):
                raise DatasetPathError("cache path now resolves outside the approved root")
            # Keep bounded request metadata; stores are retained only by the two handles.
            if len(self._requests) > 64:
                protected = {r.token.request_id for r in (self._latest_request, self._active_request, self._pending_request) if r}
                for key, old in list(self._requests.items()):
                    if key not in protected and old.state in _TERMINAL_STATES:
                        del self._requests[key]
                        if len(self._requests) <= 48:
                            break

            self._generation += 1
            token = DatasetRequestToken(
                request_id=uuid.uuid4().hex,
                dataset_id=dataset_id,
                generation=self._generation,
            )
            request = _LifecycleRequest(
                token=token,
                descriptor=descriptor,
                cancellation=DatasetCancellation(),
                created_at=time.monotonic(),
            )
            self._requests[token.request_id] = request
            self._latest_request = request

            if (
                self._active_request is not None
                and self._active_request.state in _ACTIVE_STATES
            ):
                self._supersede_request(self._active_request)
            if self._pending_request is not None:
                self._supersede_request(self._pending_request)

            self._pending_request = request
            self._condition.notify_all()
            return token

    def _supersede_request(self, request: _LifecycleRequest) -> None:
        request.superseded = True
        request.state = DatasetState.SUPERSEDED
        request.ended_at = time.monotonic()
        request.cancellation.request()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while self._pending_request is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                request = self._pending_request
                self._pending_request = None
                if request.superseded:
                    continue
            self._process_request(request)

    def _process_request(self, request: _LifecycleRequest) -> None:
        with self._condition:
            self._active_request = request
            request.state = DatasetState.LOADING
            request.started_at = time.monotonic()
            request.error = None

        try:
            stage = (
                "cache_reopen"
                if request.descriptor.cache_path is not None
                and request.descriptor.cache_path.is_file()
                else "cold_source_load"
            )
            performance_operation = None
            if self._performance_recorder is not None:
                dataset_token = request.descriptor.dataset_id
                request_token = request.token.request_id
                selection_token = request.descriptor.dataset_id
                performance_operation = self._performance_recorder.begin(
                    stage,
                    dataset_token=dataset_token,
                    request_token=request_token,
                    selection_token=selection_token,
                )
            try:
                store = self._loader(request.descriptor.args, request.cancellation)
            except Exception as load_error:
                if performance_operation is not None:
                    self._performance_recorder.record_exception(performance_operation, load_error)
                raise
            if performance_operation is not None:
                self._performance_recorder.finish(
                    performance_operation,
                    result_identity=MetricIdentity(
                        dataset_token=request.descriptor.dataset_id,
                        request_token=request.token.request_id,
                        selection_token=request.descriptor.dataset_id,
                    ),
                    payload_bytes=0,
                )
        except Exception as error:
            with self._condition:
                request.ended_at = time.monotonic()
                if request.superseded:
                    request.state = DatasetState.SUPERSEDED
                elif request.cancelled:
                    request.state = DatasetState.CANCELLED
                else:
                    request.state = DatasetState.FAILED
                    request.error = str(error)
                if self._active_request is request:
                    self._active_request = None
                self._condition.notify_all()
            return

        with self._condition:
            request.ended_at = time.monotonic()
            if request.superseded:
                request.state = DatasetState.SUPERSEDED
            elif request.cancelled:
                request.state = DatasetState.CANCELLED
            else:
                handle = DatasetHandle(
                    token=request.token,
                    descriptor=request.descriptor,
                    store=store,
                )
                self._current_handle = handle
                self._last_good_handle = handle
                request.state = DatasetState.READY
            if self._active_request is request:
                self._active_request = None
            self._condition.notify_all()

    def _snapshot(self, request: _LifecycleRequest) -> DatasetStatusSnapshot:
        state = request.state
        if state is DatasetState.READY:
            progress = DatasetProgress(stage="ready", fraction=1.0)
        elif state is DatasetState.LOADING:
            progress = DatasetProgress(stage="loading", fraction=None)
        elif state is DatasetState.QUEUED:
            progress = DatasetProgress(stage="queued", fraction=0.0)
        elif state is DatasetState.CANCELLING:
            progress = DatasetProgress(stage="cancelling", fraction=None)
        elif state is DatasetState.CANCELLED:
            progress = DatasetProgress(stage="cancelled", fraction=0.0)
        elif state is DatasetState.SUPERSEDED:
            progress = DatasetProgress(stage="superseded", fraction=0.0)
        elif state is DatasetState.FAILED:
            progress = DatasetProgress(stage="failed", fraction=0.0)
        else:
            progress = DatasetProgress(stage="idle")

        return DatasetStatusSnapshot(
            token=request.token,
            state=state,
            progress=progress,
            error=request.error,
            current_dataset_id=self._current_handle.descriptor.dataset_id
            if self._current_handle is not None
            else None,
            last_good_dataset_id=self._last_good_handle.descriptor.dataset_id
            if self._last_good_handle is not None
            else None,
            created_at=request.created_at,
            started_at=request.started_at,
            ended_at=request.ended_at,
            cancelled=request.cancelled,
            superseded=request.superseded,
        )
