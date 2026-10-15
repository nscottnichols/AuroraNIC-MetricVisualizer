from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest

import dataset_lifecycle
from dataset_lifecycle import (
    DatasetDescriptor,
    DatasetLifecycleManager,
    DatasetManagerClosedError,
    DatasetPathError,
    DatasetState,
    UnknownDatasetError,
)


def _descriptor(root: Path, dataset_id: str, *, cache: bool = False):
    dataset_dir = root / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    trace_path = dataset_dir / "trace"
    trace_path.write_text(f"{dataset_id}\n", encoding="utf-8")
    cache_path = dataset_dir / "cache.npz" if cache else None
    args = SimpleNamespace(
        trace=str(trace_path),
        trace_cache=str(cache_path) if cache_path is not None else None,
    )
    return DatasetDescriptor(
        dataset_id=dataset_id,
        trace_path=trace_path,
        cache_path=cache_path,
        args=args,
    )


def _wait_for_state(
    manager: DatasetLifecycleManager,
    token,
    state: DatasetState,
    *,
    timeout: float = 2.0,
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = manager.status(token)
        if snapshot.state is state:
            return snapshot
        time.sleep(0.005)
    raise AssertionError(f"request did not reach {state}: {manager.status(token)}")


def test_rejects_unknown_dataset(tmp_path):
    root = tmp_path
    manager = DatasetLifecycleManager(root, [], loader=lambda args, cancellation: object())
    try:
        with pytest.raises(UnknownDatasetError):
            manager.open("missing")
    finally:
        manager.close()


def test_rejects_path_outside_approved_root(tmp_path):
    root = tmp_path / "approved"
    root.mkdir()
    outside = tmp_path / "outside" / "trace"
    outside.parent.mkdir()
    outside.write_text("outside\n", encoding="utf-8")
    descriptor = DatasetDescriptor(
        dataset_id="outside",
        trace_path=outside,
        cache_path=None,
        args=SimpleNamespace(trace=str(outside), trace_cache=None),
    )
    with pytest.raises(DatasetPathError):
        DatasetLifecycleManager(root, [descriptor], loader=lambda args, cancellation: object())


def test_open_and_switch_track_generation_and_current_dataset(tmp_path):
    root = tmp_path
    descriptors = [_descriptor(root, "alpha"), _descriptor(root, "beta")]

    def loader(args, cancellation):
        return SimpleNamespace(source=args.trace)

    manager = DatasetLifecycleManager(root, descriptors, loader=loader)
    try:
        first = manager.open("alpha")
        first_status = manager.wait(first, timeout=2.0)
        assert first_status.state is DatasetState.READY
        assert first.generation == 1
        assert manager.current().descriptor.dataset_id == "alpha"
        assert manager.last_good().descriptor.dataset_id == "alpha"

        second = manager.switch("beta")
        second_status = manager.wait(second, timeout=2.0)
        assert second_status.state is DatasetState.READY
        assert second.generation == 2
        assert manager.current().descriptor.dataset_id == "beta"
        assert manager.last_good().descriptor.dataset_id == "beta"
    finally:
        manager.close()


def test_failed_switch_preserves_active_and_last_good(tmp_path):
    root = tmp_path
    descriptors = [_descriptor(root, "alpha"), _descriptor(root, "beta")]

    def loader(args, cancellation):
        if Path(args.trace).parent.name == "beta":
            raise RuntimeError("synthetic load failure")
        return SimpleNamespace(source=args.trace)

    manager = DatasetLifecycleManager(root, descriptors, loader=loader)
    try:
        good = manager.open("alpha")
        manager.wait(good, timeout=2.0)

        failed = manager.switch("beta")
        failed_status = manager.wait(failed, timeout=2.0)
        assert failed_status.state is DatasetState.FAILED
        assert failed_status.error == "synthetic load failure"
        assert manager.current().descriptor.dataset_id == "alpha"
        assert manager.last_good().descriptor.dataset_id == "alpha"
    finally:
        manager.close()


def test_cancel_is_cooperative_and_reports_cancelled(tmp_path):
    root = tmp_path
    descriptor = _descriptor(root, "alpha")

    def loader(args, cancellation):
        while not cancellation.should_cancel():
            time.sleep(0.005)
        return SimpleNamespace(source=args.trace)

    manager = DatasetLifecycleManager(root, [descriptor], loader=loader)
    try:
        token = manager.open("alpha")
        manager.cancel(token)
        status = manager.wait(token, timeout=2.0)
        assert status.state is DatasetState.CANCELLED
        assert status.cancelled is True
        assert manager.current() is None
        assert manager.last_good() is None
    finally:
        manager.close()


def test_superseded_request_cannot_replace_newer_dataset(tmp_path):
    root = tmp_path
    descriptors = [_descriptor(root, "alpha"), _descriptor(root, "beta")]
    release = threading.Event()

    def loader(args, cancellation):
        if Path(args.trace).parent.name == "alpha":
            release.wait(timeout=2.0)
            return SimpleNamespace(source="stale-alpha")
        return SimpleNamespace(source=args.trace)

    manager = DatasetLifecycleManager(root, descriptors, loader=loader)
    try:
        stale = manager.open("alpha")
        _wait_for_state(manager, stale, DatasetState.LOADING)

        fresh = manager.switch("beta")
        stale_status = manager.status(stale)
        assert stale_status.state is DatasetState.SUPERSEDED
        assert stale_status.superseded is True

        release.set()
        fresh_status = manager.wait(fresh, timeout=2.0)
        assert fresh_status.state is DatasetState.READY
        assert Path(manager.current().store.source).parent.name == "beta"
        assert Path(manager.last_good().store.source).parent.name == "beta"
        assert manager.status(stale).state is DatasetState.SUPERSEDED
    finally:
        release.set()
        manager.close()


def test_close_supersedes_active_request_and_rejects_new_work(tmp_path):
    root = tmp_path
    descriptor = _descriptor(root, "alpha")

    def loader(args, cancellation):
        while not cancellation.should_cancel():
            time.sleep(0.005)
        return SimpleNamespace(source=args.trace)

    manager = DatasetLifecycleManager(root, [descriptor], loader=loader)
    token = manager.open("alpha")
    _wait_for_state(manager, token, DatasetState.LOADING)
    manager.close()

    assert manager.status(token).state is DatasetState.SUPERSEDED
    with pytest.raises(DatasetManagerClosedError):
        manager.open("alpha")


def test_default_loader_delegates_to_production_loader(tmp_path, monkeypatch):
    root = tmp_path
    descriptor = _descriptor(root, "alpha", cache=True)
    captured = {}

    def fake_production_loader(args, cancellation):
        captured["args"] = args
        captured["cancelled"] = cancellation.should_cancel()
        return SimpleNamespace(source="production")

    monkeypatch.setattr(dataset_lifecycle, "_production_loader", fake_production_loader)
    manager = DatasetLifecycleManager(root, [descriptor])
    try:
        token = manager.open("alpha")
        status = manager.wait(token, timeout=2.0)
        assert status.state is DatasetState.READY
        assert captured["args"] is descriptor.args
        assert captured["cancelled"] is False
        assert manager.current().store.source == "production"
    finally:
        manager.close()
