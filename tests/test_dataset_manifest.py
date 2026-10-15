"""Focused tests for the approved-dataset manifest loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import main
import dataset_lifecycle
from main import _load_dataset_manifest, parse_args, run_trace_app


def write_manifest(path: Path, document: dict) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_manifest_builds_approved_descriptors(tmp_path):
    root = tmp_path / "approved"
    root.mkdir()
    trace = root / "alpha.ctf"
    trace.write_text("trace", encoding="utf-8")
    manifest = write_manifest(
        tmp_path / "datasets.json",
        {
            "approved_root": str(root),
            "initial_dataset_id": "alpha",
            "datasets": [
                {
                    "dataset_id": "alpha",
                    "trace_path": "alpha.ctf",
                    "trace_format": "ctf",
                }
            ],
        },
    )

    approved_root, descriptors, initial_dataset_id = _load_dataset_manifest(
        parse_args([]), manifest
    )

    assert approved_root == root.resolve()
    assert initial_dataset_id == "alpha"
    assert len(descriptors) == 1
    assert descriptors[0].dataset_id == "alpha"
    assert descriptors[0].trace_path == trace.resolve()
    assert descriptors[0].args.trace_format == "ctf"


def test_manifest_rejects_unapproved_initial_dataset(tmp_path):
    root = tmp_path / "approved"
    root.mkdir()
    manifest = write_manifest(
        tmp_path / "datasets.json",
        {
            "approved_root": str(root),
            "initial_dataset_id": "missing",
            "datasets": [
                {
                    "dataset_id": "alpha",
                    "trace_path": "alpha.ctf",
                    "trace_format": "ctf",
                }
            ],
        },
    )

    with pytest.raises(RuntimeError, match="initial dataset is not approved"):
        _load_dataset_manifest(parse_args([]), manifest)


def test_manifest_requires_at_least_one_dataset(tmp_path):
    manifest = write_manifest(
        tmp_path / "datasets.json",
        {"approved_root": str(tmp_path), "initial_dataset_id": "alpha", "datasets": []},
    )

    with pytest.raises(RuntimeError, match="at least one dataset"):
        _load_dataset_manifest(parse_args([]), manifest)


def test_run_trace_app_uses_manifest_and_closes_manager(
    tmp_path, monkeypatch
):
    root = tmp_path / "approved"
    root.mkdir()
    trace = root / "alpha.ctf"
    trace.write_text("trace", encoding="utf-8")
    manifest = write_manifest(
        tmp_path / "datasets.json",
        {
            "approved_root": str(root),
            "initial_dataset_id": "alpha",
            "datasets": [
                {
                    "dataset_id": "alpha",
                    "trace_path": "alpha.ctf",
                    "trace_format": "ctf",
                }
            ],
        },
    )

    class FakeHandle:
        store = object()

    class FakeManager:
        def __init__(self, approved_root, descriptors, performance_recorder=None):
            self.approved_root = approved_root
            self.descriptors = descriptors
            self.performance_recorder = performance_recorder
            self.opened = None
            self.closed = False

        def open(self, dataset_id):
            self.opened = dataset_id
            return object()

        def wait(self, token, timeout=None):
            return dataset_lifecycle.DatasetStatusSnapshot(
                token=token,
                state=dataset_lifecycle.DatasetState.READY,
                progress=dataset_lifecycle.DatasetProgress(
                    stage="ready", fraction=1.0
                ),
            )

        def current(self):
            return FakeHandle()

        def close(self):
            self.closed = True

    fake_manager = FakeManager
    monkeypatch.setattr(main, "DatasetLifecycleManager", fake_manager)
    created_apps = []
    monkeypatch.setattr(
        main,
        "create_analysis_app",
        lambda store, dataset_manager=None, performance_recorder=None, **kwargs: created_apps.append(
            (store, dataset_manager, performance_recorder)
        ),
    )
    server_calls = []
    monkeypatch.setattr(
        main,
        "run_wsgi_app",
        lambda app, host, port, server: server_calls.append(
            (app, host, port, server)
        ),
    )

    args = parse_args(["--dataset-manifest", str(manifest)])
    run_trace_app(args)

    assert len(created_apps) == 1
    assert created_apps[0][1] is not None
    assert created_apps[0][1].opened == "alpha"
    assert created_apps[0][1].closed is True
    assert len(server_calls) == 1
