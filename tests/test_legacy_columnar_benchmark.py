from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = APP_ROOT / "scripts" / "legacy_columnar_benchmark.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "legacy_columnar_benchmark_under_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_logical_fixture_is_deterministic():
    module = _load_benchmark_module()
    first = module.build_logical_fixture(8)
    second = module.build_logical_fixture(8)

    assert np.array_equal(first.before_values, second.before_values)
    assert np.array_equal(first.after_values, second.after_values)
    assert np.array_equal(first.increments, second.increments)
    assert first.series_count == second.series_count
    assert first.batch.row_count if hasattr(first.batch, "row_count") else True


def test_node_labels_preserve_numeric_order_above_999():
    module = _load_benchmark_module()
    fixture = module.build_logical_fixture(1024)
    labels = [node.label for node in fixture.nodes]

    assert labels[0] == "node-0000"
    assert labels[-1] == "node-1023"
    assert labels == sorted(labels)


def test_benchmark_result_schema_and_small_end_to_end_run():
    module = _load_benchmark_module()
    config = module.BenchmarkConfig(
        node_count=4,
        repetitions=1,
        warmup_repetitions=0,
        max_runtime_seconds=15,
        max_rss_mib=4096,
    )
    result = module.run_benchmark(config)

    assert result["schema_version"] == 1
    assert result["status"] == "ok"
    assert result["benchmark"] == "legacy_dense_vs_columnar_trace_store"
    assert set(result["measurements"]) == {
        "legacy_text_load_and_dense_normalization",
        "legacy_dense_whole_dataset_aggregation",
        "columnar_store_construction",
        "columnar_whole_dataset_aggregation",
    }
    assert all(
        {"repetitions", "median_ms", "p95_ms", "min_ms", "max_ms"}.issubset(summary)
        for summary in result["measurements"].values()
    )
    assert result["dataset"]["same_logical_dataset"] is True
    assert result["dataset"]["columnar_row_count"] == 4 * 8 * 2 * 2
    aggregate = result["aggregate"]
    assert len(aggregate["totals"]) == 8
    assert len(aggregate["counts"]) == 8
    assert len(aggregate["minimums"]) == 8
    assert len(aggregate["maximums"]) == 8
    assert len(aggregate["means"]) == 8
    assert aggregate["legacy_matches_columnar"] is True
    assert all(count == 4 * 2 for count in aggregate["counts"])
    assert all(result["correctness"].values())
    assert result["resources"]["peak_rss_mib"] > 0


def test_cli_small_end_to_end_run(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--nodes",
            "4",
            "--repetitions",
            "1",
            "--warmup",
            "0",
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
    assert all(result["correctness"].values())
