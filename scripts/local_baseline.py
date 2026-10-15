#!/usr/bin/env python3

import argparse
import hashlib
import importlib.metadata
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
RUNTIME_ROOT = ROOT / "tests" / "runtime"
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(RUNTIME_ROOT))

from app import create_app
from main import read_versioned_cache, source_fingerprint, write_versioned_cache
from metrics_processing import prepare_metric_array, process_file_pair
from oneccl_processing import prepare_benchmark_array
from synthetic_fixture import create_legacy_fixture, create_minimal_app_data


@dataclass
class StageResult:
    median_ms: float
    minimum_ms: float
    maximum_ms: float
    rss_after_bytes: int


def memory_bytes():
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except ImportError:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def peak_memory_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def elapsed_milliseconds(start):
    return (time.perf_counter_ns() - start) / 1_000_000


def measure(function, repeats):
    function()
    timings = []
    rss_samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = function()
        timings.append(elapsed_milliseconds(start))
        rss_samples.append(memory_bytes())
    return result, StageResult(
        median_ms=statistics.median(timings),
        minimum_ms=min(timings),
        maximum_ms=max(timings),
        rss_after_bytes=int(statistics.median(rss_samples)),
    )


def discover_metric_pairs(base_dir):
    pairs = []
    for directory, directories, filenames in os.walk(base_dir):
        directories.sort()
        filenames.sort()
        before_files = {
            filename.split("metric_before.", 1)[1]: Path(directory) / filename
            for filename in filenames
            if "metric_before." in filename
        }
        after_files = {
            filename.split("metric_after.", 1)[1]: Path(directory) / filename
            for filename in filenames
            if "metric_after." in filename
        }
        for identifier in sorted(before_files):
            if identifier in after_files:
                pairs.append((before_files[identifier], after_files[identifier], identifier))
    return pairs


def parse_metric_pairs(pairs):
    results = {(1, 8): []}
    result_nodes = {(1, 8): []}
    for before_path, after_path, identifier in pairs:
        results[(1, 8)].append(process_file_pair(before_path, after_path))
        result_nodes[(1, 8)].append(identifier)
    return results, result_nodes


def fixture_digest(fixture_root):
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(fixture_root.rglob("*")):
        if path.is_file():
            content = path.read_bytes()
            total_bytes += len(content)
            digest.update(path.relative_to(fixture_root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(content)
    return digest.hexdigest(), total_bytes


def file_hash(path):
    if not path.is_file():
        return "not-present"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(command):
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def module_version(module_name):
    try:
        return importlib.metadata.version(module_name)
    except importlib.metadata.PackageNotFoundError:
        try:
            module = __import__(module_name)
        except ImportError:
            return "unavailable"
        return str(getattr(module, "__version__", "unknown"))


def format_number(value):
    return f"{value:,.1f}"


def run_baseline(repeats, output_path):
    with tempfile.TemporaryDirectory(prefix="nic-legacy-baseline-") as temporary_directory:
        fixture_root = Path(temporary_directory)
        create_legacy_fixture(fixture_root)
        digest, fixture_bytes = fixture_digest(fixture_root)
        pairs, discovery_result = measure(
            lambda: discover_metric_pairs(fixture_root), repeats
        )
        parsed, parse_result = measure(lambda: parse_metric_pairs(pairs), repeats)
        parsed_results, parsed_node_names = parsed

        node_map = {"node-a": 0, "node-b": 1}
        result_nodes = {
            (1, 8): [node_map[node_name] for node_name in parsed_node_names[(1, 8)]]
        }
        metric_array, node_counts, element_counts = prepare_metric_array(
            parsed_results,
            num_interfaces=2,
            max_nodes=2,
            results_nodes=result_nodes,
        )

        def normalize_metrics():
            return prepare_metric_array(
                parsed_results,
                num_interfaces=2,
                max_nodes=2,
                results_nodes=result_nodes,
            )

        normalized, normalize_result = measure(normalize_metrics, repeats)
        metric_array = normalized[0]

        benchmark_data = {
            1: {
                "element_counts": [8],
                "data": create_minimal_app_data()["bench_array"].reshape(1, 3, 4),
            }
        }
        benchmark, benchmark_normalize_result = measure(
            lambda: prepare_benchmark_array(benchmark_data), repeats
        )
        bench_array, bench_node_counts, bench_elements = benchmark

        allocation_shape = metric_array.shape

        def allocate_metric_array():
            return metric_array.copy()

        _, allocation_result = measure(allocate_metric_array, repeats)

        cache_payload = {
            "node_map": node_map,
            "results_nodes": result_nodes,
            "metric_array": metric_array,
            "node_counts": node_counts,
            "element_counts": element_counts,
            "metric_names": ["rx_bytes", "tx_bytes"],
            "bench_array": bench_array,
            "bench_node_counts": bench_node_counts,
            "bench_elements": bench_elements,
        }
        cache_path = fixture_root / "cached_data.json"
        fingerprint = source_fingerprint([fixture_root])

        def write_cache():
            write_versioned_cache(cache_path, cache_payload, fingerprint)

        def read_cache():
            return read_versioned_cache(cache_path, fingerprint)

        _, cache_write_result = measure(write_cache, repeats)
        _, cache_read_result = measure(read_cache, repeats)

        app_data = {
            "metric_array": metric_array,
            "node_counts": node_counts,
            "element_counts": element_counts,
            "metric_names": ["rx_bytes", "tx_bytes"],
            "node_map": node_map,
            "results_nodes": result_nodes,
            "bench_array": bench_array,
            "bench_node_counts": bench_node_counts,
            "bench_elements": bench_elements,
        }
        _, app_creation_result = measure(lambda: create_app(**app_data), repeats)

        stages = [
            ("discovery", discovery_result),
            ("parse_pairs", parse_result),
            ("normalize_metrics", normalize_result),
            ("normalize_benchmarks", benchmark_normalize_result),
            ("array_allocation", allocation_result),
            ("cache_write_json", cache_write_result),
            ("cache_read_json", cache_read_result),
            ("app_creation", app_creation_result),
        ]

        expected_differences = [[1, 1, 2, 2], [1, 1, 2, 2]]
        if parsed_results[(1, 8)] != expected_differences:
            raise AssertionError(
                f"synthetic fixture values differ from reference: {parsed_results[(1, 8)]}"
            )

        commit = git_output(["git", "rev-parse", "HEAD"])
        status = git_output(["git", "status", "--short", "--untracked-files=all"])
        owned_files = [
            ROOT / "src" / "app.py",
            ROOT / "src" / "callbacks.py",
            ROOT / "src" / "expressions.py",
            ROOT / "src" / "main.py",
            ROOT / "src" / "metrics_processing.py",
            ROOT / "pyproject.toml",
            ROOT / "scripts" / "local_baseline.py",
            ROOT / "tests" / "test_expressions.py",
            ROOT / "tests" / "test_metrics_processing.py",
            ROOT / "tests" / "test_runtime_cache.py",
            ROOT / "tests" / "test_entrypoint_runtime.py",
            ROOT / "scripts" / "capture_ui.py",
            ROOT / "tests" / "runtime" / "synthetic_fixture.py",
            ROOT / "tests" / "runtime" / "test_app_runtime.py",
        ]
        owned_hashes = {
            str(path.relative_to(ROOT)): file_hash(path) for path in owned_files
        }
        owned_hashes.update(
            {
                str(path.relative_to(ROOT)): file_hash(path)
                for path in sorted(
                    (ROOT / "examples" / "legacy_fixture").rglob("*")
                )
                if path.is_file()
            }
        )

        lines = [
            "# Local legacy baseline",
            "",
            "**Status:** local only",
            f"**Date/time and timezone:** {datetime.now(timezone.utc).isoformat()}",
            f"**App commit:** `{commit}`",
            "**Dirty-file status:**",
            "```text",
            status,
            "```",
            "**Owned source hashes:**",
            "```text",
            *(
                f"{hash_value}  {path}"
                for path, hash_value in sorted(owned_hashes.items())
            ),
            "```",
            "**THAPI version / source commit / conversion version:** not applicable; synthetic local data only.",
            "**Environment:**",
            f"- Python `{platform.python_version()}` on `{platform.platform()}`",
            f"- CPU count `{os.cpu_count()}`, Dash `{module_version('dash')}`, Flask `{module_version('flask')}`, NumPy `{module_version('numpy')}`, Plotly `{module_version('plotly')}`, psutil `{module_version('psutil')}`",
            f"- Repeats per stage: `{repeats}` after one warm-up call",
            "",
            "**Dataset:** deterministic synthetic legacy fixture",
            f"- Logical fixture: one job, one `(node_count=1, elements=8)` cell, two nodes, two interfaces, four interface-major metric values.",
            f"- On-disk bytes: `{fixture_bytes:,}`; SHA-256 `{digest}`.",
            "- Benchmark data: one synthetic oneCCL-style node count, element count, three iterations, and four statistics.",
            "",
            "## Commands and expected results",
            "",
            "```bash",
            "cd AuroraNIC-MetricVisualizer",
            "python -m pip install -e '.[test]'",
            "PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q",
            "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/runtime .venv/bin/python scripts/local_baseline.py --output artifacts/local-baseline.md",
            "python -m pip install -e '.[browser]' && python -m playwright install chromium",
            "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/capture_ui.py --output artifacts/legacy-ui.png",
            "```",
            "",
            "The first command must exercise health/readiness and SIGTERM shutdown. The second command must regenerate this summarized record and fail if fixture values differ from the independent reference.",
            "",
            "## Observed results",
            "",
            "- Synthetic parse reference passed: each node has interface-major differences `[1, 1, 2, 2]`; zero remains distinct from missing.",
            "- Runtime tests covered loopback defaults, environment overrides, invalid ports, health/readiness responses, zero-valued readiness, and clean SIGTERM exit.",
            "- Timings are warm in-process medians after one warm-up; they are not cold-cache or multi-user Sunspot measurements.",
            "",
            "| Stage | Median (ms) | Min (ms) | Max (ms) | Median RSS after (MiB) |",
            "|---|---:|---:|---:|---:|",
        ]
        lines.extend(
            (
                f"| {name} | {result.median_ms:.3f} | {result.minimum_ms:.3f} "
                f"| {result.maximum_ms:.3f} | {result.rss_after_bytes / (1024 * 1024):.1f} |"
            )
            for name, result in stages
        )
        lines.extend(
            [
                "",
                f"- Metric array allocation shape: `{allocation_shape}`.",
                f"- Process peak RSS: `{format_number(peak_memory_bytes() / (1024 * 1024))} MiB`.",
                "- Versioned JSON cache I/O is measured with the production cache functions and a source fingerprint.",
                "",
                "## Artifacts",
                "",
                "- Runtime tests: `tests/runtime/test_app_runtime.py`",
                "- Deterministic fixture interfaces: `tests/runtime/synthetic_fixture.py`",
                "- Baseline harness: `scripts/local_baseline.py`",
                "- Browser capture harness: `scripts/capture_ui.py`",
                "",
                "## Cleanup and remaining limitations",
                "",
                "- The fixture and versioned JSON cache live in a Python temporary directory and are removed automatically.",
                "- No server is left running by the harness; the shutdown test owns and terminates its child process.",
                "- This harness measures synthetic local data only. Run the separate browser capture and runtime tests to check rendering, health/readiness and shutdown; no real THAPI trace, Sunspot allocation or tunnel is exercised here.",
            "- Entry-point runtime integration uses the shared `app.run_app` loopback/shutdown implementation.",
            ]
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Measure the local legacy baseline.")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "local-baseline.md",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    run_baseline(args.repeats, args.output)


if __name__ == "__main__":
    main()
