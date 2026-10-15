"""Deterministic ingestion for the legacy before/after text counters."""

from __future__ import annotations

import concurrent.futures
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np


MAX_FILE_PAIR_WORKERS = 4
_MAX_COUNTER_VALUE = np.iinfo(np.int64).max
_COUNTER_LINE_PATTERN = re.compile(r"^([0-9]+)@([^@\s]+)$")
_SUBDIRECTORY_PATTERN = re.compile(r"^([1-9][0-9]*)_([1-9][0-9]*)$")


class MetricIngestionError(ValueError):
    """Raised when legacy counter input cannot be ingested unambiguously."""


def parse_metric_records(filepath: str | os.PathLike[str]) -> tuple[list[int], list[str]]:
    """Return counter values and labels in file order."""
    path = Path(filepath)
    values: list[int] = []
    labels: list[str] = []
    seen_labels: set[str] = set()

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise MetricIngestionError(f"Cannot read metric file {path}: {exc}") from exc

    if not lines:
        raise MetricIngestionError(f"Metric file {path} is empty")

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        match = _COUNTER_LINE_PATTERN.fullmatch(line)
        if match is None:
            raise MetricIngestionError(
                f"Malformed counter at {path}:{line_number}: expected "
                "'<nonnegative integer>@<label>'"
            )

        value = int(match.group(1), 10)
        if value > _MAX_COUNTER_VALUE:
            raise MetricIngestionError(
                f"Counter at {path}:{line_number} exceeds signed 64-bit range: {value}"
            )
        label = match.group(2)
        if label in seen_labels:
            raise MetricIngestionError(
                f"Duplicate metric label {label!r} at {path}:{line_number}"
            )
        seen_labels.add(label)
        values.append(value)
        labels.append(label)

    return values, labels


def parse_metric_file(filepath: str | os.PathLike[str]) -> np.ndarray:
    """Parse legacy counter values as a one-dimensional int64 array."""
    values, _ = parse_metric_records(filepath)
    return np.asarray(values, dtype=np.int64)


def process_file_pair(
    before_path: str | os.PathLike[str],
    after_path: str | os.PathLike[str],
    expected_run: str | None = None,
    expected_node: str | None = None,
) -> list[int]:
    """Return ordered after-minus-before differences for one node."""
    before_values, before_labels = parse_metric_records(before_path)
    after_values, after_labels = parse_metric_records(after_path)

    if len(before_values) != len(after_values):
        raise MetricIngestionError(
            f"Before/after files have different lengths for {before_path} "
            f"({len(before_values)}) and {after_path} ({len(after_values)})"
        )
    if before_labels != after_labels:
        raise MetricIngestionError(
            f"Before/after metric labels differ for {before_path} and {after_path}"
        )
    _validate_metric_identity(
        before_labels,
        before_path,
        expected_run=expected_run,
        expected_node=expected_node,
    )

    return [after - before for before, after in zip(before_values, after_values)]


def _validate_metric_identity(labels, path, expected_run, expected_node):
    if expected_run is None and expected_node is None:
        return

    for label in labels:
        parts = label.split(".")
        if len(parts) < 4 or any(not part for part in parts):
            raise MetricIngestionError(
                f"Metric label {label!r} in {path} must encode "
                "<run>.<node>.<NIC>.<counter>"
            )
        if expected_run is not None and parts[0] != expected_run:
            raise MetricIngestionError(
                f"Metric label {label!r} in {path} does not identify run "
                f"{expected_run!r}"
            )
        if expected_node is not None and parts[1] != expected_node:
            raise MetricIngestionError(
                f"Metric label {label!r} in {path} does not identify node "
                f"{expected_node!r}"
            )


def _exact_metric_filename(filename: str) -> tuple[str, str] | None:
    if filename.startswith("metric_before."):
        return "before", filename.removeprefix("metric_before.")
    if filename.startswith("metric_after."):
        return "after", filename.removeprefix("metric_after.")
    return None


def _collect_file_pairs(subdir_path: Path) -> list[tuple[Path, Path, str]]:
    before_files: dict[str, Path] = {}
    after_files: dict[str, Path] = {}

    try:
        entries = sorted(os.listdir(subdir_path))
    except OSError as exc:
        raise MetricIngestionError(f"Cannot list {subdir_path}: {exc}") from exc

    for filename in entries:
        entry_path = subdir_path / filename
        if not entry_path.is_file():
            raise MetricIngestionError(
                f"Unexpected directory in metric pair directory: {entry_path}"
            )

        parsed = _exact_metric_filename(filename)
        if parsed is None or not parsed[1]:
            raise MetricIngestionError(f"Invalid metric filename: {entry_path}")

        kind, identifier = parsed
        files = before_files if kind == "before" else after_files
        if identifier in files:
            raise MetricIngestionError(
                f"Duplicate {kind} identifier {identifier!r} in {subdir_path}"
            )
        files[identifier] = entry_path

    if not before_files and not after_files:
        raise MetricIngestionError(f"Metric pair directory is empty: {subdir_path}")

    missing_before = sorted(set(after_files) - set(before_files))
    missing_after = sorted(set(before_files) - set(after_files))
    if missing_before or missing_after:
        details = []
        if missing_before:
            details.append(f"missing before for {missing_before}")
        if missing_after:
            details.append(f"missing after for {missing_after}")
        raise MetricIngestionError(
            f"Incomplete metric file pairs in {subdir_path}: " + "; ".join(details)
        )

    return [
        (before_files[identifier], after_files[identifier], identifier)
        for identifier in sorted(before_files)
    ]


def process_single_job(job_dir: str, base_dir: str | os.PathLike[str]):
    """Process one legacy job directory deterministically.

    Returned node identifiers are filename identifiers. Run-qualified
    identities are introduced by :func:`process_all_jobs`.
    """
    if not job_dir or job_dir in {".", ".."} or Path(job_dir).name != job_dir:
        raise MetricIngestionError(f"Invalid job directory name: {job_dir!r}")

    job_path = Path(base_dir) / job_dir
    if not job_path.is_dir():
        raise MetricIngestionError(f"Job directory does not exist: {job_path}")

    metrics_path = job_path / f"metrics_{job_dir}"
    if not metrics_path.is_dir():
        raise MetricIngestionError(
            f"Job metrics directory does not exist: {metrics_path}"
        )

    try:
        entries = sorted(os.listdir(metrics_path))
    except OSError as exc:
        raise MetricIngestionError(f"Cannot list {metrics_path}: {exc}") from exc
    if not entries:
        raise MetricIngestionError(f"Job metrics directory is empty: {metrics_path}")

    all_pairs: list[tuple[tuple[int, int], Path, Path, str]] = []
    for subdir_name in entries:
        subdir_path = metrics_path / subdir_name
        if not subdir_path.is_dir():
            raise MetricIngestionError(
                f"Expected metric subdirectory, found file: {subdir_path}"
            )

        match = _SUBDIRECTORY_PATTERN.fullmatch(subdir_name)
        if match is None:
            raise MetricIngestionError(
                f"Invalid metric subdirectory name {subdir_name!r} in {metrics_path}; "
                "expected '<positive node count>_<positive element count>'"
            )
        key = (int(match.group(1)), int(match.group(2)))
        all_pairs.extend(
            (key, before_path, after_path, identifier)
            for before_path, after_path, identifier in _collect_file_pairs(subdir_path)
        )

    if not all_pairs:
        raise MetricIngestionError(f"No metric file pairs found in {metrics_path}")

    worker_count = min(MAX_FILE_PAIR_WORKERS, len(all_pairs))
    if worker_count == 1:
        results = [
            process_file_pair(
                before_path,
                after_path,
                expected_run=job_dir,
                expected_node=identifier,
            )
            for _, before_path, after_path, identifier in all_pairs
        ]
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count
        ) as executor:
            futures = [
                executor.submit(
                    process_file_pair,
                    before_path,
                    after_path,
                    expected_run=job_dir,
                    expected_node=identifier,
                )
                for _, before_path, after_path, identifier in all_pairs
            ]
            results = [future.result() for future in futures]

    job_results: dict[tuple[int, int], list[list[int]]] = {}
    job_results_nodes: dict[tuple[int, int], list[str]] = {}
    for (key, _, _, identifier), differences in zip(all_pairs, results):
        job_results.setdefault(key, []).append(differences)
        job_results_nodes.setdefault(key, []).append(identifier)

    return job_results, job_results_nodes


def process_all_jobs(base_dir: str | os.PathLike[str]):
    """Process every job while qualifying node identities by run.

    This function does not use a process pool. Per-file work is bounded to
    ``MAX_FILE_PAIR_WORKERS`` threads within each job, and jobs are processed
    serially so results are deterministic.
    """
    base_path = Path(base_dir)
    if not base_path.is_dir():
        raise MetricIngestionError(f"Metric data directory does not exist: {base_path}")

    job_dirs = sorted(
        entry for entry in os.listdir(base_path)
        if (base_path / entry).is_dir()
    )
    if not job_dirs:
        raise MetricIngestionError(f"No job directories found in {base_path}")

    aggregated_results: dict[tuple[int, int], list[list[int]]] = {}
    aggregated_results_nodes: dict[tuple[int, int], list[str]] = {}

    for job_dir in job_dirs:
        job_results, job_results_nodes = process_single_job(job_dir, base_path)
        for key, differences in job_results.items():
            aggregated_results.setdefault(key, []).extend(differences)
            aggregated_results_nodes.setdefault(key, []).extend(
                f"{job_dir}/{identifier}" for identifier in job_results_nodes[key]
            )

    node_map: dict[str, int] = {}
    for id_list in aggregated_results_nodes.values():
        for identifier in id_list:
            if identifier not in node_map:
                node_map[identifier] = len(node_map)

    for key, id_list in aggregated_results_nodes.items():
        aggregated_results_nodes[key] = [node_map[identifier] for identifier in id_list]

    return node_map, aggregated_results, aggregated_results_nodes


def prepare_metric_array(
    results: dict[tuple[int, int], Iterable[list[int]]],
    num_interfaces: int,
    max_nodes: int,
    results_nodes: dict[tuple[int, int], Iterable[int]] | None = None,
):
    """Create the legacy interface-major dense metric array.

    Input vectors are interface-major: all metrics for interface 0, then all
    metrics for interface 1. If ``results_nodes`` is omitted, callers get the
    legacy per-key enumeration. Supplying it preserves run-qualified node IDs.
    """
    if not results:
        raise MetricIngestionError("No metric results to normalize")
    if not isinstance(num_interfaces, int) or num_interfaces < 1:
        raise MetricIngestionError(
            f"num_interfaces must be positive, got {num_interfaces}"
        )
    if not isinstance(max_nodes, int) or max_nodes < 1:
        raise MetricIngestionError(f"max_nodes must be positive, got {max_nodes}")

    node_counts = sorted({key[0] for key in results})
    element_counts = sorted({key[1] for key in results})
    first_key = next(iter(results))
    first_vectors = list(results[first_key])
    if not first_vectors:
        raise MetricIngestionError(f"No metric vectors for {first_key}")
    num_metrics = len(first_vectors[0])
    if any(len(vector) != num_metrics for vector in first_vectors):
        raise MetricIngestionError(f"Inconsistent metric vector lengths for {first_key}")
    if num_metrics == 0 or num_metrics % num_interfaces:
        raise MetricIngestionError(
            f"Metric vector length {num_metrics} is not divisible by "
            f"num_interfaces={num_interfaces}"
        )
    metrics_per_interface = num_metrics // num_interfaces

    metric_array = np.full(
        (
            num_interfaces,
            metrics_per_interface,
            len(node_counts),
            len(element_counts),
            max_nodes,
        ),
        -1,
        dtype=np.int64,
    )

    for key, vectors in results.items():
        differences_by_key = list(vectors)
        if any(len(differences) != num_metrics for differences in differences_by_key):
            raise MetricIngestionError(f"Inconsistent metric vector lengths for {key}")

        if results_nodes is None:
            node_ids = list(range(len(differences_by_key)))
        else:
            node_ids = list(results_nodes.get(key, []))
        if len(node_ids) != len(differences_by_key):
            raise MetricIngestionError(
                f"Results and node IDs have different lengths for {key}"
            )
        if len(set(node_ids)) != len(node_ids):
            raise MetricIngestionError(f"Duplicate node IDs for {key}")

        x = node_counts.index(key[0])
        y = element_counts.index(key[1])
        for node_id, differences in zip(node_ids, differences_by_key):
            if not isinstance(node_id, int) or isinstance(node_id, bool):
                raise MetricIngestionError(f"Non-integer node ID {node_id!r} for {key}")
            if not 0 <= node_id < max_nodes:
                raise MetricIngestionError(
                    f"Node ID {node_id} for {key} is outside [0, {max_nodes})"
                )
            for interface_index in range(num_interfaces):
                for metric_index in range(metrics_per_interface):
                    flattened_index = (
                        interface_index * metrics_per_interface + metric_index
                    )
                    metric_array[
                        interface_index, metric_index, x, y, node_id
                    ] = differences[flattened_index]

    return metric_array, node_counts, element_counts
