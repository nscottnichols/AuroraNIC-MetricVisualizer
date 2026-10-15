from pathlib import Path

import numpy as np
import pytest

from metrics_processing import (
    MAX_FILE_PAIR_WORKERS,
    MetricIngestionError,
    parse_metric_file,
    parse_metric_records,
    prepare_metric_array,
    process_all_jobs,
    process_file_pair,
    process_single_job,
)


FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "examples" / "legacy_fixture"
)


def write_metric_file(path, values, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{value}@{label}\n" for value, label in zip(values, labels)),
        encoding="utf-8",
    )


def make_valid_job(base_dir, job_name="job-001", node_names=("node-a",)):
    metrics_dir = base_dir / job_name / f"metrics_{job_name}" / "1_8"
    for node_name in node_names:
        labels = [
            f"{job_name}.{node_name}.cxi0.rx_bytes",
            f"{job_name}.{node_name}.cxi0.tx_bytes",
            f"{job_name}.{node_name}.cxi1.rx_bytes",
            f"{job_name}.{node_name}.cxi1.tx_bytes",
        ]
        write_metric_file(
            metrics_dir / f"metric_before.{node_name}", [10, 20, 30, 40], labels
        )
        write_metric_file(
            metrics_dir / f"metric_after.{node_name}", [13, 22, 39, 49], labels
        )
    return metrics_dir


def test_static_fixture_labels_values_and_interface_major_order():
    expected = {
        "job-001": {
            "node-a": [7, 11, 12, 22],
            "node-b": [9, 13, 19, 29],
        },
        "job-002": {
            "node-a": [8, 12, 24, 34],
            "node-b": [10, 14, 20, 30],
        },
    }

    for job_name, nodes in expected.items():
        results, result_nodes = process_single_job(job_name, FIXTURE_ROOT)
        assert list(results) == [(2, 16)]
        assert result_nodes[(2, 16)] == ["node-a", "node-b"]
        assert results[(2, 16)] == [nodes["node-a"], nodes["node-b"]]

        for node_name in nodes:
            pair_dir = (
                FIXTURE_ROOT / job_name / f"metrics_{job_name}" / "2_16"
            )
            _, labels = parse_metric_records(
                pair_dir / f"metric_before.{node_name}"
            )
            assert labels == [
                f"{job_name}.{node_name}.cxi0.rx_bytes",
                f"{job_name}.{node_name}.cxi0.tx_bytes",
                f"{job_name}.{node_name}.cxi1.rx_bytes",
                f"{job_name}.{node_name}.cxi1.tx_bytes",
            ]


def test_record_identity_must_match_job_and_filename(tmp_path):
    metrics_dir = make_valid_job(tmp_path, node_names=("node-a",))
    labels = [
        "job-001.node-b.cxi0.rx_bytes",
        "job-001.node-b.cxi0.tx_bytes",
        "job-001.node-b.cxi1.rx_bytes",
        "job-001.node-b.cxi1.tx_bytes",
    ]
    write_metric_file(
        metrics_dir / "metric_before.node-a", [10, 20, 30, 40], labels
    )
    write_metric_file(
        metrics_dir / "metric_after.node-a", [13, 22, 39, 49], labels
    )

    with pytest.raises(MetricIngestionError, match="does not identify node"):
        process_single_job("job-001", tmp_path)


def test_static_fixture_preserves_multi_run_node_identity():
    node_map, results, result_nodes = process_all_jobs(FIXTURE_ROOT)

    assert node_map == {
        "job-001/node-a": 0,
        "job-001/node-b": 1,
        "job-002/node-a": 2,
        "job-002/node-b": 3,
    }
    assert result_nodes[(2, 16)] == [0, 1, 2, 3]

    metric_array, node_counts, element_counts = prepare_metric_array(
        results,
        num_interfaces=2,
        max_nodes=len(node_map),
        results_nodes=result_nodes,
    )
    assert node_counts == [2]
    assert element_counts == [16]
    assert metric_array.shape == (2, 2, 1, 1, 4)

    expected_by_identity = {
        "job-001/node-a": [7, 11, 12, 22],
        "job-001/node-b": [9, 13, 19, 29],
        "job-002/node-a": [8, 12, 24, 34],
        "job-002/node-b": [10, 14, 20, 30],
    }
    for identity, differences in expected_by_identity.items():
        node_id = node_map[identity]
        assert metric_array[0, 0, 0, 0, node_id] == differences[0]
        assert metric_array[0, 1, 0, 0, node_id] == differences[1]
        assert metric_array[1, 0, 0, 0, node_id] == differences[2]
        assert metric_array[1, 1, 0, 0, node_id] == differences[3]


def test_before_and_after_must_have_matching_labels_and_lengths(tmp_path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    write_metric_file(before, [1, 2], ["metric-a", "metric-b"])
    write_metric_file(after, [3], ["metric-a"])

    with pytest.raises(MetricIngestionError, match="different lengths"):
        process_file_pair(before, after)

    write_metric_file(after, [3, 4], ["metric-a", "metric-c"])
    with pytest.raises(MetricIngestionError, match="labels differ"):
        process_file_pair(before, after)


def test_missing_before_or_after_file_is_rejected(tmp_path):
    metrics_dir = make_valid_job(tmp_path)
    (metrics_dir / "metric_after.node-a").unlink()

    with pytest.raises(MetricIngestionError, match="missing after"):
        process_single_job("job-001", tmp_path)


@pytest.mark.parametrize(
    "content",
    [
        "",
        "\n",
        "   \n",
        "not-a-number@metric-a\n",
        "1@\n",
        "@metric-a\n",
        "1\n",
        "1@metric-a@extra\n",
        "1@metric a\n",
        "-1@metric-a\n",
        "1.0@metric-a\n",
        "+1@metric-a\n",
    ],
)
def test_malformed_or_empty_counter_lines_fail(tmp_path, content):
    metric_file = tmp_path / "metric_before.node-a"
    metric_file.write_text(content, encoding="utf-8")

    with pytest.raises(MetricIngestionError):
        parse_metric_file(metric_file)


def test_duplicate_metric_labels_and_node_identifiers_fail(tmp_path):
    metric_file = tmp_path / "metric"
    metric_file.write_text(
        "1@metric-a\n2@metric-a\n", encoding="utf-8"
    )
    with pytest.raises(MetricIngestionError, match="Duplicate metric label"):
        parse_metric_file(metric_file)

    metrics_dir = make_valid_job(tmp_path)
    (metrics_dir / "metric_before").write_text(
        "1@duplicate\n", encoding="utf-8"
    )
    with pytest.raises(MetricIngestionError, match="Invalid metric filename"):
        process_single_job("job-001", tmp_path)

    results = {(1, 8): [[1, 2, 3, 4], [5, 6, 7, 8]]}
    with pytest.raises(MetricIngestionError, match="Duplicate node IDs"):
        prepare_metric_array(
            results,
            num_interfaces=2,
            max_nodes=2,
            results_nodes={(1, 8): [0, 0]},
        )


def test_invalid_subdirectories_and_filenames_fail(tmp_path):
    metrics_dir = make_valid_job(tmp_path)
    job_metrics_dir = metrics_dir.parent
    (job_metrics_dir / "1_8_notes").mkdir()
    with pytest.raises(MetricIngestionError, match="Invalid metric subdirectory"):
        process_single_job("job-001", tmp_path)

    (job_metrics_dir / "1_8_notes").rmdir()
    (job_metrics_dir / "README.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(MetricIngestionError, match="found file"):
        process_single_job("job-001", tmp_path)

    (job_metrics_dir / "README.txt").unlink()
    (metrics_dir / "metric_before").write_text(
        "1@metric-a\n", encoding="utf-8"
    )
    with pytest.raises(MetricIngestionError, match="Invalid metric filename"):
        process_single_job("job-001", tmp_path)


def test_integer_range_is_checked_and_maximum_difference_is_preserved(tmp_path):
    maximum = np.iinfo(np.int64).max
    metric_file = tmp_path / "too_large"
    metric_file.write_text(
        f"{maximum + 1}@metric-a\n", encoding="utf-8"
    )
    with pytest.raises(MetricIngestionError, match="exceeds signed 64-bit range"):
        parse_metric_file(metric_file)

    before = tmp_path / "before"
    after = tmp_path / "after"
    write_metric_file(before, [0], ["metric-a"])
    write_metric_file(after, [maximum], ["metric-a"])
    assert process_file_pair(before, after) == [maximum]


def test_empty_or_missing_directories_do_not_succeed_silently(tmp_path):
    with pytest.raises(MetricIngestionError, match="No job directories"):
        process_all_jobs(tmp_path)

    (tmp_path / "not-a-job").mkdir()
    with pytest.raises(MetricIngestionError, match="metrics directory does not exist"):
        process_all_jobs(tmp_path)

    empty_base = tmp_path / "empty-base"
    empty_base.mkdir()
    empty_metrics = empty_base / "empty-job" / "metrics_empty-job"
    empty_metrics.mkdir(parents=True)
    with pytest.raises(MetricIngestionError, match="Job metrics directory is empty"):
        process_all_jobs(empty_base)


def test_zero_differences_are_valid_not_missing(tmp_path):
    metrics_dir = make_valid_job(tmp_path)
    for node_file in metrics_dir.glob("metric_after.*"):
        node_file.write_text(
            "10@job-001.node-a.cxi0.rx_bytes\n"
            "20@job-001.node-a.cxi0.tx_bytes\n"
            "30@job-001.node-a.cxi1.rx_bytes\n"
            "40@job-001.node-a.cxi1.tx_bytes\n",
            encoding="utf-8",
        )

    results, result_nodes = process_single_job("job-001", tmp_path)
    assert result_nodes[(1, 8)] == ["node-a"]
    assert results[(1, 8)] == [[0, 0, 0, 0]]


def test_normalization_validates_interface_and_node_contracts():
    results = {(1, 8): [[1, 2, 3, 4]]}
    with pytest.raises(MetricIngestionError, match="not divisible"):
        prepare_metric_array(results, num_interfaces=3, max_nodes=1)

    with pytest.raises(MetricIngestionError, match="outside"):
        prepare_metric_array(
            results,
            num_interfaces=2,
            max_nodes=1,
            results_nodes={(1, 8): [1]},
        )

    with pytest.raises(MetricIngestionError, match="different lengths"):
        prepare_metric_array(
            results,
            num_interfaces=2,
            max_nodes=2,
            results_nodes={(1, 8): []},
        )


def test_worker_count_is_bounded_without_process_pools():
    assert 1 <= MAX_FILE_PAIR_WORKERS <= 4
