import numpy as np


def create_minimal_app_data(zero_values=False):
    metric_array = np.zeros((2, 2, 1, 1, 2), dtype=np.int64)
    if not zero_values:
        metric_array[0, 0, 0, 0, 0] = 7
        metric_array[0, 1, 0, 0, 0] = 11
        metric_array[1, 0, 0, 0, 0] = 13
        metric_array[1, 1, 0, 0, 0] = 17
        metric_array[0, 0, 0, 0, 1] = 19
        metric_array[0, 1, 0, 0, 1] = 23
        metric_array[1, 0, 0, 0, 1] = 29
        metric_array[1, 1, 0, 0, 1] = 31

    bench_array = np.full((1, 1, 3, 4), np.nan, dtype=np.float64)
    if not zero_values:
        bench_array[0, 0] = np.array(
            [
                [1.0, 2.0, 1.5, 0.1],
                [1.1, 2.1, 1.6, 0.1],
                [1.2, 2.2, 1.7, 0.1],
            ],
            dtype=np.float64,
        )

    return {
        "metric_array": metric_array,
        "node_counts": [1],
        "element_counts": [8],
        "metric_names": ["rx_bytes", "tx_bytes"],
        "node_map": {"node-a": 0, "node-b": 1},
        "results_nodes": {(1, 8): [0, 1]},
        "bench_array": bench_array,
        "bench_node_counts": [1],
        "bench_elements": [8],
    }


def create_legacy_fixture(root, num_interfaces=2, metrics_per_interface=2):
    job_name = "job-001"
    metrics_dir = root / job_name / f"metrics_{job_name}" / "1_8"
    metrics_dir.mkdir(parents=True)
    nodes = ["node-a", "node-b"]

    for node_index, node in enumerate(nodes):
        before_values = []
        after_values = []
        for interface_index in range(num_interfaces):
            for metric_index in range(metrics_per_interface):
                base = interface_index * 100 + metric_index * 10 + node_index
                before_values.append(base)
                after_values.append(base + interface_index + 1)
        before_path = metrics_dir / f"metric_before.{node}"
        after_path = metrics_dir / f"metric_after.{node}"
        before_path.write_text(
            "".join(f"{value}@metric-{index}\n" for index, value in enumerate(before_values))
        )
        after_path.write_text(
            "".join(f"{value}@metric-{index}\n" for index, value in enumerate(after_values))
        )

    return {
        "base_dir": root,
        "job_name": job_name,
        "nodes": nodes,
        "num_interfaces": num_interfaces,
        "metrics_per_interface": metrics_per_interface,
    }
