import os
import pickle
import argparse

from metrics_processing import (
    process_all_jobs,
    prepare_metric_array
)

from oneccl_processing import (
    process_oneccl_benchmarks,
    prepare_benchmark_array
)

from osu_processing import (
    process_osu_benchmarks
)

from app import run_interactive_dash_app


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process metrics and benchmarks for interactive Dash app.")
    parser.add_argument("--base_dir", type=str, default="/lus/gila/projects/atlas_aesp_CNDA/oneCCL_test/jobs_test",
                        help="Base directory path containing all job/benchmark data.")
    parser.add_argument("--num_interfaces", type=int, default=8,
                        help="Number of NIC interfaces.")
    parser.add_argument("--metric_names_file", type=str, default="metric_names.txt",
                        help="File containing metric names (one metric name per line).")
    parser.add_argument("--benchmark", type=str, choices=["oneccl", "osu"], default="oneccl",
                        help="Select which benchmark to process: 'oneccl' or 'osu'.")
    return parser.parse_args()


def main():
    # (0) Parse command line arguments
    args = parse_args()

    base_directory = args.base_dir
    num_interfaces = args.num_interfaces
    metric_names_file = args.metric_names_file
    benchmark_type = args.benchmark  # "oneccl" or "osu"

    # (1) Load or process metric data
    cache_file = os.path.join(base_directory, "cached_data.pkl")
    if os.path.isfile(cache_file):
        # Load cached data
        with open(cache_file, "rb") as f:
            node_map, job_results, job_results_nodes, metric_array, node_counts, element_counts, metric_names = pickle.load(f)
        print("Loaded data from cache.")
    else:
        # Process all jobs and retrieve results
        node_map, job_results, job_results_nodes = process_all_jobs(base_directory)

        # Prepare the multidimensional array
        metric_array, node_counts, element_counts = prepare_metric_array(
            job_results,
            num_interfaces,
            max((len(node_list) for node_list in job_results_nodes.values()), default=0)
        )

        # Early exit if no data found
        if metric_array is None:
            print("No metric data found.")
            exit(0)

        # Determine how many metrics per interface
        # metric_array shape: (num_interfaces, metrics_per_interface, num_nodes, ...)
        _, metrics_per_interface, _, _, _ = metric_array.shape

        # Read metric names from file
        if os.path.isfile(metric_names_file):
            with open(metric_names_file, 'r') as f:
                all_names = [line.strip() for line in f.readlines()]
        else:
            # Fallback if file not found
            all_names = []

        # If the file doesn't have enough names, fill with generic placeholders
        if len(all_names) < metrics_per_interface:
            all_names += [f"Metric_{i+1}" for i in range(len(all_names), metrics_per_interface)]

        metric_names = all_names

        # Save the processed metric data to cache
        with open(cache_file, "wb") as f:
            pickle.dump((node_map, job_results, job_results_nodes, metric_array, node_counts, element_counts, metric_names), f)
        print("Processed metric data and saved to cache.")

    # (2) Load or process benchmark data depending on --benchmark argument
    bench_cache_file = os.path.join(base_directory, f"cached_bench_{benchmark_type}.pkl")
    if os.path.isfile(bench_cache_file):
        # Load cached benchmark data
        with open(bench_cache_file, "rb") as f:
            bench_array, bench_node_counts, bench_elements = pickle.load(f)
        print(f"Loaded {benchmark_type} benchmark data from cache.")
    else:
        # Process benchmarks depending on the chosen type
        if benchmark_type == "oneccl":
            bench_results = process_oneccl_benchmarks(base_directory)
            bench_array, bench_node_counts, bench_elements = prepare_benchmark_array(bench_results)
        else:
            bench_results = process_osu_benchmarks(base_directory)
            bench_array, bench_node_counts, bench_elements = prepare_benchmark_array(bench_results)

        # Save the processed benchmark data to cache
        with open(bench_cache_file, "wb") as f:
            pickle.dump((bench_array, bench_node_counts, bench_elements), f)
        print(f"Processed {benchmark_type} benchmark data and saved to cache.")

    # (3) Run the interactive Dash app
    run_interactive_dash_app(
        metric_array, node_counts, element_counts,
        metric_names, node_map, job_results_nodes,
        bench_array, bench_node_counts, bench_elements
    )

if __name__ == "__main__":
    main()
