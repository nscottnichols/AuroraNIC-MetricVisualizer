import os
import pickle

from metrics_processing import (
    process_all_jobs,
    prepare_metric_array
)

from oneccl_processing import (
    process_oneccl_benchmarks,
    prepare_benchmark_array
)

from app import run_interactive_dash_app

def main():
    base_directory = "/lus/gila/projects/atlas_aesp_CNDA/oneCCL_test/jobs_test"  # Update to your base directory path
    num_interfaces = 8  # Number of interfaces
    metric_names_file = "metric_names.txt"  # Update if needed
    cache_file = os.path.join(base_directory, "cached_data.pkl")

    # (1) Load or process metric data
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
            print("No data found.")
            exit(0)

        # Determine how many metrics per interface
        # metric_array shape: (num_interfaces, metrics_per_interface, ...)
        _, metrics_per_interface, _, _, _ = metric_array.shape

        # Read metric names from file
        if os.path.isfile(metric_names_file):
            with open(metric_names_file, 'r') as f:
                all_names = [line.strip() for line in f.readlines()]
        else:
            # Fallback if file not found
            all_names = []

        # If the file doesn't have enough names, fill with generic names
        if len(all_names) < metrics_per_interface:
            all_names += [f"Metric_{i+1}" for i in range(len(all_names), metrics_per_interface)]

        metric_names = all_names

        # Save the processed metric data to cache
        with open(cache_file, "wb") as f:
            pickle.dump((node_map, job_results, job_results_nodes, metric_array, node_counts, element_counts, metric_names), f)
        print("Processed metric data and saved to cache.")

    # (2) Load or process oneCCL benchmarks
    bench_cache_file = os.path.join(base_directory, "cached_bench.pkl")
    if os.path.isfile(bench_cache_file):
        with open(bench_cache_file, "rb") as f:
            bench_array, bench_node_counts, bench_elements = pickle.load(f)
        print("Loaded oneCCL benchmark data from cache.")
    else:
        bench_results = process_oneccl_benchmarks(base_directory)
        bench_array, bench_node_counts, bench_elements = prepare_benchmark_array(bench_results)

        # Save the processed benchmark data to cache
        with open(bench_cache_file, "wb") as f:
            pickle.dump((bench_array, bench_node_counts, bench_elements), f)
        print("Processed oneCCL benchmark data and saved to cache.")

    # (3) Run the interactive Dash app
    run_interactive_dash_app(
        metric_array, node_counts, element_counts,
        metric_names, node_map, job_results_nodes,
        bench_array, bench_node_counts, bench_elements
    )

if __name__ == "__main__":
    main()
