# src/metrics_processing.py

import os
import numpy as np
import concurrent.futures  # For process and thread pooling

def parse_metric_file(filepath):
    """
    Parse the metric file and return a list of metric values (int).
    """
    with open(filepath, 'r') as file:
        data = file.readlines()
    return [int(line.split('@')[0]) for line in data]

def process_file_pair(before_path, after_path):
    """
    Helper function to process a pair of metric files and compute differences.

    Note: This function is used inside a thread pool.
    """
    before_metrics = parse_metric_file(before_path)
    after_metrics = parse_metric_file(after_path)

    # Compute raw differences for each metric entry
    differences = [a - b for b, a in zip(before_metrics, after_metrics)]
    return differences

def process_single_job(job_dir, base_dir):
    """
    Process a single job directory.
    
    Return per-job results:
      - job_results: { (node_count, num_elements): [ [differences from a file pair], ... ] }
      - job_results_nodes: { (node_count, num_elements): [ identifier (str) for each file pair ] }

    Note: This function is called in parallel for each job directory.
    """
    job_results = {}
    job_results_nodes = {}
    job_path = os.path.join(base_dir, job_dir)
    metrics_dir = os.path.join(job_path, f"metrics_{job_dir}")

    if not os.path.isdir(metrics_dir):
        return job_results, job_results_nodes  # Skip if metrics directory doesn't exist

    # Iterate over subdirectories in the metrics directory
    for subdir in os.listdir(metrics_dir):
        if '_' in subdir:
            try:
                node_count, num_elements = map(int, subdir.split('_'))
            except ValueError:
                continue  # Skip directories that don't match the expected pattern

            subdir_path = os.path.join(metrics_dir, subdir)

            # Collect 'before' and 'after' files along with their identifiers
            before_files = {}
            after_files = {}
            for file in os.listdir(subdir_path):
                if 'metric_before' in file:
                    identifier = file.split('metric_before.')[1]
                    before_files[identifier] = os.path.join(subdir_path, file)
                elif 'metric_after' in file:
                    identifier = file.split('metric_after.')[1]
                    after_files[identifier] = os.path.join(subdir_path, file)

            # Process each file pair concurrently using a ThreadPoolExecutor.
            differences_list = []
            identifier_list = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_identifier = {}
                for identifier in before_files:
                    if identifier in after_files:
                        before_path = before_files[identifier]
                        after_path = after_files[identifier]
                        future = executor.submit(process_file_pair, before_path, after_path)
                        future_to_identifier[future] = identifier

                for future in concurrent.futures.as_completed(future_to_identifier):
                    identifier = future_to_identifier[future]
                    try:
                        differences = future.result()
                        differences_list.append(differences)
                        identifier_list.append(identifier)
                    except Exception as exc:
                        print(f"Error processing identifier {identifier} in job {job_dir}: {exc}")

            key = (node_count, num_elements)
            if differences_list:
                job_results.setdefault(key, []).extend(differences_list)
                job_results_nodes.setdefault(key, []).extend(identifier_list)

    return job_results, job_results_nodes

def process_all_jobs(base_dir):
    """
    Process all job directories in parallel and aggregate results.

    Uses ProcessPoolExecutor to process multiple job directories concurrently.
    Each job directory (via process_single_job) returns its own results.
    The main process aggregates all results and then builds a global node_map.
    The node_map is built after collecting all identifiers, and then results_nodes are converted to integer ids.

    Returns:
      - node_map: {node_name: integer}
      - results: { (node_count, num_elements): [[raw differences per metric entry from each job]] }
      - results_nodes: { (node_count, num_elements): [list of node ids corresponding to node_map] }
    """
    aggregated_results = {}
    aggregated_results_nodes = {}

    # List all job directories in base_dir
    job_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # Process each job directory in parallel using ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_job = {executor.submit(process_single_job, job_dir, base_dir): job_dir
                         for job_dir in job_dirs}

        for future in concurrent.futures.as_completed(future_to_job):
            job_dir = future_to_job[future]
            try:
                job_results, job_results_nodes = future.result()

                # Merge the results from each job directory into the aggregated dictionaries
                for key, diffs in job_results.items():
                    aggregated_results.setdefault(key, []).extend(diffs)
                for key, ids in job_results_nodes.items():
                    aggregated_results_nodes.setdefault(key, []).extend(ids)

            except Exception as exc:
                print(f"Job directory {job_dir} generated an exception: {exc}")

    # Build a global node_map from all identifiers encountered
    node_map = {}
    for id_list in aggregated_results_nodes.values():
        for identifier in id_list:
            if identifier not in node_map:
                node_map[identifier] = len(node_map)

    # Replace identifier strings in aggregated_results_nodes with their integer mapping
    for key, id_list in aggregated_results_nodes.items():
        aggregated_results_nodes[key] = [node_map[identifier] for identifier in id_list]

    return node_map, aggregated_results, aggregated_results_nodes

def prepare_metric_array(results, num_interfaces, max_nodes):
    """
    Create a multidimensional array to store the data.
    Shape: {num_interfaces} x {metrics_per_interface} x {len(node_counts)} x {len(element_counts)} x {max_nodes}
    """
    # Early exit if no results
    if not results:
        # Handle empty results case
        return None, [], []
    # Extract unique node counts and element counts
    node_counts = sorted({key[0] for key in results.keys()})
    element_counts = sorted({key[1] for key in results.keys()})

    # Determine number of metrics based on the first entry
    example_key = next(iter(results.keys()))
    num_metrics = len(results[example_key][0])

    # Adjust metrics to consider interfaces
    metrics_per_interface = num_metrics // num_interfaces

    # Initialize the array with -1
    metric_array = np.full((num_interfaces, metrics_per_interface, len(node_counts), len(element_counts), max_nodes), -1, dtype=int)

    for (node_count, num_elements), all_differences in results.items():
        x = node_counts.index(node_count)
        y = element_counts.index(num_elements)

        for interface_index in range(num_interfaces):
            for metric_index in range(metrics_per_interface):
                z = interface_index * metrics_per_interface + metric_index
                metric_differences = [differences[z] for differences in all_differences]

                for node_id, value in enumerate(metric_differences):
                    metric_array[interface_index, metric_index, x, y, node_id] = value

    return metric_array, node_counts, element_counts
