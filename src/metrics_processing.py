# src/metrics_processing.py

import os
import numpy as np

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
    """
    job_node_map = {}
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
            identifiers = set()

            for file in os.listdir(subdir_path):
                if 'metric_before' in file:
                    identifier = file.split('metric_before.')[1]
                    before_files[identifier] = os.path.join(subdir_path, file)
                    identifiers.add(identifier)
                elif 'metric_after' in file:
                    identifier = file.split('metric_after.')[1]
                    after_files[identifier] = os.path.join(subdir_path, file)
                    identifiers.add(identifier)

            # Update job_node_map with any new identifiers
            for identifier in identifiers:
                if identifier not in job_node_map:
                    job_node_map[identifier] = len(job_node_map)

            # Ensure each 'before' file is paired with an 'after' file
            for identifier in before_files.keys():
                if identifier in after_files:
                    differences = process_file_pair(before_files[identifier], after_files[identifier])

                    key = (node_count, num_elements)
                    job_results.setdefault(key, []).append(differences)
                    job_results_nodes.setdefault(key, []).append(job_node_map[identifier])
    return job_results, job_results_nodes

def process_all_jobs(base_dir):
    """
    Process all job directories, create an integer map for nodes, and collect raw differences into dictionaries.

    Returns:
    - node_map: {node_name: integer}
    - results: { (node_count, num_elements): [[raw_differences_per_metric_entry_from_each_job]] }
    - results_nodes: { (node_count, num_elements): [list_of_node_ids_corresponding_to_node_map] }
    """
    node_map = {}
    results = {}
    results_nodes = {}

    for job_dir in os.listdir(base_dir):
        job_path = os.path.join(base_dir, job_dir)
        job_results, job_results_nodes = process_single_job(job_dir, base_dir)
        # Merge the results from each job directory into the aggregated dictionaries
        for key, diffs in job_results.items():
            results.setdefault(key, []).extend(diffs)
        for key, ids in job_results_nodes.items():
            results_nodes.setdefault(key, []).extend(ids)
    # Build a global node_map from all identifiers encountered
    for id_list in aggregated_results_nodes.values():
        for identifier in id_list:
            if identifier not in node_map:
                node_map[identifier] = len(node_map)

    # Fix identifier strings in results_nodes
    for key, id_list in results_nodes.items():
        aggregated_results_nodes[key] = [node_map[identifier] for identifier in id_list]
    return node_map, results, results_nodes

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
