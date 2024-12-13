import os
import numpy as np
import matplotlib.pyplot as plt

def parse_metric_file(filepath):
    """
    Parse the metric file and return a list of metric values (int).
    """
    with open(filepath, 'r') as file:
        data = file.readlines()
    return [int(line.split('@')[0]) for line in data]

def process_all_jobs(base_dir):
    """
    Process all job directories, create an integer map for nodes, and collect raw differences into dictionaries.

    Returns:
    - node_map: {node_name: integer}
    - results: { (node_count, num_elements): [[raw_differences_per_metric_entry_from_each_job]] }
    - results_int: { (node_count, num_elements): [[raw_differences_with_node_ids]] }
    """
    results = {}
    node_map = {}
    results_int = {}

    for job_dir in os.listdir(base_dir):
        job_path = os.path.join(base_dir, job_dir)
        metrics_dir = os.path.join(job_path, f"metrics_{job_dir}")

        if not os.path.isdir(metrics_dir):
            continue  # Skip if metrics directory doesn't exist

        for subdir in os.listdir(metrics_dir):
            if '_' in subdir:
                try:
                    node_count, num_elements = map(int, subdir.split('_'))
                except ValueError:
                    continue  # Skip directories that don't match the pattern

                subdir_path = os.path.join(metrics_dir, subdir)

                # Collect 'before' and 'after' files
                before_files = {}
                after_files = {}

                for file in os.listdir(subdir_path):
                    if 'metric_before' in file:
                        # Extract unique identifier for pairing
                        identifier = file.split('metric_before.')[1]
                        before_files[identifier] = os.path.join(subdir_path, file)
                    elif 'metric_after' in file:
                        # Extract unique identifier for pairing
                        identifier = file.split('metric_after.')[1]
                        after_files[identifier] = os.path.join(subdir_path, file)

                    # Add the identifier to the node_map if it's new
                    if identifier not in node_map:
                        node_map[identifier] = len(node_map)

                # Ensure each 'before' file is paired with the correct 'after' file
                for identifier in before_files.keys():
                    if identifier in after_files:
                        before_metrics = parse_metric_file(before_files[identifier])
                        after_metrics = parse_metric_file(after_files[identifier])

                        # Compute raw differences for each metric entry
                        differences = [after - before for before, after in zip(before_metrics, after_metrics)]

                        # Store results with node names
                        key = (node_count, num_elements)
                        results.setdefault(key, []).append(differences)

                        # Store results with node IDs
                        key_int = (node_count, num_elements)
                        differences_with_node_id = {node_map[identifier]: differences}
                        results_int.setdefault(key_int, []).append(differences_with_node_id)

    return node_map, results, results_int

def plot_metric_heatmaps(results, output_dir):
    """
    Generate and save separate heatmaps for each metric entry across node counts and elements.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract unique node counts and element counts
    node_counts = sorted({key[0] for key in results.keys()})
    element_counts = sorted({key[1] for key in results.keys()})

    # Determine number of metrics based on the first entry
    example_key = next(iter(results.keys()))
    num_metrics = len(results[example_key][0])

    for metric_index in range(num_metrics):
        # Create a 2D array for the current metric
        heatmap_data = np.zeros((len(element_counts), len(node_counts)))

        for (node_count, num_elements), all_differences in results.items():
            x = node_counts.index(node_count)
            y = element_counts.index(num_elements)

            # Use the raw difference for this metric index
            metric_differences = [differences[metric_index] for differences in all_differences]
            heatmap_data[y, x] = np.mean(metric_differences)  # Use the mean if multiple jobs provide values

        # Plot the heatmap for the current metric
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap_data, origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(label=f'Metric {metric_index + 1} Raw Difference')
        plt.xticks(ticks=np.arange(len(node_counts)), labels=node_counts)
        plt.yticks(ticks=np.arange(len(element_counts)), labels=element_counts)
        plt.xlabel('Node Count')
        plt.ylabel('Number of Elements')
        plt.title(f'Heatmap for Metric {metric_index + 1}')
        plt.grid(False)

        # Save the plot
        output_file = os.path.join(output_dir, f'metric_{metric_index + 1}_heatmap.png')
        plt.savefig(output_file)
        plt.close()

if __name__ == "__main__":
    base_directory = "/lus/gila/projects/atlas_aesp_CNDA/oneCCL_test/jobs_test"  # Update to your base directory path
    figures_directory = "figures"  # Directory to save plots

    # Process all jobs and retrieve results
    node_map, job_results, job_results_int = process_all_jobs(base_directory)

    # Save and plot heatmaps for each metric
    plot_metric_heatmaps(job_results, figures_directory)

    # Check node_map
    print("Node Map:", node_map)
