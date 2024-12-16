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
    - results_nodes: { (node_count, num_elements): [[node_list_corresponding_to_node_map]] }
    """
    results = {}
    node_map = {}
    results_nodes = {}

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
                        results_nodes.setdefault(key, []).append(node_map[identifier])

    return node_map, results, results_nodes

def prepare_metric_array(results, num_interfaces, max_node_map_size):
    """
    Create a multidimensional array to store the data.
    Shape: {num_interfaces} x {metrics_per_interface} x {node_counts} x {number_of_elements} x {max_node_map_size}

    Returns:
    - metric_array: Initialized and populated array.
    - node_counts: Sorted list of unique node counts.
    - element_counts: Sorted list of unique element counts.
    """
    # Extract unique node counts and element counts
    node_counts = sorted({key[0] for key in results.keys()})
    element_counts = sorted({key[1] for key in results.keys()})

    # Determine number of metrics based on the first entry
    example_key = next(iter(results.keys()))
    num_metrics = len(results[example_key][0])

    # Adjust metrics to consider interfaces
    metrics_per_interface = num_metrics // num_interfaces

    # Initialize the array with -1
    metric_array = np.full((num_interfaces, metrics_per_interface, len(node_counts), len(element_counts), max_node_map_size), -1, dtype=int)

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

def plot_metric_heatmaps_from_array(metric_array, node_counts, element_counts, output_dir):
    """
    Generate a 3x3 grid of plots for each metric index: the outer 8 plots are heatmaps for each interface,
    and the center plot shows the sum of all the heatmaps for the current metric index.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    num_interfaces, metrics_per_interface, _, _, _ = metric_array.shape

    for metric_index in range(metrics_per_interface):
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        combined_heatmap = np.zeros((len(node_counts), len(element_counts)))

        plot_idx = 0
        for interface_index in range(num_interfaces):
            # Create a 2D array for the current metric and interface
            heatmap_data = metric_array[interface_index, metric_index, :, :, :].mean(axis=-1)
            combined_heatmap += heatmap_data

            # Plot the heatmap for the current metric and interface
            ax = axes[plot_idx // 3, plot_idx % 3]
            im = ax.imshow(heatmap_data, origin='lower', aspect='auto', cmap='viridis')
            fig.colorbar(im, ax=ax)
            ax.set_title(f'Interface {interface_index + 1}, Metric {metric_index + 1}')
            ax.set_xticks(np.arange(len(node_counts)))
            ax.set_xticklabels(node_counts)
            ax.set_yticks(np.arange(len(element_counts)))
            ax.set_yticklabels(element_counts)
            ax.set_xlabel('Node Count')
            ax.set_ylabel('Number of Elements')
            plot_idx += 1

        # Plot the combined heatmap in the center
        ax = axes[1, 1]
        im = ax.imshow(combined_heatmap, origin='lower', aspect='auto', cmap='plasma')
        fig.colorbar(im, ax=ax)
        ax.set_title(f'Combined Heatmap for Metric {metric_index + 1}')
        ax.set_xticks(np.arange(len(node_counts)))
        ax.set_xticklabels(node_counts)
        ax.set_yticks(np.arange(len(element_counts)))
        ax.set_yticklabels(element_counts)
        ax.set_xlabel('Node Count')
        ax.set_ylabel('Number of Elements')

        # Remove any unused subplots
        for i in range(plot_idx, 9):
            if i != 4:  # Skip the center plot
                fig.delaxes(axes[i // 3, i % 3])

        plt.tight_layout()
        output_file = os.path.join(output_dir, f'metric_{metric_index + 1}_heatmap_grid.png')
        plt.savefig(output_file)
        plt.close()

if __name__ == "__main__":
    base_directory = "/lus/gila/projects/atlas_aesp_CNDA/oneCCL_test/jobs_test"  # Update to your base directory path
    figures_directory = "figures"  # Directory to save plots
    num_interfaces = 8  # Number of interfaces

    # Process all jobs and retrieve results
    node_map, job_results, job_results_nodes = process_all_jobs(base_directory)

    # Prepare the multidimensional array
    metric_array, node_counts, element_counts = prepare_metric_array(job_results, num_interfaces, max((len(node_list) for node_list in job_results_nodes.values()), default=0))

    # Save and plot heatmaps for each metric
    plot_metric_heatmaps_from_array(metric_array, node_counts, element_counts, figures_directory)
