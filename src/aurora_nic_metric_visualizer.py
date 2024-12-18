import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output

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
    - results_nodes: { (node_count, num_elements): [list_of_node_ids_corresponding_to_node_map] }
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

                # Update node_map with any new identifiers
                for identifier in identifiers:
                    if identifier not in node_map:
                        node_map[identifier] = len(node_map)

                # Ensure each 'before' file is paired with an 'after' file
                for identifier in before_files.keys():
                    if identifier in after_files:
                        before_metrics = parse_metric_file(before_files[identifier])
                        after_metrics = parse_metric_file(after_files[identifier])

                        # Compute raw differences for each metric entry
                        differences = [a - b for b, a in zip(before_metrics, after_metrics)]

                        key = (node_count, num_elements)
                        results.setdefault(key, []).append(differences)
                        results_nodes.setdefault(key, []).append(node_map[identifier])

    return node_map, results, results_nodes

def prepare_metric_array(results, num_interfaces, max_nodes):
    """
    Create a multidimensional array to store the data.
    Shape: {num_interfaces} x {metrics_per_interface} x {len(node_counts)} x {len(element_counts)} x {max_nodes}
    """
    # Extract unique node counts and element counts
    node_counts = sorted({key[0] for key in results.keys()})
    element_counts = sorted({key[1] for key in results.keys()})

    # Determine number of metrics based on the first entry
    if not results:
        # Handle empty results case
        return None, [], []
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

# New interactive plotting function with Plotly and Dash
def run_interactive_dash_app(metric_array, node_counts, element_counts, metric_names, node_map, results_nodes):
    # Dimensions
    num_interfaces, metrics_per_interface, Nx, Ny, max_nodes = metric_array.shape

    # Reverse the node_map to get node_name from node_id
    reverse_node_map = {v: k for k, v in node_map.items()}

    # Determine which metrics are all zero:
    # For each metric_index, check all interfaces and all data points
    # Convert -1 to nan for checking actual sums
    mask = (metric_array == -1)
    data_filled = np.where(mask, np.nan, metric_array)

    # Sum absolute values to ensure we catch negative differences if any
    metric_sums = np.nansum(np.abs(data_filled), axis=(0,2,3,4)) # sum over interfaces, node_counts, element_counts, max_nodes
    zero_metrics = (metric_sums == 0)

    # All metric options (including zero ones)
    all_metric_options = [{'label': metric_names[i], 'value': i} for i in range(metrics_per_interface)]

    # Non-zero metric options
    nonzero_metric_options = [{'label': metric_names[i], 'value': i} for i in range(metrics_per_interface) if not zero_metrics[i]]

    # Build a dictionary of nodes_for_node_count
    # nodes_for_node_count[node_count] = set of node names that appear in that node_count column (across all elements)
    nodes_for_node_count = {nc: set() for nc in node_counts}
    for (nc, ne), node_ids in results_nodes.items():
        for nid in node_ids:
            node_name = reverse_node_map[nid]
            nodes_for_node_count[nc].add(node_name)

    # Create node dropdowns for each node_count
    node_dropdowns = []
    for nc in node_counts:
        node_dropdowns.append(
            html.Div([
                html.Label(f"Select Nodes for Node Count {nc}:"),
                dcc.Dropdown(
                    id=f'node-dropdown-{nc}',
                    options=[{'label': n, 'value': n} for n in sorted(nodes_for_node_count[nc])],
                    multi=True,
                    placeholder="Select nodes (default: all)",
                    style={'width': '400px'}
                )
            ], style={'margin': '20px'})
        )

    # Create the Dash app
    app = Dash(__name__)

    # Layout: Dropdown + Figure
    app.layout = html.Div([
        html.H1("Interactive Heatmap Analysis for Metric Differences", style={'textAlign': 'center'}),
        html.Div([
            html.Label("Show only non-zero metrics:"),
            dcc.Checklist(
                id='hide-zero-metrics',
                options=[{'label': '', 'value': 'hide'}],
                value=[],  # by default show all metrics
                style={'display': 'inline-block', 'margin-right': '20px'}
            ),
            html.Label("Select Metric:"),
            dcc.Dropdown(
                id='metric-dropdown',
                value=0,  # default selected metric index
                clearable=False,
                style={'width': '300px', 'display': 'inline-block', 'margin-left': '20px'}
            )
        ], style={'textAlign': 'center', 'margin': '20px'}),

        html.Div(node_dropdowns, style={'textAlign': 'center'}),

        dcc.Graph(id='heatmap-figure')
    ])

    # Callback to update the dropdown options based on the hide-zero-metrics checkbox
    @app.callback(
        Output('metric-dropdown', 'options'),
        Input('hide-zero-metrics', 'value')
    )
    def update_dropdown_options(hide_zero):
        if 'hide' in hide_zero:
            # Return only non-zero metrics
            return nonzero_metric_options
        else:
            # Return all metrics
            return all_metric_options

    # Build a list of Inputs for each node-dropdown
    node_inputs = [Input(f'node-dropdown-{nc}', 'value') for nc in node_counts]

    @app.callback(
        Output('heatmap-figure', 'figure'),
        [Input('metric-dropdown', 'value'), Input('hide-zero-metrics', 'value')] + node_inputs
    )
    def update_figure(selected_metric_index, hide_zero, *node_selections):
        # node_selections corresponds to each node_count in node_counts
        selected_nodes_by_nc = {}
        for i, nc in enumerate(node_counts):
            selected_nodes = node_selections[i]
            # If None or empty, use all nodes for that node_count
            if not selected_nodes:
                selected_nodes = nodes_for_node_count[nc]
            selected_nodes_by_nc[nc] = set(selected_nodes)

        subplot_titles = [
            "Interface 1",
            "Interface 2",
            "Interface 3",
            "Interface 4",
            "Combined",
            "Interface 5",
            "Interface 6",
            "Interface 7",
            "Interface 8"
        ]

        # Create a 3x3 subplot figure
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=subplot_titles,
            vertical_spacing=0.02, horizontal_spacing=0.02,
            shared_xaxes=False,
            shared_yaxes=False
        )

        # Compute the 8 interface heatmaps for the selected metric
        # Each heatmap is (node_counts, element_counts), averaged over nodes
        # metric_array: (num_interfaces, metrics_per_interface, node_counts, element_counts, max_nodes)
        # We'll compute the interface_heatmaps with per-(node_count, element_count) filtering
        interface_heatmaps = []
        for interface_index in range(num_interfaces):
            # We'll build a 2D array (Nx, Ny) by summing over the filtered nodes
            hm_data = np.zeros((Nx, Ny))
            hm_data[:] = np.nan  # start with nan to use nanmean/sum if needed

            # Walk through each (x,y) cell
            # (x,y) corresponds to node_counts[x], element_counts[y]
            for x in range(Nx):
                nc = node_counts[x]
                allowed_nodes = selected_nodes_by_nc[nc]  # allowed node names for this node_count
                for y in range(Ny):
                    ne = element_counts[y]
                    node_ids = results_nodes.get((nc, ne), [])
                    # Filter node_ids by allowed_nodes
                    filtered_values = []
                    for nid in node_ids:
                        node_name = reverse_node_map[nid]
                        if node_name in allowed_nodes:
                            node_pos = node_ids.index(nid)
                            val = metric_array[interface_index, selected_metric_index, x, y, node_pos]
                            if val != -1:
                                filtered_values.append(val)

                    if filtered_values:
                        hm_data[x, y] = np.nansum(filtered_values)
                    else:
                        hm_data[x, y] = 0  # or np.nan if you prefer

            # If there are nans, replace them with 0
            hm_data = np.nan_to_num(hm_data, nan=0)
            interface_heatmaps.append(hm_data)

        # Compute combined heatmap (sum of all interfaces)
        combined_heatmap = np.sum(interface_heatmaps, axis=0)

        # Positions of the subplots:
        # We'll place interfaces in the outer 8 subplots (ignoring the center)
        # and the combined in the center (2, 2) in 1-based indexing.
        # The layout (0-based indexing for code reference):
        # (1, 1) is the center subplot in 0-based, but plotly is 1-based indexing so center is (2, 2).
        # Let's define the subplot order (row, col) for the 8 interfaces:
        positions = [(1,1), (1,2), (1,3),
                     (2,1),        (2,3),
                     (3,1), (3,2), (3,3)]

        # We'll use indices for plotting, then map ticks to correct ranges
        x_indices = list(range(Nx))
        y_indices = list(range(Ny))

        # Add the 8 interface heatmaps
        for i, hm_data in enumerate(interface_heatmaps):
            r, c = positions[i]
            fig.add_trace(
                go.Heatmap(
                    z=hm_data.T,
                    x=x_indices,
                    y=y_indices,
                    colorscale='Viridis',
                    hovertemplate="Node Count: %{x}<br>Elements: %{y}<br>Value: %{z}<extra></extra>",
                    xgap=1,  # spacing between cells
                    ygap=1,
                    coloraxis="coloraxis"  # Shared coloraxis for all outer plots
                ),
                row=r, col=c
            )

        # Add the combined heatmap in the center (2, 2)
        fig.add_trace(
            go.Heatmap(
                z=combined_heatmap.T,
                x=x_indices,
                y=y_indices,
                colorscale='Plasma',
                hovertemplate="Node Count: %{x}<br>Elements: %{y}<br>Combined: %{z}<extra></extra>",
                xgap=1,
                ygap=1,
                coloraxis="coloraxis2"  # Separate coloraxis for combined
            ),
            row=2, col=2
        )

        # Update axes for each subplot to show labels and correct ticks and ranges
        for r in range(1, 4):
            for c in range(1, 4):
                # Determine whether to show labels based on position
                show_x_labels = (r == 3)
                show_y_labels = (c == 1)

                # Map tickvals to actual node/element values
                fig.update_xaxes(
                    title_text="Node Count" if show_x_labels else None,
                    tickmode='array',
                    tickvals=x_indices,
                    ticktext=node_counts if show_x_labels else [],
                    range=[-0.5, Nx - 0.5],
                    showticklabels=show_x_labels,
                    row=r, col=c
                )
                
                fig.update_yaxes(
                    title_text="Elements" if show_y_labels else None,
                    tickmode='array',
                    tickvals=y_indices,
                    ticktext=element_counts if show_y_labels else [],
                    range=[-0.5, Ny - 0.5],
                    showticklabels=show_y_labels,
                    row=r, col=c
                )

        # Make the figure square
        fig.update_layout(
            title=f"Selected Metric: {metric_names[selected_metric_index]}",
            width=1000,
            height=1000,
            coloraxis=dict(
                colorscale='Viridis',
                colorbar=dict(
                    title='Value',
                    x=1.01,  # move the colorbar slightly to the right of the subplots
                    y=0.5,
                    len=1.0
                )
            ),
            coloraxis2=dict(
                colorscale='Plasma',
                colorbar=dict(
                    title='Combined',
                    x=1.15,  # place this colorbar a bit further to avoid overlap
                    y=0.5,
                    len=1.0
                )
            ),
            margin=dict(l=50, r=150, t=50, b=50)
        )

        return fig

    # Run the Dash app
    #app.run_server(debug=False, host='0.0.0.0', port=8050)
    app.run_server(debug=False, host='0.0.0.0', port=13717)

if __name__ == "__main__":
    base_directory = "/lus/gila/projects/atlas_aesp_CNDA/oneCCL_test/jobs_test"  # Update to your base directory path
    num_interfaces = 8  # Number of interfaces

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
    metric_names_file = "metric_names.txt"  # Update to your metric names file path
    if os.path.isfile(metric_names_file):
        with open(metric_names_file, 'r') as f:
            all_names = [line.strip() for line in f.readlines()]
    else:
        # Fallback if file not found
        all_names = []

    # If the file doesn't have enough names, fill with generic names
    if len(all_names) < metrics_per_interface:
        all_names += [f"Metric_{i+1}" for i in range(len(all_names), metrics_per_interface)]

    # Run the interactive Dash app
    run_interactive_dash_app(metric_array, node_counts, element_counts, metric_names=all_names, node_map=node_map, results_nodes=job_results_nodes)
