import os
import re
import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output
from plotly.colors import qualitative

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

def parse_oneccl_file(filepath):
    """
    Parse a single oneCCL benchmark file and extract three lines of timing data.

    Returns:
        A list of up to three dictionaries, each with the keys:
            't_min', 't_max', 't_avg', 'stddev'.
        Returns an empty list if parsing fails or if data is incomplete.
    """
    data_lines = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the performance table header
    header_index = None
    for i, line in enumerate(lines):
        if '#bytes' in line and 't_min[usec]' in line:
            header_index = i + 1
            break
    if header_index is None:
        return data_lines

    # Extract up to three lines of data following the header
    for i in range(3):
        idx = header_index + i
        if idx < len(lines):
            row = lines[idx].strip()
            if not row or row.startswith('#'):
                continue
            parts = row.split()
            if len(parts) < 7:
                continue
            try:
                t_min = float(parts[3])
                t_max = float(parts[4])
                t_avg = float(parts[5])
                stddev = float(parts[6])
                data_lines.append({
                    't_min': t_min,
                    't_max': t_max,
                    't_avg': t_avg,
                    'stddev': stddev
                })
            except ValueError:
                pass

    return data_lines


def process_oneccl_benchmarks(base_dir):
    """
    Parse oneCCL benchmark data from multiple job directories.

    For each job directory, searches an "out_{job_dir}" subdirectory for
    files named in the pattern:
        oneccl_allreduce_{job_name}_{node_count}_{ranks}_{rpn}_{elem}_sycl_ccl_gpu_out_w1.txt

    The 'node_count' is inferred from the matching filename, and 'elem' indicates
    the element count. Each file is parsed for timing data via parse_oneccl_file().

    Returns:
        A dictionary of the form:
            {
              node_count: {
                'element_counts': [...],
                'data': np.array of shape (num_elements, 3, 4)
              },
              ...
            }
        where the 3 dimension corresponds to the three lines of timing data,
        and 4 corresponds to (t_min, t_max, t_avg, stddev).
    """
    bench_data = {}

    for job_dir in os.listdir(base_dir):
        job_path = os.path.join(base_dir, job_dir)
        out_dir = os.path.join(job_path, f"out_{job_dir}")
        if not os.path.isdir(out_dir):
            continue

        # Parse node_count from matching filenames
        node_count = None
        file_map = {}

        for fname in os.listdir(out_dir):
            if not fname.startswith("oneccl_allreduce_") or not fname.endswith("_sycl_ccl_gpu_out_w1.txt"):
                continue

            # Example filename:
            #   oneccl_allreduce_10281099.amn-0001_64_768_12_1024_sycl_ccl_gpu_out_w1.txt
            # Pattern: oneccl_allreduce_{job_name}_{node_count}_{ranks}_{rpn}_{elem}_sycl_ccl_gpu_out_w1.txt
            pattern = r"oneccl_allreduce_([^_]+)_(\d+)_(\d+)_(\d+)_(\d+)_sycl_ccl_gpu_out_w1\.txt"
            match = re.match(pattern, fname)
            if not match:
                continue

            # job_name = match.group(1)     # e.g. 10281099.amn-0001
            ncount = int(match.group(2))    # e.g. 64
            # ranks = int(match.group(3))   # e.g. 768
            # rpn = int(match.group(4))     # e.g. 12
            elem_count = int(match.group(5))  # e.g. 1024

            if node_count is None:
                node_count = ncount
            if node_count != ncount:
                # Skip files that have a mismatched node_count
                continue

            file_map[elem_count] = os.path.join(out_dir, fname)

        if node_count is None:
            continue

        sorted_elems = sorted(file_map.keys())
        if not sorted_elems:
            continue

        # Prepare an array to store 3 lines x 4 stats
        data_array = np.zeros((len(sorted_elems), 3, 4), dtype=np.float64)

        for i, elem_count in enumerate(sorted_elems):
            fpath = file_map[elem_count]
            lines_data = parse_oneccl_file(fpath)
            if len(lines_data) == 3:
                for j, d in enumerate(lines_data):
                    data_array[i, j, 0] = d['t_min']
                    data_array[i, j, 1] = d['t_max']
                    data_array[i, j, 2] = d['t_avg']
                    data_array[i, j, 3] = d['stddev']
            else:
                # Fill with NaN if data is incomplete
                data_array[i, :, :] = np.nan

        bench_data[node_count] = {
            'element_counts': sorted_elems,
            'data': data_array
        }

    return bench_data

def prepare_benchmark_array(bench_results):
    """
    Reorganize the benchmark data into a consolidated NumPy array.

    Parameters
    ----------
    bench_results : dict
        A dictionary of the form returned by process_oneccl_benchmarks, i.e.:
        {
          node_count: {
            'element_counts': [e1, e2, ...],
            'data': np.array of shape (num_elems, 3, 4)
                    where:
                      - num_elems is len(element_counts)
                      - the second dimension (3) corresponds to 3 iterations
                      - the third dimension (4) corresponds to [t_min, t_max, t_avg, stddev]
          },
          ...
        }

    Returns
    -------
    bench_array : np.ndarray
        A 4D array of shape:
            (num_node_counts, num_element_counts, 3, 4)
        where:
          - The first axis is over sorted node_counts
          - The second axis is over sorted unique element_counts
          - The third axis is the iteration index (0..2)
          - The fourth axis is the statistic index:
                0 -> t_min
                1 -> t_max
                2 -> t_avg
                3 -> stddev
          Any missing (node_count, element_count) pair will be filled with np.nan.

    node_counts : list of int
        Sorted list of node counts extracted from bench_results.

    element_counts : list of int
        Sorted list of all unique element counts combined across all node counts.

    Notes
    -----
    - This function merges all node counts and element counts from bench_results
      into a single consistent 4D array for simpler heatmap or line-plot usage.
    - For a node_count that does not have a particular element_count, the
      corresponding array slice is filled with np.nan.
    """
    # 1. Gather all node_counts and sort them.
    all_node_counts = sorted(bench_results.keys())

    # 2. Gather all element_counts from all node_counts and build a global sorted list.
    all_element_sets = []
    for nc in all_node_counts:
        all_element_sets.append(bench_results[nc]['element_counts'])
    # Flatten and deduplicate
    unique_elements = sorted(set().union(*all_element_sets))

    # 3. Prepare the 4D array:
    #    shape = (num_node_counts, num_element_counts, 3, 4)
    #    Fill with np.nan by default.
    num_nodes = len(all_node_counts)
    num_elems = len(unique_elements)
    bench_array = np.full((num_nodes, num_elems, 3, 4), np.nan, dtype=np.float64)

    # 4. Build a small helper index for quick lookup:
    #    For each node_count, 'element_counts' is already sorted,
    #    so create a dict: {elem_count -> row index in the data array}.
    #    The map allows data to be easily copied into the correct location in bench_array.
    for i, node_count in enumerate(all_node_counts):
        elem_list = bench_results[node_count]['element_counts']
        data_3d = bench_results[node_count]['data']  # shape = (nLocal, 3, 4)
        # Build a map from elem -> local row index.
        local_index_map = {elem_val: idx for idx, elem_val in enumerate(elem_list)}

        # Fill bench_array for current node_count
        for j, global_elem in enumerate(unique_elements):
            if global_elem in local_index_map:
                local_idx = local_index_map[global_elem]
                # data_3d[local_idx] is shape (3,4) for the 3 iterations x 4 stats
                bench_array[i, j, :, :] = data_3d[local_idx, :, :]

    return bench_array, all_node_counts, unique_elements

# New interactive plotting function with Plotly and Dash
def run_interactive_dash_app(metric_array, node_counts, element_counts, metric_names,
                             node_map, results_nodes,
                             bench_array, bench_node_counts, bench_elements):

    # Calculate log2 of element counts
    log2_element_counts = np.log2(element_counts)

    # Dimensions of metric_array
    num_interfaces, metrics_per_interface, Nx, Ny, max_nodes = metric_array.shape

    # Reverse the node_map to get node_name from node_id
    reverse_node_map = {v: k for k, v in node_map.items()}

    # Determine which metrics are all zero:
    # For each metric_index, check all interfaces and all data points
    # Convert -1 to nan for checking actual sums
    mask = (metric_array == -1)
    data_filled = np.where(mask, np.nan, metric_array)

    # Sum absolute values to ensure we catch negative differences if any
    metric_sums = np.nansum(np.abs(data_filled), axis=(0, 2, 3, 4)) # sum over interfaces, node_counts, element_counts, max_nodes
    zero_metrics = (metric_sums == 0)

    # Build metric dropdown options
    all_metric_options = [{'label': metric_names[i], 'value': i} for i in range(metrics_per_interface)]
    nonzero_metric_options = [{'label': metric_names[i], 'value': i} 
                              for i in range(metrics_per_interface) if not zero_metrics[i]]

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

    # Dropdown for selecting node_count for line plots
    node_count_line_options = [{'label': str(nc), 'value': nc} for nc in node_counts]

    # ------------------- OneCCL Benchmarks Controls -------------------
    # 1) Dropdown to select iteration index (0..2)
    bench_iteration_dropdown = html.Div([
        html.Label("Select Benchmark Iteration:"),
        dcc.Dropdown(
            id='bench-iteration-dropdown',
            options=[{'label': f"Iteration {i}", 'value': i} for i in [0, 1, 2]],
            value=0,  # default iteration
            clearable=False,
            style={'width': '200px', 'display': 'inline-block', 'margin-left': '20px'}
        )
    ], style={'textAlign': 'center', 'margin': '20px'})

    # 2) Dropdown to select stat index ([t_min, t_max, t_avg, stddev] -> [0,1,2,3])
    bench_stat_dropdown = html.Div([
        html.Label("Select Benchmark Stat:"),
        dcc.Dropdown(
            id='bench-stat-dropdown',
            options=[
                {'label': 't_min',   'value': 0},
                {'label': 't_max',   'value': 1},
                {'label': 't_avg',   'value': 2},
                {'label': 'stddev',  'value': 3},
            ],
            value=2,  # default to t_avg
            clearable=False,
            style={'width': '200px', 'display': 'inline-block', 'margin-left': '20px'}
        )
    ], style={'textAlign': 'center', 'margin': '20px'})
    # -----------------------------------------------------------------------

    # Create the Dash app
    app = Dash(__name__)

    # Layout: metric controls + metric figures + benchmark controls + benchmark figures
    app.layout = html.Div([

        html.H1("Interactive Heatmap Analysis for Metric Differences and Line Plot Slices per Node Count",
                style={'textAlign': 'center'}),

        # Metric controls (show/hide zero metrics, metric selection, etc.)
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

        html.Div([
            html.Label("Select Node Count for Line Plots:"),
            dcc.Dropdown(
                id='node-count-line-dropdown',
                options=node_count_line_options,
                value=node_counts[0] if node_counts else None,
                clearable=False,
                style={'width': '300px', 'display': 'inline-block', 'margin-left': '20px'}
            )
        ], style={'textAlign': 'center', 'margin': '20px'}),

        # Metric Plots (2 figures: heatmap + line)
        html.Div([
            html.Div([
                html.H2("Heatmaps (Metric Data)", style={'textAlign': 'center'}),
                dcc.Graph(id='heatmap-figure')
            ], style={'display': 'inline-block', 'verticalAlign': 'top'}),

            html.Div([
                html.H2("Line Plots (Metric Data)", style={'textAlign': 'center'}),
                dcc.Graph(id='line-figure')
            ], style={'display': 'inline-block', 'verticalAlign': 'top'})
        ], style={'width': '100%', 'textAlign': 'center', 'marginBottom': '50px'}),

        # OneCCL Benchmark Section
        html.Hr(),
        html.H2("oneCCL Benchmark Data", style={'textAlign': 'center'}),
        html.Div([
            bench_iteration_dropdown,
            bench_stat_dropdown
        ], style={'textAlign': 'center'}),

        html.Div([
            html.Div([
                html.H2("Benchmark Heatmap", style={'textAlign': 'center'}),
                dcc.Graph(id='bench-heatmap-figure')
            ], style={'display': 'inline-block', 'verticalAlign': 'top'}),

            html.Div([
                html.H2("Benchmark Line Plots", style={'textAlign': 'center'}),
                dcc.Graph(id='bench-line-figure')
            ], style={'display': 'inline-block', 'verticalAlign': 'top'})
        ], style={'width': '100%', 'textAlign': 'center'}),

    ])

    # Callback to update the metric controls
    @app.callback(
        Output('metric-dropdown', 'options'),
        Input('hide-zero-metrics', 'value')
    )
    def update_dropdown_options(hide_zero):
        if 'hide' in hide_zero:
            return nonzero_metric_options
        else:
            return all_metric_options

    # Build a list of Inputs for each node-dropdown
    node_inputs = [Input(f'node-dropdown-{nc}', 'value') for nc in node_counts]

    # ------------------- UPDATE FIGURES CALLBACK -------------------
    @app.callback(
        # Return *four* figures: two metric figures + two benchmark figures
        [
            Output('heatmap-figure', 'figure'),
            Output('line-figure', 'figure'),
            Output('bench-heatmap-figure', 'figure'),
            Output('bench-line-figure', 'figure')
        ],
        [
            Input('metric-dropdown', 'value'),
            Input('hide-zero-metrics', 'value'),
            Input('node-count-line-dropdown', 'value'),
            *node_inputs,  # one input per node-dropdown (one per node_count)
            Input('bench-iteration-dropdown', 'value'),
            Input('bench-stat-dropdown', 'value')
        ]
    )
    def update_figure(selected_metric_index,
                      hide_zero,
                      selected_line_node_count,
                      *args):
        """
        args = node_selections + (bench_iteration, bench_stat)
        where node_selections is a tuple of length len(node_counts).
        """
        # Separate the node-dropdown selections from the new bench iteration/stat
        node_selections = args[:len(node_counts)]
        bench_iteration = args[len(node_counts)]
        bench_stat      = args[len(node_counts)+1]

        # ------------------- (1) Metric Plots -------------------
        # Map selected nodes per node_count
        selected_nodes_by_nc = {}
        for i, nc in enumerate(node_counts):
            selected_nodes = node_selections[i]
            if not selected_nodes:  # If None or empty, use all nodes
                selected_nodes = nodes_for_node_count[nc]
            selected_nodes_by_nc[nc] = set(selected_nodes)

        # Subplot titles for the metric plots
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

        # Compute the 8 interface heatmaps for the selected metric
        # Each heatmap is (node_counts, element_counts), averaged over nodes
        # metric_array: (num_interfaces, metrics_per_interface, node_counts, element_counts, max_nodes)
        # The interface_heatmaps are computed with per-(node_count, element_count) filtering
        interface_heatmaps = []
        for interface_index in range(num_interfaces):
            # Build a 2D array (Nx, Ny) by summing over the filtered nodes
            hm_data = np.zeros((Nx, Ny))
            hm_data[:] = np.nan  # start with nan to use nanmean/sum

            # Walk through each (x,y) cell
            # (x, y) corresponds to node_counts[x], element_counts[y]
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
                        hm_data[x, y] = 0

            # If there are nans, replace them with 0
            hm_data = np.nan_to_num(hm_data, nan=0)
            interface_heatmaps.append(hm_data)

        # Compute combined heatmap (sum of all interfaces)
        combined_heatmap = np.sum(interface_heatmaps, axis=0)

        # Create a 3x3 subplot figure for the metric heatmaps
        heatmap_fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=subplot_titles,
            vertical_spacing=0.02, horizontal_spacing=0.02,
            shared_xaxes=False, shared_yaxes=False
        )

        # Positions of the subplots:
        # Interfaces are placed in the outer 8 subplots (ignoring the center)
        # and the combined is placed in the center (2, 2) in 1-based indexing.
        # The layout (0-based indexing for code reference):
        # (1, 1) is the center subplot in 0-based, but plotly is 1-based indexing so center is (2, 2).
        # Below is the subplot order (row, col) for the 8 interfaces:
        positions = [(1,1), (1,2), (1,3),
                     (2,1),        (2,3),
                     (3,1), (3,2), (3,3)]

        # Use indices for plotting, then map ticks to correct ranges
        x_indices = list(range(Nx))
        y_indices = list(range(Ny))

        # Add the 8 interface heatmaps
        for i, hm_data in enumerate(interface_heatmaps):
            r, c = positions[i]
            heatmap_fig.add_trace(
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
        heatmap_fig.add_trace(
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
                heatmap_fig.update_xaxes(
                    title_text="Node Count" if show_x_labels else None,
                    tickmode='array',
                    tickvals=x_indices,
                    ticktext=node_counts if show_x_labels else [],
                    range=[-0.5, Nx - 0.5],
                    showticklabels=show_x_labels,
                    row=r, col=c
                )

                heatmap_fig.update_yaxes(
                    title_text="Elements" if show_y_labels else None,
                    tickmode='array',
                    tickvals=y_indices,
                    ticktext=element_counts if show_y_labels else [],
                    range=[-0.5, Ny - 0.5],
                    showticklabels=show_y_labels,
                    row=r, col=c
                )

        # Make the figure square
        heatmap_fig.update_layout(
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

        # ---- Build the line plots for the metric data ----
        line_fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=subplot_titles,
            vertical_spacing=0.05, horizontal_spacing=0.05,
            shared_xaxes=False, shared_yaxes=False
        )

        # For the chosen selected_line_node_count, show one line per node
        if selected_line_node_count not in node_counts:
            # If invalid node_count, return empty lines for the metric line plot
            return heatmap_fig, line_fig, go.Figure(), go.Figure()

        line_nc = selected_line_node_count
        allowed_nodes_line = selected_nodes_by_nc[line_nc]  # nodes selected for this node_count

        # Collect data per interface per node
        # line_values[node_name][interface_index][element] = value
        # store per-interface node lines, then sum for combined
        node_line_data_per_interface = []

        for interface_index in range(num_interfaces):
            # For this interface, build a dict of node_name -> array of Ny values
            interface_node_lines = {node_name: np.zeros(Ny) for node_name in allowed_nodes_line}

            # x_idx for the chosen node_count column
            x_idx = node_counts.index(line_nc)

            # For each element_count, filter nodes and collect values
            for y in range(Ny):
                ne = element_counts[y]
                node_ids = results_nodes.get((line_nc, ne), [])
                # Build a map from node_id to val for this interface, metric, x_idx,y
                val_map = {}
                for nid_i, nid in enumerate(node_ids):
                    node_name = reverse_node_map[nid]
                    if node_name in allowed_nodes_line:
                        val = metric_array[interface_index, selected_metric_index, x_idx, y, nid_i]
                        if val == -1:
                            val = 0
                        val_map[node_name] = val

                # Assign values to interface_node_lines
                for node_name in allowed_nodes_line:
                    interface_node_lines[node_name][y] = val_map.get(node_name, 0)

            node_line_data_per_interface.append(interface_node_lines)

        # Compute combined node lines by summing across interfaces
        combined_node_lines = {node_name: np.zeros(Ny) for node_name in allowed_nodes_line}
        for node_name in allowed_nodes_line:
            for interface_lines in node_line_data_per_interface:
                combined_node_lines[node_name] += interface_lines[node_name]

        # Assign colors/styles per node and use legendgroup to show single legend entry
        color_cycle = qualitative.Dark24
        node_colors = {}
        seen_nodes = set()

        def get_node_style(node_name):
            if node_name not in node_colors:
                node_colors[node_name] = color_cycle[len(node_colors) % len(color_cycle)]
            return node_colors[node_name]

        # Plot lines for each interface
        for i, interface_lines in enumerate(node_line_data_per_interface):
            r, c = positions[i]
            for node_name in allowed_nodes_line:
                node_color = get_node_style(node_name)
                show_legend_flag = (node_name not in seen_nodes)
                if show_legend_flag:
                    seen_nodes.add(node_name)

                # Extract a truncated version of the node name
                truncated_name = node_name.split('.', 1)[0]

                line_fig.add_trace(
                    go.Scatter(
                        x=log2_element_counts,
                        y=interface_lines[node_name],
                        mode='lines+markers',
                        name=truncated_name,
                        legendgroup=truncated_name,
                        showlegend=show_legend_flag,
                        line=dict(color=node_color),
                        customdata=element_counts,
                        hovertemplate=(
                            "Node: " + node_name +
                            "<br>Elements: %{customdata}" +
                            "<br>Value: %{y}<extra></extra>"
                        )
                    ),
                    row=r, col=c
                )

        # Plot combined lines in the center
        combined_r, combined_c = (2, 2)
        for node_name in allowed_nodes_line:
            node_color = get_node_style(node_name)

            # Extract a truncated version of the node name
            truncated_name = node_name.split('.', 1)[0]

            line_fig.add_trace(
                go.Scatter(
                    x=log2_element_counts,
                    y=combined_node_lines[node_name],
                    mode='lines+markers',
                    name=truncated_name,
                    legendgroup=truncated_name,
                    showlegend=False,  # Already shown above
                    line=dict(color=node_color),
                    customdata=element_counts,
                    hovertemplate=(
                        "Node: " + node_name +
                        "<br>Elements: %{customdata}" +
                        "<br>Value: %{y}<extra></extra>"
                    )
                ),
                row=combined_r, col=combined_c
            )

        # Update axes for line plots
        for r in range(1,4):
            for c in range(1,4):
                show_x_labels = (r == 3)
                show_y_labels = (c == 1)
                line_fig.update_xaxes(
                    title_text="Elements" if show_x_labels else None,
                    tickmode='array',
                    tickvals=log2_element_counts,
                    ticktext=element_counts if show_x_labels else [],
                    showticklabels=show_x_labels,
                    row=r, col=c
                )
                line_fig.update_yaxes(
                    title_text="Value" if show_y_labels else None,
                    showticklabels=show_y_labels,
                    row=r, col=c
                )

        line_fig.update_layout(
            title=f"Line Plots for Node Count {selected_line_node_count} (Metric: {metric_names[selected_metric_index]})",
            width=1000,
            height=1000,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        # ------------------- (2) OneCCL Benchmark Plots -------------------

        # Prepare log2 versions of the benchmark axes
        log2_bench_node_counts = np.log2(bench_node_counts)
        log2_bench_elements    = np.log2(bench_elements)

        # (a) Benchmark Heatmap
        #   x-axis -> bench_node_counts
        #   y-axis -> bench_elements
        #   z      -> bench_array[x, y, bench_iteration, bench_stat]
        #            shape = (num_node_counts, num_element_counts)

        # Build a 2D array of (nc, ne) pairs for customdata
        # shape => (len(bench_elements), len(bench_node_counts), 2)
        xx, yy = np.meshgrid(bench_node_counts, bench_elements)
        custom_data_2d = np.dstack((xx, yy))  # final shape (num_elements, num_node_counts, 2)
        bench_heatmap_fig = go.Figure()
        # Note: bench_array has shape (len(bench_node_counts), len(bench_elements), 3, 4)
        # A 2D slice is picke out bench_array[:, :, bench_iteration, bench_stat]
        # with shape (num_node_counts, num_element_counts).
        z_data = bench_array[:, :, bench_iteration, bench_stat]  # shape (node_counts, elements)

        bench_heatmap_fig.add_trace(
            go.Heatmap(
                z=z_data.T,
                x=log2_bench_node_counts,
                y=log2_bench_elements,
                colorscale='Viridis',
                coloraxis="coloraxis",
                customdata=custom_data_2d,
                hovertemplate=(
                    "Node Count: %{customdata[0]}<br>" +
                    "Elements: %{customdata[1]}<br>" +
                    "Value: %{z}<extra></extra>"
                )
            )
        )

        bench_heatmap_fig.update_layout(
            title=f"OneCCL Bench Heatmap (Iteration={bench_iteration}, Stat={['t_min','t_max','t_avg','stddev'][bench_stat]})",
            width=600,
            height=600,
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis=dict(
                title="Nodes",
                tickmode='array',
                tickvals=log2_bench_node_counts,
                ticktext=[str(nc) for nc in bench_node_counts]
            ),
            yaxis=dict(
                title="Elements",
                tickmode='array',
                tickvals=log2_bench_elements,
                ticktext=[str(ne) for ne in bench_elements]
            ),
            coloraxis=dict(
                colorscale='Viridis',
                colorbar=dict(
                    title='Value'
                )
            )
        )

        # (b) Benchmark Line Plot
        #   x-axis -> bench_elements
        #   Each line -> one node_count
        #   y-values -> bench_array[node_count_index, :, bench_iteration, bench_stat]
        bench_line_fig = go.Figure()
        for i, nc in enumerate(bench_node_counts):
            yvals = bench_array[i, :, bench_iteration, bench_stat]
            bench_line_fig.add_trace(
                go.Scatter(
                    x=log2_bench_elements,
                    y=yvals,
                    mode='lines+markers',
                    name=f"Node Count {nc}",
                    customdata=bench_elements,
                    hovertemplate=(
                        "Node Count: " + str(nc) +
                        "<br>Elements: %{customdata}" +
                        "<br>Value: %{y}<extra></extra>"
                    )
                )
            )

        bench_line_fig.update_layout(
            title=f"OneCCL Benchmark Lines (Iteration={bench_iteration}, Stat={['t_min','t_max','t_avg','stddev'][bench_stat]})",
            width=600,
            height=600,
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis=dict(
                title="Elements",
                tickmode='array',
                tickvals=log2_bench_elements,
                ticktext=[str(ne) for ne in bench_elements]
            ),
            yaxis=dict(
                title="Value"
            )
        )

        # Return figures
        return heatmap_fig, line_fig, bench_heatmap_fig, bench_line_fig

    # Run the Dash app
    #app.run_server(debug=False, host='0.0.0.0', port=8050)
    app.run(debug=False, host='0.0.0.0', port=13717)

if __name__ == "__main__":
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
