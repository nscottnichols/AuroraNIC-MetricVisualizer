# src/callbacks.py

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative
from dash import Input, Output, State

def register_callbacks(app,
                       metric_array,
                       node_counts,
                       element_counts,
                       metric_names,
                       node_map,
                       results_nodes,
                       bench_array,
                       bench_node_counts,
                       bench_elements):
    """
    All Dash callbacks and their logic are defined here.
    The main app calls register_callbacks(...) to wire them up.
    """
    # -----------------------------------------------------------
    # HELPER DATA
    # -----------------------------------------------------------

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
    all_metric_options = [
        {'label': f"{i}: {metric_names[i]}", 'value': i}
        for i in range(metrics_per_interface)
    ]
    nonzero_metric_options = [
        {'label': f"{i}: {metric_names[i]}", 'value': i}
        for i in range(metrics_per_interface) if not zero_metrics[i]
    ]

    # Build a dictionary of nodes_for_node_count
    # nodes_for_node_count[node_count] = set of node names that appear in that node_count column (across all elements)
    nodes_for_node_count = {nc: set() for nc in node_counts}
    for (nc, ne), node_ids in results_nodes.items():
        for nid in node_ids:
            node_name = reverse_node_map[nid]
            nodes_for_node_count[nc].add(node_name)

    # -----------------------------------------------------------
    # (A) Callback: Add a new calculated metric to the store
    # -----------------------------------------------------------
    @app.callback(
        [Output('calculated-metrics-store', 'data'),
         Output('calc-metric-message', 'children')],
        [Input('add-calc-metric-button', 'n_clicks')],
        [State('calc-metric-name', 'value'),
         State('calc-metric-formula', 'value'),
         State('calculated-metrics-store', 'data')]
    )
    def add_calculated_metric(n_clicks, name, formula, calc_metrics):
        if n_clicks is None or n_clicks == 0:
            # No clicks yet; do nothing.
            return calc_metrics, ""
        if not name or not formula:
            return calc_metrics, "Please provide both a name and a formula."
        # Check for duplicate names.
        if name in calc_metrics:
            return calc_metrics, f"A calculated metric named '{name}' already exists."

        # Test the formula with a dummy array (optional)
        try:
            dummy = np.zeros((metrics_per_interface, 1, 1, 1))
            _ = eval(formula, {"np": np, "m": dummy})
        except Exception as e:
            return calc_metrics, f"Error in formula: {e}"

        # Add the new calculated metric
        calc_metrics[name] = {"name": name, "formula": formula}

        # Clear any cached result for this metric if it already exists.
        if name in app.computed_calc_metrics_cache:
            del app.computed_calc_metrics_cache[name]

        return calc_metrics, f"Calculated metric '{name}' added."

    # -----------------------------------------------------------
    # (B) Callback: Update the metric-dropdown options
    # -----------------------------------------------------------
    @app.callback(
        Output('metric-dropdown', 'options'),
        [Input('hide-zero-metrics', 'value'),
         Input('calculated-metrics-store', 'data')]
    )
    def update_dropdown_options(hide_zero, calc_metrics):
        # Start with built-in options
        if 'hide' in hide_zero:
            options = nonzero_metric_options.copy()
        else:
            options = all_metric_options.copy()

        # Add calculated metrics: use value as "calc:<name>"
        if calc_metrics:
            for name, details in calc_metrics.items():
                options.append({'label': f"Calculated: {details['name']}", 'value': f"calc:{details['name']}"})
        return options

    # -----------------------------------------------------------
    # (C) Main Callback: Update figures (Heatmap & Line Plot, plus Bench)
    # -----------------------------------------------------------
    node_inputs = [Input(f'node-dropdown-{nc}', 'value') for nc in node_counts]

    @app.callback(
        [Output('heatmap-figure', 'figure'),
         Output('line-figure', 'figure'),
         Output('bench-heatmap-figure', 'figure'),
         Output('bench-line-figure', 'figure')],
        [Input('metric-dropdown', 'value'),
         Input('hide-zero-metrics', 'value'),
         Input('node-count-line-dropdown', 'value'),
         *node_inputs,  # one input per node-dropdown (one per node_count)
         Input('heatmap-stat-tabs', 'value'),
         Input('bench-iteration-dropdown', 'value'),
         Input('bench-stat-dropdown', 'value'),
         Input('bench-line-axis-dropdown', 'value'),
         Input('calculated-metrics-store', 'data')]
    )
    def update_figure(selected_metric_value,
                      hide_zero,
                      selected_line_node_count,
                      *args):
        """
        The 'args' contain:
        - node_selections (a tuple of length len(node_counts))
        - bench_iteration, bench_stat, bench_line_axis
        - calc_metrics_store
        in that order.
        """
        # Separate the node-dropdown selections from others (bench iteration/stat/axis and calculated metrics)
        num_node_dropdowns = len(node_counts)
        node_selections = args[:num_node_dropdowns]
        selected_aggregation = args[num_node_dropdowns]
        bench_iteration = args[num_node_dropdowns + 1]
        bench_stat      = args[num_node_dropdowns + 2]
        bench_line_axis = args[num_node_dropdowns + 3]
        calc_metrics_store = args[num_node_dropdowns + 4]

        # Map selected nodes per node_count
        selected_nodes_by_nc = {}
        for i, nc in enumerate(node_counts):
            selected_nodes = node_selections[i]
            if not selected_nodes:  # if none selected, use all nodes
                selected_nodes = nodes_for_node_count[nc]
            selected_nodes_by_nc[nc] = set(selected_nodes)

        # Determine if the selected metric is built-in or calculated
        is_calculated = False
        formula = None
        calc_metric_name = None

        if isinstance(selected_metric_value, int):
            # It's a built-in metric index
            pass
        elif isinstance(selected_metric_value, str) and selected_metric_value.startswith("calc:"):
            is_calculated = True
            calc_metric_name = selected_metric_value[5:]
            if calc_metrics_store and calc_metric_name in calc_metrics_store:
                formula = calc_metrics_store[calc_metric_name]["formula"]
            else:
                # Fallback to built-in index 0 if formula not found
                selected_metric_value = 0
                is_calculated = False

        # Possibly compute the calculated metrics array
        # For calculated metrics, compute an array per interface using the custom formula.
        calc_metric_arrays = None
        if is_calculated:
            # Use the metric name as key.
            key = calc_metric_name
            if key in app.computed_calc_metrics_cache:
                calc_metric_arrays = app.computed_calc_metrics_cache[key]
            else:
                calc_metric_arrays = []
                for interface_index in range(num_interfaces):
                    # Get the full built-in metric data for this interface.
                    m_data = metric_array[interface_index].copy()  # shape: (metrics_per_interface, Nx, Ny, max_nodes)
                    # Replace -1 with np.nan
                    m_data = np.where(m_data == -1, np.nan, m_data)
                    try:
                        calc_array = eval(formula, {"np": np, "m": m_data})
                        # Expect calc_array shape to be (Nx, Ny, max_nodes)
                    except Exception:
                        calc_array = np.full((Nx, Ny, m_data.shape[-1]), np.nan)
                    calc_metric_arrays.append(calc_array)
                app.computed_calc_metrics_cache[key] = calc_metric_arrays

        # ---------------- Heatmap Figures for metric data ----------------
        heatmap_fig = build_metric_heatmaps(
            metric_array, calc_metric_arrays, is_calculated, selected_metric_value,
            selected_nodes_by_nc, node_counts, element_counts, reverse_node_map, 
            results_nodes, metric_names, calc_metric_name, selected_aggregation
        )

        # ---------------- Line Figures for metric data ----------------
        line_fig = build_metric_line_plots(
            metric_array, calc_metric_arrays, is_calculated, selected_metric_value,
            selected_nodes_by_nc, node_counts, element_counts, reverse_node_map, 
            results_nodes, metric_names, calc_metric_name, selected_line_node_count
        )

        # ---------------- Benchmark Heatmap and Lines ----------------
        bench_heatmap_fig = build_bench_heatmap(
            bench_array, bench_node_counts, bench_elements,
            bench_iteration, bench_stat
        )
        bench_line_fig = build_bench_line_plots(
            bench_array, bench_node_counts, bench_elements,
            bench_iteration, bench_stat, bench_line_axis
        )

        return heatmap_fig, line_fig, bench_heatmap_fig, bench_line_fig


# ------------------------------------------------------------------------
# Helper Plotting Functions
# ------------------------------------------------------------------------

def build_metric_heatmaps(metric_array, calc_metric_arrays, is_calculated, selected_metric_value,
                          selected_nodes_by_nc, node_counts, element_counts, reverse_node_map, 
                          results_nodes, metric_names, calc_metric_name, selected_aggregation):
    """
    Constructs the 3x3 subplot figure with 8 interface heatmaps (outer subplots) and a combined heatmap (in the center),
    using the specified statistical aggregation (min, max, avg, std, or sum) over the node dimension.
    """
    num_interfaces, _, Nx, Ny, _ = metric_array.shape

    subplot_titles = [
        "Interface 1", "Interface 2", "Interface 3",
        "Interface 4", "Combined", "Interface 5",
        "Interface 6", "Interface 7", "Interface 8"
    ]

    # Compute the 8 interface heatmaps for the selected metric and aggregation
    # Each heatmap is (node_counts, element_counts), averaged over nodes
    # metric_array: (num_interfaces, metrics_per_interface, node_counts, element_counts, max_nodes)
    # The interface_heatmaps are computed with per-(node_count, element_count) filtering
    interface_heatmaps = []
    for interface_index in range(num_interfaces):
        # Build a 2D array (Nx, Ny) by summing over the filtered nodes
        hm_data = np.empty((Nx, Ny))
        hm_data.fill(np.nan)  # start with nan to use nanmean/sum

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
                        # Use index of node in the list for proper alignment.
                        node_pos = node_ids.index(nid)
                        if is_calculated:
                            val = calc_metric_arrays[interface_index][x, y, node_pos]
                        else:
                            val = metric_array[interface_index, selected_metric_value, x, y, node_pos]
                        if val != -1 and not np.isnan(val):
                            filtered_values.append(val)
                if filtered_values:
                    if selected_aggregation == 'min':
                        hm_data[x, y] = np.nanmin(filtered_values)
                    elif selected_aggregation == 'max':
                        hm_data[x, y] = np.nanmax(filtered_values)
                    elif selected_aggregation == 'avg':
                        hm_data[x, y] = np.nanmean(filtered_values)
                    elif selected_aggregation == 'std':
                        hm_data[x, y] = np.nanstd(filtered_values)
                    elif selected_aggregation == 'sum':
                        hm_data[x, y] = np.nansum(filtered_values)
                else:
                    hm_data[x, y] = 0

        # Replace any remaining NaNs with 0
        hm_data = np.nan_to_num(hm_data, copy=False, nan=0)
        interface_heatmaps.append(hm_data)

    # Compute combined heatmap (sum of all interfaces) using selected aggregation
    if selected_aggregation == 'min':
        combined_heatmap = np.nanmin(interface_heatmaps, axis=0)
    elif selected_aggregation == 'max':
        combined_heatmap = np.nanmax(interface_heatmaps, axis=0)
    elif selected_aggregation == 'avg':
        combined_heatmap = np.nanmean(interface_heatmaps, axis=0)
    elif selected_aggregation == 'std':
        combined_heatmap = np.nanstd(interface_heatmaps, axis=0)
    elif selected_aggregation == 'sum':
        combined_heatmap = np.nansum(interface_heatmaps, axis=0)

    # Prepare customdata arrays for hover information
    nc_mesh, ne_mesh = np.meshgrid(node_counts, element_counts)
    metric_customdata = np.dstack((nc_mesh, ne_mesh))  # shape: (Nx, Ny, 2)

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
    positions = [(1, 1), (1, 2), (1, 3),
                 (2, 1),         (2, 3),
                 (3, 1), (3, 2), (3, 3)]

    # Use indices for plotting, then map ticks to correct ranges
    x_indices = list(range(Nx))
    y_indices = list(range(Ny))

    # Add each interface heatmap
    for i, hm in enumerate(interface_heatmaps):
        r, c = positions[i]
        heatmap_fig.add_trace(
            go.Heatmap(
                z=hm.T,
                x=x_indices,
                y=y_indices,
                colorscale='Viridis',
                customdata=metric_customdata,
                hovertemplate=(
                    "Node Count: %{customdata[0]}<br>" +
                    "Elements: %{customdata[1]}<br>" +
                    "Value: %{z}<extra></extra>"
                ),
                xgap=1,  # spacing between cells
                ygap=1,
                coloraxis="coloraxis"  # Shared coloraxis for all outer plots
            ),
            row=r, col=c
        )

    # Add the combined heatmap in the center (2, 2).
    heatmap_fig.add_trace(
        go.Heatmap(
            z=combined_heatmap.T,
            x=x_indices,
            y=y_indices,
            colorscale='Plasma',
            customdata=metric_customdata,
            hovertemplate=(
                "Node Count: %{customdata[0]}<br>" +
                "Elements: %{customdata[1]}<br>" +
                "Value: %{z}<extra></extra>"
            ),
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

    # Determine the metric label
    if is_calculated:
        metric_label = f"Calculated: {calc_metric_name}"
    else:
        metric_label = metric_names[selected_metric_value]

    # Set the metric label and make the figure square
    heatmap_fig.update_layout(
        title=f"Selected Metric: {metric_label} (Aggregation: {aggregation.capitalize()})",
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

    return heatmap_fig


def build_metric_line_plots(metric_array, calc_metric_arrays, is_calculated, selected_metric_value,
                            selected_nodes_by_nc, node_counts, element_counts, reverse_node_map,
                            results_nodes, metric_names, calc_metric_name, selected_line_node_count):
    """
    Constructs the 3x3 subplot figure for line plots for each interface and
    a combined line plot in the center for the selected node count.
    """

    num_interfaces, _, _, Ny, _ = metric_array.shape

    subplot_titles = [
        "Interface 1", "Interface 2", "Interface 3",
        "Interface 4", "Combined", "Interface 5",
        "Interface 6", "Interface 7", "Interface 8"
    ]

    line_fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=subplot_titles,
        vertical_spacing=0.05, horizontal_spacing=0.05,
        shared_xaxes=False, shared_yaxes=False
    )

    # If the selected node count is invalid, return an empty figure.
    if selected_line_node_count not in node_counts:
        return line_fig

    line_nc = selected_line_node_count
    allowed_nodes_line = selected_nodes_by_nc[line_nc]  # nodes selected for this node_count

    # Compute log2 of element counts.
    log2_element_counts = np.log2(element_counts)
    positions = [(1, 1), (1, 2), (1, 3),
                 (2, 1),         (2, 3),
                 (3, 1), (3, 2), (3, 3)]

    # Build line data per interface.
    #     - Collect data per interface per node
    #     - line_values[node_name][interface_index][element] = value
    #     - store per-interface node lines, then sum for combined
    node_line_data_per_interface = []
    for interface_index in range(num_interfaces):
        # For this interface, build a dict of node_name -> array of Ny values
        interface_node_lines = {node_name: np.zeros(len(element_counts)) for node_name in allowed_nodes_line}

        # x_idx for the chosen node_count column
        x_idx = node_counts.index(line_nc)

        # For each element_count, filter nodes and collect values
        for y in range(len(element_counts)):
            ne = element_counts[y]
            node_ids = results_nodes.get((line_nc, ne), [])

            # Build a map from node_id to val for this interface, metric, x_idx,y
            val_map = {}
            for nid_i, nid in enumerate(node_ids):
                node_name = reverse_node_map[nid]
                if node_name in allowed_nodes_line:
                    if is_calculated:
                        val = calc_metric_arrays[interface_index][x_idx, y, nid_i]
                    else:
                        val = metric_array[interface_index, selected_metric_value, x_idx, y, nid_i]
                    if val == -1 or np.isnan(val):
                        val = 0
                    val_map[node_name] = val

            # Assign values to interface_node_lines
            for node_name in allowed_nodes_line:
                interface_node_lines[node_name][y] = val_map.get(node_name, 0)
        node_line_data_per_interface.append(interface_node_lines)

    # Compute combined node lines by summing across interfaces
    combined_node_lines = {node_name: np.zeros(len(element_counts)) for node_name in allowed_nodes_line}
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

    # Plot lines for each interface.
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

    # Plot the combined lines in the center (2, 2)
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

    # Update axes for all subplots
    for r in range(1, 4):
        for c in range(1, 4):
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

    # Determine the metric label
    if is_calculated:
        metric_label = f"Calculated: {calc_metric_name}"
    else:
        metric_label = metric_names[selected_metric_value]

    line_fig.update_layout(
        title=f"Line Plots for Node Count {selected_line_node_count} (Metric: {metric_label})",
        width=1000,
        height=1000,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return line_fig


def build_bench_heatmap(bench_array, bench_node_counts, bench_elements, bench_iteration, bench_stat):
    """
    Constructs the benchmark heatmap.
    - x-axis: log2(bench_node_counts)
    - y-axis: log2(bench_elements)
    - z-values: bench_array[:, :, bench_iteration, bench_stat]
    """

    # Prepare log2 versions of the benchmark axes
    log2_bench_node_counts = np.log2(bench_node_counts)
    log2_bench_elements = np.log2(bench_elements)

    # (a) Benchmark Heatmap
    #   x-axis -> bench_node_counts
    #   y-axis -> bench_elements
    #   z      -> bench_array[x, y, bench_iteration, bench_stat]
    #            shape = (num_node_counts, num_element_counts)

    # Create customdata for hover info.
    #     - Build a 2D array of (nc, ne) pairs for customdata
    #     - shape => (len(bench_elements), len(bench_node_counts), 2)
    xx, yy = np.meshgrid(bench_node_counts, bench_elements)
    custom_data_2d = np.dstack((xx, yy))  # final shape (num_elements, num_node_counts, 2)

    bench_heatmap_fig = go.Figure()
    # Note: bench_array has shape (len(bench_node_counts), len(bench_elements), 3, 4)
    #     - A 2D slice is picked out of bench_array[:, :, bench_iteration, bench_stat]
    #       with shape (num_node_counts, num_element_counts).
    z_data = bench_array[:, :, bench_iteration, bench_stat]  # shape (num_node_counts, num_element_counts)
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
        title=f"Bench Heatmap (Iteration={bench_iteration}, Stat={['t_min','t_max','t_avg','stddev'][bench_stat]})",
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

    return bench_heatmap_fig


def build_bench_line_plots(bench_array, bench_node_counts, bench_elements,
                           bench_iteration, bench_stat, bench_line_axis):
    """
    Constructs the benchmark line plot.
    Depending on bench_line_axis, plots either:
    - x-axis: log2(bench_elements) with one line per node_count, or
    - x-axis: log2(bench_node_counts) with one line per element.
    """

    bench_line_fig = go.Figure()
    log2_bench_node_counts = np.log2(bench_node_counts)
    log2_bench_elements = np.log2(bench_elements)

    if bench_line_axis == 'elements':
        # x-axis = bench_elements (log2), one line per node_count
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
            xaxis=dict(
                title="Elements",
                tickmode='array',
                tickvals=log2_bench_elements,
                ticktext=[str(ne) for ne in bench_elements]
            )
        )
    else:
        # x-axis = bench_node_counts (log2), one line per element_count
        for j, ne in enumerate(bench_elements):
            yvals = bench_array[:, j, bench_iteration, bench_stat]
            bench_line_fig.add_trace(
                go.Scatter(
                    x=log2_bench_node_counts,
                    y=yvals,
                    mode='lines+markers',
                    name=f"Elements {ne}",
                    customdata=bench_node_counts,
                    hovertemplate=(
                        "Elements: " + str(ne) +
                        "<br>Node Count: %{customdata}" +
                        "<br>Value: %{y}<extra></extra>"
                    )
                )
            )
        bench_line_fig.update_layout(
            xaxis=dict(
                title="Nodes",
                tickmode='array',
                tickvals=log2_bench_node_counts,
                ticktext=[str(nc) for nc in bench_node_counts]
            )
        )

    bench_line_fig.update_layout(
        title=f"Benchmark Lines (Iteration={bench_iteration}, Stat={['t_min','t_max','t_avg','stddev'][bench_stat]})",
        width=600,
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        yaxis=dict(title="Value")
    )

    return bench_line_fig
