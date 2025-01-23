import os
import re
import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output
from plotly.colors import qualitative

from metrics_processing import (
    process_all_jobs,
    prepare_metric_array
)

from oneccl_processing import (
    process_oneccl_benchmarks,
    prepare_benchmark_array
)

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
            value=2,  # default t_avg
            clearable=False,
            style={'width': '200px', 'display': 'inline-block', 'margin-left': '20px'}
        )
    ], style={'textAlign': 'center', 'margin': '20px'})

    # 3) Dropdown to select which dimension appears on the x-axis of line plot
    bench_line_axis_dropdown = html.Div([
        html.Label("Select Line X-Axis:"),
        dcc.Dropdown(
            id='bench-line-axis-dropdown',
            options=[
                {'label': 'Elements', 'value': 'elements'},
                {'label': 'Nodes',   'value': 'nodes'},
            ],
            value='elements',  # default
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
            bench_stat_dropdown,
            bench_line_axis_dropdown
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
            Input('bench-stat-dropdown', 'value'),
            Input('bench-line-axis-dropdown', 'value')
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
        # Separate the node-dropdown selections from the new bench iteration/stat/axis
        node_selections = args[:len(node_counts)]
        bench_iteration = args[len(node_counts)]
        bench_stat      = args[len(node_counts) + 1]
        bench_line_axis = args[len(node_counts) + 2]

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

        # Prepare customdata arrays for hover information
        nc_mesh, ne_mesh = np.meshgrid(node_counts, element_counts)
        metric_customdata = np.dstack((nc_mesh, ne_mesh))  # shape (Nx, Ny, 2)

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

        # Add the combined heatmap in the center (2, 2)
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

        if bench_line_axis == 'elements':
            #   x-axis = bench_elements (log2), one line per node_count
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
            #   x-axis = bench_node_counts (log2), one line per element_count
            for j, ne in enumerate(bench_elements):
                yvals = bench_array[:, j, bench_iteration, bench_stat]  # shape=(len(bench_node_counts),)
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
            title=f"OneCCL Benchmark Lines (Iteration={bench_iteration}, Stat={['t_min','t_max','t_avg','stddev'][bench_stat]})",
            width=600,
            height=600,
            margin=dict(l=50, r=50, t=80, b=50),
            yaxis=dict(title="Value")
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
