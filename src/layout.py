# src/layout.py

import numpy as np
from dash import dcc, html

def create_layout(metric_array,
                  node_counts,
                  element_counts,
                  metric_names,
                  node_map,
                  results_nodes,
                  bench_array,
                  bench_node_counts,
                  bench_elements):
    """
    Returns the entire Dash layout (all html.Div() wrappers).
    """

    # 1. Prepare data needed for the layout
    # Calculate log2 of element counts
    log2_element_counts = np.log2(element_counts)

    # Dimensions of metric_array
    num_interfaces, metrics_per_interface, Nx, Ny, max_nodes = metric_array.shape

    # Reverse the node_map to get node_name from node_id
    reverse_node_map = {v: k for k, v in node_map.items()}

    # Determine which built-in metrics are all zero
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

    # 2. Build the top-level layout
    layout = html.Div([
        html.H1("Interactive Heatmap Analysis for Metric Differences and Line Plot Slices per Node Count",
                style={'textAlign': 'center'}),

        # --- Calculated Metric Editor ---
        html.Div([
            html.H3("Calculated Metric Editor"),
            html.Div([
                html.Label("Metric Name:"),
                dcc.Input(
                    id="calc-metric-name", 
                    type="text", 
                    placeholder="Enter calculated metric name", 
                    style={'width': '300px'}
                ),
                html.Br(),
                html.Label("Formula (use 'm' for built-in metrics):"),
                dcc.Input(
                    id="calc-metric-formula", 
                    type="text", 
                    placeholder="e.g., (m[0]-m[1])/(m[2]+1)", 
                    style={'width': '300px'}
                ),
                html.Br(),
                html.Button("Add Calculated Metric", id="add-calc-metric-button", n_clicks=0),
                html.Div(id="calc-metric-message", style={'color': 'green', 'margin-top': '10px'})
            ], style={'display': 'inline-block', 'margin': '20px'})
        ], style={'textAlign': 'center'}),

        # dcc.Store to keep calculated metrics
        dcc.Store(id='calculated-metrics-store', data={}),

        # --- Metric controls ---
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

        #Aggregation Tabs for the heatmap
        html.Div([
            html.Label("Select Aggregation:"),
            dcc.Tabs(
                id='heatmap-stat-tabs',
                value='avg',  # default aggregation is average
                children=[
                    dcc.Tab(label='Min', value='min'),
                    dcc.Tab(label='Max', value='max'),
                    dcc.Tab(label='Average', value='avg'),
                    dcc.Tab(label='Std Dev', value='std'),
                    dcc.Tab(label='Sum', value='sum')
                ]
            )
        ], style={'textAlign': 'center', 'margin': '20px'}),

        # Node dropdowns
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

        # --- Metric Plots ---
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

        # --- Benchmark Section ---
        html.Hr(),
        html.H2("Benchmark Data", style={'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.Label("Select Benchmark Iteration:"),
                dcc.Dropdown( # Dropdown to select iteration index (0..2)
                    id='bench-iteration-dropdown',
                    options=[{'label': f"Iteration {i}", 'value': i} for i in [0, 1, 2]],
                    value=0,  # default iteration
                    clearable=False,
                    style={'width': '200px', 'display': 'inline-block', 'margin-left': '20px'}
                )
            ], style={'textAlign': 'center', 'margin': '20px'}),

            html.Div([
                html.Label("Select Benchmark Stat:"),
                dcc.Dropdown( # Dropdown to select stat index ([t_min, t_max, t_avg, stddev] -> [0,1,2,3])
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
            ], style={'textAlign': 'center', 'margin': '20px'}),

            html.Div([
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
        ], style={'display': 'inline-block'}),

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

    return layout
