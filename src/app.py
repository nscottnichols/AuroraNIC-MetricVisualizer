# src/app.py

from layout import create_layout
from callbacks import register_callbacks

def create_app(metric_array,
               node_counts,
               element_counts,
               metric_names,
               node_map,
               results_nodes,
               bench_array,
               bench_node_counts,
               bench_elements):
    """
    Create and configure the Dash application instance.
    """
    app = Dash(__name__)

    # Attach cache for computed calculated metric arrays.
    # Keys are the calculated metric names.
    app.computed_calc_metrics_cache = {}

    # Build layout
    app.layout = create_layout(
        metric_array=metric_array,
        node_counts=node_counts,
        element_counts=element_counts,
        metric_names=metric_names,
        node_map=node_map,
        results_nodes=results_nodes,
        bench_array=bench_array,
        bench_node_counts=bench_node_counts,
        bench_elements=bench_elements
    )

    # Register all callbacks
    register_callbacks(
        app=app,
        metric_array=metric_array,
        node_counts=node_counts,
        element_counts=element_counts,
        metric_names=metric_names,
        node_map=node_map,
        results_nodes=results_nodes,
        bench_array=bench_array,
        bench_node_counts=bench_node_counts,
        bench_elements=bench_elements
    )

    return app

