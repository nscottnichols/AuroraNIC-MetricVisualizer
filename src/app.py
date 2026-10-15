import os
import signal
from collections import OrderedDict
from dataclasses import dataclass

from dash import Dash
from layout import create_layout
from callbacks import register_callbacks


@dataclass(frozen=True)
class RuntimeConfig:
    host: str
    port: int


def resolve_runtime_config(host=None, port=None):
    configured_host = host if host is not None else os.environ.get(
        "AURORA_APP_HOST", "127.0.0.1"
    )
    configured_port = port if port is not None else os.environ.get(
        "AURORA_APP_PORT", "8050"
    )
    try:
        resolved_port = int(configured_port)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Dash server port: {configured_port!r}") from exc
    if not 1 <= resolved_port <= 65535:
        raise ValueError(f"Dash server port is out of range: {resolved_port}")
    if not configured_host:
        raise ValueError("Dash server host must not be empty")
    return RuntimeConfig(host=configured_host, port=resolved_port)


def readiness_report(app):
    state = app.nic_runtime_state
    metric_shape = tuple(getattr(state["metric_array"], "shape", ()))
    bench_shape = tuple(getattr(state["bench_array"], "shape", ()))
    checks = {
        "layout_initialized": app.layout is not None,
        "callbacks_registered": bool(app.nic_callbacks_registered),
        "metric_array_5d": len(metric_shape) == 5,
        "metric_axis_2_matches_node_counts": (
            len(metric_shape) == 5 and len(state["node_counts"]) == metric_shape[2]
        ),
        "metric_axis_3_matches_element_counts": (
            len(metric_shape) == 5 and len(state["element_counts"]) == metric_shape[3]
        ),
        "metric_names_match_axis_1": (
            len(metric_shape) >= 2 and len(state["metric_names"]) == metric_shape[1]
        ),
        "node_map_nonempty": len(state["node_map"]) > 0,
        "node_results_present": len(state["results_nodes"]) > 0,
        "benchmark_array_4d": len(bench_shape) == 4,
        "benchmark_axes_match": (
            len(bench_shape) == 4
            and len(state["bench_node_counts"]) == bench_shape[0]
            and len(state["bench_elements"]) == bench_shape[1]
        ),
    }
    return {
        "status": "ready" if all(checks.values()) else "not_ready",
        "checks": checks,
    }


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
    Create and configure a Dash application with health and readiness routes.
    """
    app = Dash(__name__)

    app.computed_calc_metrics_cache = OrderedDict()
    app.nic_runtime_state = {
        "metric_array": metric_array,
        "node_counts": node_counts,
        "element_counts": element_counts,
        "metric_names": metric_names,
        "node_map": node_map,
        "results_nodes": results_nodes,
        "bench_array": bench_array,
        "bench_node_counts": bench_node_counts,
        "bench_elements": bench_elements,
    }

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

    app.nic_callbacks_registered = True

    @app.server.get("/healthz")
    def healthz():
        return {"status": "ok"}, 200

    @app.server.get("/readyz")
    def readyz():
        report = readiness_report(app)
        status_code = 200 if report["status"] == "ready" else 503
        return report, status_code

    return app


def run_app(app, host=None, port=None):
    """
    Run the Dash server on a loopback address by default and stop it cleanly.
    """
    config = resolve_runtime_config(host=host, port=port)
    app.runtime_config = config
    original_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def request_shutdown(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_shutdown)
    try:
        app.run(
            host=config.host,
            port=config.port,
            debug=False,
            use_reloader=False,
        )
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, original_sigterm_handler)
