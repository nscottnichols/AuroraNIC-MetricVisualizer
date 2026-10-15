"""Single physical NIC traffic screen."""

from __future__ import annotations

import math
from typing import Any

from dash import dcc, html
import plotly.graph_objects as go

from .services import _format_bytes, _format_percent, _format_rate


def _display_rates(record: dict[str, Any]) -> tuple[list[float], list[float]]:
    timestamps = record.get("display_timestamps_ns", ())
    values = record.get("display_values", ())
    x_values: list[float] = []
    y_values: list[float] = []
    previous_timestamp = None
    previous_value = None
    for timestamp, value in zip(timestamps, values, strict=True):
        x_values.append((int(timestamp)) / 1_000_000_000)
        if previous_timestamp is None or int(timestamp) <= previous_timestamp:
            y_values.append(math.nan)
        else:
            duration_s = (int(timestamp) - previous_timestamp) / 1_000_000_000
            if value < previous_value or duration_s <= 0:
                y_values.append(math.nan)
            else:
                y_values.append((value - previous_value) / duration_s)
        previous_timestamp = int(timestamp)
        previous_value = value
    return x_values, y_values


def render_nic_detail(
    view: dict[str, Any], chart_mode: str = "rate", *, chart_mode_id: str
) -> html.Section:
    detail = view["nic_detail"]
    start_ns = int(view["capture"]["selected_start_ns"])
    figure = go.Figure()
    missing_directions = []
    y_title = (
        "Traffic rate (bytes/s)"
        if chart_mode == "rate"
        else "Cumulative bytes"
    )
    for direction, color in (("TX", "#0b6fa4"), ("RX", "#a4460b")):
        record = next(
            (record for record in detail["series"] if record["direction"] == direction),
            None,
        )
        if record is None or record.get("state") in {"missing", "unsupported", "invalid"}:
            missing_directions.append(direction)
            figure.add_trace(go.Scatter(x=[], y=[], name=direction, line={"color": color}))
            continue
        timestamps = record.get("display_timestamps_ns", ())
        values = record.get("display_values", ())
        if chart_mode == "rate":
            x_values, y_values = _display_rates(record)
            y_title = "Traffic rate (bytes/s)"
        else:
            x_values = [(timestamp - start_ns) / 1_000_000_000 for timestamp in timestamps]
            y_values = list(values)
            y_title = "Cumulative bytes"
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines+markers",
                name=direction,
                line={"color": color},
                hovertemplate="%{x:.3f} s<br>%{y:,.0f}<extra>"
                + direction
                + "</extra>",
            )
        )
    if missing_directions:
        figure.add_annotation(
            text=", ".join(missing_directions)
            + " unavailable (not observed as zero); select another direction, collector, or interval.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.94,
            showarrow=False,
        )
    figure.update_layout(
        xaxis_title="Time from window start (s)",
        yaxis_title=y_title,
        template="plotly_white",
        margin={"l": 68, "r": 18, "t": 24, "b": 48},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )

    cards = []
    for record in detail["series"]:
        cards.extend(
            [
                html.Div(
                    className="compact-card",
                    children=[
                        html.P(f"{record['direction']} total", className="stat-label"),
                        html.P(_format_bytes(record["total_bytes"]), className="stat-value"),
                        html.P(record["status"], className="state-detail"),
                    ],
                ),
                html.Div(
                    className="compact-card",
                    children=[
                        html.P(f"{record['direction']} mean rate", className="stat-label"),
                        html.P(_format_rate(record["mean_rate_bytes_per_s"]), className="stat-value"),
                    ],
                ),
                html.Div(
                    className="compact-card",
                    children=[
                        html.P(f"{record['direction']} P99 rate", className="stat-label"),
                        html.P(_format_rate(record["p99_rate_bytes_per_s"]), className="stat-value"),
                        html.P("Population: valid intervals", className="state-detail"),
                    ],
                ),
                html.Div(
                    className="compact-card",
                    children=[
                        html.P(f"{record['direction']} coverage", className="stat-label"),
                        html.P(_format_percent(record["coverage_fraction"]), className="stat-value"),
                        html.P(record["aggregation_population"], className="state-detail"),
                    ],
                ),
            ]
        )

    first = detail["series"][0] if detail["series"] else {}
    collectors = first.get("collector_identities", ())
    collector_policy = first.get("collector_policy", "not observed")
    if len(collectors) > 1:
        collector_names = ", ".join(
            f"{identity['name']} (ID {identity['collector_id']})"
            for identity in collectors
        )
        collector_policy = (
            f"Ambiguous: {len(collectors)} collector streams ({collector_names}). "
            "Select a collector or treat the result as unavailable."
        )
    evidence_items = []
    for record in detail["series"]:
        evidence_items.append(
            html.Li(
                f"{record['direction']}: {record['counter']}; "
                f"{record['point_count']} source samples, "
                f"{record['valid_interval_count']} valid intervals, "
                f"{record['invalid_interval_count']} invalid intervals; "
                f"quality {', '.join(record.get('quality_flags', ())) or 'none'}"
            )
        )

    return html.Section(
        className="nic-workspace",
        children=[
            html.Div(
                className="section-heading",
                children=[
                    html.H2("Single-NIC traffic", id="nic-detail-heading"),
                    html.P(
                        f"{first.get('node', 'No node')} / {first.get('nic', 'No NIC')} · "
                        f"node ID {first.get('node_id', 'unknown')} · "
                        f"NIC ID {first.get('nic_id', 'unknown')}"
                    ),
                    html.P(f"Collector policy: {collector_policy}"),
                ],
            ),
            html.P(
                "Chart modes: Rate (bytes/s) · Cumulative bytes",
                className="chart-mode-note",
            ),
            dcc.RadioItems(
                id=chart_mode_id,
                options=[
                    {"label": "Rate", "value": "rate"},
                    {"label": "Cumulative", "value": "cumulative"},
                ],
                value=chart_mode,
                inline=True,
                className="chart-mode",
            ),
            dcc.Graph(
                id="analysis-nic-graph",
                figure=figure,
                config={"displayModeBar": True, "responsive": True},
                className="analysis-graph",
            ),
            html.Div(className="compact-grid", children=cards),
            html.Details(
                className="evidence-drawer",
                children=[
                    html.Summary("Traffic evidence, formula, and quality"),
                    html.Ul(children=evidence_items or [html.Li("No traffic evidence available.")]),
                    html.P(
                        "Rate formula: delta_bytes / duration_seconds. Display series are bounded; summaries use full-resolution server-side analysis."
                    ),
                    html.P(
                        f"Exact interval: {view['capture']['selected_start_ns']}–"
                        f"{view['capture']['selected_end_ns']} ns; clock: {view['run']['clock_domain']}."
                    ),
                ],
            ),
        ],
    )
