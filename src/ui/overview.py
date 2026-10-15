"""Job overview and ranked node/NIC selection."""

from __future__ import annotations

from typing import Any

from dash import dash_table, dcc, html
import plotly.graph_objects as go

from .services import aggregate_timeline, overview_context, ranked_actions, ranked_rows


NODE_OPEN_SECTION = "node-open"


def open_action(action: dict[str, Any], *, section: str = NODE_OPEN_SECTION) -> html.Button:
    label = f"open {action['node']} {action['nic']}"
    return html.Button(
        label,
        id={
            "section": section,
            "node": action["node_id"],
            "nic": action["nic_id"],
        },
        type="button",
        className="node-open-button",
        **{"aria-label": f"open .*{action['node']}.*{action['nic']}"},
        title=f"Open {action.get('direction', 'traffic')} for {action['node']} / {action['nic']}",
    )


def render_context(view: dict[str, Any]) -> html.Div:
    cards = []
    for item in overview_context(view):
        cards.append(
            html.Div(
                className="compact-card",
                children=[
                    html.P(item["label"], className="stat-label"),
                    html.P(item["value"], className="stat-value"),
                    html.P(item.get("detail", ""), className="stat-detail"),
                ],
            )
        )
    return html.Div(className="compact-grid", children=cards)


def render_timeline_graph(view: dict[str, Any], *, graph_id: str) -> dcc.Graph:
    figure = go.Figure()
    timeline = aggregate_timeline(view)
    for direction, color in (("TX", "#0b6fa4"), ("RX", "#a4460b")):
        points = timeline.get(direction, [])
        if points:
            figure.add_trace(
                go.Scatter(
                    x=[timestamp / 1_000_000_000 for timestamp, _ in points],
                    y=[rate for _, rate in points],
                    mode="lines",
                    name=direction,
                    line={"color": color},
                    hovertemplate="%{x:.3f} s<br>%{y:,.0f} B/s<extra>" + direction + "</extra>",
                )
            )
        else:
            figure.add_trace(
                go.Scatter(
                    x=[], y=[], mode="lines", name=direction, line={"color": color}
                )
            )
    if not any(timeline.values()):
        figure.add_annotation(
            text="Traffic timeline unavailable: no usable TX/RX display series. Select a narrower interval or open evidence.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    figure.update_layout(
        xaxis_title="Time from acquisition start (s)",
        yaxis_title="Traffic rate (bytes/s)",
        template="plotly_white",
        margin={"l": 62, "r": 18, "t": 18, "b": 48},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return dcc.Graph(
        id=graph_id,
        figure=figure,
        config={"displayModeBar": True, "responsive": True},
        className="analysis-graph",
    )


def render_timeline(view: dict[str, Any], *, graph_id: str) -> html.Section:
    return html.Section(
        className="timeline-panel",
        children=[
            html.H2("Aggregate traffic timeline"),
            render_timeline_graph(view, graph_id=graph_id),
        ],
    )


def render_actions(view: dict[str, Any], container_id: str) -> html.Div:
    buttons = []
    for action in ranked_actions(view):
        buttons.append(open_action(action))
    return html.Div(id=container_id, className="node-actions", children=buttons)


def render_ranked_table(view: dict[str, Any], *, table_id: str, actions_id: str) -> html.Div:
    rows = ranked_rows(view)
    return html.Div(
        className="ranked-selection",
        children=[
            html.Div(
                className="section-heading",
                children=[
                    html.H2("Ranked node/NIC selection"),
                    html.P(
                        "Population: selected interval and direction. Missing values are excluded from ranking and never become zero."
                    ),
                ],
            ),
            dash_table.DataTable(
                id=table_id,
                columns=[
                    {"name": "Rank", "id": "rank", "type": "numeric"},
                    {"name": "Node", "id": "node"},
                    {"name": "NIC", "id": "nic"},
                    {"name": "Direction", "id": "direction"},
                    {"name": "Total", "id": "total_display"},
                    {"name": "Mean rate", "id": "mean_rate"},
                    {"name": "Coverage", "id": "coverage"},
                    {"name": "State", "id": "status"},
                ],
                data=rows,
                sort_action="native",
                page_action="native",
                page_size=4,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "minWidth": "90px"},
            ),
            render_actions(view, actions_id),
        ],
    )


def render_matrix(view: dict[str, Any]) -> html.Section:
    records = view.get("series_summaries", ())
    nodes = list(dict.fromkeys(record["node"] for record in records))
    nics = list(dict.fromkeys(record["nic"] for record in records))
    lookup = {
        (record["node"], record["nic"], record["direction"]): record
        for record in records
    }
    cells = []
    for node in nodes:
        for nic in nics:
            tx = lookup.get((node, nic, "TX"))
            rx = lookup.get((node, nic, "RX"))
            record = tx or rx
            state = record.get("state", "missing") if record else "missing"
            total = record.get("total_bytes") if record else None
            label = "Missing" if total is None else f"{total:,} B"
            cells.append(
                html.Td(
                    className=f"matrix-cell state-{state}",
                    children=[
                        html.Strong(label),
                        html.Span(state),
                    ],
                )
            )
    rows = []
    cell_index = 0
    for node in nodes:
        row_cells = []
        for _ in nics:
            row_cells.append(cells[cell_index])
            cell_index += 1
        rows.append(html.Tr([html.Th(node, scope="row"), *row_cells]))

    return html.Section(
        className="matrix-panel",
        children=[
            html.H2("Node×NIC evidence matrix"),
            html.P(
                "Population matrix only. Color and text describe observed evidence; this is not inferred topology."
            ),
            html.Div(
                className="matrix-scroll",
                children=html.Table(
                    children=[
                        html.Thead(html.Tr([html.Th("Node"), *(html.Th(nic) for nic in nics)])),
                        html.Tbody(rows),
                    ]
                ),
            ),
        ],
    )


def render_overview(view: dict[str, Any]) -> html.Section:
    return html.Section(
        className="overview-workspace",
        children=[
            html.Div(
                className="section-heading",
                children=[
                    html.H2("Job overview"),
                    html.P(
                        "Find the node and physical NIC that carried traffic in the shared interval."
                    ),
                ],
            ),
            render_matrix(view),
            render_context(view),
        ],
    )
