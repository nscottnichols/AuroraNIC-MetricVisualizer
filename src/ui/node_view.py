"""Node comparison screen for traffic investigations."""

from __future__ import annotations

from typing import Any

from dash import html

from .services import _format_bytes, _format_percent, _format_rate, node_records


def render_node_view(view: dict[str, Any]) -> html.Section:
    selected_node = view["filters"].get("node_id")
    records = node_records(view)
    cards = []
    for record in records:
        tx = record.get("TX") or {}
        rx = record.get("RX") or {}
        selected = tx or rx or {}
        state = selected.get("status", "Missing")
        cards.append(
            html.Article(
                className="node-nic-card",
                children=[
                    html.H3(f"{record['nic']} (ID {record['nic_id']})"),
                    html.P(
                        f"TX {_format_bytes(tx.get('total_bytes'))} · "
                        f"RX {_format_bytes(rx.get('total_bytes'))}",
                        className="stat-value",
                    ),
                    html.P(_format_rate(selected.get("mean_rate_bytes_per_s"))),
                    html.P(
                        f"Coverage {_format_percent(selected.get('coverage_fraction'))} · "
                        f"{selected.get('collector_policy', 'not observed')}"
                    ),
                    html.P(state, className=f"state-badge state-{selected.get('state', 'missing')}"),
                ],
            )
        )

    return html.Section(
        className="node-workspace",
        children=[
            html.Div(
                className="section-heading",
                children=[
                    html.H2("Node comparison"),
                    html.P(
                        (
                            f"Selected interval {view['capture']['selected_start_ns'] / 1_000_000_000:.3f} s – "
                            f"{view['capture']['selected_end_ns'] / 1_000_000_000:.3f} s. "
                            if selected_node is not None
                            else "Select a node from the ranked overview. "
                        )
                        + "Traffic distribution is observational and does not imply topology, congestion, or causation."
                    ),
                ],
            ),
            html.Div(
                className="node-card-grid",
                children=cards
                or [
                    html.P(
                        "No node selected. Open a ranked row to compare its physical NICs.",
                        className="empty-state",
                    )
                ],
            ),
        ],
    )
