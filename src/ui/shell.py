"""Persistent shell for the traffic investigation workspace."""

from __future__ import annotations

from typing import Any

from dash import dcc, html

from .services import shell_context
from .state import InvestigationState


TABS_ID = "analysis-tabs"
BACK_BUTTON_ID = "analysis-back-button"
RESET_SELECTION_ID = "analysis-reset-selection"
RESET_INTERVAL_ID = "analysis-reset-interval"
STATE_STORE_ID = "analysis-investigation-state"
CHART_MODE_ID = "analysis-chart-mode"
SHELL_FACTS_ID = "analysis-shell-facts"
QUESTION_TAB_VALUE = "questions"


def render_shell(
    view: dict[str, Any],
    state: InvestigationState,
    *,
    export_button_id: str,
    export_download_id: str,
    filter_status_id: str,
) -> html.Header:
    context = shell_context(view)
    return html.Header(
        className="investigation-shell",
        children=[
            html.Div(
                className="shell-title",
                children=[
                    html.H1("Aurora NIC traffic investigation"),
                    html.P(
                        f"Run {context['run_id']} · {context['source_format']} · "
                        f"{context['source_fingerprint']} · {context['provenance']}",
                        className="shell-run",
                    ),
                ],
            ),
            html.Dl(
                id=SHELL_FACTS_ID,
                className="shell-facts",
                children=[
                    html.Div([html.Dt("Identity"), html.Dd(context["identity"])]),
                    html.Div([html.Dt("Collector"), html.Dd(context["collector_policy"])]),
                    html.Div([html.Dt("Direction"), html.Dd(context["direction"])]),
                    html.Div([html.Dt("Interval"), html.Dd(context["interval"])]),
                    html.Div([html.Dt("Coverage"), html.Dd(context["coverage"])]),
                    html.Div([html.Dt("State"), html.Dd(context["state"])]),
                ],
            ),
            html.Div(
                className="shell-actions",
                children=[
                    html.Button(
                        "Back",
                        id=BACK_BUTTON_ID,
                        type="button",
                        className="secondary-button",
                    ),
                    html.Button(
                        "Reset selection",
                        id=RESET_SELECTION_ID,
                        type="button",
                        className="secondary-button",
                    ),
                    html.Button(
                        "Reset interval",
                        id=RESET_INTERVAL_ID,
                        type="button",
                        className="secondary-button",
                    ),
                    html.Button(
                        id=export_button_id,
                        type="button",
                        className="secondary-button",
                        children=[
                            html.Span(
                                "Export",
                                id=f"{export_button_id}-name",
                            ),
                        ],
                        **{
                            "aria-label": "Export the currently filtered view model as JSON",
                            "aria-labelledby": f"{export_button_id}-name",
                        },
                    ),
                ],
            ),
            html.Div(
                id=filter_status_id,
                className="filter-status",
                role="status",
                **{"aria-live": "polite"},
                children=[],
            ),
            dcc.Download(id=export_download_id),
            dcc.Store(id=STATE_STORE_ID, data=state.to_dict(), storage_type="memory"),
        ],
    )


def render_shell_facts(view: dict[str, Any]) -> list[html.Div]:
    context = shell_context(view)
    return [
        html.Div([html.Dt("Identity"), html.Dd(context["identity"])]),
        html.Div([html.Dt("Collector"), html.Dd(context["collector_policy"])]),
        html.Div([html.Dt("Direction"), html.Dd(context["direction"])]),
        html.Div([html.Dt("Interval"), html.Dd(context["interval"])]),
        html.Div([html.Dt("Coverage"), html.Dd(context["coverage"])]),
        html.Div([html.Dt("State"), html.Dd(context["state"])]),
    ]


def render_filter_status(view: dict[str, Any]) -> html.Div:
    return html.Div(
        className="filter-status",
        role="status",
        **{"aria-live": "polite"},
        children=[
            html.Strong("Active filters: "),
            html.Span(
                f"node {view['filters']['node_id'] or 'all'}; "
                f"NIC {view['filters']['nic_id'] or 'all'}; "
                f"direction {view['filters']['direction']}; "
                f"interval {view['capture']['selected_start_ns'] / 1_000_000_000:.3f} s – "
                f"{view['capture']['selected_end_ns'] / 1_000_000_000:.3f} s; "
                f"window {view['capture']['selected_start_ns']}–"
                f"{view['capture']['selected_end_ns']} ns"
            ),
        ],
    )
