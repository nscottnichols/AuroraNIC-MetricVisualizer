"""Legacy question index and capability-to-answerability navigation."""

from __future__ import annotations

from typing import Any

from dash import dcc, html


QUESTION_SELECT_ID = "analysis-question-select"
QUESTION_RESULT_ID = "analysis-question-result"
QUESTION_LIST_ID = "analysis-question-list"
QUESTION_TAB_VALUE = "questions"
INTERVAL_STORE_ID = "analysis-question-interval-store"
INTERVAL_NAME_ID = "analysis-question-interval-name"
INTERVAL_START_ID = "analysis-question-interval-start"
INTERVAL_END_ID = "analysis-question-interval-end"
INTERVAL_ADD_ID = "analysis-question-interval-add"
INTERVAL_CLEAR_ID = "analysis-question-interval-clear"
INTERVAL_RESULT_ID = "analysis-question-interval-result"


QUESTION_ROUTES: dict[str, dict[str, str]] = {
    "NIC-01": {"title": "NIC traffic", "scope": "NIC"},
    "NIC-02": {"title": "Flow control", "scope": "NIC"},
    "NIC-03": {"title": "Receive matching", "scope": "NIC"},
    "NIC-04": {"title": "Host pressure", "scope": "NIC"},
    "NIC-05": {"title": "Operations", "scope": "NIC"},
    "NIC-06": {"title": "Frame sizes", "scope": "NIC"},
    "NIC-07": {"title": "Address translation", "scope": "NIC"},
    "NIC-08": {"title": "Latency", "scope": "NIC"},
    "NIC-09": {"title": "Reliability", "scope": "NIC"},
    "NODE-01": {"title": "NIC load balance", "scope": "node"},
    "NODE-02": {"title": "Cross-NIC consistency", "scope": "node"},
    "JOB-01": {"title": "Node hotspots", "scope": "job"},
    "JOB-02": {"title": "Node balance", "scope": "job"},
    "JOB-03": {"title": "Time phases", "scope": "job"},
    "JOB-04": {"title": "Communication profile", "scope": "job"},
}


def _answerability(
    question_id: str,
    view: dict[str, Any],
    payload: dict[str, Any] | None,
) -> tuple[str, str]:
    filters = view["filters"]
    if question_id.startswith("NIC-") and filters.get("nic_id") is None:
        return (
            "Not answerable now",
            "Select one physical NIC. The current view may show an automatically selected first NIC, but it is not an explicit answer population.",
        )
    if payload is None:
        return ("Not answerable now", "No bounded result is available for this route.")

    state = str(payload.get("state", "missing"))
    if state in {"ready", "ok", "zero", "observed"}:
        return ("Answerable now", "The selected population has usable evidence.")
    if state in {"partial", "ambiguous", "zero_with_gaps"}:
        return (
            "Partially answerable",
            "Review the result before using it; coverage or compatibility is incomplete.",
        )
    reason = "; ".join(payload.get("status_reasons", ())) or str(
        payload.get("status", "No result")
    )
    return ("Not answerable now", reason)


def question_rows(view: dict[str, Any], payload_by_question: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for capability in view.get("question_coverage", ()):
        question_id = str(capability.get("id"))
        route = QUESTION_ROUTES.get(question_id)
        if route is None:
            continue
        payload = payload_by_question.get(question_id)
        answerability, answer_reason = _answerability(question_id, view, payload)
        rows.append(
            {
                "id": question_id,
                "question": capability.get("question", ""),
                "dataset_capability": str(
                    capability.get("state", capability.get("status", "unknown"))
                ),
                "current_selection": answerability,
                "reason": answer_reason,
                "scope": route["scope"],
                "title": route["title"],
            }
        )
    return rows


def _status_class(value: str) -> str:
    normalized = value.lower().replace(" ", "-")
    return f"state-badge state-{normalized}"


def question_list_children(
    view: dict[str, Any], payload_by_question: dict[str, Any]
) -> list[html.Li]:
    return [
        html.Li(
            children=[
                html.Strong(f"{row['id']} · {row['title']}"),
                html.P(row["question"]),
                html.Span(
                    f"Dataset: {row['dataset_capability']}",
                    className=_status_class(row["dataset_capability"]),
                ),
                html.Span(
                    f"Current selection: {row['current_selection']}",
                    className=_status_class(row["current_selection"]),
                ),
            ]
        )
        for row in question_rows(view, payload_by_question)
    ]


def render_question_index(
    view: dict[str, Any], payload_by_question: dict[str, Any] | None = None
) -> html.Section:
    payloads = payload_by_question or {}
    rows = question_rows(view, payloads)
    options = [
        {"label": f"{row['id']} — {row['title']}", "value": row["id"]}
        for row in rows
    ]
    return html.Section(
        className="question-index",
        **{"aria-labelledby": "question-index-heading"},
        children=[
            html.Div(
                className="section-heading",
                children=[
                    html.H2("Question index", id="question-index-heading"),
                    html.P(
                        "Dataset capability and current-selection answerability are separate. "
                        "Open a question to render only that diagnostic route."
                    ),
                ],
            ),
            dcc.Dropdown(
                id=QUESTION_SELECT_ID,
                options=options,
                value=None,
                placeholder="Choose a catalog question",
                clearable=False,
                searchable=True,
            ),
            html.Ol(
                id=QUESTION_LIST_ID,
                className="question-list",
                children=question_list_children(view, payloads),
            ),
        ],
    )


def render_interval_controls() -> html.Fieldset:
    return html.Fieldset(
        className="named-interval-controls",
        children=[
            html.Legend("Named interval comparison"),
            html.P(
                "Enter relative seconds. The server converts each boundary to exact "
                "nanoseconds on the canonical trace clock; names and boundaries are not inferred."
            ),
            html.Div(
                className="filter-grid",
                children=[
                    html.Label(
                        [html.Span("Interval name", className="control-label"), dcc.Input(
                            id=INTERVAL_NAME_ID, type="text", value="", debounce=True
                        )],
                        className="control",
                    ),
                    html.Label(
                        [html.Span("Start (s)", className="control-label"), dcc.Input(
                            id=INTERVAL_START_ID,
                            type="number",
                            value=None,
                            min=0,
                            step=0.001,
                            debounce=True,
                        )],
                        className="control",
                    ),
                    html.Label(
                        [html.Span("End (s)", className="control-label"), dcc.Input(
                            id=INTERVAL_END_ID,
                            type="number",
                            value=None,
                            min=0,
                            step=0.001,
                            debounce=True,
                        )],
                        className="control",
                    ),
                    html.Button("Add interval", id=INTERVAL_ADD_ID, type="button"),
                    html.Button("Clear intervals", id=INTERVAL_CLEAR_ID, type="button"),
                ],
            ),
        ],
    )
