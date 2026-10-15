"""Focused, bounded diagnostic views for one catalog question at a time."""

from __future__ import annotations

from typing import Any

from dash import dcc, html
import plotly.graph_objects as go

from .question_index import QUESTION_ROUTES


def diagnostic_payload(view: dict[str, Any], question_id: str) -> dict[str, Any] | None:
    detail = view.get("nic_detail", {})
    sources = {
        "NIC-02": detail.get("flow_control"),
        "NIC-03": detail.get("matching"),
        "NIC-04": detail.get("host_pressure"),
        "NIC-05": detail.get("operations"),
        "NIC-06": detail.get("frames"),
        "NIC-07": detail.get("translation"),
        "NIC-08": detail.get("latency"),
        "NIC-09": detail.get("reliability"),
        "NODE-01": view.get("node_balance"),
        "NODE-02": view.get("node_comparison"),
        "JOB-01": view.get("job_balance"),
        "JOB-02": view.get("job_balance"),
        "JOB-03": view.get("phase_analysis"),
        "JOB-04": view.get("job_profile"),
    }
    if question_id == "NIC-01":
        series = detail.get("series", ())
        if not series:
            return {
                "state": "missing",
                "status": "No selected NIC traffic evidence",
                "status_reasons": ("Select one physical NIC and compatible direction.",),
            }
        states = {str(record.get("state", "missing")) for record in series}
        if states <= {"ready", "zero", "ok", "observed"}:
            state = "ready"
        elif len(states) == 1:
            state = next(iter(states))
        else:
            state = "partial"
        return {
            "state": state,
            "status": "Selected NIC TX/RX traffic",
            "status_reasons": tuple(
                dict.fromkeys(
                    reason
                    for record in series
                    for reason in record.get("status_reasons", ())
                )
            ),
            "series": series,
        }
    return sources.get(question_id)


def payload_by_question(view: dict[str, Any]) -> dict[str, Any]:
    return {
        question_id: diagnostic_payload(view, question_id)
        for question_id in QUESTION_ROUTES
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "Missing"
    if isinstance(value, float):
        return f"{value:,.3f}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(item) for item in value) or "None"
    return str(value)


def _format_bytes(value: Any) -> str:
    if value is None:
        return "Missing"
    if isinstance(value, (int, float)) and value == 0:
        return "0 bytes"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if abs(amount) < 1024 or unit == units[-1]:
            precision = 0 if unit == "B" else 2
            human = f"{amount:,.{precision}f} {unit}"
            break
        amount /= 1024
    return f"{human} ({float(value):.3e} bytes)"


def _format_rate(value: Any) -> str:
    if value is None:
        return "Missing"
    if isinstance(value, (int, float)) and value == 0:
        return "0 bytes/s"
    units = ("B/s", "KiB/s", "MiB/s", "GiB/s", "TiB/s")
    amount = float(value)
    for unit in units:
        if abs(amount) < 1024 or unit == units[-1]:
            precision = 0 if unit == "B/s" else 2
            human = f"{amount:,.{precision}f} {unit}"
            break
        amount /= 1024
    return f"{human} ({float(value):.3e} bytes/s)"


def _traffic_summary(payload: dict[str, Any]) -> list[tuple[str, str]]:
    rows = []
    for record in payload.get("series", ()):
        rows.extend(
            [
                (
                    f"{record.get('direction')} total",
                    _format_bytes(record.get("total_bytes")),
                ),
                (
                    f"{record.get('direction')} mean rate",
                    _format_rate(record.get("mean_rate_bytes_per_s")),
                ),
            ]
        )
    return rows[:12]


def _summary_items(question_id: str, payload: dict[str, Any]) -> list[tuple[str, str]]:
    if question_id == "NIC-01":
        return _traffic_summary(payload)
    items = [
        ("State", str(payload.get("status", payload.get("state", "Unknown")))),
        ("Collector count", _format_value(payload.get("collector_count"))),
        ("Coverage scope", _format_value(payload.get("coverage_scope"))),
    ]
    for key in ("hit_ratio_formula", "priority_fraction_formula", "traffic_share_formula"):
        if key in payload:
            items.append(("Formula", _format_value(payload[key])))
    if question_id in {"NODE-01", "JOB-01", "JOB-02"}:
        for row in payload.get("rows", payload.get("directions", ()))[:8]:
            items.append(
                (
                    f"{row.get('node', row.get('direction'))} total",
                    f"{_format_value(row.get('total_bytes'))} bytes",
                )
            )
    if question_id == "JOB-03":
        for phase in payload.get("phases", ())[:8]:
            items.append((str(phase.get("name")), f"{_format_value(phase.get('total_bytes'))} bytes"))
    if question_id == "JOB-04":
        items.extend(
            [
                ("Supported dimensions", _format_value(payload.get("supported_dimensions"))),
                ("Partial dimensions", _format_value(payload.get("partial_dimensions"))),
                ("Unsupported dimensions", _format_value(payload.get("unsupported_dimensions"))),
            ]
        )
    return items[:16]


def _diagnostic_figure(question_id: str, payload: dict[str, Any]) -> dcc.Graph | None:
    figure = go.Figure()
    if question_id == "NIC-03":
        records = payload.get("collector_priority_fractions", ())
        if records:
            labels = [str(record.get("collector_id")) for record in records]
            figure.add_bar(x=labels, y=[record.get("priority_fraction") for record in records], name="Priority")
            figure.add_bar(x=labels, y=[record.get("overflow_fraction") for record in records], name="Overflow")
            figure.update_layout(
                title="Receive matching by collector",
                yaxis_title="Fraction of compatible counts",
                barmode="group",
                template="plotly_white",
            )
            return dcc.Graph(id="analysis-diagnostic-graph", figure=figure)
    if question_id in {"NIC-06", "NIC-08"}:
        bins: list[str] = []
        values: list[Any] = []
        for collector in payload.get("collector_analyses", ()):
            if question_id == "NIC-06":
                for direction in collector.get("direction_results", ()):
                    bins.extend(
                        f"{direction.get('direction')} {record.get('bin_id')}"
                        for record in direction.get("bin_results", ())
                    )
                    values.extend(record.get("total") for record in direction.get("bin_results", ()))
            else:
                bins.extend(str(record.get("bin")) for record in collector.get("bin_results", ()))
                values.extend(record.get("count") for record in collector.get("bin_results", ()))
        if bins:
            figure.add_bar(x=bins, y=values)
            figure.update_layout(
                title="Histogram bins by collector",
                yaxis_title="Count (source units)",
                template="plotly_white",
            )
            return dcc.Graph(id="analysis-diagnostic-graph", figure=figure)
    return None


def render_diagnostic(view: dict[str, Any], question_id: str) -> html.Section:
    route = QUESTION_ROUTES.get(question_id)
    if route is None:
        return html.Section(
            className="diagnostic-view",
            children=[
                html.P(
                    "Unknown question. Choose a question from the index.",
                    className="empty-state",
                )
            ],
        )
    payload = diagnostic_payload(view, question_id)
    capture = view["capture"]
    detail = view.get("nic_detail", {})
    state = str(payload.get("state", "missing")) if payload else "missing"
    status = str(payload.get("status", "Missing result")) if payload else "No bounded result"
    collector_policy = detail.get("collector_policy", "not applicable")
    summary = _summary_items(question_id, payload or {})
    figure = _diagnostic_figure(question_id, payload or {})
    evidence: list[Any] = []
    if payload:
        evidence.extend(payload.get("status_reasons", ()))
        evidence.extend(payload.get("reasons", ()))
        evidence.extend(payload.get("limitations", ()))
        evidence.extend(payload.get("missing_counter_ids", ()))

    return html.Section(
        className="diagnostic-view",
        **{"aria-labelledby": f"diagnostic-{question_id}-heading"},
        children=[
            html.Div(
                className="section-heading",
                children=[
                    html.H2(
                        f"{question_id}: {route['title']}",
                        id=f"diagnostic-{question_id}-heading",
                    ),
                    html.Span(status, className=f"state-badge state-{state}"),
                ],
            ),
            html.Dl(
                className="diagnostic-provenance",
                children=[
                    html.Div([html.Dt("Run"), html.Dd(view["run"]["run_id"])]),
                    html.Div([html.Dt("Clock"), html.Dd(view["run"]["clock_domain"])]),
                    html.Div(
                        [
                            html.Dt("Identity"),
                            html.Dd(
                                f"node {view['filters']['node_id'] or 'all'} / "
                                f"NIC {view['filters']['nic_id'] or 'all'}"
                            ),
                        ]
                    ),
                    html.Div([html.Dt("Collector policy"), html.Dd(str(collector_policy))]),
                    html.Div(
                        [
                            html.Dt("Exact window"),
                            html.Dd(f"{capture['selected_start_ns']}–{capture['selected_end_ns']} ns"),
                        ]
                    ),
                ],
            ),
            html.P(
                "This route is rendered only when requested and sends bounded summaries, not raw samples.",
                className="state-detail",
            ),
            html.Div(
                className="compact-grid",
                children=[
                    html.Div(
                        className="compact-card",
                        children=[
                            html.P(label, className="stat-label"),
                            html.P(value, className="stat-value"),
                        ],
                    )
                    for label, value in summary
                ]
                or [html.P("No diagnostic summary is available.", className="empty-state")],
            ),
            figure,
            html.Details(
                className="evidence-drawer",
                children=[
                    html.Summary("Evidence, reasons, and limitations"),
                    html.Ul(
                        children=[html.Li(str(item)) for item in evidence[:24]]
                        or [html.Li("None recorded.")]
                    ),
                ],
            ),
        ],
    )


def render_interval_comparison(
    intervals: list[dict[str, Any]], views: list[dict[str, Any]]
) -> html.Section:
    cards = []
    for interval, view in zip(intervals, views, strict=True):
        for record in view.get("nic_detail", {}).get("series", ()):
            cards.append(
                html.Div(
                    className="compact-card",
                    children=[
                        html.P(
                            f"{interval['name']} {record.get('direction')} total",
                            className="stat-label",
                        ),
                        html.P(
                            f"{_format_value(record.get('total_bytes'))} bytes",
                            className="stat-value",
                        ),
                        html.P(
                            f"{_format_value(record.get('mean_rate_bytes_per_s'))} bytes/s; "
                            f"coverage {_format_value(record.get('coverage_fraction'))}; "
                            f"exact window {interval['start_ns']}–{interval['end_ns']} ns",
                            className="state-detail",
                        ),
                    ],
                )
            )
    changes = []
    for previous, current in zip(views, views[1:]):
        for previous_record in previous.get("nic_detail", {}).get("series", ()):
            current_record = next(
                (
                    record
                    for record in current.get("nic_detail", {}).get("series", ())
                    if record.get("direction") == previous_record.get("direction")
                ),
                None,
            )
            if current_record is None:
                continue
            previous_total = previous_record.get("total_bytes")
            current_total = current_record.get("total_bytes")
            change = (
                current_total - previous_total
                if previous_total is not None and current_total is not None
                else None
            )
            changes.append(
                html.Li(
                    f"{previous_record.get('direction')}: observed total change "
                    f"{_format_value(change)} bytes. This is a direct comparison of "
                    "two independently named windows, not contiguous phase segmentation."
                )
            )
    return html.Section(
        className="named-interval-comparison",
        children=[
            html.H3("Named interval comparison"),
            html.Span(
                f"{len(intervals)} interval(s)",
                className="state-badge state-ready" if intervals else "state-badge state-missing",
            ),
            html.Div(className="compact-grid", children=cards),
            html.Details(
                className="evidence-drawer",
                children=[
                    html.Summary("Comparisons, formulas, and limitations"),
                    html.Ul(children=changes or [html.Li("Add at least two intervals to compare.")]),
                    html.P("Total formula: sum(valid interval byte deltas)"),
                    html.P("Mean-rate formula: total bytes / covered duration seconds"),
                    html.P(
                        "Named windows may be arbitrary and non-overlapping. Differences are "
                        "observational and do not infer workload phases or causes."
                    ),
                ],
            ),
        ],
    )
