"""Central navigation, state, chart-mode, and interval callbacks."""

from __future__ import annotations


from dash import ALL, Input, Output, State, callback_context, html, no_update

from . import overview
from .diagnostics import payload_by_question, render_diagnostic, render_interval_comparison
from .node_view import render_node_view
from .nic_view import render_nic_detail
from .question_index import (
    INTERVAL_ADD_ID,
    INTERVAL_CLEAR_ID,
    INTERVAL_END_ID,
    INTERVAL_NAME_ID,
    INTERVAL_RESULT_ID,
    INTERVAL_START_ID,
    INTERVAL_STORE_ID,
    QUESTION_RESULT_ID,
    QUESTION_SELECT_ID,
    QUESTION_LIST_ID,
    question_list_children,
)
from .state import InvestigationState, resolve_state


def _state_from_store(data):
    try:
        return InvestigationState.from_dict(data)
    except (TypeError, ValueError, KeyError):
        return None


def register_investigation_callbacks(
    app,
    *,
    store_provider,
    build_view_model,
    filter_state_factory,
    initial_state,
    time_max_provider,
    component_ids,
):
    state_store_id = component_ids["state_store"]
    tabs_id = component_ids["tabs"]
    back_id = component_ids["back"]
    reset_selection_id = component_ids["reset_selection"]
    reset_interval_id = component_ids["reset_interval"]
    chart_mode_id = component_ids["chart_mode"]
    node_filter_id = component_ids["node_filter"]
    nic_filter_id = component_ids["nic_filter"]
    direction_filter_id = component_ids["direction_filter"]
    time_filter_id = component_ids["time_filter"]
    time_origin_id = component_ids["time_origin"]
    nic_detail_id = component_ids["nic_detail"]
    node_view_id = component_ids["node_view"]
    overview_id = component_ids["overview"]
    filter_status_id = component_ids["filter_status"]
    filter_bar_id = component_ids["filter_bar"]
    question_select_id = component_ids["question_select"]
    question_result_id = component_ids["question_result"]
    question_list_id = component_ids["question_list"]
    interval_store_id = component_ids["interval_store"]
    interval_result_id = component_ids["interval_result"]

    def current_store():
        return store_provider()

    def current_time_max():
        return time_max_provider()

    @app.callback(
        Output(question_list_id, "children"),
        Input(node_filter_id, "value"),
        Input(nic_filter_id, "value"),
        Input(direction_filter_id, "value"),
        Input(time_filter_id, "value"),
        State(time_origin_id, "data"),
        prevent_initial_call=True,
    )
    def update_question_index_answerability(
        node_value,
        nic_value,
        direction_value,
        time_value,
        time_origin,
    ):
        node_id = _parse_identity(node_value, "node")
        nic_id = _parse_identity(nic_value, "nic")
        if node_id is not None and not any(
            nic.node_id == node_id and nic.nic_id == nic_id for nic in current_store().nics
        ):
            nic_id = None
        origin = int(time_origin["acquisition_start_ns"])
        window = time_value or [0, current_time_max()]
        filters = filter_state_factory(
            node_id=node_id,
            nic_id=nic_id,
            direction=direction_value or "all",
            start_ns=origin + int(window[0]),
            end_ns=origin + int(window[-1]),
        )
        view = build_view_model(current_store(), filters)
        return question_list_children(view, payload_by_question(view))

    @app.callback(
        Output(question_result_id, "children"),
        Input(question_select_id, "value"),
        Input(node_filter_id, "value"),
        Input(nic_filter_id, "value"),
        Input(direction_filter_id, "value"),
        Input(time_filter_id, "value"),
        State(time_origin_id, "data"),
        prevent_initial_call=True,
    )
    def update_requested_diagnostic(
        question_id,
        node_value,
        nic_value,
        direction_value,
        time_value,
        time_origin,
    ):
        if not question_id:
            return [html.P("No diagnostic requested yet.", className="empty-state")]
        node_id = _parse_identity(node_value, "node")
        nic_id = _parse_identity(nic_value, "nic")
        if node_id is not None and not any(
            nic.node_id == node_id and nic.nic_id == nic_id for nic in current_store().nics
        ):
            nic_id = None
        origin = int(time_origin["acquisition_start_ns"])
        window = time_value or [0, current_time_max()]
        filters = filter_state_factory(
            node_id=node_id,
            nic_id=nic_id,
            direction=direction_value or "all",
            start_ns=origin + int(window[0]),
            end_ns=origin + int(window[-1]),
        )
        return render_diagnostic(build_view_model(current_store(), filters), question_id)

    @app.callback(
        Output(interval_store_id, "data"),
        Output(interval_result_id, "children"),
        Input(INTERVAL_ADD_ID, "n_clicks"),
        Input(INTERVAL_CLEAR_ID, "n_clicks"),
        Input(node_filter_id, "value"),
        Input(nic_filter_id, "value"),
        Input(direction_filter_id, "value"),
        Input(time_filter_id, "value"),
        State(INTERVAL_NAME_ID, "value"),
        State(INTERVAL_START_ID, "value"),
        State(INTERVAL_END_ID, "value"),
        State(interval_store_id, "data"),
        State(time_origin_id, "data"),
        prevent_initial_call=True,
    )
    def update_named_intervals(
        add_clicks,
        clear_clicks,
        node_value,
        nic_value,
        direction_value,
        time_value,
        interval_name,
        interval_start_s,
        interval_end_s,
        current_intervals,
        time_origin,
    ):
        intervals = list(current_intervals or [])
        triggered = callback_context.triggered_id
        if triggered == INTERVAL_CLEAR_ID and clear_clicks:
            intervals = []
        elif triggered == INTERVAL_ADD_ID and add_clicks:
            if not isinstance(interval_name, str) or not interval_name.strip():
                return intervals, [html.P("Enter a nonempty interval name.", className="empty-state")]
            if interval_start_s is None or interval_end_s is None:
                return intervals, [html.P("Enter numeric start and end times in seconds.", className="empty-state")]
            origin = int(time_origin["acquisition_start_ns"])
            try:
                start_ns = origin + int(round(float(interval_start_s) * 1_000_000_000))
                end_ns = origin + int(round(float(interval_end_s) * 1_000_000_000))
            except (TypeError, ValueError, OverflowError):
                return intervals, [html.P("Interval boundaries must be finite numbers.", className="empty-state")]
            if end_ns <= start_ns:
                return intervals, [html.P("Interval end must be later than its start.", className="empty-state")]
            if any(item["name"] == interval_name.strip() for item in intervals):
                return intervals, [html.P("Interval names must be unique.", className="empty-state")]
            if len(intervals) >= 8:
                return intervals, [html.P("At most 8 named intervals are supported.", className="empty-state")]
            acquisition_start_ns = int(time_origin["acquisition_start_ns"])
            if start_ns < acquisition_start_ns or end_ns > acquisition_start_ns + current_time_max():
                return intervals, [html.P("Interval boundaries must remain within the acquisition window.", className="empty-state")]
            intervals.append({"name": interval_name.strip(), "start_ns": start_ns, "end_ns": end_ns})

        node_id = _parse_identity(node_value, "node")
        nic_id = _parse_identity(nic_value, "nic")
        if node_id is not None and not any(
            nic.node_id == node_id and nic.nic_id == nic_id for nic in current_store().nics
        ):
            nic_id = None
        if not intervals:
            return intervals, [html.P("No named intervals selected.", className="empty-state")]
        views = []
        for interval in intervals:
            views.append(
                build_view_model(
                    current_store(),
                    filter_state_factory(
                        node_id=node_id,
                        nic_id=nic_id,
                        direction=direction_value or "all",
                        start_ns=interval["start_ns"],
                        end_ns=interval["end_ns"],
                    ),
                )
            )
        return intervals, render_interval_comparison(intervals, views)

    @app.callback(
        Output(overview_id, "style"),
        Input(filter_status_id, "children"),
        prevent_initial_call=False,
    )
    def reveal_overview_after_initial_analysis(filter_status_children):
        if not filter_status_children:
            return {"display": "none"}
        return {
            "display": "block",
            "visibility": "visible",
            "transition": "visibility 0s linear 500ms",
        }

    @app.callback(
        Output(filter_bar_id, "style"),
        Input(overview_id, "style"),
        prevent_initial_call=False,
    )
    def reveal_filter_bar_after_overview(overview_style):
        if not isinstance(overview_style, dict) or overview_style.get("display") != "block":
            return {"display": "none"}
        return {"display": "grid"}

    @app.callback(
        Output(state_store_id, "data"),
        Input(node_filter_id, "value"),
        Input(nic_filter_id, "value"),
        Input(direction_filter_id, "value"),
        Input(time_filter_id, "value"),
        State(time_origin_id, "data"),
        State(state_store_id, "data"),
        prevent_initial_call=True,
    )
    def sync_state(node_value, nic_value, direction_value, time_value, time_origin, current_data):
        current = _state_from_store(current_data) or initial_state
        node_id = _parse_identity(node_value, "node")
        nic_id = _parse_identity(nic_value, "nic")
        if node_id is not None and not any(
            nic.node_id == node_id and nic.nic_id == nic_id for nic in current_store().nics
        ):
            nic_id = None
        window = time_value or [0, current_time_max()]
        origin = int(time_origin["acquisition_start_ns"])
        return resolve_state(
            current,
            node_id=node_id,
            nic_id=nic_id,
            direction=direction_value or "all",
            start_ns=origin + int(window[0]),
            end_ns=origin + int(window[-1]),
        ).to_dict()

    @app.callback(
        Output(tabs_id, "value"),
        Output(state_store_id, "data", allow_duplicate=True),
        Input({"section": ALL, "node": ALL, "nic": ALL}, "n_clicks"),
        State(state_store_id, "data"),
        prevent_initial_call=True,
    )
    def open_nic(clicks, current_data):
        identity = callback_context.triggered_id
        if not clicks or not any(clicks):
            return no_update, no_update
        if not isinstance(identity, dict) or identity.get("section") not in {
            overview.NODE_OPEN_SECTION,
            "node-nic-open",
        }:
            return no_update, no_update
        current = _state_from_store(current_data) or initial_state
        next_screen = "node" if current.screen == "overview" else "nic"
        next_state = resolve_state(
            current,
            node_id=int(identity["node"]),
            nic_id=int(identity["nic"]),
            direction=current.direction,
            start_ns=current.interval.start_ns,
            end_ns=current.interval.end_ns,
            screen=next_screen,
            collector_policy=current.selection.collector_policy,
        )
        return next_screen, next_state.to_dict()

    @app.callback(
        Output(tabs_id, "value"),
        Output(state_store_id, "data", allow_duplicate=True),
        Input(back_id, "n_clicks"),
        State(state_store_id, "data"),
        prevent_initial_call=True,
    )
    def go_back(_, current_data):
        if not _:
            return no_update, no_update
        current = _state_from_store(current_data) or initial_state
        if len(current.history) < 2:
            return "overview", no_update
        previous = current.history[-2]
        next_state = resolve_state(
            current,
            node_id=previous.get("node_id"),
            nic_id=previous.get("nic_id"),
            direction=current.direction,
            start_ns=current.interval.start_ns,
            end_ns=current.interval.end_ns,
            screen=previous.get("screen", "overview"),
        )
        next_state.history = current.history[:-1]
        return previous.get("screen", "overview"), next_state.to_dict()

    @app.callback(
        Output(node_filter_id, "value", allow_duplicate=True),
        Output(nic_filter_id, "value", allow_duplicate=True),
        Output(direction_filter_id, "value", allow_duplicate=True),
        Output(tabs_id, "value"),
        Output(state_store_id, "data", allow_duplicate=True),
        Input(reset_selection_id, "n_clicks"),
        State(state_store_id, "data"),
        prevent_initial_call=True,
    )
    def reset_selection(_, current_data):
        if not _:
            return no_update, no_update, no_update, no_update, no_update
        current = _state_from_store(current_data) or initial_state
        next_state = resolve_state(
            current,
            node_id=None,
            nic_id=None,
            direction="all",
            start_ns=current.interval.start_ns,
            end_ns=current.interval.end_ns,
            screen="overview",
        )
        return "all", "all", "all", "overview", next_state.to_dict()

    @app.callback(
        Output(time_filter_id, "value"),
        Output(state_store_id, "data", allow_duplicate=True),
        Input(reset_interval_id, "n_clicks"),
        State(state_store_id, "data"),
        prevent_initial_call=True,
    )
    def reset_interval(_, current_data):
        if not _:
            return no_update, no_update
        current = _state_from_store(current_data) or initial_state
        next_state = resolve_state(
            current,
            node_id=current.selection.node_id,
            nic_id=current.selection.nic_id,
            direction=current.direction,
            start_ns=initial_state.interval.start_ns,
            end_ns=initial_state.interval.end_ns,
            screen=current.screen,
        )
        return [0, current_time_max()], next_state.to_dict()

    @app.callback(
        Output(nic_detail_id, "children", allow_duplicate=True),
        Input(chart_mode_id, "value"),
        State(node_filter_id, "value"),
        State(nic_filter_id, "value"),
        State(direction_filter_id, "value"),
        State(time_filter_id, "value"),
        State(time_origin_id, "data"),
        prevent_initial_call=True,
    )
    def update_chart_mode(chart_mode, node_value, nic_value, direction_value, time_value, time_origin):
        node_id = _parse_identity(node_value, "node")
        nic_id = _parse_identity(nic_value, "nic")
        if node_id is not None and not any(
            nic.node_id == node_id and nic.nic_id == nic_id for nic in current_store().nics
        ):
            nic_id = None
        window = time_value or [0, current_time_max()]
        origin = int(time_origin["acquisition_start_ns"])
        filters = filter_state_factory(
            node_id=node_id,
            nic_id=nic_id,
            direction=direction_value or "all",
            start_ns=origin + int(window[0]),
            end_ns=origin + int(window[-1]),
        )
        return render_nic_detail(
            build_view_model(current_store(), filters),
            chart_mode or "rate",
            chart_mode_id=chart_mode_id,
        )

    @app.callback(
        Output(node_view_id, "children"),
        Input(node_filter_id, "value"),
        Input(nic_filter_id, "value"),
        Input(direction_filter_id, "value"),
        Input(time_filter_id, "value"),
        State(time_origin_id, "data"),
        prevent_initial_call=True,
    )
    def update_node_view(node_value, nic_value, direction_value, time_value, time_origin):
        node_id = _parse_identity(node_value, "node")
        nic_id = _parse_identity(nic_value, "nic")
        if node_id is not None and not any(
            nic.node_id == node_id and nic.nic_id == nic_id for nic in current_store().nics
        ):
            nic_id = None
        window = time_value or [0, current_time_max()]
        origin = int(time_origin["acquisition_start_ns"])
        filters = filter_state_factory(
            node_id=node_id,
            nic_id=nic_id,
            direction=direction_value or "all",
            start_ns=origin + int(window[0]),
            end_ns=origin + int(window[-1]),
        )
        return render_node_view(build_view_model(current_store(), filters))


def _parse_identity(value, prefix):
    if value in (None, "", "all"):
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.startswith(f"{prefix}:"):
        return int(value.split(":", 1)[1])
    raise ValueError(f"invalid {prefix} value {value!r}")
