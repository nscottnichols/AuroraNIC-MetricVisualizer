"""Reusable Dash controls for server-side dataset lifecycle management."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from dash import Dash, Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate

from dataset_lifecycle import (
    DatasetProgress,
    DatasetRequestToken,
    DatasetState,
    DatasetStatusSnapshot,
)


def _serialize_token(token: DatasetRequestToken) -> dict[str, Any]:
    return {
        "request_id": token.request_id,
        "dataset_id": token.dataset_id,
        "generation": token.generation,
    }


def _deserialize_token(value: Any) -> DatasetRequestToken | None:
    if not isinstance(value, dict):
        return None
    try:
        return DatasetRequestToken(
            request_id=str(value["request_id"]),
            dataset_id=str(value["dataset_id"]),
            generation=int(value["generation"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


class DatasetControls:
    """A generic approved-dataset control panel and callback layer.

    The component only accepts dataset IDs that were supplied by the caller.
    It never discovers files or constructs paths itself; all path validation
    remains the responsibility of :class:`DatasetLifecycleManager`.
    """

    def __init__(
        self,
        dataset_ids: Iterable[str],
        *,
        poll_interval_ms: int = 500,
        id_prefix: str = "dataset",
    ) -> None:
        unique_ids: list[str] = []
        seen: set[str] = set()
        for dataset_id in dataset_ids:
            if not isinstance(dataset_id, str) or not dataset_id.strip():
                raise ValueError("dataset IDs must be nonempty strings")
            if dataset_id not in seen:
                seen.add(dataset_id)
                unique_ids.append(dataset_id)
        if not unique_ids:
            raise ValueError("at least one approved dataset ID is required")
        if not isinstance(poll_interval_ms, int) or poll_interval_ms < 100:
            raise ValueError("poll_interval_ms must be at least 100")
        if not isinstance(id_prefix, str) or not id_prefix.strip():
            raise ValueError("id_prefix must be a nonempty string")

        self._dataset_ids = unique_ids
        self._poll_interval_ms = poll_interval_ms
        self._id_prefix = id_prefix.strip()

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        return tuple(self._dataset_ids)

    @property
    def poll_interval_ms(self) -> int:
        return self._poll_interval_ms

    @property
    def heading_id(self) -> str:
        return f"{self._id_prefix}-heading"

    @property
    def dataset_select_id(self) -> str:
        return f"{self._id_prefix}-select"

    @property
    def request_button_id(self) -> str:
        return f"{self._id_prefix}-request-button"

    @property
    def cancel_button_id(self) -> str:
        return f"{self._id_prefix}-cancel-button"

    @property
    def status_id(self) -> str:
        return f"{self._id_prefix}-status"

    @property
    def progress_id(self) -> str:
        return f"{self._id_prefix}-progress"

    @property
    def error_id(self) -> str:
        return f"{self._id_prefix}-error"

    @property
    def request_store_id(self) -> str:
        return f"{self._id_prefix}-request-store"

    @property
    def ready_store_id(self) -> str:
        return f"{self._id_prefix}-ready-store"

    @property
    def current_dataset_store_id(self) -> str:
        return f"{self._id_prefix}-current-store"

    @property
    def cancel_requested_store_id(self) -> str:
        return f"{self._id_prefix}-cancel-requested-store"

    @property
    def poll_interval_id(self) -> str:
        return f"{self._id_prefix}-poll-interval"

    def render(self) -> html.Section:
        return html.Section(
            id=f"{self._id_prefix}-controls",
            className="dataset-controls-panel",
            role="region",
            **{"aria-labelledby": self.heading_id},
            children=[
                html.H2("Dataset", id=self.heading_id),
                html.Div(
                    className="dataset-control-row",
                    children=[
                        html.Label(
                            "Approved dataset",
                            htmlFor=self.dataset_select_id,
                            className="control-label",
                        ),
                        dcc.Dropdown(
                            id=self.dataset_select_id,
                            options=[
                                {"label": dataset_id, "value": dataset_id}
                                for dataset_id in self._dataset_ids
                            ],
                            value=self._dataset_ids[0],
                            clearable=False,
                            searchable=True,
                        ),
                        html.Button(
                            "Open / switch dataset",
                            id=self.request_button_id,
                            type="button",
                            className="primary-button",
                        ),
                        html.Button(
                            "Cancel",
                            id=self.cancel_button_id,
                            type="button",
                            className="secondary-button",
                            disabled=True,
                        ),
                    ],
                ),
                html.Div(id=self.status_id, role="status", className="dataset-status"),
                html.Progress(
                    id=self.progress_id,
                    max=100,
                    value=0,
                    className="dataset-progress",
                ),
                html.Div(id=self.error_id, role="alert", className="dataset-error"),
                dcc.Store(id=self.request_store_id, storage_type="memory"),
                dcc.Store(id=self.ready_store_id, data=False, storage_type="memory"),
                dcc.Store(
                    id=self.current_dataset_store_id,
                    storage_type="memory",
                ),
                dcc.Store(
                    id=self.cancel_requested_store_id,
                    data=False,
                    storage_type="memory",
                ),
                dcc.Interval(
                    id=self.poll_interval_id,
                    interval=self._poll_interval_ms,
                    n_intervals=0,
                ),
            ],
        )

    def register_callbacks(self, app: Dash, manager: Any) -> None:
        @app.callback(
            Output(self.request_store_id, "data"),
            Output(self.status_id, "children"),
            Output(self.error_id, "children"),
            Output(self.progress_id, "value"),
            Input(self.request_button_id, "n_clicks"),
            State(self.dataset_select_id, "value"),
            prevent_initial_call=True,
        )
        def request_dataset(clicks: int, dataset_id: str | None):
            if not clicks or not dataset_id:
                raise PreventUpdate
            try:
                if manager.current() is None:
                    token = manager.open(dataset_id)
                else:
                    token = manager.switch(dataset_id)
            except Exception as exc:
                return (
                    no_update,
                    "Dataset request failed",
                    str(exc),
                    0,
                )
            return (
                _serialize_token(token),
                f"Requested dataset {dataset_id}",
                "",
                0,
            )

        @app.callback(
            Output(self.status_id, "children"),
            Output(self.progress_id, "value"),
            Output(self.error_id, "children"),
            Output(self.cancel_button_id, "disabled"),
            Output(self.ready_store_id, "data"),
            Output(self.current_dataset_store_id, "data"),
            Input(self.poll_interval_id, "n_intervals"),
            State(self.request_store_id, "data"),
            prevent_initial_call=False,
        )
        def poll_dataset_status(
            intervals: int,
            token_data: Any,
        ):
            token = _deserialize_token(token_data)
            try:
                snapshot = manager.status(token)
            except Exception as exc:
                return (
                    "Dataset status unavailable",
                    0,
                    str(exc),
                    True,
                    False,
                    None,
                )

            state = snapshot.state
            progress = snapshot.progress
            fraction = progress.fraction if progress.fraction is not None else 0.0
            progress_value = max(0, min(100, int(round(fraction * 100))))
            state_label = state.value.replace("_", " ").title()
            stage_label = progress.stage.replace("_", " ").title()
            status_text = f"{state_label}: {stage_label}"
            if progress.message:
                status_text += f" — {progress.message}"
            if snapshot.current_dataset_id:
                status_text += f" — {snapshot.current_dataset_id}"

            return (
                status_text,
                progress_value,
                snapshot.error or "",
                state
                not in {
                    DatasetState.QUEUED,
                    DatasetState.LOADING,
                    DatasetState.CANCELLING,
                },
                state is DatasetState.READY,
                snapshot.current_dataset_id,
            )

        @app.callback(
            Output(self.cancel_requested_store_id, "data"),
            Input(self.cancel_button_id, "n_clicks"),
            State(self.request_store_id, "data"),
            prevent_initial_call=True,
        )
        def cancel_dataset(clicks: int, token_data: Any):
            if not clicks:
                raise PreventUpdate
            token = _deserialize_token(token_data)
            if token is None:
                return False
            try:
                manager.cancel(token)
            except Exception:
                return False
            return True
