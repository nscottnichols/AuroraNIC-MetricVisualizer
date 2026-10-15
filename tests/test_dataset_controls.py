"""Focused tests for the reusable dataset-control layer."""

from __future__ import annotations

from typing import Any

import pytest
from dash import Dash, html

from dataset_lifecycle import (
    DatasetProgress,
    DatasetRequestToken,
    DatasetState,
    DatasetStatusSnapshot,
)
from ui.dataset_controls import DatasetControls, _serialize_token


class FakeDatasetManager:
    def __init__(
        self,
        *,
        has_current: bool = False,
        state: DatasetState = DatasetState.LOADING,
        progress: DatasetProgress | None = None,
        error: str | None = None,
        current_dataset_id: str | None = None,
    ) -> None:
        self.has_current = has_current
        self.state = state
        self.progress = progress or DatasetProgress(stage="idle")
        self.error = error
        self.current_dataset_id = current_dataset_id
        self.open_calls: list[str] = []
        self.switch_calls: list[str] = []
        self.cancel_calls: list[DatasetRequestToken] = []
        self.token = DatasetRequestToken(
            request_id="request-1",
            dataset_id="alpha",
            generation=1,
        )

    def current(self) -> Any:
        return object() if self.has_current else None

    def open(self, dataset_id: str) -> DatasetRequestToken:
        self.open_calls.append(dataset_id)
        return self.token

    def switch(self, dataset_id: str) -> DatasetRequestToken:
        self.switch_calls.append(dataset_id)
        return self.token

    def cancel(self, token: DatasetRequestToken) -> None:
        self.cancel_calls.append(token)

    def status(self, token: DatasetRequestToken | None = None) -> DatasetStatusSnapshot:
        return DatasetStatusSnapshot(
            token=token,
            state=self.state,
            progress=self.progress,
            error=self.error,
            current_dataset_id=self.current_dataset_id,
            last_good_dataset_id=self.current_dataset_id,
        )


def _raw_callback(app: Dash, id_part: str):
    for entry in app._callback_list:
        input_ids = {item["id"] for item in entry["inputs"]}
        state_ids = {item["id"] for item in entry.get("state", [])}
        if id_part in input_ids or id_part in state_ids:
            return app.callback_map[entry["output"]]["callback"].__wrapped__
    raise AssertionError(f"callback not found for {id_part}")


def test_component_renders_only_approved_dataset_ids():
    controls = DatasetControls(["alpha", "beta", "alpha"])
    layout = controls.render()

    control_row = next(
        child
        for child in layout.children
        if isinstance(child, html.Div)
        and getattr(child, "className", None) == "dataset-control-row"
    )
    dropdown = next(
        child
        for child in control_row.children
        if getattr(child, "id", None) == controls.dataset_select_id
    )
    assert dropdown.options == [
        {"label": "alpha", "value": "alpha"},
        {"label": "beta", "value": "beta"},
    ]
    assert dropdown.value == "alpha"
    assert not any("/" in option["label"] for option in dropdown.options)


def test_rejects_invalid_dataset_ids_and_poll_interval():
    with pytest.raises(ValueError):
        DatasetControls([])
    with pytest.raises(ValueError):
        DatasetControls([""])
    with pytest.raises(ValueError):
        DatasetControls(["alpha"], poll_interval_ms=50)


def test_request_callback_uses_open_when_no_current_dataset():
    manager = FakeDatasetManager(has_current=False)
    controls = DatasetControls(["alpha"])
    app = Dash(__name__)
    controls.register_callbacks(app, manager)

    request_dataset = _raw_callback(app, controls.request_button_id)
    result = request_dataset(1, "alpha")

    assert manager.open_calls == ["alpha"]
    assert manager.switch_calls == []
    assert result[0] == _serialize_token(manager.token)
    assert result[1] == "Requested dataset alpha"
    assert result[2] == ""
    assert result[3] == 0


def test_request_callback_uses_switch_when_current_dataset_exists():
    manager = FakeDatasetManager(has_current=True)
    controls = DatasetControls(["alpha"])
    app = Dash(__name__)
    controls.register_callbacks(app, manager)

    request_dataset = _raw_callback(app, controls.request_button_id)
    result = request_dataset(1, "alpha")

    assert manager.open_calls == []
    assert manager.switch_calls == ["alpha"]
    assert result[0] == _serialize_token(manager.token)


def test_poll_callback_reports_loading_progress_and_current_dataset():
    manager = FakeDatasetManager(
        state=DatasetState.LOADING,
        progress=DatasetProgress(
            stage="reading_trace",
            fraction=0.42,
            message="Reading trace",
        ),
        current_dataset_id="alpha",
    )
    controls = DatasetControls(["alpha"])
    app = Dash(__name__)
    controls.register_callbacks(app, manager)

    poll_status = _raw_callback(app, controls.poll_interval_id)
    result = poll_status(1, _serialize_token(manager.token))

    status_text, progress_value, error_text, cancel_disabled, ready, current_id = result
    assert "Loading: Reading Trace" in status_text
    assert "Reading trace" in status_text
    assert "alpha" in status_text
    assert progress_value == 42
    assert error_text == ""
    assert cancel_disabled is False
    assert ready is False
    assert current_id == "alpha"


def test_poll_callback_reports_ready_and_error_states():
    ready_manager = FakeDatasetManager(
        state=DatasetState.READY,
        progress=DatasetProgress(stage="ready", fraction=1.0),
        current_dataset_id="alpha",
    )
    controls = DatasetControls(["alpha"])
    app = Dash(__name__)
    controls.register_callbacks(app, ready_manager)
    poll_status = _raw_callback(app, controls.poll_interval_id)
    result = poll_status(1, _serialize_token(ready_manager.token))
    assert result[3] is True
    assert result[4] is True
    assert result[5] == "alpha"

    failed_manager = FakeDatasetManager(
        state=DatasetState.FAILED,
        progress=DatasetProgress(stage="failed", fraction=0.25),
        error="synthetic load failure",
        current_dataset_id=None,
    )
    failed_controls = DatasetControls(["alpha"])
    failed_app = Dash(__name__)
    failed_controls.register_callbacks(failed_app, failed_manager)
    failed_poll = _raw_callback(failed_app, failed_controls.poll_interval_id)
    failed_result = failed_poll(1, _serialize_token(failed_manager.token))
    assert failed_result[2] == "synthetic load failure"
    assert failed_result[3] is True
    assert failed_result[4] is False
    assert failed_result[5] is None


def test_cancel_callback_calls_manager_and_sets_cancel_requested_store():
    manager = FakeDatasetManager(state=DatasetState.LOADING)
    controls = DatasetControls(["alpha"])
    app = Dash(__name__)
    controls.register_callbacks(app, manager)

    cancel_dataset = _raw_callback(app, controls.cancel_button_id)
    result = cancel_dataset(1, _serialize_token(manager.token))

    assert manager.cancel_calls == [manager.token]
    assert result is True
