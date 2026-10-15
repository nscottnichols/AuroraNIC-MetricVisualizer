"""Matching export snapshot for traffic investigation state."""

from __future__ import annotations

from typing import Any, Callable

from .state import InvestigationState


def build_investigation_export(
    view: dict[str, Any],
    state: InvestigationState,
    base_document_builder: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    document = base_document_builder(view)
    document["investigation"] = {
        "schema_version": state.schema_version,
        "dataset_token": state.dataset_token,
        "screen": state.screen,
        "selection": {
            "node_id": state.selection.node_id,
            "nic_id": state.selection.nic_id,
            "collector_id": state.selection.collector_id,
            "collector_policy": state.selection.collector_policy,
        },
        "direction": state.direction,
        "interval": {
            "origin_token": state.interval.origin_token,
            "start_ns": state.interval.start_ns,
            "end_ns": state.interval.end_ns,
        },
        "chart_mode": state.chart_mode,
        "state_version": state.state_version,
    }
    document.setdefault("methodology", {})["units"] = {
        "traffic_totals": "bytes",
        "rates": "bytes/s",
        "coverage": "fraction of requested duration",
    }
    return document


def build_investigation_export_payload(
    view: dict[str, Any],
    state: InvestigationState,
    *,
    base_document_builder: Callable[[dict[str, Any]], dict[str, Any]],
    serialize_document: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    document = build_investigation_export(view, state, base_document_builder)
    return {
        "content": serialize_document(document),
        "filename": "aurora-nic-investigation.json",
        "type": "application/json",
    }
