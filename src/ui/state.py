"""Serializable state and normalization rules for traffic investigations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from typing import Any, Literal


ScreenName = Literal["overview", "node", "nic"]
DirectionName = Literal["all", "tx", "rx"]
ChartMode = Literal["rate", "cumulative"]


@dataclass(frozen=True)
class SelectionState:
    node_id: int | None
    nic_id: int | None
    collector_id: int | None
    collector_policy: str | None


@dataclass(frozen=True)
class IntervalState:
    origin_token: str
    start_ns: int
    end_ns: int


@dataclass
class InvestigationState:
    schema_version: int = 1
    dataset_token: str = ""
    screen: ScreenName = "overview"
    selection: SelectionState = field(
        default_factory=lambda: SelectionState(None, None, None, None)
    )
    direction: DirectionName = "all"
    interval: IntervalState = field(
        default_factory=lambda: IntervalState("canonical-epoch", 0, 1)
    )
    chart_mode: ChartMode = "rate"
    history: list[dict[str, Any]] = field(
        default_factory=lambda: [{"screen": "overview", "node_id": None, "nic_id": None}]
    )
    state_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> "InvestigationState":
        if not isinstance(value, dict):
            raise TypeError("state must be a dictionary")
        selection = value.get("selection") or {}
        interval = value.get("interval") or {}
        return cls(
            schema_version=int(value.get("schema_version", 1)),
            dataset_token=str(value.get("dataset_token", "")),
            screen=value.get("screen", "overview"),
            selection=SelectionState(
                node_id=selection.get("node_id"),
                nic_id=selection.get("nic_id"),
                collector_id=selection.get("collector_id"),
                collector_policy=selection.get("collector_policy"),
            ),
            direction=value.get("direction", "all"),
            interval=IntervalState(
                origin_token=str(interval.get("origin_token", "canonical-epoch")),
                start_ns=int(interval.get("start_ns", 0)),
                end_ns=int(interval.get("end_ns", 1)),
            ),
            chart_mode=value.get("chart_mode", "rate"),
            history=list(value.get("history", [])),
            state_version=int(value.get("state_version", 1)),
        )


def dataset_token(run_id: str, source_fingerprint: str, schema_version: int) -> str:
    digest = hashlib.sha256(
        f"{schema_version}:{run_id}:{source_fingerprint}".encode("utf-8")
    ).hexdigest()
    return f"urn:aurora-nic:dataset:{digest[:24]}"


def initial_state(
    *,
    run_id: str,
    source_fingerprint: str,
    schema_version: int,
    start_ns: int,
    end_ns: int,
) -> InvestigationState:
    return InvestigationState(
        dataset_token=dataset_token(run_id, source_fingerprint, schema_version),
        interval=IntervalState("canonical-epoch", int(start_ns), int(end_ns)),
    )


def resolve_state(
    current: InvestigationState,
    *,
    node_id: int | None,
    nic_id: int | None,
    direction: str,
    start_ns: int,
    end_ns: int,
    screen: ScreenName | None = None,
    collector_id: int | None = None,
    collector_policy: str | None = None,
) -> InvestigationState:
    normalized_direction: DirectionName
    if direction in {"tx", "rx", "all"}:
        normalized_direction = direction  # type: ignore[assignment]
    else:
        normalized_direction = "all"

    start = int(start_ns)
    end = int(end_ns)
    if end <= start:
        start, end = current.interval.start_ns, current.interval.end_ns

    next_screen = screen or current.screen
    history = list(current.history)
    entry = {
        "screen": next_screen,
        "node_id": node_id,
        "nic_id": nic_id,
    }
    if not history or history[-1] != entry:
        history.append(entry)
        history = history[-12:]

    return InvestigationState(
        schema_version=1,
        dataset_token=current.dataset_token,
        screen=next_screen,
        selection=SelectionState(node_id, nic_id, collector_id, collector_policy),
        direction=normalized_direction,
        interval=IntervalState(current.interval.origin_token, start, end),
        chart_mode=current.chart_mode,
        history=history,
        state_version=current.state_version + 1,
    )
