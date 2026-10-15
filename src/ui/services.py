"""Bounded view-model adapters for traffic investigations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _format_bytes(value: int | float | None) -> str:
    if value is None:
        return "Missing"
    if value == 0:
        return "0 B"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if abs(amount) < 1024 or unit == units[-1]:
            precision = 0 if unit == "B" else 2
            return f"{amount:.{precision}f} {unit}"
        amount /= 1024
    return "Missing"


def _format_rate(value: int | float | None) -> str:
    return "Missing" if value is None else f"{float(value):,.0f} B/s"


def _format_percent(value: float | None) -> str:
    return "Missing" if value is None else f"{float(value) * 100:.1f}%"


def _format_interval(start_ns: int, end_ns: int) -> str:
    return f"{start_ns / 1_000_000_000:.3f} s – {end_ns / 1_000_000_000:.3f} s"


def overview_context(view: dict[str, Any]) -> list[dict[str, str]]:
    run = view["run"]
    capture = view["capture"]
    overview = view["overview"]
    return [
        {"label": "Run", "value": run["run_id"], "detail": run["source_format"]},
        {"label": "TX total", "value": _format_bytes(overview.get("tx_total_bytes"))},
        {"label": "RX total", "value": _format_bytes(overview.get("rx_total_bytes"))},
        {
            "label": "Nodes / NICs",
            "value": f"{capture['node_count']} / {capture['nic_count']}",
            "detail": "observed population; expected topology is not inferred",
        },
        {
            "label": "Minimum traffic coverage",
            "value": _format_percent(overview.get("coverage_fraction")),
            "detail": "minimum across compatible selected series",
        },
        {
            "label": "Intervals",
            "value": (
                f"{overview.get('valid_interval_count', 0)} valid / "
                f"{overview.get('invalid_interval_count', 0)} invalid"
            ),
            "detail": "invalid intervals are not treated as zero traffic",
        },
        {
            "label": "Result state",
            "value": str(overview.get("traffic_status", "Missing")),
            "detail": "; ".join(overview.get("traffic_status_reasons", ())) or "no additional reason",
        },
    ]


def ranked_rows(view: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in view.get("series_summaries", ()):
        total = record.get("total_bytes")
        rows.append(
            {
                "rank": len(rows) + 1,
                "node": record["node"],
                "nic": record["nic"],
                "direction": record["direction"],
                "total_bytes": total,
                "total_display": _format_bytes(total),
                "mean_rate": _format_rate(record.get("mean_rate_bytes_per_s")),
                "coverage": _format_percent(record.get("coverage_fraction")),
                "state": record.get("state", "missing"),
                "status": record.get("status", "Missing"),
            }
        )
    direction_order = {"TX": 0, "RX": 1}
    rows.sort(
        key=lambda row: (
            direction_order.get(row["direction"], 2),
            -(row["total_bytes"] if row["total_bytes"] is not None else -1),
            row["node"],
            row["nic"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows[:200]


def ranked_actions(view: dict[str, Any]) -> list[dict[str, Any]]:
    actions = []
    seen = set()
    for record in view.get("series_summaries", ()):
        identity = (record["node_id"], record["nic_id"])
        if identity in seen:
            continue
        seen.add(identity)
        actions.append(
            {
                "node_id": record["node_id"],
                "nic_id": record["nic_id"],
                "node": record["node"],
                "nic": record["nic"],
                "direction": record["direction"],
            }
        )
    return actions


def aggregate_timeline(view: dict[str, Any]) -> dict[str, list[tuple[int, float]]]:
    by_direction: dict[str, dict[int, float]] = {
        "TX": defaultdict(float),
        "RX": defaultdict(float),
    }
    for record in view.get("series_summaries", ()):
        direction = record.get("direction")
        if direction not in by_direction:
            continue
        timestamps = record.get("display_timestamps_ns", ())
        values = record.get("display_values", ())
        if not timestamps:
            continue
        previous_timestamp = int(timestamps[0])
        previous_value = values[0]
        for timestamp, value in zip(timestamps[1:], values[1:], strict=True):
            duration_ns = int(timestamp) - previous_timestamp
            if duration_ns <= 0 or value < previous_value:
                by_direction[direction][int(timestamp)] = float("nan")
            else:
                by_direction[direction][int(timestamp)] += (
                    (value - previous_value) / (duration_ns / 1_000_000_000)
                )
            previous_timestamp = int(timestamp)
            previous_value = value
    return {
        direction: sorted(points.items()) for direction, points in by_direction.items()
    }


def selected_identity(view: dict[str, Any]) -> dict[str, str]:
    detail = view["nic_detail"]
    records = detail.get("series", ())
    if not records:
        return {
            "node": "All nodes",
            "nic": "All NICs",
            "collector_policy": "not selected",
        }
    first = records[0]
    collectors = first.get("collector_identities", ())
    policy = first.get("collector_policy") or "ambiguous"
    if len(collectors) > 1:
        policy = f"ambiguous ({len(collectors)} collector streams)"
    return {
        "node": f"{first['node']} (ID {first['node_id']})",
        "nic": f"{first['nic']} (ID {first['nic_id']})",
        "collector_policy": policy,
    }


def shell_context(view: dict[str, Any]) -> dict[str, str]:
    run = view["run"]
    capture = view["capture"]
    overview = view["overview"]
    identity = selected_identity(view)
    return {
        "run_id": view["run"]["run_id"],
        "source_format": view["run"]["source_format"],
        "source_fingerprint": view["run"]["source_fingerprint"],
        "identity": f"{identity['node']} / {identity['nic']}",
        "collector_policy": identity["collector_policy"],
        "direction": str(view["filters"]["direction"]).upper(),
        "interval": _format_interval(
            capture["selected_start_ns"], capture["selected_end_ns"]
        ),
        "coverage": _format_percent(overview.get("coverage_fraction")),
        "state": str(overview.get("traffic_status", "Missing")),
        "provenance": (
            "Synthetic fixture"
            if "synthetic" in run["source_format"].lower()
            else "Real trace"
        ),
    }


def node_records(view: dict[str, Any]) -> list[dict[str, Any]]:
    selected_node = view["filters"].get("node_id")
    records = [
        record
        for record in view.get("series_summaries", ())
        if selected_node is None or record.get("node_id") == selected_node
    ]
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for record in records:
        key = (record["node_id"], record["nic_id"])
        if key not in grouped:
            grouped[key] = {
                "node": record["node"],
                "nic": record["nic"],
                "node_id": record["node_id"],
                "nic_id": record["nic_id"],
                "TX": None,
                "RX": None,
                "state": "missing",
            }
        grouped[key][record["direction"]] = record
    return list(grouped.values())
