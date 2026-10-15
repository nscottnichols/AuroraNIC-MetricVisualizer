"""Adapter between normalized trace storage and traffic analysis."""

from __future__ import annotations

from trace_model import CounterSemantics, SeriesResult, ValueKind
from counter_semantic_registry import counter_semantic_definition
from traffic_analysis import (
    MetricSeries,
    SeriesIdentity,
    SeriesQualityFlag,
    TrafficDirection,
    ValueKind as TrafficValueKind,
)


_SEMANTIC_MAP = {
    CounterSemantics.CUMULATIVE: TrafficValueKind.CUMULATIVE,
    CounterSemantics.DELTA: TrafficValueKind.DELTA,
    CounterSemantics.GAUGE: TrafficValueKind.GAUGE,
    CounterSemantics.UNKNOWN: TrafficValueKind.UNKNOWN,
}

_QUALITY_MAP = {
    1 << 0: SeriesQualityFlag.REORDERED,
    1 << 1: SeriesQualityFlag.REORDERED,
    1 << 2: SeriesQualityFlag.DUPLICATE,
    1 << 3: SeriesQualityFlag.BASELINE_RELATIVE,
    1 << 4: SeriesQualityFlag.PRECISION_LOSS_POSSIBLE,
    1 << 5: SeriesQualityFlag.DECREASE_OR_RESET,
    1 << 6: SeriesQualityFlag.WRAP_SUSPECT,
    1 << 7: SeriesQualityFlag.GAP,
    1 << 8: SeriesQualityFlag.UNVALIDATED,
}


def infer_traffic_direction(counter_name: str) -> TrafficDirection:
    normalized_name = counter_name.casefold()
    if "tx" in normalized_name:
        return TrafficDirection.TX
    if "rx" in normalized_name:
        return TrafficDirection.RX
    raise ValueError(
        f"cannot infer TX/RX direction from counter {counter_name!r}; provide direction"
    )


def metric_series_from_series_result(
    result: SeriesResult,
    *,
    run_id: str | None = None,
    direction: TrafficDirection | None = None,
) -> MetricSeries:
    source_run_id = result.run_identity.run_id
    if run_id is not None and run_id != source_run_id:
        raise ValueError(
            "run_id does not match the run identity carried by SeriesResult"
        )
    resolved_direction = direction or infer_traffic_direction(result.counter.name)

    quality_flags: set[SeriesQualityFlag] = set()
    for bit, traffic_flag in _QUALITY_MAP.items():
        if bool(((result.quality_flags & bit) != 0).any()):
            quality_flags.add(traffic_flag)
    if result.counter.baseline_relative:
        quality_flags.add(SeriesQualityFlag.BASELINE_RELATIVE)
    if result.observation_stream_count > 1:
        quality_flags.add(SeriesQualityFlag.MULTIPLE_OBSERVATION_STREAMS)

    semantic_kind = _SEMANTIC_MAP[result.counter.semantics]
    counter_definition = counter_semantic_definition(result.counter.name)
    return MetricSeries(
        identity=SeriesIdentity(
            run_id=source_run_id,
            node_id=result.node.label,
            nic_id=result.nic.label,
            collector_id=result.collector.name,
            counter_id=result.counter.name,
            direction=resolved_direction,
        ),
        timestamps_ns=result.timestamps_ns,
        values=result.values,
        value_dtype=(
            "uint64"
            if result.counter.value_kind is ValueKind.EXACT_UINT64
            else "float64"
        ),
        value_kind=semantic_kind,
        semantics=(
            counter_definition.measurement
            if counter_definition is not None
            else "bytes" if result.counter.unit == "bytes" else result.counter.unit
        ),
        unit=result.counter.unit,
        baseline_relative=result.counter.baseline_relative,
        quality_flags=tuple(
            sorted(quality_flags, key=lambda flag: flag.value)
        ),
        counter_definition=counter_definition,
    )
