"""Compact reliability view model for the selection-driven explorer."""
from __future__ import annotations
from trace_store import UnknownSeriesError

from dataclasses import asdict
import math
from typing import Any

import numpy as np

from reliability_analysis import (
    CounterIdentity,
    CounterRole,
    ReliabilityCounterSeries,
    ReliabilityCounterSpec,
    ReliabilityDataset,
    ReliabilityPopulation,
    ReliabilityQualityFlag,
    ValueKind as ReliabilityValueKind,
    analyze_reliability,
)
from trace_model import CounterSemantics, QualityFlag, ValueKind


RELIABILITY_METRIC = "octet_errors"

RELIABILITY_COUNTER_ROLES = {
    "hni_sts_rx_bad_octets": CounterRole.ERROR,
    "hni_sts_tx_bad_octets": CounterRole.ERROR,
}

_QUALITY_FLAG_MAP = {
    QualityFlag.REORDERED: ReliabilityQualityFlag.REORDERED,
    QualityFlag.NONMONOTONIC_TIMESTAMP: ReliabilityQualityFlag.REORDERED,
    QualityFlag.DUPLICATE_REMOVED: ReliabilityQualityFlag.DUPLICATE,
    QualityFlag.BASELINE_RELATIVE: ReliabilityQualityFlag.BASELINE_RELATIVE,
    QualityFlag.PRECISION_LOSS_POSSIBLE: ReliabilityQualityFlag.UNSUPPORTED_SERIES,
    QualityFlag.RESET_SUSPECT: ReliabilityQualityFlag.DECREASE_OR_RESET,
    QualityFlag.WRAP_SUSPECT: ReliabilityQualityFlag.DECREASE_OR_RESET,
    QualityFlag.GAP: ReliabilityQualityFlag.PARTIAL_WINDOW,
    QualityFlag.UNKNOWN_METADATA: ReliabilityQualityFlag.UNSUPPORTED_SERIES,
}


def _quality_flags(result) -> tuple[ReliabilityQualityFlag, ...]:
    combined = int(result.counter.quality_flags)
    if result.quality_flags.size:
        combined |= int(np.bitwise_or.reduce(result.quality_flags))
    return tuple(
        dict.fromkeys(
            mapped
            for source_flag, mapped in _QUALITY_FLAG_MAP.items()
            if combined & source_flag
        )
    )


def _counter_series(result, spec: ReliabilityCounterSpec):
    if result.counter.value_kind is not ValueKind.EXACT_UINT64:
        return None
    values = tuple(int(value) for value in result.values)
    if any(value < 0 or value > 2**64 - 1 for value in values):
        return None
    return ReliabilityCounterSeries(
        spec=spec,
        identity=CounterIdentity(
            run_id=result.run_identity.run_id,
            node_id=str(result.node.node_id),
            nic_id=str(result.nic.nic_id),
            collector_id=f"{result.collector.name} (ID {result.collector.collector_id})",
            source_counter_id=result.counter.name,
        ),
        timestamps_ns=tuple(int(timestamp) for timestamp in result.timestamps_ns),
        values=values,
        value_kind=ReliabilityValueKind.CUMULATIVE,
        semantics=(
            "cumulative counter"
            if result.counter.semantics is CounterSemantics.CUMULATIVE
            else result.counter.semantics.value
        ),
        baseline_relative=result.counter.baseline_relative,
        quality_flags=_quality_flags(result),
    )


def _finite(value):
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _counter_record(result):
    return {
        "counter": result.spec.counter_id,
        "role": result.spec.role.value,
        "unit": result.unit,
        "state": result.status.value.replace("_", " ").title(),
        "total": result.total,
        "rate_per_second": _finite(result.rate_per_second),
        "coverage": _finite(result.coverage_fraction) or 0.0,
        "sample_count": result.sample_count,
        "valid_intervals": result.valid_interval_count,
        "invalid_intervals": result.invalid_interval_count,
        "first_nonzero_ns": result.first_nonzero_sample_ns,
        "zero_observed": result.zero_observed,
        "persistent_zero": result.persistent_zero,
        "quality_flags": sorted(flag.value for flag in result.quality_flags),
    }


def _record(nic, collector_id, analysis, engine_nodes):
    observed_counters = [
        result
        for result in analysis.counter_results
        if result.total is not None
    ]
    zero_counters = [result for result in observed_counters if result.total == 0]
    nonzero_counters = [result for result in observed_counters if result.total > 0]
    return {
        "id": str(nic.nic_id),
        "node_id": nic.node_id,
        "nic_id": nic.nic_id,
        "node": engine_nodes[nic.node_id].label,
        "nic": nic.label,
        "direction": "Errors",
        "counter": RELIABILITY_METRIC,
        "state": analysis.status.value.replace("_", " ").title(),
        "reason": " ".join(analysis.status_reasons)
        or "Bad-octet observations use complete valid intervals.",
        "total": analysis.error_total,
        "mean": _finite(analysis.aggregate_error_rate_per_second),
        "p99": None,
        "cv": None,
        "coverage": min((_finite(result.coverage_fraction) or 0.0 for result in observed_counters), default=0.0),
        "collector_id": collector_id,
        "unit": analysis.aggregate_rate_unit or "mixed source units",
        "sample_count": sum(result.sample_count for result in analysis.counter_results),
        "observed_counter_count": len(observed_counters),
        "zero_counter_count": len(zero_counters),
        "nonzero_counter_count": len(nonzero_counters),
        "flags": sorted(flag.value for flag in analysis.quality_flags),
        "good_total": analysis.good_total,
        "zero_observed": analysis.zero_observed,
        "persistent_zero": analysis.persistent_zero,
        "counter_results": [
            _counter_record(result) for result in analysis.counter_results
        ],
        "historical_zero_note": analysis.historical_zero_note,
    }


def build_reliability_view(store, query, engine_nodes):
    counter_by_name = {counter.name: counter for counter in store.counters}
    available = {
        name: role
        for name, role in RELIABILITY_COUNTER_ROLES.items()
        if name in counter_by_name
    }
    specs = tuple(
        ReliabilityCounterSpec(
            counter_id=name,
            role=role,
            unit=counter_by_name[name].unit,
            aggregation_group="octet_errors",
        )
        for name, role in available.items()
    )

    selected_nics = [
        nic
        for nic in store.nics
        if (not query.node_ids or nic.node_id in query.node_ids)
        and (not query.nic_ids or nic.nic_id in query.nic_ids)
    ]
    choices = dict(query.collectors)
    records = []
    for nic in selected_nics:
        chosen = choices.get(nic.nic_id)
        if chosen is None and specs:
            try:
                representative = store.query_series(nic.node_id, nic.nic_id, counter_by_name[specs[0].counter_id])
            except UnknownSeriesError:
                from types import SimpleNamespace
                representative = SimpleNamespace(results=())
            if len(representative.results) != 1:
                records.append(
                    {
                        "id": str(nic.nic_id),
                        "node_id": nic.node_id,
                        "nic_id": nic.nic_id,
                        "node": engine_nodes[nic.node_id].label,
                        "nic": nic.label,
                        "direction": "Errors",
                        "counter": RELIABILITY_METRIC,
                        "state": "Choose collector" if representative.results else "No samples",
                        "reason": (
                            f"{len(representative.results)} observation streams; "
                            "select this NIC and choose one. Streams are never added together."
                        ),
                        "total": None,
                        "mean": None,
                        "p99": None,
                        "cv": None,
                        "coverage": 0.0,
                        "collector_id": None,
                        "unit": "mixed source units",
                        "sample_count": 0,
                        "observed_counter_count": 0,
                        "zero_counter_count": 0,
                        "nonzero_counter_count": 0,
                        "flags": [],
                        "good_total": None,
                        "zero_observed": False,
                        "persistent_zero": False,
                        "counter_results": [],
                        "historical_zero_note": "",
                    }
                )
                continue
            chosen = representative.results[0].collector.collector_id

        series = []
        collector_name = None
        for spec in specs:
            counter = counter_by_name[spec.counter_id]
            try:
                result = store.query_series(nic.node_id, nic.nic_id, counter, collector_id=chosen).unique()
            except UnknownSeriesError:
                continue
            collector_name = result.collector.name
            reliability_series = _counter_series(result, spec)
            if reliability_series is not None:
                series.append(reliability_series)

        analysis = analyze_reliability(
            ReliabilityDataset(
                population=ReliabilityPopulation(
                    run_id=store.identity.run_id,
                    node_id=str(nic.node_id),
                    nic_id=str(nic.nic_id),
                    collector_id=f"{collector_name} (ID {chosen})",
                ),
                requested_counters=specs,
                series=tuple(series),
                window_ns=(query.start_ns, query.end_ns),
            )
        )
        records.append(_record(nic, chosen, analysis, engine_nodes))

    ranked = sorted(
        records,
        key=lambda record: (
            record["total"] is None,
            -(float(record["total"]) if record["total"] is not None else 0),
            record["node"],
            record["nic"],
        ),
    )
    return {
        "query": query,
        "duration_s": (query.end_ns - query.start_ns) / 1e9,
        "records": records,
        "ranked": ranked,
        "nic_count": len(selected_nics),
        "node_count": len({nic.node_id for nic in selected_nics}),
        "origin_ns": store.coverage.acquisition_start_ns if store.coverage.acquisition_start_ns is not None else int(store.timestamps_ns.min()),
        "run": asdict(store.identity),
        "counter": RELIABILITY_METRIC,
        "source_rows": store.row_count,
    }
