"""Bounded explorer queries over the existing immutable TraceStore.

Scientific traffic summaries use the project's traffic analyser, not chart points.
The display integral reuses its ordered/validated transitions; aggregation occurs
on one common time grid, never by adding unrelated sample timestamps. Raw counter
inspection deliberately makes no derived diagnostic claims.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import math
import threading
from typing import Any

import numpy as np

from trace_analysis_bridge import metric_series_from_series_result
from trace_model import CounterSemantics, QualityFlag
from trace_store import TraceStore, UnknownSeriesError
from traffic_analysis import (
    TrafficDirection, analyze_traffic_series, _ordered_summary,
    SeriesQualityFlag,
)
from .reliability import RELIABILITY_METRIC, build_reliability_view
from .diagnostics import METRICS, build_view as build_diagnostic_view

TX = "hni_sts_tx_ok_octets"
RX = "hni_sts_rx_ok_octets"
MAX_BINS = 240
MAX_TABLE_ROWS = 200
MAX_HEATMAP_ROWS = 48


def finite(value):
    return None if value is None or not math.isfinite(float(value)) else float(value)


def human(value, unit="B") -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    amount = float(value)
    if unit in {"B", "B/s"}:
        for prefix in ("", "k", "M", "G", "T", "P"):
            if abs(amount) < 1000 or prefix == "P":
                return f"{amount:,.2f} {prefix}{unit}" if prefix else f"{amount:,.0f} {unit}"
            amount /= 1000
    return f"{amount:,.3g} {unit}".strip()


def boundaries(store: TraceStore) -> tuple[int, int]:
    start = store.coverage.acquisition_start_ns
    end = store.coverage.acquisition_end_ns
    if start is None:
        start = int(store.timestamps_ns.min()) if store.row_count else 0
    if end is None:
        end = int(store.timestamps_ns.max()) if store.row_count else start + 1
    return int(start), max(int(start) + 1, int(end))


@dataclass(frozen=True)
class Query:
    """Only selection metadata crosses the browser boundary."""
    start_ns: int
    end_ns: int
    node_ids: tuple[int, ...] = ()
    nic_ids: tuple[int, ...] = ()
    metric: str = "traffic"
    mode: str = "rate"
    direction: str = "TX"
    collectors: tuple[tuple[int, int], ...] = ()  # physical nic_id -> collector_id

    def to_dict(self):
        # Epoch ns are strings in browser/export metadata to avoid JS int rounding.
        result = asdict(self)
        result["start_ns"] = str(self.start_ns)
        result["end_ns"] = str(self.end_ns)
        return result

    @classmethod
    def from_dict(cls, value):
        return cls(
            start_ns=int(value["start_ns"]), end_ns=int(value["end_ns"]),
            node_ids=tuple(int(v) for v in value.get("node_ids", [])),
            nic_ids=tuple(int(v) for v in value.get("nic_ids", [])),
            metric=value.get("metric", "traffic"), mode=value.get("mode", "rate"),
            direction=value.get("direction", "TX"),
            collectors=tuple((int(k), int(v)) for k, v in value.get("collectors", [])),
        )


def _integral_at(edges, starts, ends, amounts):
    """Integral of disjoint constant-rate intervals. All axes are relative ns."""
    if not len(starts):
        return np.zeros(len(edges), dtype=float)
    prefix = np.r_[0., np.cumsum(np.asarray(amounts, dtype=float))]
    positions = np.searchsorted(ends, edges, side="right")
    output = prefix[positions].copy()
    active = positions < len(starts)
    p = positions[active]
    overlap = np.clip(edges[active] - starts[p], 0, ends[p] - starts[p])
    output[active] += np.asarray(amounts, dtype=float)[p] * overlap / (ends[p] - starts[p])
    return output


def traffic_bins(series, start_ns, end_ns, bin_count=MAX_BINS):
    """Return bounded display mass and coverage, after full-resolution validation.

    Rates are interval averages, not instantaneous peaks. A transition crossing a
    selection boundary is excluded, matching analyze_traffic_series. Invalid or
    reset transitions are not interpolated. Chart amounts are float approximations;
    the summary/export totals are independently computed exact integer deltas.
    """
    count = min(bin_count, max(1, end_ns - start_ns))
    edges = np.linspace(0, end_ns - start_ns, count + 1, dtype=np.int64)
    summary, flags = _ordered_summary(series, None)
    valid = summary.valid & (summary.starts >= start_ns) & (summary.ends <= end_ns)
    if SeriesQualityFlag.INVALID in flags or SeriesQualityFlag.MISSING in flags:
        valid[:] = False
    starts = summary.starts[valid] - start_ns
    ends = summary.ends[valid] - start_ns
    amounts = (summary.delta_uint64 if summary.delta_uint64 is not None else summary.delta_float64)[valid]
    mass = np.diff(_integral_at(edges, starts, ends, amounts))
    coverage_ns = np.diff(_integral_at(edges, starts, ends, ends - starts))
    widths = np.diff(edges).astype(float)
    full = coverage_ns >= widths - 0.5  # ns rounding tolerance, not a coverage waiver
    rates = np.full(count, np.nan)
    rates[full] = mass[full] / (widths[full] / 1e9)
    return edges, mass, coverage_ns, rates


class ExplorerEngine:
    """Small process-local cache of bounded results, never an unbounded raw cache."""
    def __init__(self, store: TraceStore, analysis_config=None):
        self.analysis_config = analysis_config or {}
        self.store = store
        self.origin, self.end = boundaries(store)
        self.nodes = {v.node_id: v for v in store.nodes}
        self.nics = {v.nic_id: v for v in store.nics}
        self.counters = {v.name: v for v in store.counters}
        # Build once; diagnostics formerly rescanned the complete index per NIC.
        self._collectors_by_nic = {}
        for (_, nic_id, _), streams in store._index.items():
            self._collectors_by_nic.setdefault(nic_id, set()).update(streams)
        self._cache: OrderedDict[Query, dict] = OrderedDict()
        self._lock = threading.RLock()
        self.query_executions = 0

    def collector_options(self, nic_id: int) -> list[dict]:
        nic = self.nics[nic_id]
        # Discovery uses the store's existing identity index, not a scan of rows.
        ids = sorted(self._collectors_by_nic.get(nic_id, ()))
        by_id = {c.collector_id: c for c in self.store.collectors}
        return [{"label": f"Collector {cid}" + (f" · rank {by_id[cid].rank}" if by_id[cid].rank is not None else ""),
                 "value": str(cid)} for cid in ids]

    def query(self, query: Query) -> dict:
        key = replace(
            query,
            mode="rate",
            direction="TX",
            node_ids=tuple(sorted(set(query.node_ids))),
            nic_ids=tuple(sorted(set(query.nic_ids))),
            collectors=tuple(sorted(set(query.collectors))),
        )
        with self._lock:
            result = self._cache.get(key)
            if result is not None:
                self._cache.move_to_end(key)
        if result is None:
            result = self._build(key)
            with self._lock:
                self.query_executions += 1
                self._cache[key] = result
                while len(self._cache) > 4:
                    self._cache.popitem(last=False)
        # Mode/direction changes reuse the numerical result; no source re-query.
        view = dict(result)
        view["query"] = query
        direction = query.direction if query.metric == "traffic" else "Errors" if query.metric == RELIABILITY_METRIC else "Counter"
        rows = [r for r in result["records"] if r["direction"] == direction]
        view["ranked"] = sorted(rows, key=lambda r: (r["total"] is None,
                               -(float(r["total"]) if r["total"] is not None else 0),
                               r["node"], r["nic"]))
        return view

    def _build(self, query: Query):
        if not self.origin <= query.start_ns < query.end_ns <= self.end:
            raise ValueError("Time selection lies outside this trace.")
        nics = [nic for nic in self.store.nics
                if (not query.node_ids or nic.node_id in query.node_ids)
                and (not query.nic_ids or nic.nic_id in query.nic_ids)]
        if query.node_ids and not set(query.node_ids) <= set(self.nodes):
            raise ValueError("Unknown node selection.")
        if query.nic_ids and not set(query.nic_ids) <= set(self.nics):
            raise ValueError("Unknown physical NIC selection.")
        if query.metric in METRICS:
            return build_diagnostic_view(self, query)
        if query.metric == RELIABILITY_METRIC:
            return build_reliability_view(self.store, query, self.nodes)
        names = [("TX", TX), ("RX", RX)] if query.metric == "traffic" else [("Counter", query.metric)]
        choices = dict(query.collectors)
        records = []
        for nic in nics:
            for direction, name in names:
                records.append(self._record(nic, name, direction, query, choices.get(nic.nic_id)))
        duration = (query.end_ns - query.start_ns) / 1e9
        selected_records = [r for r in records if r["direction"] == (query.direction if query.metric == "traffic" else "Counter")]
        ranked = sorted(selected_records, key=lambda r: (r["total"] is None,
                          -(float(r["total"]) if r["total"] is not None else 0), r["node"], r["nic"]))
        aggregate = {}
        for direction, _ in names:
            candidates = [r for r in records if r["direction"] == direction]
            usable = [r for r in candidates if r["total"] is not None]
            if query.metric == "traffic":
                total = sum(r["total"] for r in usable) if usable else None
                if usable:
                    rate_arrays = np.asarray([r["rates"] for r in usable], dtype=float)
                    complete = np.all(np.isfinite(rate_arrays), axis=0)
                    rates = np.sum(np.where(np.isfinite(rate_arrays), rate_arrays, 0), axis=0)
                    rates[~complete] = np.nan
                    # Cumulative observed mass is for display only; breaks indicate missing intervals.
                    mass_arrays = np.asarray([r["mass"] for r in usable], dtype=float)
                    cumulative = np.cumsum(np.sum(mass_arrays, axis=0))
                    cumulative[~complete] = np.nan
                else:
                    rates = np.full(min(MAX_BINS, query.end_ns-query.start_ns), np.nan)
                    cumulative = rates.copy()
                aggregate[direction] = {"total": total, "usable": len(usable), "expected": len(nics),
                                        "rates": rates, "cumulative": cumulative,
                                        "complete": len(usable) == len(nics) and all(r["coverage"] >= .999999 for r in usable)}
        if query.metric == 'traffic':
            for direction, _ in names:
                population = [row for row in records if row['direction'] == direction]
                total = sum(row['total'] for row in population if row['total'] is not None)
                node_totals = {}
                for r in population:
                    if r['total'] is not None: node_totals[r['node_id']] = node_totals.get(r['node_id'], 0) + r['total']
                for row in population:
                    node_total = node_totals.get(row['node_id'], 0)
                    row['population_share'] = row['total']/total if row['total'] is not None and total > 0 else None
                    row['node_share'] = row['total']/node_total if row['total'] is not None and node_total > 0 else None
        return {"query": query, "duration_s": duration, "records": records, "ranked": ranked,
                "aggregate": aggregate, "node_count": len({n.node_id for n in nics}),
                "nic_count": len(nics), "origin_ns": self.origin,
                "run": asdict(self.store.identity), "counter": query.metric,
                "x_s": np.linspace((query.start_ns-self.origin)/1e9,
                                    (query.end_ns-self.origin)/1e9,
                                    min(MAX_BINS, query.end_ns-query.start_ns)+1),
                "source_rows": self.store.row_count}

    def _record(self, nic, name, direction, query, collector):
        row = {"id": str(nic.nic_id), "node_id": nic.node_id, "nic_id": nic.nic_id,
               "node": self.nodes[nic.node_id].label, "nic": nic.label, "direction": direction,
               "counter": name, "state": "No samples", "reason": "Counter was not observed on this NIC.",
               "total": None, "mean": None, "p50": None, "p99": None, "cv": None,
               "coverage": 0., "collector_id": None, "collector_options": [],
               "unit": "B" if direction in {"TX", "RX"} else "",
               "exact_last_value": None, "sample_count": 0, "flags": [],
               "rates": np.full(min(MAX_BINS,query.end_ns-query.start_ns), np.nan),
               "mass": np.zeros(min(MAX_BINS,query.end_ns-query.start_ns)),
               "raw_x": [], "raw_y": []}
        if name not in self.counters:
            return row
        counter = self.counters[name]
        row["metadata"] = {"semantics": counter.semantics.value, "unit": counter.unit,
                           "value_kind": counter.value_kind.value, "baseline_relative": counter.baseline_relative,
                           "registry_version": counter.registry_version}
        try:
            all_results = self.store.query_series(nic.node_id, nic.nic_id, name).results
        except UnknownSeriesError:
            return row
        row["collector_options"] = [r.collector.collector_id for r in all_results]
        if collector is None and len(all_results) != 1:
            row.update(state="Choose collector", reason=f"{len(all_results)} observation streams; select this NIC and choose one. Streams are never added together.")
            return row
        result = next((r for r in all_results if r.collector.collector_id == collector), None) if collector is not None else all_results[0]
        if result is None:
            row.update(state="No samples", reason="The chosen collector did not observe this counter.")
            return row
        row["collector_id"] = result.collector.collector_id
        row["collector_name"] = result.collector.name
        if query.metric == "traffic":
            series = metric_series_from_series_result(result, direction=TrafficDirection(direction))
            summary = analyze_traffic_series(series, window_start_ns=query.start_ns,
                                             window_end_ns=query.end_ns, include_intervals=False)
            row.update(total=summary.total_bytes, mean=finite(summary.mean_rate_bytes_per_s),
                       p50=finite(summary.p50_rate_bytes_per_s), p99=finite(summary.p99_rate_bytes_per_s),
                       cv=finite(summary.coefficient_of_variation), coverage=summary.coverage_fraction or 0.,
                       sample_count=summary.sample_count, flags=sorted(f.value for f in summary.quality_flags),
                       reason=" ".join(summary.status_reasons),
                       state="No valid intervals" if summary.total_bytes is None else "Idle" if summary.total_bytes == 0 else "Observed")
            if summary.total_bytes is not None:
                _, mass, coverage_ns, rates = traffic_bins(series, query.start_ns, query.end_ns)
                row.update(rates=rates, mass=mass)
                if summary.coverage_fraction is not None and summary.coverage_fraction < .999999:
                    row["state"] = "Partial"
            row["formula"] = summary.mean_rate_formula
            row["caveats"] = summary.counter_caveats
        else:
            selected = (result.timestamps_ns >= query.start_ns) & (result.timestamps_ns <= query.end_ns)
            t, v = result.timestamps_ns[selected], result.values[selected]
            row["sample_count"] = len(t)
            row["unit"] = counter.unit
            if not len(t):
                return row
            row.update(exact_last_value=str(v[-1]), state="Observed samples", total=v[-1].item(),
                       reason="Raw observations; absolute cumulative values are not selected-window totals.")
            if counter.baseline_relative:
                row["reason"] += " Baseline-relative/change-only source: absent samples do not establish idle time."
            # Last actual observation in each display bin; no carry-forward or
            # rate derivation. Use every source point, not the extrema envelope.
            count=min(MAX_BINS,query.end_ns-query.start_ns)
            edges=np.linspace(0,query.end_ns-query.start_ns,count+1,dtype=np.int64)
            bins=np.minimum(count-1,np.searchsorted(edges,t-query.start_ns,side='right')-1)
            heat=np.full(count,np.nan)
            for index,value in zip(bins,v):
                if index>=0: heat[index]=float(value)
            row['heat_y']=heat
            indices = extrema_indices(v, MAX_BINS)
            row.update(raw_x=((t[indices]-self.origin)/1e9).tolist(), raw_y=v[indices].astype(float).tolist())
            if v.dtype == np.uint64 and np.any(v > 2**53):
                row["reason"] += " Plot positions are floating-point approximations; exact last value is in the export."
        if query.metric == 'traffic' and counter.baseline_relative:
            row.update(p50=None, p99=None, cv=None, active_fraction=None,
                first_emitted_progress=format(float(result.values[0]),'.17g'),
                last_emitted_progress=format(float(result.values[-1]),'.17g'))
            row['reason'] += ' Change-only source: totals are changes between observed endpoints. Hidden initial progress is retained separately; native-cadence percentiles/idle time are unavailable.'
        elif query.metric == 'traffic':
            summary, _ = _ordered_summary(series, None)
            valid = summary.valid & (summary.starts >= query.start_ns) & (summary.ends <= query.end_ns)
            amounts = (summary.delta_uint64 if summary.delta_uint64 is not None else summary.delta_float64)[valid]
            durations = (summary.ends-summary.starts)[valid]
            row['active_fraction'] = float(durations[amounts>0].sum()/durations.sum()) if durations.sum() else None
        return row

    def export(self, query: Query):
        view = self.query(query)
        records = []
        for row in view["records"]:
            item = {k:v for k,v in row.items() if k not in {"rates","mass","raw_x","raw_y","heat_y"}}
            # Exact integers are exported as decimal strings alongside their units.
            if isinstance(item["total"], int):
                item["total"] = str(item["total"])
            records.append(item)
        traffic_totals = ({d:{"observed_bytes": str(a["total"]) if a["total"] is not None else None,
                              "usable_nics": a["usable"], "selected_nics": a["expected"],
                              "complete": a["complete"]} for d,a in view["aggregate"].items()}
                          if "aggregate" in view else {})
        balance = None
        if query.metric == 'traffic':
            from .diagnostic_ui import balance_rows
            balance = {'population': balance_rows(view['ranked']),
                'nodes': {node: balance_rows([r for r in view['ranked'] if r['node'] == node]) for node in sorted({r['node'] for r in view['ranked']})},
                'direction': query.direction,
                'caveat': 'Observed volumes on complete intervals; unequal coverage and background device traffic limit imbalance/participation conclusions.'}
        return {"schema": "aurora-nic-explorer/1", "run": view["run"], "selection": query.to_dict(),
                "units": "SI bytes/rates; counter-native units in counter mode",
                "window_policy": "Complete sample intervals only. Boundary-straddling intervals excluded.",
                "collector_policy": "One explicit stream per physical NIC; ambiguous observations excluded, never summed.",
                "display_policy": "240 equal-time bins, interval-average rates. Summary statistics use all valid source intervals.",
                "records": records,
                "diagnostic": view.get('diagnostic'),
                "formula": view.get('formula'),
                "alignment_method": view.get('method'),
                "source_reference": view.get('source_reference'),
                "traffic_totals": traffic_totals, "balance": balance,
                "topology_groups": self._topology_export(view)}

    def _topology_export(self, view):
        from .diagnostic_ui import topology_distribution
        return topology_distribution(view, self) if self.analysis_config.get('topology') else []


def extrema_indices(values, maximum):
    """Bounded min/max envelope, retaining endpoints instead of uniform decimation."""
    if len(values) <= maximum:
        return np.arange(len(values))
    indices = {0, len(values)-1}
    for bucket in np.array_split(np.arange(len(values)), max(1,(maximum-2)//2)):
        indices.add(int(bucket[np.argmin(values[bucket])]))
        indices.add(int(bucket[np.argmax(values[bucket])]))
    return np.asarray(sorted(indices))
