"""Legacy question-driven Dash application for synthetic traffic analysis."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import uuid
import threading
from typing import Any, Literal
import weakref

from flask import request

import numpy as np
import plotly.graph_objects as go
from dash import (
    Dash,
    ALL,
    Input,
    Output,
    State,
    callback_context,
    dcc,
    dash_table,
    html,
    no_update,
)
from dash.exceptions import PreventUpdate

from frame_analysis import (
    FRAME_ANALYSIS_VERSION,
    FRAME_INTERPRETATION,
    FRAME_LIMITATIONS,
    FrameBinEdgeSet,
    FrameBytePopulationCompatibility,
    FrameCounterIdentity,
    FrameCounterKind,
    FrameCounterSeries,
    FrameDataset,
    FrameDirection,
    FramePopulation,
    FrameQualityFlag,
    FrameValueKind,
    analyze_frames,
    byte_count_spec,
    frame_bin_spec,
)
from counter_semantic_registry import (
    COUNTER_SEMANTIC_REGISTRY_VERSION,
    CounterQuality,
    counter_semantic_definition,
)

from job_analysis import analyze_job_node_balance
from operation_analysis import (
    OPERATION_ANALYSIS_VERSION,
    OPERATION_LIMITATIONS,
    OperationCounterKind,
    OperationQualityFlag,
    OperationCounterSeries,
    OperationDataset,
    OperationPopulation,
    OperationSeriesIdentity,
    OpcodeMapping,
    OpcodeMappingEntry,
    OperationValueKind,
    analyze_operations,
    standard_operation_specs,
)
from host_pressure_analysis import (
    HOST_PRESSURE_ANALYSIS_VERSION,
    HOST_PRESSURE_LIMITATIONS,
    HostPressureCounterKind,
    HostPressureDataset,
    HostPressurePath,
    HostPressurePopulation,
    HostPressureQualityFlag,
    HostPressureSeries,
    HostPressureSeriesIdentity,
    HostPressureSeriesSpec,
    HostPressureStallKind,
    HostPressureValueKind,
    analyze_host_pressure,
)
from matching_analysis import (
    MATCHING_ANALYSIS_VERSION,
    MATCHING_INTERPRETATION,
    MATCHING_LIMITATIONS,
    MatchingCounterIdentity,
    MatchingCounterSeries,
    MatchingCounterSpec,
    MatchingDataset,
    MatchingPopulation,
    MatchingQualityFlag,
    MatchingRole,
    MatchingStatus,
    ValueKind as MatchingValueKind,
    analyze_matching,
)
from job_profile import (
    JobProfileProvenance,
    ProfileSummary,
    SummaryKind,
    SummaryStatus,
    build_job_communication_profile,
)
from latency_analysis import (
    LATENCY_ANALYSIS_VERSION,
    LATENCY_INTERPRETATION,
    LATENCY_LIMITATIONS,
    LatencyBinDefinition,
    LatencyBinEncoding,
    LatencyHistogramGroup,
    analyze_latency_histograms,
)
from phase_analysis import (
    ManualPhaseAnalysisResult,
    PhaseGapPolicy,
    PhaseTrafficRecord,
    PhaseWindow,
    analyze_manual_phase_windows,
)
from trace_analysis_bridge import infer_traffic_direction, metric_series_from_series_result
from trace_model import (
    CounterMetadata,
    CounterSemantics,
    QualityFlag,
    SCHEMA_VERSION,
    TraceValidationError,
    ValueKind,
)
from question_capabilities import build_question_capability_matrix
from flow_control_analysis import (
    DEFAULT_FLOW_CONTROL_PRIORITIES,
    FLOW_CONTROL_ANALYSIS_VERSION,
    FLOW_CONTROL_LIMITATIONS,
    analyze_flow_control,
)
from node_analysis import analyze_node_nic_balance
from reliability_analysis import (
    CounterIdentity,
    CounterRole,
    ReliabilityCounterSeries,
    ReliabilityCounterSpec,
    ReliabilityDataset,
    ReliabilityPopulation,
    ReliabilityQualityFlag,
    ReliabilityStatus,
    ValueKind as ReliabilityValueKind,
    analyze_reliability,
)
from trace_store import TraceStore
from translation_analysis import (
    TRANSLATION_ANALYSIS_VERSION,
    TRANSLATION_LIMITATIONS,
    TranslationCounterIdentity,
    TranslationCounterRole,
    TranslationCounterSeries,
    TranslationCounterSpec,
    TranslationDataset,
    TranslationMissView,
    TranslationPopulation,
    TranslationQualityFlag,
    TrafficDirection as TranslationTrafficDirection,
    TranslationTrafficTotal,
    ValueKind as TranslationValueKind,
    analyze_translation,
)
from traffic_analysis import (
    AnalysisStatus,
    MetricSeries,
    SeriesQualityFlag,
    SeriesIdentity,
    TrafficDirection,
    ValueKind as TrafficValueKind,
    analyze_traffic_series,
)
from performance_metrics import MetricIdentity, PerformanceMetricsRecorder
from ui import callbacks as investigation_callbacks
from ui.dataset_controls import DatasetControls
from ui import diagnostics as investigation_diagnostics
from ui import export_view as investigation_export
from ui import node_view as investigation_node_view
from ui import nic_view as investigation_nic_view
from ui import overview as investigation_overview
from ui import question_index as investigation_question_index
from ui import shell as investigation_shell
from ui.state import initial_state, resolve_state


MAX_DISPLAY_POINTS_PER_SERIES = 500
MAX_NODE_COMPARISON_ROWS = 200
MAX_NODE_COMPARISON_BARS = 100
MAX_EXPORT_SERIES = 200
MAX_EXPORT_COLLECTION_ITEMS = 500
MAX_EXPORT_JSON_BYTES = 2 * 1024 * 1024
MAX_PHASE_WINDOWS = 100
MAX_VIEW_MODEL_CACHE_ENTRIES = 32
MAX_RELIABILITY_COLLECTORS = 8
MAX_RELIABILITY_COUNTER_RESULTS = 32
MAX_FLOW_CONTROL_COLLECTORS = 8
MAX_FLOW_CONTROL_COUNTER_RESULTS = 64
MAX_MATCHING_COLLECTORS = 8
MAX_MATCHING_COUNTER_RESULTS = 32
MAX_MATCHING_PRIORITY_RESULTS = 8
MAX_HOST_PRESSURE_COLLECTORS = 8
MAX_HOST_PRESSURE_PATH_RESULTS = 16
MAX_HOST_PRESSURE_STALL_RESULTS = 32
MAX_OPERATION_COLLECTORS = 8
MAX_OPERATION_OPCODE_RESULTS = 16
MAX_OPERATION_TOTAL_RESULTS = 16
MAX_FRAME_COLLECTORS = 8
MAX_FRAME_BIN_RESULTS = 24
MAX_FRAME_QUANTILE_RESULTS = 4
MAX_TRANSLATION_COLLECTORS = 8
MAX_TRANSLATION_COUNTER_RESULTS = 32
MAX_TRANSLATION_VIEW_RESULTS = 4
MAX_LATENCY_COLLECTORS = 8
MAX_LATENCY_BIN_RESULTS = 24
MAX_LATENCY_QUANTILE_RESULTS = 4
EXPORT_SCHEMA_VERSION = 1

_VIEW_MODEL_CACHE: OrderedDict[
    tuple[Any, ...], tuple[weakref.ReferenceType[TraceStore], dict[str, Any]]
] = OrderedDict()
_VIEW_MODEL_CACHE_LOCK = threading.RLock()

OVERVIEW_ID = "analysis-overview"
FILTER_STATUS_ID = "analysis-filter-status"
NODE_FILTER_ID = "analysis-node-filter"
NIC_FILTER_ID = "analysis-nic-filter"
DIRECTION_FILTER_ID = "analysis-direction-filter"
TIME_FILTER_ID = "analysis-time-filter"
TIME_ORIGIN_ID = "analysis-time-origin"
NODE_TABLE_ID = "analysis-node-table"
NODE_ACTIONS_ID = "analysis-node-actions"
NODE_OPEN_SECTION = "node-open"
NODE_GRAPH_ID = "analysis-node-graph"
NIC_DETAIL_ID = "analysis-nic-detail"
PHASE_PANEL_ID = "analysis-phase-panel"
PHASE_RESULT_ID = "analysis-phase-result"
PHASE_STORE_ID = "analysis-phase-store"
PHASE_NAME_ID = "analysis-phase-name"
PHASE_START_ID = "analysis-phase-start"
PHASE_END_ID = "analysis-phase-end"
PHASE_ADD_ID = "analysis-phase-add"
JOB_PROFILE_ID = "analysis-job-profile"
QUESTION_TABLE_ID = "analysis-question-table"
COUNTER_TABLE_ID = "analysis-counter-table"
DATASET_LOADED_STORE_ID = "analysis-dataset-loaded-store"
EXPORT_BUTTON_ID = "analysis-export-button"
EXPORT_DOWNLOAD_ID = "analysis-export-download"
FILTER_RESET_ID = "analysis-filter-reset"

_RELIABILITY_ERROR_COUNTERS = (
    "hni_llr_rx_replay_event",
    "hni_llr_tx_replay_event",
    "hni_pcs_corrected_cw",
    "hni_pcs_uncorrected_cw",
    "pct_no_mst_nacks",
    "pct_no_tct_nacks",
    "pct_no_trs_nacks",
    "pct_retry_srb_requests",
    "pct_sct_timeouts",
    "pct_spt_timeouts",
    "pct_trs_rsp_nack_drops",
    "hni_sts_rx_bad_octets",
    "hni_sts_tx_bad_octets",
    *(f"hni_discard_cntr_{index}" for index in range(8)),
    "ixe_rx_pkt_drop_ixe_parser",
    "ixe_rx_pkt_drop_pct",
    "ixe_rx_pkt_drop_rmu_norsp",
    "ixe_rx_pkt_drop_rmu_wrsp",
)
_RELIABILITY_COUNTER_ROLES = {
    **{name: CounterRole.ERROR for name in _RELIABILITY_ERROR_COUNTERS},
    "hni_pcs_good_cw": CounterRole.GOOD,
}
_FLOW_CONTROL_TX_COUNTERS = tuple(
    dict.fromkeys(
        (
            *(f"hni_tx_paused_{priority}" for priority in DEFAULT_FLOW_CONTROL_PRIORITIES),
            "hni_pause_refresh",
            "hni_pause_sent",
            *(
                f"hni_pause_xoff_sent_{priority}"
                for priority in DEFAULT_FLOW_CONTROL_PRIORITIES
            ),
        )
    )
)
_FLOW_CONTROL_RX_COUNTERS = tuple(
    dict.fromkeys(
        (
            *(f"hni_rx_paused_{priority}" for priority in DEFAULT_FLOW_CONTROL_PRIORITIES),
            *(f"hni_pause_recv_{priority}" for priority in DEFAULT_FLOW_CONTROL_PRIORITIES),
        )
    )
)
_FLOW_CONTROL_UNDIRECTED_COUNTERS = (
    "hni_fgfc_discard",
    "hni_fgfc_event_xoff",
)
_FLOW_CONTROL_COUNTERS = tuple(
    dict.fromkeys(
        (*_FLOW_CONTROL_TX_COUNTERS, *_FLOW_CONTROL_RX_COUNTERS, *_FLOW_CONTROL_UNDIRECTED_COUNTERS)
    )
)
_FLOW_CONTROL_DIRECTION_BY_COUNTER = {
    **{counter: TrafficDirection.TX for counter in _FLOW_CONTROL_TX_COUNTERS},
    **{counter: TrafficDirection.RX for counter in _FLOW_CONTROL_RX_COUNTERS},
}
_MATCHING_PRIORITIES = range(4)
_MATCHING_COUNTERS = tuple(
    dict.fromkeys(
        (
            *(f"lpe_net_match_priority_{priority}" for priority in _MATCHING_PRIORITIES),
            *(f"lpe_net_match_overflow_{priority}" for priority in _MATCHING_PRIORITIES),
        )
    )
)
_MATCHING_POPULATION_ID = "nic03-receive-match-v1"
_HOST_PRESSURE_PACKET_COUNTERS = {
    HostPressurePath.POSTED: "parbs_tarb_pi_posted_pkts",
    HostPressurePath.NONPOSTED: "parbs_tarb_pi_non_posted_pkts",
}
_HOST_PRESSURE_BLOCKED_COUNTERS = {
    HostPressurePath.POSTED: "parbs_tarb_pi_posted_blocked_cnt",
    HostPressurePath.NONPOSTED: "parbs_tarb_pi_non_posted_blocked_cnt",
}
_HOST_PRESSURE_STALL_COUNTERS = (
    ("ixe_dmawr_stall_np_cdt", HostPressureStallKind.HOST_DMA),
    ("ixe_dmawr_stall_p_cdt", HostPressureStallKind.HOST_DMA),
    ("ixe_disp_stall_atu_cdt", HostPressureStallKind.DISPATCHER),
    ("ixe_disp_stall_atu_fifo", HostPressureStallKind.DISPATCHER),
    ("ixe_disp_stall_conflict", HostPressureStallKind.DISPATCHER),
    ("ixe_disp_stall_mdid_cdts", HostPressureStallKind.DISPATCHER),
    ("ixe_disp_stall_put_resp", HostPressureStallKind.DISPATCHER),
    ("ixe_disp_stall_resp_fifo", HostPressureStallKind.DISPATCHER),
    *((f"oxe_stall_pbuf_bc_{index}", HostPressureStallKind.TX) for index in range(10)),
)
_OPERATION_SPECS = standard_operation_specs()
_OPERATION_COUNTERS_BY_NAME = {spec.counter_id: spec for spec in _OPERATION_SPECS}
_OPERATION_OPCODE_MAPPING: OpcodeMapping | None = None
_FRAME_BIN_IDS = (
    "27",
    "35",
    "36_to_63",
    "64",
    "65_to_127",
    "128_to_255",
    "256_to_511",
    "512_to_1023",
    "1024_to_2047",
    "2048_to_4095",
    "4096_to_8191",
    "8192_to_max",
)
_FRAME_SPECS = tuple(
    dict.fromkeys(
        (
            *(
                frame_bin_spec(
                    f"hni_tx_ok_{bin_id}",
                    FrameDirection.TX,
                    bin_id,
                    f"nic06-{FrameDirection.TX.value}-wire-frames-v1",
                )
                for bin_id in _FRAME_BIN_IDS
            ),
            byte_count_spec(
                "hni_sts_tx_ok_octets",
                FrameDirection.TX,
                f"nic06-{FrameDirection.TX.value}-wire-bytes-v1",
            ),
            *(
                frame_bin_spec(
                    f"hni_rx_ok_{bin_id}",
                    FrameDirection.RX,
                    bin_id,
                    f"nic06-{FrameDirection.RX.value}-wire-frames-v1",
                )
                for bin_id in _FRAME_BIN_IDS
            ),
            byte_count_spec(
                "hni_sts_rx_ok_octets",
                FrameDirection.RX,
                f"nic06-{FrameDirection.RX.value}-wire-bytes-v1",
            ),
        )
    )
)
_FRAME_COUNTERS_BY_NAME = {spec.counter_id: spec for spec in _FRAME_SPECS}
_FRAME_BIN_EDGE_SETS: tuple[FrameBinEdgeSet, ...] = ()
_FRAME_BYTE_COMPATIBILITY: tuple[FrameBytePopulationCompatibility, ...] = ()
_TRANSLATION_POPULATION_ID = "nic07-address-translation-v1"
_TRANSLATION_SPECS = (
    *(
        TranslationCounterSpec(
            counter_id=counter_id,
            role=TranslationCounterRole.REQUEST,
            unit="events",
            population_id=_TRANSLATION_POPULATION_ID,
        )
        for counter_id in (
            "atu_client_req_ee",
            "atu_client_req_ixe",
            "atu_client_req_oxe",
        )
    ),
    *(
        TranslationCounterSpec(
            counter_id=counter_id,
            role=TranslationCounterRole.HIT,
            unit="events",
            population_id=_TRANSLATION_POPULATION_ID,
        )
        for counter_id in (
            *(f"atu_cache_hit_base_page_size_{index}" for index in range(4)),
            *(f"atu_cache_hit_derivative1_page_size_{index}" for index in range(4)),
            *(f"atu_cache_hit_derivative2_page_size_{index}" for index in range(4)),
        )
    ),
    *(
        TranslationCounterSpec(
            counter_id=counter_id,
            role=TranslationCounterRole.MISS,
            unit="events",
            population_id=_TRANSLATION_POPULATION_ID,
            miss_view=TranslationMissView.POOL,
        )
        for counter_id in (f"atu_cache_miss_{index}" for index in range(4))
    ),
    *(
        TranslationCounterSpec(
            counter_id=counter_id,
            role=TranslationCounterRole.MISS,
            unit="events",
            population_id=_TRANSLATION_POPULATION_ID,
            miss_view=TranslationMissView.CLIENT,
        )
        for counter_id in (
            "atu_cache_miss_ee",
            "atu_cache_miss_ixe",
            "atu_cache_miss_oxe",
        )
    ),
    TranslationCounterSpec(
        counter_id="atu_cache_evictions",
        role=TranslationCounterRole.EVICTION,
        unit="events",
        population_id=_TRANSLATION_POPULATION_ID,
    ),
)
_TRANSLATION_COUNTERS_BY_NAME = {
    spec.counter_id: spec for spec in _TRANSLATION_SPECS
}
_LATENCY_COUNTERS = tuple(
    dict.fromkeys(
        (
            *(f"pct_req_rsp_latency_{index}" for index in range(32)),
            *(f"pct_host_access_latency_{index}" for index in range(16)),
        )
    )
)
_LATENCY_BIN_DEFINITIONS: dict[
    LatencyHistogramGroup, LatencyBinDefinition
] = {
    group: LatencyBinDefinition(
        definition_version=1,
        group=group,
        bin_names=tuple(f"{group.value}_{index}" for index in range(bin_count)),
        encoding=LatencyBinEncoding.UNKNOWN,
        source="counter-name catalog only; no authoritative histogram metadata",
    )
    for group, bin_count in (
        (LatencyHistogramGroup.REQUEST_RESPONSE, 32),
        (LatencyHistogramGroup.HOST_ACCESS, 16),
    )
}

_EXPORT_FORBIDDEN_KEYS = {
    "trace_store",
    "raw_store",
    "uint64_values",
    "float64_values",
    "timestamps_ns",
    "record_ids",
    "quality_flags_array",
    "values",
    "samples",
    "display_values",
    "display_timestamps_ns",
}
_EXPORT_FORBIDDEN_KEY_SUFFIXES = ("_values", "_samples")


def _sanitize_export_value(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return None
    if isinstance(value, dict):
        return {
            str(key): _sanitize_export_value(child, depth + 1)
            for key, child in value.items()
            if isinstance(key, str)
            and key not in _EXPORT_FORBIDDEN_KEYS
            and not key.endswith(_EXPORT_FORBIDDEN_KEY_SUFFIXES)
        }
    if isinstance(value, (list, tuple)):
        return [
            _sanitize_export_value(child, depth + 1)
            for child in list(value)[:MAX_EXPORT_COLLECTION_ITEMS]
        ]
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is Ellipsis:
        return None
    return value


def _export_series_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exported_records = []
    for record in records[:MAX_EXPORT_SERIES]:
        exported = {
            key: child
            for key, child in record.items()
            if not key.startswith("display_")
        }
        if (
            exported.get("state") == "partial"
            and int(exported.get("stream_count") or 0) > 1
            and str(exported.get("collector_policy", "")).startswith("ambiguous:")
        ):
            exported["state"] = "ambiguous"
        exported_records.append(exported)
    return exported_records


def _export_reliability_summary(reliability: Any) -> dict[str, Any]:
    if not isinstance(reliability, dict):
        return _empty_reliability_analysis()
    records: list[dict[str, Any]] = []
    collector_records = []
    for collector in reliability.get("collector_analyses", ())[:MAX_RELIABILITY_COLLECTORS]:
        copied = dict(collector)
        counters = copied.get("counter_results", [])
        remaining = MAX_RELIABILITY_COUNTER_RESULTS - len(records)
        selected = counters[: max(remaining, 0)]
        records.extend(selected)
        copied["counter_results"] = selected
        copied["counter_count"] = len(counters)
        copied["counter_truncated"] = len(counters) > len(selected)
        collector_records.append(copied)
    return {
        **reliability,
        "collector_analyses": collector_records,
        "counter_count": len(records),
        "counter_truncated": any(
            collector.get("counter_truncated") for collector in collector_records
        ),
    }


def _empty_node_balance() -> dict[str, Any]:
    return {
        "rows": [],
        "traffic_share_formula": "nic_total_bytes / node_total_bytes",
        "effective_nic_count_formula": "1 / sum((nic_total_bytes / node_total_bytes)^2)",
        "active_nic_rule": "mean_rate_bytes_per_s > 0 bytes/s",
        "spatial_quantile_interpolation": "nearest-rank across NIC total bytes",
    }


def _empty_job_balance() -> dict[str, Any]:
    return {
        "directions": [],
        "traffic_share_formula": "node_total_bytes / job_total_bytes",
        "effective_node_count_formula": "1 / sum((node_total_bytes / job_total_bytes)^2)",
        "spatial_quantile_interpolation": (
            "nearest-rank across node total bytes; temporal P99s are not averaged"
        ),
        "topology_limitation": (
            "No topology mapping was supplied; only node-level results are available."
        ),
        "edge_limitation": (
            "Communication-edge metadata is required for peer or near/far claims."
        ),
    }


def _empty_phase_analysis() -> dict[str, Any]:
    return {
        "state": "missing_data",
        "status": "No phase selected",
        "status_reasons": ("No manual phase windows have been added.",),
        "phases": [],
        "changes": [],
        "interpretation_limits": (
            "Phase names and boundaries are user-provided and do not infer workload labels or causality."
        ),
    }


def _empty_reliability_analysis() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "question_id": "NIC-09",
        "state": "missing",
        "status": "Missing reliability evidence",
        "status_reasons": ("No node/NIC reliability counter series is available.",),
        "collector_analyses": [],
        "collector_count": 0,
        "collector_limit": MAX_RELIABILITY_COLLECTORS,
        "counter_limit": MAX_RELIABILITY_COUNTER_RESULTS,
        "collector_truncated": False,
        "counter_truncated": False,
        "limitations": (
            "Reliability counters are correlational evidence, not a causal diagnosis.",
            "Missing counters and samples are not zero-valued observations.",
            "Decrease or reset intervals are excluded from totals and rate denominators.",
        ),
    }


def _empty_flow_control_analysis() -> dict[str, Any]:
    return {
        "schema_version": FLOW_CONTROL_ANALYSIS_VERSION,
        "question_id": "NIC-02",
        "state": "missing",
        "status": "Missing flow-control evidence",
        "status_reasons": ("No node/NIC flow-control counter series is available.",),
        "collector_analyses": [],
        "collector_count": 0,
        "collector_limit": MAX_FLOW_CONTROL_COLLECTORS,
        "counter_limit": MAX_FLOW_CONTROL_COUNTER_RESULTS,
        "collector_truncated": False,
        "counter_truncated": False,
        "expected_priorities": DEFAULT_FLOW_CONTROL_PRIORITIES,
        "direction_source": "explicit flow-control counter catalog",
        "priority_source": "explicit flow-control counter catalog",
        "clock_conversion": "unsupported without verified clock evidence",
        "coverage_scope": (
            "Coverage is reported separately for each collector; no "
            "collector-merged or job-wide coverage value is reported."
        ),
        "limitations": FLOW_CONTROL_LIMITATIONS,
    }


def _export_flow_control_summary(flow_control: Any) -> dict[str, Any]:
    if not isinstance(flow_control, dict):
        return _empty_flow_control_analysis()
    collector_records = []
    counter_count = 0
    for collector in flow_control.get("collector_analyses", ())[
        :MAX_FLOW_CONTROL_COLLECTORS
    ]:
        copied = dict(collector)
        counters = copied.get("counter_results", [])
        remaining = MAX_FLOW_CONTROL_COUNTER_RESULTS - counter_count
        selected = counters[: max(remaining, 0)]
        counter_count += len(selected)
        copied["counter_results"] = selected
        copied["counter_count"] = len(counters)
        copied["counter_truncated"] = len(counters) > len(selected)
        collector_records.append(copied)
    return {
        **flow_control,
        "collector_analyses": collector_records,
        "counter_count": counter_count,
        "counter_truncated": (
            counter_count > MAX_FLOW_CONTROL_COUNTER_RESULTS
            or any(collector["counter_truncated"] for collector in collector_records)
        ),
    }


def _empty_matching_analysis() -> dict[str, Any]:
    return {
        "schema_version": MATCHING_ANALYSIS_VERSION,
        "question_id": "NIC-03",
        "state": "missing",
        "status": "Missing matching evidence",
        "status_reasons": ("No node/NIC matching counter series is available.",),
        "collector_analyses": [],
        "collector_count": 0,
        "collector_limit": MAX_MATCHING_COLLECTORS,
        "counter_limit": MAX_MATCHING_COUNTER_RESULTS,
        "collector_truncated": False,
        "counter_truncated": False,
        "priority_limit": MAX_MATCHING_PRIORITY_RESULTS,
        "priority_truncated": False,
        "collector_priority_fractions": [],
        "priority_fraction_scope": (
            "Collector-separated values; incompatible or heterogeneous collectors are not merged."
        ),
        "expected_counter_names": _MATCHING_COUNTERS,
        "population_id": _MATCHING_POPULATION_ID,
        "population_compatibility_source": (
            "Explicit catalog pairing of the four priority and overflow receive-match "
            "subcounters; duplicate collectors remain separate."
        ),
        "coverage_scope": (
            "Coverage is reported separately for each collector; no "
            "collector-merged or job-wide coverage value is reported."
        ),
        "interpretation": MATCHING_INTERPRETATION,
        "limitations": MATCHING_LIMITATIONS,
    }


def _export_matching_summary(matching: Any) -> dict[str, Any]:
    if not isinstance(matching, dict):
        return _empty_matching_analysis()
    collector_records = []
    counter_count = 0
    priority_count = 0
    for collector in matching.get("collector_analyses", ())[:MAX_MATCHING_COLLECTORS]:
        copied = dict(collector)
        counters = copied.get("counter_results", [])
        remaining = MAX_MATCHING_COUNTER_RESULTS - counter_count
        selected_counters = counters[: max(remaining, 0)]
        counter_count += len(selected_counters)
        priorities = copied.get("priority_results", [])
        remaining_priorities = MAX_MATCHING_PRIORITY_RESULTS - priority_count
        selected_priorities = priorities[: max(remaining_priorities, 0)]
        priority_count += len(selected_priorities)
        copied["counter_results"] = selected_counters
        copied["priority_results"] = selected_priorities
        copied["counter_count"] = len(counters)
        copied["priority_count"] = len(priorities)
        copied["counter_truncated"] = len(counters) > len(selected_counters)
        copied["priority_truncated"] = len(priorities) > len(selected_priorities)
        collector_records.append(copied)
    return {
        **matching,
        "collector_analyses": collector_records,
        "counter_count": counter_count,
        "priority_count": priority_count,
        "counter_truncated": (
            counter_count > MAX_MATCHING_COUNTER_RESULTS
            or any(collector["counter_truncated"] for collector in collector_records)
        ),
        "priority_truncated": (
            priority_count > MAX_MATCHING_PRIORITY_RESULTS
            or any(collector["priority_truncated"] for collector in collector_records)
        ),
    }


def _export_operation_summary(operation: Any) -> dict[str, Any]:
    if not isinstance(operation, dict):
        return _empty_operation_analysis()
    collector_records = []
    opcode_count = 0
    total_count = 0
    dma_bin_count = 0
    for collector in operation.get("collector_analyses", ())[:MAX_OPERATION_COLLECTORS]:
        copied = dict(collector)
        opcodes = copied.get("opcode_results", [])
        selected_opcodes = opcodes[: max(MAX_OPERATION_OPCODE_RESULTS - opcode_count, 0)]
        opcode_count += len(selected_opcodes)
        dma_bins = copied.get("dma_opcode_bin_results", ())
        selected_dma_bins = dma_bins[
            : max(MAX_OPERATION_OPCODE_RESULTS - dma_bin_count, 0)
        ]
        dma_bin_count += len(selected_dma_bins)
        dma_totals = copied.get("dma_total_results", ())
        idc_totals = copied.get("idc_total_results", ())
        available_totals = (*dma_totals, *idc_totals)
        selected_totals = available_totals[: max(MAX_OPERATION_TOTAL_RESULTS - total_count, 0)]
        total_count += len(selected_totals)
        dma_count = min(len(dma_totals), len(selected_totals))
        copied["opcode_results"] = selected_opcodes
        copied["dma_opcode_bin_results"] = selected_dma_bins
        copied["dma_total_results"] = selected_totals[:dma_count]
        copied["idc_total_results"] = selected_totals[dma_count:]
        copied["opcode_count"] = len(opcodes)
        copied["dma_opcode_bin_count"] = len(dma_bins)
        copied["total_count"] = len(available_totals)
        copied["opcode_truncated"] = len(opcodes) > len(selected_opcodes)
        copied["dma_opcode_bin_truncated"] = len(dma_bins) > len(selected_dma_bins)
        copied["total_truncated"] = len(available_totals) > len(selected_totals)
        collector_records.append(copied)
    return {
        **operation,
        "collector_analyses": collector_records,
        "opcode_count": opcode_count,
        "dma_opcode_bin_count": dma_bin_count,
        "total_count": total_count,
        "opcode_truncated": (
            opcode_count > MAX_OPERATION_OPCODE_RESULTS
            or any(collector["opcode_truncated"] for collector in collector_records)
        ),
        "dma_opcode_bin_truncated": (
            dma_bin_count > MAX_OPERATION_OPCODE_RESULTS
            or any(
                collector["dma_opcode_bin_truncated"]
                for collector in collector_records
            )
        ),
        "total_truncated": (
            total_count > MAX_OPERATION_TOTAL_RESULTS
            or any(collector["total_truncated"] for collector in collector_records)
        ),
    }


def _empty_frame_analysis() -> dict[str, Any]:
    return {
        "schema_version": FRAME_ANALYSIS_VERSION,
        "question_id": "NIC-06",
        "state": "missing",
        "status": "No wire-frame evidence",
        "status_reasons": (
            "No cataloged frame-bin counter series is available for this NIC.",
        ),
        "quality_flags": [],
        "collector_analyses": (),
        "collector_count": 0,
        "collector_limit": MAX_FRAME_COLLECTORS,
        "bin_limit": MAX_FRAME_BIN_RESULTS,
        "quantile_limit": MAX_FRAME_QUANTILE_RESULTS,
        "collector_truncated": False,
        "bin_truncated": False,
        "quantile_truncated": False,
        "duplicate_stream_count": 0,
        "requested_window_ns": None,
        "collector_coverage": (),
        "coverage_scope": (
            "Coverage is reported separately for each collector and direction; "
            "bins combine only on identical valid interval boundaries."
        ),
        "expected_counter_names": tuple(spec.counter_id for spec in _FRAME_SPECS),
        "missing_counter_ids": tuple(spec.counter_id for spec in _FRAME_SPECS),
        "unsupported_counter_names": (),
        "bin_edges_source": None,
        "bin_edges_complete": False,
        "population_compatibility": (),
        "interpretation": FRAME_INTERPRETATION,
        "limitations": FRAME_LIMITATIONS,
    }


def _export_frame_summary(frames: Any) -> dict[str, Any]:
    if not isinstance(frames, dict):
        return _empty_frame_analysis()
    collector_records = []
    bin_count = 0
    quantile_count = 0
    for collector in frames.get("collector_analyses", ())[:MAX_FRAME_COLLECTORS]:
        copied = dict(collector)
        directions = []
        for direction in copied.get("direction_results", ()):
            direction_copy = dict(direction)
            bins = direction_copy.get("bin_results", ())
            selected_bins = bins[: max(MAX_FRAME_BIN_RESULTS - bin_count, 0)]
            bin_count += len(selected_bins)
            quantiles = direction_copy.get("quantile_results", ())
            selected_quantiles = quantiles[
                : max(MAX_FRAME_QUANTILE_RESULTS - quantile_count, 0)
            ]
            quantile_count += len(selected_quantiles)
            direction_copy["bin_results"] = selected_bins
            direction_copy["quantile_results"] = selected_quantiles
            direction_copy["bin_count"] = len(bins)
            direction_copy["quantile_count"] = len(quantiles)
            direction_copy["bin_truncated"] = len(bins) > len(selected_bins)
            direction_copy["quantile_truncated"] = len(quantiles) > len(selected_quantiles)
            directions.append(direction_copy)
        copied["direction_results"] = directions
        collector_records.append(copied)
    return {
        **frames,
        "collector_analyses": collector_records,
        "bin_count": bin_count,
        "quantile_count": quantile_count,
        "bin_truncated": (
            bin_count > MAX_FRAME_BIN_RESULTS
            or any(
                direction["bin_truncated"]
                for record in collector_records
                for direction in record["direction_results"]
            )
        ),
        "quantile_truncated": (
            quantile_count > MAX_FRAME_QUANTILE_RESULTS
            or any(
                direction["quantile_truncated"]
                for record in collector_records
                for direction in record["direction_results"]
            )
        ),
    }


def _empty_translation_analysis() -> dict[str, Any]:
    return {
        "schema_version": TRANSLATION_ANALYSIS_VERSION,
        "question_id": "NIC-07",
        "state": "missing_data",
        "status": "No address-translation evidence",
        "status_reasons": (
            "No cataloged address-translation counter series is available for this NIC.",
        ),
        "quality_flags": [],
        "collector_analyses": (),
        "collector_count": 0,
        "collector_limit": MAX_TRANSLATION_COLLECTORS,
        "counter_limit": MAX_TRANSLATION_COUNTER_RESULTS,
        "view_limit": MAX_TRANSLATION_VIEW_RESULTS,
        "collector_truncated": False,
        "counter_truncated": False,
        "view_truncated": False,
        "duplicate_stream_count": 0,
        "requested_window_ns": None,
        "collector_coverage": (),
        "coverage_scope": (
            "Coverage is reported separately for each collector; request, hit, miss, "
            "and eviction roles combine only on exact common valid interval boundaries."
        ),
        "expected_counter_names": tuple(
            spec.counter_id for spec in _TRANSLATION_SPECS
        ),
        "missing_counter_ids": tuple(
            spec.counter_id for spec in _TRANSLATION_SPECS
        ),
        "unsupported_counter_names": (),
        "eviction_total": None,
        "eviction_totals": (),
        "eviction_status": "missing_data",
        "eviction_statuses": (),
        "eviction_zero_observed": False,
        "traffic_status": "not_supplied",
        "hit_ratio_formula": "hit_total / (hit_total + miss_total)",
        "misses_per_request_formula": "miss_total / request_total",
        "misses_per_gb_formula": "miss_total / traffic_total_gb",
        "delta_formula": "end_value - start_value",
        "rate_formula": (
            "sum(valid interval deltas) / sum(valid interval durations)"
        ),
        "limitations": TRANSLATION_LIMITATIONS,
    }


def _translation_quality_flags(flags: int) -> tuple[Any, ...]:
    mappings = {
        QualityFlag.REORDERED: TranslationQualityFlag.REORDERED,
        QualityFlag.NONMONOTONIC_TIMESTAMP: TranslationQualityFlag.CONFLICTING_TIMESTAMP,
        QualityFlag.DUPLICATE_REMOVED: TranslationQualityFlag.DUPLICATE,
        QualityFlag.BASELINE_RELATIVE: TranslationQualityFlag.BASELINE_RELATIVE,
        QualityFlag.PRECISION_LOSS_POSSIBLE: TranslationQualityFlag.UNSUPPORTED_SERIES,
        QualityFlag.RESET_SUSPECT: TranslationQualityFlag.DECREASE_OR_RESET,
        QualityFlag.WRAP_SUSPECT: TranslationQualityFlag.DECREASE_OR_RESET,
        QualityFlag.GAP: TranslationQualityFlag.PARTIAL_WINDOW,
        QualityFlag.UNKNOWN_METADATA: TranslationQualityFlag.UNSUPPORTED_SERIES,
    }
    return tuple(
        dict.fromkeys(
            mapped
            for source_flag, mapped in mappings.items()
            if int(flags) & source_flag
        )
    )


def _translation_series(
    result: Any, spec: TranslationCounterSpec, source_unit: str
) -> TranslationCounterSeries | None:
    values: list[int | float] = []
    exact_uint64 = True
    for value in result.values:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            if not math.isfinite(value) or not value.is_integer():
                exact_uint64 = False
                values.append(value)
                continue
            value = int(value)
        if type(value) is not int or not 0 <= value <= 2**64 - 1:
            exact_uint64 = False
        values.append(value)
    quality_flags = set(_translation_quality_flags(int(result.counter.quality_flags)))
    quality_flags.update(
        _translation_quality_flags(int(np.bitwise_or.reduce(result.quality_flags)))
    )
    if not exact_uint64:
        quality_flags.add(TranslationQualityFlag.UNSUPPORTED_SERIES)
    collector = result.collector
    return TranslationCounterSeries(
        spec=spec,
        identity=TranslationCounterIdentity(
            run_id=result.run_identity.run_id,
            node_id=str(result.node.node_id),
            nic_id=str(result.nic.nic_id),
            collector_id=f"{collector.name} (ID {collector.collector_id})",
            source_counter_id=result.counter.name,
        ),
        timestamps_ns=tuple(int(timestamp) for timestamp in result.timestamps_ns),
        values=tuple(values),
        value_kind=(
            TranslationValueKind.CUMULATIVE
            if result.counter.value_kind is ValueKind.EXACT_UINT64
            and result.counter.semantics is CounterSemantics.CUMULATIVE
            and source_unit == spec.unit
            and exact_uint64
            else TranslationValueKind.UNKNOWN
        ),
        semantics=(
            result.counter.semantics.value
            if result.counter.semantics is not CounterSemantics.CUMULATIVE
            else "cumulative counter"
        ),
        baseline_relative=result.counter.baseline_relative,
        quality_flags=tuple(quality_flags),
    )


def _translation_rate(total: int | None, covered_duration_ns: int) -> float | None:
    if total is None or covered_duration_ns <= 0:
        return None
    return _finite_float(total / (covered_duration_ns / 1_000_000_000))


def _translation_counter_record(result: Any) -> dict[str, Any]:
    identity = result.source_identity
    return {
        "counter": result.spec.counter_id,
        "role": result.spec.role.value,
        "miss_view": result.spec.miss_view.value if result.spec.miss_view else None,
        "state": result.status.value,
        "status": result.status.value.replace("_", " ").title(),
        "status_reasons": result.status_reasons,
        "quality_flags": sorted(flag.value for flag in result.quality_flags),
        "run_id": identity.run_id if identity else None,
        "node_id": int(identity.node_id) if identity else None,
        "nic_id": int(identity.nic_id) if identity else None,
        "collector_id": identity.collector_id if identity else None,
        "source_counter_id": (
            identity.source_counter_id if identity else result.spec.counter_id
        ),
        "requested_window_ns": result.requested_window_ns,
        "represented_window_ns": result.represented_window_ns,
        "requested_duration_ns": result.requested_duration_ns,
        "covered_duration_ns": result.covered_duration_ns,
        "coverage_fraction": _finite_float(result.coverage_fraction),
        "sample_count": result.sample_count,
        "valid_interval_count": result.valid_interval_count,
        "invalid_interval_count": result.invalid_interval_count,
        "outside_window_interval_count": result.outside_window_interval_count,
        "total": result.total,
        "unit": result.unit,
        "rate_per_second": _translation_rate(
            result.total, result.covered_duration_ns
        ),
        "zero_observed": result.zero_observed,
        "persistent_zero": result.persistent_zero,
    }


def _translation_view_record(result: Any) -> dict[str, Any]:
    return {
        "miss_view": result.miss_view.value,
        "state": result.status.value,
        "status": result.status.value.replace("_", " ").title(),
        "status_reasons": result.status_reasons,
        "quality_flags": sorted(flag.value for flag in result.quality_flags),
        "request_total": result.request_total,
        "hit_total": result.hit_total,
        "miss_total": result.miss_total,
        "hit_ratio": _finite_float(result.hit_ratio),
        "hit_denominator": result.hit_denominator,
        "misses_per_request": _finite_float(result.misses_per_request),
        "request_denominator": result.request_denominator,
        "misses_per_gb": _finite_float(result.misses_per_gb),
        "traffic_total_gb": _finite_float(result.traffic_total_gb),
        "traffic_direction": (
            result.traffic_direction.value if result.traffic_direction else None
        ),
        "traffic_status": result.traffic_status.value,
        "covered_duration_ns": result.covered_duration_ns,
        "represented_window_ns": result.represented_window_ns,
        "coverage_fraction": _finite_float(result.coverage_fraction),
        "zero_observed": result.zero_observed,
        "persistent_zero": result.persistent_zero,
        "request_counter_ids": result.request_counter_ids,
        "hit_counter_ids": result.hit_counter_ids,
        "miss_counter_ids": result.miss_counter_ids,
        "missing_counter_ids": result.missing_counter_ids,
    }


def _translation_collector_record(
    analysis: Any, collector: Any, traffic: Any
) -> dict[str, Any]:
    counter_results = analysis.counter_results[:MAX_TRANSLATION_COUNTER_RESULTS]
    view_results = analysis.view_results[:MAX_TRANSLATION_VIEW_RESULTS]
    return {
        "schema_version": TRANSLATION_ANALYSIS_VERSION,
        "state": analysis.status.value,
        "status": analysis.status.value.replace("_", " ").title(),
        "status_reasons": analysis.status_reasons,
        "quality_flags": sorted(flag.value for flag in analysis.quality_flags),
        "population": {
            "run_id": analysis.population.run_id,
            "node_id": int(analysis.population.node_id),
            "nic_id": int(analysis.population.nic_id),
            "collector_id": analysis.population.collector_id,
            "collector_name": collector.name,
            "collector_registry_id": collector.collector_id,
            "collector_rank": collector.rank,
            "collector_process": collector.process,
            "collector_type": collector.collector_type,
        },
        "requested_window_ns": analysis.requested_window_ns,
        "represented_window_ns": analysis.represented_window_ns,
        "covered_duration_ns": analysis.covered_duration_ns,
        "coverage_fraction": _finite_float(analysis.coverage_fraction),
        "counter_results": [
            _translation_counter_record(result) for result in counter_results
        ],
        "counter_count": len(analysis.counter_results),
        "view_results": [
            _translation_view_record(result) for result in view_results
        ],
        "view_count": len(analysis.view_results),
        "eviction_total": analysis.eviction_total,
        "eviction_status": analysis.eviction_status.value,
        "eviction_zero_observed": analysis.eviction_zero_observed,
        "traffic": traffic,
    }


def _translation_traffic_total(
    store: TraceStore,
    node: Any,
    nic: Any,
    collector: Any,
    start_ns: int,
    end_ns: int,
    direction: Any,
) -> TranslationTrafficTotal | None:
    totals: list[int | float] = []
    source_counter_ids: list[str] = []
    windows: list[tuple[int, int]] = []
    for selected_direction in _directions(direction):
        counter, _ = _selected_counter(store.counters, selected_direction)
        if counter is None:
            continue
        try:
            query_result = store.query_series(
                node,
                nic,
                counter,
                start_ns=start_ns,
                end_ns=end_ns,
            )
        except TraceValidationError:
            continue
        matching = [
            result
            for result in query_result.results
            if result.collector.collector_id == collector.collector_id
        ]
        if len(matching) != 1:
            return None
        metric_series = metric_series_from_series_result(
            matching[0],
            run_id=store.identity.run_id,
            direction=selected_direction,
        )
        traffic_analysis = analyze_traffic_series(
            metric_series,
            window_start_ns=start_ns,
            window_end_ns=end_ns,
            include_intervals=False,
        )
        if traffic_analysis.total_bytes is None:
            return None
        totals.append(traffic_analysis.total_bytes)
        source_counter_ids.append(counter.name)
        if traffic_analysis.represented_window_ns is not None:
            windows.append(traffic_analysis.represented_window_ns)
    if not totals:
        return None
    if not windows or len(windows) != len(totals):
        return None
    represented_window = (
        min(start for start, _ in windows),
        max(end for _, end in windows),
    )
    if represented_window[0] >= represented_window[1]:
        return None
    direction_value = (
        _directions(direction)[0].value.lower()
        if len(_directions(direction)) == 1
        else "both"
    )
    return TranslationTrafficTotal(
        total=sum(totals),
        unit="bytes",
        direction=TranslationTrafficDirection(direction_value),
        population_id=_TRANSLATION_POPULATION_ID,
        window_ns=represented_window,
        source_counter_ids=tuple(source_counter_ids),
    )


def _build_translation_analysis(
    store: TraceStore,
    node: Any,
    nic: Any,
    start_ns: int,
    end_ns: int,
    direction: Any,
) -> dict[str, Any]:
    counter_by_name = {counter.name: counter for counter in store.counters}
    series_by_collector: dict[int, list[TranslationCounterSeries]] = {}
    collectors: dict[int, Any] = {}
    unsupported_by_collector: dict[int, set[str]] = {}
    duplicate_stream_count = 0
    for spec in _TRANSLATION_SPECS:
        counter = counter_by_name.get(spec.counter_id)
        if counter is None:
            continue
        try:
            query_result = store.query_series(
                node, nic, counter, start_ns=start_ns, end_ns=end_ns
            )
        except TraceValidationError:
            continue
        for result in query_result.results:
            collector_id = result.collector.collector_id
            collectors[collector_id] = result.collector
            duplicate_stream_count = max(
                duplicate_stream_count, result.observation_stream_count
            )
            series = _translation_series(result, spec, counter.unit)
            if series is None:
                unsupported_by_collector.setdefault(collector_id, set()).add(
                    spec.counter_id
                )
                continue
            series_by_collector.setdefault(collector_id, []).append(series)

    collector_ids = sorted(series_by_collector)
    records = []
    for collector_id in collector_ids[:MAX_TRANSLATION_COLLECTORS]:
        collector = collectors[collector_id]
        analysis = analyze_translation(
            TranslationDataset(
                population=TranslationPopulation(
                    run_id=store.identity.run_id,
                    node_id=str(node.node_id),
                    nic_id=str(nic.nic_id),
                    collector_id=series_by_collector[collector_id][0].identity.collector_id,
                ),
                requested_counters=_TRANSLATION_SPECS,
                series=tuple(series_by_collector[collector_id]),
                window_ns=(start_ns, end_ns - 1),
                traffic_total=_translation_traffic_total(
                    store,
                    node,
                    nic,
                    collector,
                    start_ns,
                    end_ns,
                    direction,
                ),
            )
        )
        records.append(
            _translation_collector_record(
                analysis,
                collector,
                analysis.view_results[0].traffic_total_gb
                if analysis.view_results
                else None,
            )
        )
    if not records:
        unsupported_names = {
            name for names in unsupported_by_collector.values() for name in names
        }
        return {
            **_empty_translation_analysis(),
            "status": (
                "Unsupported series" if unsupported_names else "No address-translation evidence"
            ),
            "status_reasons": (
                "One or more translation series have unsupported precision, value type, semantics, or unit."
                if unsupported_names
                else "No cataloged address-translation counter series is available for this NIC."
            ),
            "requested_window_ns": (start_ns, end_ns - 1),
            "missing_counter_ids": tuple(
                spec.counter_id
                for spec in _TRANSLATION_SPECS
                if counter_by_name.get(spec.counter_id) is None
            ),
            "unsupported_counter_names": tuple(sorted(unsupported_names)),
            "duplicate_stream_count": duplicate_stream_count,
        }

    states = {record["state"] for record in records}
    state = next(iter(states)) if len(states) == 1 else "partial"
    counter_count = sum(
        len(collector["counter_results"]) for collector in records
    )
    view_count = sum(len(collector["view_results"]) for collector in records)
    analyzer_unsupported_names = {
        result["counter"]
        for record in records
        for result in record["counter_results"]
        if result["state"] == "unsupported"
    }
    return {
        "schema_version": TRANSLATION_ANALYSIS_VERSION,
        "question_id": "NIC-07",
        "state": state,
        "status": state.replace("_", " ").title(),
        "status_reasons": tuple(
            dict.fromkeys(
                reason
                for record in records
                for reason in record["status_reasons"]
            )
        ),
        "quality_flags": sorted(
            {flag for record in records for flag in record["quality_flags"]}
        ),
        "collector_analyses": records,
        "collector_count": len(series_by_collector),
        "collector_limit": MAX_TRANSLATION_COLLECTORS,
        "counter_limit": MAX_TRANSLATION_COUNTER_RESULTS,
        "view_limit": MAX_TRANSLATION_VIEW_RESULTS,
        "collector_truncated": len(series_by_collector) > MAX_TRANSLATION_COLLECTORS,
        "counter_truncated": counter_count > MAX_TRANSLATION_COUNTER_RESULTS,
        "view_truncated": view_count > MAX_TRANSLATION_VIEW_RESULTS,
        "duplicate_stream_count": duplicate_stream_count,
        "requested_window_ns": records[0]["requested_window_ns"],
        "collector_coverage": [
            {
                "collector_id": record["population"]["collector_id"],
                "collector_registry_id": record["population"]["collector_registry_id"],
                "covered_duration_ns": record["covered_duration_ns"],
                "coverage_fraction": record["coverage_fraction"],
            }
            for record in records
        ],
        "coverage_scope": (
            "Coverage is reported separately for each collector; request, hit, miss, "
            "and eviction roles combine only on exact common valid interval boundaries."
        ),
        "expected_counter_names": tuple(
            spec.counter_id for spec in _TRANSLATION_SPECS
        ),
        "missing_counter_ids": tuple(
            dict.fromkeys(
                counter_id
                for record in records
                for result in record["counter_results"]
                if result["state"] == "missing_data"
                for counter_id in (result["counter"],)
            )
        ),
        "unsupported_counter_names": tuple(
            dict.fromkeys(
                (
                    *analyzer_unsupported_names,
                *(
                    name
                    for collector_id, names in unsupported_by_collector.items()
                    if collector_id in {
                        record["population"]["collector_registry_id"]
                        for record in records
                    }
                    for name in names
                ),
                ),
            )
        ),
        "eviction_total": (
            records[0]["eviction_total"] if len(records) == 1 else None
        ),
        "eviction_totals": tuple(
            record["eviction_total"] for record in records
        ),
        "eviction_status": (
            records[0]["eviction_status"]
            if len({record["eviction_status"] for record in records}) == 1
            else "partial"
        ),
        "eviction_statuses": tuple(
            record["eviction_status"] for record in records
        ),
        "eviction_zero_observed": all(
            record["eviction_zero_observed"] for record in records
        ),
        "traffic_status": next(
            (
                view["traffic_status"]
                for record in records
                for view in record["view_results"]
                if view["traffic_status"] != "not_supplied"
            ),
            "not_supplied",
        ),
        "hit_ratio_formula": "hit_total / (hit_total + miss_total)",
        "misses_per_request_formula": "miss_total / request_total",
        "misses_per_gb_formula": "miss_total / traffic_total_gb",
        "delta_formula": "end_value - start_value",
        "rate_formula": (
            "sum(valid interval deltas) / sum(valid interval durations)"
        ),
        "limitations": TRANSLATION_LIMITATIONS,
    }


def _export_translation_summary(translation: Any) -> dict[str, Any]:
    if not isinstance(translation, dict):
        return _empty_translation_analysis()
    collector_records = []
    counter_count = 0
    view_count = 0
    for collector in translation.get("collector_analyses", ())[
        :MAX_TRANSLATION_COLLECTORS
    ]:
        copied = dict(collector)
        counters = copied.get("counter_results", ())
        selected_counters = counters[
            : max(MAX_TRANSLATION_COUNTER_RESULTS - counter_count, 0)
        ]
        counter_count += len(selected_counters)
        views = copied.get("view_results", ())
        selected_views = views[
            : max(MAX_TRANSLATION_VIEW_RESULTS - view_count, 0)
        ]
        view_count += len(selected_views)
        copied["counter_results"] = selected_counters
        copied["view_results"] = selected_views
        copied["counter_count"] = len(counters)
        copied["view_count"] = len(views)
        copied["counter_truncated"] = len(counters) > len(selected_counters)
        copied["view_truncated"] = len(views) > len(selected_views)
        collector_records.append(copied)
    return {
        **translation,
        "collector_analyses": collector_records,
        "counter_count": counter_count,
        "view_count": view_count,
        "counter_truncated": (
            counter_count > MAX_TRANSLATION_COUNTER_RESULTS
            or any(
                collector["counter_truncated"]
                for collector in collector_records
            )
        ),
        "view_truncated": (
            view_count > MAX_TRANSLATION_VIEW_RESULTS
            or any(
                collector["view_truncated"] for collector in collector_records
            )
        ),
    }


def _empty_latency_analysis() -> dict[str, Any]:
    return {
        "schema_version": LATENCY_ANALYSIS_VERSION,
        "question_id": "NIC-08",
        "state": "missing",
        "status": "No latency-histogram evidence",
        "status_reasons": (
            "No cataloged latency-histogram counter series is available for this NIC.",
        ),
        "quality_flags": [],
        "collector_analyses": (),
        "collector_count": 0,
        "collector_limit": MAX_LATENCY_COLLECTORS,
        "bin_limit": MAX_LATENCY_BIN_RESULTS,
        "quantile_limit": MAX_LATENCY_QUANTILE_RESULTS,
        "collector_truncated": False,
        "bin_truncated": False,
        "quantile_truncated": False,
        "duplicate_stream_count": 0,
        "requested_window_ns": None,
        "collector_coverage": (),
        "coverage_scope": (
            "Coverage is reported separately for each collector and histogram group; "
            "groups are not merged into a job-wide latency distribution."
        ),
        "expected_counter_names": _LATENCY_COUNTERS,
        "missing_counter_ids": _LATENCY_COUNTERS,
        "unsupported_counter_names": (),
        "bin_metadata_status": "unsourced",
        "bin_definition_sources": tuple(
            definition.source for definition in _LATENCY_BIN_DEFINITIONS.values()
        ),
        "interpretation": LATENCY_INTERPRETATION,
        "limitations": LATENCY_LIMITATIONS,
    }


def _latency_bin_record(result: Any) -> dict[str, Any]:
    return {
        "bin": result.bin_name,
        "bin_index": result.bin_index,
        "source_counter": result.source_counter,
        "count": result.count,
        "count_kind": result.count_kind.value if result.count_kind else None,
        "normalized_fraction": _finite_float(result.normalized_fraction),
        "lower_edge": _finite_float(result.lower_edge),
        "upper_edge": _finite_float(result.upper_edge),
        "open_tailed": result.open_tailed,
    }


def _latency_quantile_record(result: Any) -> dict[str, Any]:
    return {
        "quantile": result.quantile,
        "state": result.status.value,
        "status": result.status.value.replace("_", " ").title(),
        "bin": result.bin_name,
        "lower_edge": _finite_float(result.lower_edge),
        "upper_edge": _finite_float(result.upper_edge),
        "reason": result.reason,
    }


def _latency_collector_record(
    analysis: Any, collector: Any
) -> dict[str, Any]:
    bin_results = analysis.bin_results[:MAX_LATENCY_BIN_RESULTS]
    quantile_results = analysis.quantile_results[:MAX_LATENCY_QUANTILE_RESULTS]
    return {
        "schema_version": LATENCY_ANALYSIS_VERSION,
        "state": analysis.status.value,
        "status": analysis.status.value.replace("_", " ").title(),
        "status_reasons": analysis.status_reasons,
        "quality_flags": sorted(flag.value for flag in analysis.quality_flags),
        "population": {
            "run_id": analysis.population.run_id,
            "node_id": analysis.population.node_id,
            "nic_id": analysis.population.nic_id,
            "collector_id": analysis.population.collector_id,
            "collector_name": collector.name,
            "collector_registry_id": collector.collector_id,
            "collector_rank": collector.rank,
            "collector_process": collector.process,
            "collector_type": collector.collector_type,
            "direction": analysis.population.direction,
        },
        "histogram_group": analysis.group.value,
        "requested_window_ns": analysis.requested_window_ns,
        "represented_window_ns": analysis.represented_window_ns,
        "covered_duration_ns": analysis.covered_duration_ns,
        "coverage_fraction": _finite_float(analysis.coverage_fraction),
        "bin_definition_version": analysis.bin_definition_version,
        "bin_definition_source": analysis.bin_definition_source,
        "encoding": analysis.encoding.value,
        "bins_are_disjoint": analysis.bins_are_disjoint,
        "bin_unit": analysis.bin_unit,
        "bin_results": [_latency_bin_record(result) for result in bin_results],
        "bin_count": len(analysis.bin_results),
        "bin_truncated": len(analysis.bin_results) > len(bin_results),
        "total_observations": analysis.total_observations,
        "normalized_distribution_available": (
            analysis.normalized_distribution_available
        ),
        "quantile_results": [
            _latency_quantile_record(result) for result in quantile_results
        ],
        "quantile_count": len(analysis.quantile_results),
        "quantile_truncated": len(analysis.quantile_results) > len(quantile_results),
        "mean_result": {
            "state": analysis.mean_result.status.value,
            "status": analysis.mean_result.status.value.replace("_", " ").title(),
            "value": _finite_float(analysis.mean_result.value),
            "unit": analysis.mean_result.unit,
            "formula": analysis.mean_result.formula,
            "reason": analysis.mean_result.reason,
        },
        "sample_count": analysis.sample_count,
        "interval_count": analysis.interval_count,
        "valid_interval_count": analysis.valid_interval_count,
        "invalid_interval_count": analysis.invalid_interval_count,
        "reset_excluded_interval_count": analysis.reset_excluded_interval_count,
        "boundary_excluded_interval_count": (
            analysis.boundary_excluded_interval_count
        ),
        "delta_formula": analysis.delta_formula,
        "boundary_decumulation_formula": analysis.boundary_decumulation_formula,
        "normalized_fraction_formula": analysis.normalized_fraction_formula,
        "quantile_selection_formula": analysis.quantile_selection_formula,
        "mean_estimate_formula": analysis.mean_estimate_formula,
        "atu_latency_substitute": analysis.atu_latency_substitute,
        "interpretation": analysis.interpretation,
        "limitations": analysis.limitations,
    }


def _build_latency_analysis(
    store: TraceStore,
    node: Any,
    nic: Any,
    start_ns: int,
    end_ns: int,
) -> dict[str, Any]:
    counter_by_name = {counter.name: counter for counter in store.counters}
    series_by_collector: dict[int, list[MetricSeries]] = {}
    collectors: dict[int, Any] = {}
    unsupported_by_collector: dict[int, set[str]] = {}
    duplicate_stream_count = 0
    for counter_name in _LATENCY_COUNTERS:
        counter = counter_by_name.get(counter_name)
        if counter is None:
            continue
        try:
            query_result = store.query_series(
                node, nic, counter, start_ns=start_ns, end_ns=end_ns
            )
        except TraceValidationError:
            continue
        for result in query_result.results:
            collector_id = result.collector.collector_id
            collectors[collector_id] = result.collector
            duplicate_stream_count = max(
                duplicate_stream_count, result.observation_stream_count
            )
            if result.counter.semantics is not CounterSemantics.CUMULATIVE:
                unsupported_by_collector.setdefault(collector_id, set()).add(
                    counter_name
                )
            series_by_collector.setdefault(collector_id, []).append(
                metric_series_from_series_result(
                    result,
                    run_id=store.identity.run_id,
                    direction=TrafficDirection.TX,
                )
            )

    selected_collector_ids = sorted(series_by_collector)[:MAX_LATENCY_COLLECTORS]
    records = [
        _latency_collector_record(analysis, collectors[collector_id])
        for collector_id in selected_collector_ids
        for analysis in analyze_latency_histograms(
            tuple(series_by_collector[collector_id]),
            _LATENCY_BIN_DEFINITIONS,
            requested_window_ns=(start_ns, end_ns - 1),
        )
    ]
    if not records:
        unsupported_names = {
            name for names in unsupported_by_collector.values() for name in names
        }
        return {
            **_empty_latency_analysis(),
            "status": (
                "Unsupported series" if unsupported_names else "No latency-histogram evidence"
            ),
            "status_reasons": (
                "One or more latency series are not cumulative counters."
                if unsupported_names
                else "No cataloged latency-histogram counter series is available for this NIC."
            ),
            "requested_window_ns": (start_ns, end_ns - 1),
            "missing_counter_ids": tuple(
                name for name in _LATENCY_COUNTERS if counter_by_name.get(name) is None
            ),
            "unsupported_counter_names": tuple(sorted(unsupported_names)),
            "duplicate_stream_count": duplicate_stream_count,
        }

    states = {record["state"] for record in records}
    state = next(iter(states)) if len(states) == 1 else "partial"
    bin_count = sum(record["bin_count"] for record in records)
    quantile_count = sum(record["quantile_count"] for record in records)
    metadata_supported = all(
        definition.bin_edges is not None
        and definition.unit is not None
        and definition.encoding is not LatencyBinEncoding.UNKNOWN
        for definition in _LATENCY_BIN_DEFINITIONS.values()
    )
    return {
        "schema_version": LATENCY_ANALYSIS_VERSION,
        "question_id": "NIC-08",
        "state": state,
        "status": state.replace("_", " ").title(),
        "status_reasons": tuple(
            dict.fromkeys(
                reason
                for record in records
                for reason in record["status_reasons"]
            )
        ),
        "quality_flags": sorted(
            {flag for record in records for flag in record["quality_flags"]}
        ),
        "collector_analyses": records,
        "collector_count": len(series_by_collector),
        "collector_limit": MAX_LATENCY_COLLECTORS,
        "collector_truncated": len(series_by_collector) > MAX_LATENCY_COLLECTORS,
        "bin_limit": MAX_LATENCY_BIN_RESULTS,
        "quantile_limit": MAX_LATENCY_QUANTILE_RESULTS,
        "bin_count": bin_count,
        "quantile_count": quantile_count,
        "bin_truncated": any(record["bin_truncated"] for record in records),
        "quantile_truncated": any(
            record["quantile_truncated"] for record in records
        ),
        "duplicate_stream_count": duplicate_stream_count,
        "requested_window_ns": records[0]["requested_window_ns"],
        "collector_coverage": [
            {
                "collector_id": record["population"]["collector_id"],
                "collector_registry_id": record["population"][
                    "collector_registry_id"
                ],
                "histogram_group": record["histogram_group"],
                "covered_duration_ns": record["covered_duration_ns"],
                "coverage_fraction": record["coverage_fraction"],
            }
            for record in records
        ],
        "coverage_scope": (
            "Coverage is reported separately for each collector and histogram group; "
            "groups are not merged into a job-wide latency distribution."
        ),
        "expected_counter_names": _LATENCY_COUNTERS,
        "missing_counter_ids": tuple(
            name for name in _LATENCY_COUNTERS if counter_by_name.get(name) is None
        ),
        "unsupported_counter_names": tuple(
            dict.fromkeys(
                name
                for collector_id, names in unsupported_by_collector.items()
                if collector_id in selected_collector_ids
                for name in names
            )
        ),
        "bin_metadata_status": (
            "sourced" if metadata_supported else "unsourced"
        ),
        "bin_definition_sources": tuple(
            definition.source for definition in _LATENCY_BIN_DEFINITIONS.values()
        ),
        "interpretation": LATENCY_INTERPRETATION,
        "delta_formula": records[0]["delta_formula"],
        "quantile_selection_formula": records[0]["quantile_selection_formula"],
        "mean_estimate_formula": records[0]["mean_estimate_formula"],
        "limitations": LATENCY_LIMITATIONS,
    }


def _export_latency_summary(latency: Any) -> dict[str, Any]:
    if not isinstance(latency, dict):
        return _empty_latency_analysis()
    collector_records = []
    bin_count = 0
    quantile_count = 0
    for collector in latency.get("collector_analyses", ())[
        :MAX_LATENCY_COLLECTORS
    ]:
        copied = dict(collector)
        bins = copied.get("bin_results", ())
        selected_bins = bins[: max(MAX_LATENCY_BIN_RESULTS - bin_count, 0)]
        bin_count += len(selected_bins)
        quantiles = copied.get("quantile_results", ())
        selected_quantiles = quantiles[
            : max(MAX_LATENCY_QUANTILE_RESULTS - quantile_count, 0)
        ]
        quantile_count += len(selected_quantiles)
        copied["bin_results"] = selected_bins
        copied["quantile_results"] = selected_quantiles
        copied["bin_count"] = len(bins)
        copied["quantile_count"] = len(quantiles)
        copied["bin_truncated"] = len(bins) > len(selected_bins)
        copied["quantile_truncated"] = len(quantiles) > len(selected_quantiles)
        collector_records.append(copied)
    return {
        **latency,
        "collector_analyses": collector_records,
        "bin_count": bin_count,
        "quantile_count": quantile_count,
        "bin_truncated": (
            bin_count > MAX_LATENCY_BIN_RESULTS
            or any(collector["bin_truncated"] for collector in collector_records)
        ),
        "quantile_truncated": (
            quantile_count > MAX_LATENCY_QUANTILE_RESULTS
            or any(
                collector["quantile_truncated"]
                for collector in collector_records
            )
        ),
    }


def _empty_job_profile() -> dict[str, Any]:
    return {
        "version": 1,
        "status": "unavailable",
        "summary": "No filtered view-model evidence is available for a job profile.",
        "dimensions": [],
        "questions": [],
        "evidence": [],
        "observed_summary_count": 0,
        "zero_observation_count": 0,
        "missing_observation_count": 0,
        "coverage_fraction": None,
        "supported_dimensions": (),
        "partial_dimensions": (),
        "unsupported_dimensions": (),
        "topology_status": "unsupported",
        "topology_reasons": ("No topology mapping was supplied.",),
        "communication_edge_status": "unsupported",
        "communication_edge_reasons": (
            "Communication-edge metadata was not supplied.",
        ),
        "formulas": (),
        "interpretation_limits": (
            "No job-profile claim is supported without filtered evidence.",
        ),
    }


def _job_state(status: Any) -> str:
    status_name = getattr(status, "value", str(status))
    return {
        "ok": "ready",
        "zero_traffic": "zero",
        "partial_data": "partial",
        "missing_data": "missing",
        "unsupported_data": "unsupported",
    }.get(status_name, "unsupported")


def _job_balance_record(result: Any) -> dict[str, Any]:
    per_node = [
        {
            "node": item.node_id,
            "state": item.status.value,
            "state_flags": sorted(flag.value for flag in item.state_flags),
            "status_reasons": item.status_reasons,
            "total_bytes": item.total_bytes,
            "traffic_share": _finite_float(item.traffic_share),
            "known_nic_count": item.known_nic_count,
            "observed_nic_count": item.observed_nic_count,
            "active_nic_fraction": _finite_float(item.active_nic_fraction),
            "coverage_fraction": _finite_float(item.coverage_fraction),
            "topology_group_id": item.topology_group_id,
        }
        for item in result.per_node
    ]
    return {
        "direction": result.direction.value if result.direction else None,
        "state": _job_state(result.status),
        "status": result.status.value.replace("_", " ").title(),
        "status_reasons": result.status_reasons,
        "state_flags": sorted(flag.value for flag in result.state_flags),
        "source_counter": result.source_counter,
        "common_window_ns": result.common_window_ns,
        "job_total_bytes": result.job_total_bytes,
        "known_node_count": result.known_node_count,
        "observed_node_count": result.observed_node_count,
        "valid_node_count": result.valid_node_count,
        "missing_node_ids": result.missing_node_ids,
        "unknown_observed_node_ids": result.unknown_observed_node_ids,
        "unavailable_node_ids": result.unavailable_node_ids,
        "partial_node_ids": result.partial_node_ids,
        "unsupported_node_ids": result.unsupported_node_ids,
        "max_traffic_bytes": result.max_traffic_bytes,
        "min_traffic_bytes": result.min_traffic_bytes,
        "max_min_ratio": _finite_float(result.max_min_ratio),
        "effective_node_count": _finite_float(result.effective_node_count),
        "spatial_p50_bytes": result.spatial_p50_bytes,
        "spatial_p99_bytes": result.spatial_p99_bytes,
        "hotspot_concentration": _finite_float(result.hotspot_concentration),
        "hotspot_node_count": result.hotspot_node_count,
        "coverage_fraction": _finite_float(result.coverage_fraction),
        "valid_interval_count": result.valid_interval_count,
        "invalid_interval_count": result.invalid_interval_count,
        "baseline_relative_observation_count": result.baseline_relative_observation_count,
        "topology_state": result.topology_state.value,
        "topology_status_reasons": result.topology_status_reasons,
        "topology_supported": result.topology_supported,
        "unknown_topology_node_ids": result.unknown_topology_node_ids,
        "peer_near_far_supported": result.peer_near_far_supported,
        "peer_near_far_reason": result.peer_near_far_reason,
        "unsupported_dimensions": result.unsupported_dimensions,
        "per_node": per_node[:MAX_NODE_COMPARISON_ROWS],
        "per_node_truncated": len(per_node) > MAX_NODE_COMPARISON_ROWS,
        "traffic_share_formula": result.traffic_share_formula,
        "max_min_ratio_formula": result.max_min_ratio_formula,
        "effective_node_count_formula": result.effective_node_count_formula,
        "spatial_quantile_interpolation": result.spatial_quantile_interpolation,
        "hotspot_concentration_formula": result.hotspot_concentration_formula,
        "coverage_aggregation": result.coverage_aggregation,
    }


def _phase_metric_record(metric: Any) -> dict[str, Any]:
    quality_flags = sorted(flag.value for flag in metric.quality_flags)
    return {
        "name": metric.phase_name,
        "window_ns": metric.window_ns,
        "state": metric.status.value,
        "state_flags": sorted(flag.value for flag in metric.state_flags),
        "status_reasons": metric.status_reasons,
        "quality_flags": quality_flags,
        "reset_observed": "decrease_or_reset" in quality_flags,
        "source_quality_flags": sorted(
            flag.value for flag in metric.source_quality_flags
        ),
        "source_counter": metric.source_counter,
        "total_bytes": metric.total_bytes,
        "covered_duration_ns": metric.covered_duration_ns,
        "requested_duration_ns": metric.requested_duration_ns,
        "uncovered_duration_ns": metric.uncovered_duration_ns,
        "coverage_fraction": _finite_float(metric.coverage_fraction),
        "mean_rate_bytes_per_s": _finite_float(metric.mean_rate_bytes_per_s),
        "valid_interval_count": metric.valid_interval_count,
        "invalid_interval_count": metric.invalid_interval_count,
        "boundary_excluded_interval_count": metric.boundary_excluded_interval_count,
        "represented_window_ns": metric.represented_window_ns,
        "baseline_relative": metric.baseline_relative,
    }


def _phase_analysis_record(result: ManualPhaseAnalysisResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "node_id": result.node_id,
        "nic_id": result.nic_id,
        "direction": result.direction.value,
        "source_counter": result.source_counter,
        "unit": result.unit,
        "rate_unit": result.rate_unit,
        "clock_id": result.clock_id,
        "common_window_ns": result.common_window_ns,
        "gap_policy": result.gap_policy.value,
        "state": _job_state(result.status),
        "status": result.status.value.replace("_", " ").title(),
        "status_reasons": result.status_reasons,
        "state_flags": sorted(flag.value for flag in result.state_flags),
        "phases": [
            _phase_metric_record(metric) for metric in result.phases
        ],
        "changes": [
            {
                "from_phase": change.from_phase_name,
                "to_phase": change.to_phase_name,
                "from_window_ns": change.from_window_ns,
                "to_window_ns": change.to_window_ns,
                "inter_phase_gap_duration_ns": change.inter_phase_gap_duration_ns,
                "total_change_bytes": change.total_change_bytes,
                "total_change_percent": _finite_float(change.total_change_percent),
                "rate_change_bytes_per_s": _finite_float(change.rate_change_bytes_per_s),
                "rate_ratio": _finite_float(change.rate_ratio),
                "comparison_status": change.comparison_status.value,
                "is_comparable": change.is_comparable,
                "uncertainty_reasons": change.uncertainty_reasons,
            }
            for change in result.changes
        ],
        "phase_count": result.phase_count,
        "missing_phase_count": result.missing_phase_count,
        "covered_duration_ns": result.covered_duration_ns,
        "requested_duration_ns": result.requested_duration_ns,
        "uncovered_duration_ns": result.uncovered_duration_ns,
        "coverage_fraction": _finite_float(result.coverage_fraction),
        "valid_interval_count": result.valid_interval_count,
        "invalid_interval_count": result.invalid_interval_count,
        "boundary_excluded_interval_count": result.boundary_excluded_interval_count,
        "inter_phase_gap_duration_ns": result.inter_phase_gap_duration_ns,
        "missing_phase_names": result.missing_phase_names,
        "partial_phase_names": result.partial_phase_names,
        "zero_phase_names": result.zero_phase_names,
        "unsupported_phase_names": result.unsupported_phase_names,
        "baseline_relative": result.baseline_relative,
        "total_formula": result.total_formula,
        "mean_rate_formula": result.mean_rate_formula,
        "coverage_formula": result.coverage_formula,
        "total_change_formula": result.total_change_formula,
        "rate_change_formula": result.rate_change_formula,
        "interpretation_limits": result.interpretation_limits,
    }


def _invalid_export_document(reason: str) -> dict[str, Any]:
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "export": {
            "state": "invalid",
            "status": "Invalid export",
            "reasons": (reason,),
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "series_count": 0,
            "series_limit": MAX_EXPORT_SERIES,
            "series_truncated": False,
            "serialization_limit_bytes": MAX_EXPORT_JSON_BYTES,
        },
        "run": {},
        "capture": {},
        "filters": {},
        "overview": {
            "traffic_state": "invalid",
            "traffic_status": "Invalid export",
            "traffic_status_reasons": (reason,),
        },
        "node_comparison": {
            "rows": [],
            "state": "invalid",
            "status": "Invalid export",
            "truncated": False,
        },
        "node_balance": _empty_node_balance(),
        "job_balance": _empty_job_balance(),
        "nic_detail": {
            "selected_node": None,
            "selected_nic": None,
            "selection_note": reason,
            "series": [],
            "matching": _empty_matching_analysis(),
            "flow_control": _empty_flow_control_analysis(),
            "frames": _empty_frame_analysis(),
            "operations": _empty_operation_analysis(),
        },
        "phase_analysis": _empty_phase_analysis(),
        "job_profile": _empty_job_profile(),
        "question_capabilities": [],
        "methodology": {
            "formulas": {},
            "units": {},
            "windows": {},
            "populations": {},
            "quality_flags": (),
            "limitations": (
                reason,
                "This summary export excludes raw store columns and sample values.",
            ),
        },
    }


def build_export_document(
    view_model: dict[str, Any] | None,
    *,
    exported_at_utc: datetime | str | None = None,
) -> dict[str, Any]:
    """Build a finite, summary-only JSON document for the current filters."""

    required_keys = {
        "schema_version", "run", "capture", "filters", "overview",
        "series_summaries", "question_coverage",
    }
    if not isinstance(view_model, dict) or not required_keys.issubset(view_model):
        return _invalid_export_document(
            "The requested view model is unavailable or does not match the export contract."
        )

    if isinstance(exported_at_utc, datetime):
        timestamp = exported_at_utc.astimezone(timezone.utc).isoformat()
    elif exported_at_utc is not None:
        timestamp = str(exported_at_utc)
    else:
        timestamp = datetime.now(timezone.utc).isoformat()

    series = _export_series_records(view_model.get("series_summaries", []))
    overview = {
        "traffic_state": "missing",
        "traffic_status": "Missing",
        "traffic_status_reasons": (),
        **view_model.get("overview", {}),
    }
    capture = view_model.get("capture", {})
    filters = view_model.get("filters", {})
    question_rows = view_model.get("question_coverage", [])
    node_comparison = view_model.get("node_comparison", {"rows": []})
    node_balance = view_model.get("node_balance") or _empty_node_balance()
    job_balance = view_model.get("job_balance") or _empty_job_balance()
    nic_detail = view_model.get("nic_detail") or {
        "selected_node": None,
        "selected_nic": None,
        "selection_note": "No NIC matches the current filters.",
        "series": [],
    }
    nic_detail = {
        **nic_detail,
        "series": _export_series_records(nic_detail.get("series", [])),
        "reliability": _export_reliability_summary(
            nic_detail.get("reliability")
        ),
        "flow_control": _export_flow_control_summary(
            nic_detail.get("flow_control")
        ),
        "frames": _export_frame_summary(nic_detail.get("frames")),
        "translation": _export_translation_summary(
            nic_detail.get("translation")
        ),
        "latency": _export_latency_summary(nic_detail.get("latency")),
        "matching": _export_matching_summary(nic_detail.get("matching")),
        "operations": _export_operation_summary(nic_detail.get("operations")),
    }
    phase_analysis = view_model.get("phase_analysis") or _empty_phase_analysis()
    job_profile = view_model.get("job_profile") or _empty_job_profile()
    total_series_count = len(view_model.get("series_summaries", []))
    traffic_state = str(overview.get("traffic_state") or "missing")
    if not series and not node_comparison.get("rows"):
        export_state = "invalid" if traffic_state == "invalid" else "empty"
        export_status = (
            "Invalid time window"
            if export_state == "invalid"
            else "No series match the current filters"
        )
    else:
        export_state = traffic_state
        export_status = str(overview.get("traffic_status") or "Unknown")

    quality_flags = set(overview.get("quality_flags") or ())
    quality_flags.update(
        flag
        for record in series
        for flag in record.get("quality_flags", ())
    )
    reliability = nic_detail.get("reliability") or {}
    flow_control = nic_detail.get("flow_control") or {}
    matching = nic_detail.get("matching") or {}
    operations = nic_detail.get("operations") or {}
    frames = nic_detail.get("frames") or {}
    translation = nic_detail.get("translation") or {}
    latency = nic_detail.get("latency") or {}
    quality_flags.update(reliability.get("quality_flags") or ())
    quality_flags.update(
        flag
        for collector in reliability.get("collector_analyses", ())
        for flag in collector.get("quality_flags", ())
    )
    quality_flags.update(operations.get("quality_flags") or ())
    quality_flags.update(
        flag
        for collector in operations.get("collector_analyses", ())
        for flag in collector.get("quality_flags", ())
    )
    quality_flags.update(flow_control.get("quality_flags") or ())
    quality_flags.update(
        flag
        for collector in flow_control.get("collector_analyses", ())
        for flag in collector.get("quality_flags", ())
    )
    quality_flags.update(matching.get("quality_flags") or ())
    quality_flags.update(
        flag
        for collector in matching.get("collector_analyses", ())
        for flag in collector.get("quality_flags", ())
    )
    quality_flags.update(frames.get("quality_flags") or ())
    quality_flags.update(
        flag
        for collector in frames.get("collector_analyses", ())
        for flag in collector.get("quality_flags", ())
    )
    quality_flags.update(translation.get("quality_flags") or ())
    quality_flags.update(
        flag
        for collector in translation.get("collector_analyses", ())
        for flag in collector.get("quality_flags", ())
    )
    quality_flags.update(latency.get("quality_flags") or ())
    quality_flags.update(
        flag
        for collector in latency.get("collector_analyses", ())
        for flag in collector.get("quality_flags", ())
    )
    supported_question_ids = tuple(
        row["id"] for row in question_rows if row.get("state") == "supported"
    )
    formulas = {
        "interval_rate": sorted(
            {
                record["rate_formula"]
                for record in series
                if record.get("rate_formula") is not None
            }
        ),
        "mean_rate": sorted(
            {
                record["mean_rate_formula"]
                for record in series
                if record.get("mean_rate_formula") is not None
            }
        ),
        "p99_rate": (
            "duration-weighted P99 over compatible valid interval rates; "
            "per-NIC P99 values are not averaged"
        ),
        "node_traffic_share": node_balance.get("traffic_share_formula"),
        "effective_nic_count": node_balance.get("effective_nic_count_formula"),
        "spatial_quantile_interpolation": node_balance.get(
            "spatial_quantile_interpolation"
        ),
        "job_node_traffic_share": job_balance.get("traffic_share_formula"),
        "effective_node_count": job_balance.get("effective_node_count_formula"),
        "job_spatial_quantile_interpolation": job_balance.get(
            "spatial_quantile_interpolation"
        ),
        "hotspot_concentration": job_balance.get("hotspot_concentration_formula"),
        "phase_total": phase_analysis.get("total_formula"),
        "phase_mean_rate": phase_analysis.get("mean_rate_formula"),
        "reliability_error_rate": (
            "sum(error interval deltas) / common measured covered duration"
        ),
        "flow_control_delta": flow_control.get("delta_formula"),
        "flow_control_mean_rate": flow_control.get("mean_rate_formula"),
        "matching_delta": matching.get("delta_formula"),
        "matching_priority_fraction": matching.get("priority_fraction_formula"),
        "matching_overflow_fraction": matching.get("overflow_fraction_formula"),
        "operation_delta": operations.get("delta_formula"),
        "operation_rate": operations.get("rate_formula"),
        "operation_opcode_fraction": operations.get("opcode_fraction_formula"),
        "operation_dma_idc_fraction": operations.get("dma_idc_fraction_formula"),
        "frame_delta": frames.get("delta_formula"),
        "frame_rate": frames.get("rate_formula"),
        "frame_bin_share": frames.get("bin_share_formula"),
        "frame_mean": frames.get("mean_formula"),
        "frame_approximate_mean": frames.get("approximate_mean_formula"),
        "translation_delta": translation.get("delta_formula"),
        "translation_rate": translation.get("rate_formula"),
        "translation_hit_ratio": translation.get("hit_ratio_formula"),
        "translation_misses_per_request": translation.get(
            "misses_per_request_formula"
        ),
        "translation_misses_per_gb": translation.get(
            "misses_per_gb_formula"
        ),
        "latency_delta": latency.get("delta_formula"),
        "latency_quantile": latency.get("quantile_selection_formula"),
        "latency_mean": latency.get("mean_estimate_formula"),
    }
    limitations = [
        "Summary export only; raw trace records, store columns, and sample values are excluded.",
        "Statistics use complete valid intervals in the selected window; missing is not zero.",
        "Temporal and spatial quantiles are separate populations; temporal P99 values are not averaged.",
        "Counter observations suggest correlations, not causation.",
        "FCS/replay accounting and register-specific reset domains remain unresolved.",
        "Network-frame octets are not application payload or physical-wire occupancy.",
    ]
    limitations.extend(reliability.get("limitations") or ())
    limitations.extend(flow_control.get("limitations") or ())
    limitations.extend(matching.get("limitations") or [])
    limitations.extend(operations.get("limitations") or [])
    limitations.extend(frames.get("limitations") or [])
    limitations.extend(translation.get("limitations") or [])
    limitations.extend(latency.get("limitations") or [])
    if any(record.get("baseline_relative") for record in series):
        limitations.append(
            "At least one source is baseline-relative; omitted unchanged samples limit temporal detail."
        )
    limitations.extend(
        [
            job_balance.get("topology_limitation") or (
                "Topology interpretation is limited to supplied non-overlapping node groups."
            ),
            job_balance.get("edge_limitation") or (
                "Communication-edge metadata is required for peer or near/far claims."
            ),
            phase_analysis.get("interpretation_limits") or (
                "Phase changes describe observed counters, not workload causes."
            ),
        ]
    )

    document = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "export": {
            "state": export_state,
            "status": export_status,
            "reasons": tuple(overview.get("traffic_status_reasons") or ()),
            "exported_at_utc": timestamp,
            "view_model_schema_version": view_model.get("schema_version"),
            "content_scope": (
                "Currently filtered view-model summaries; no raw trace or sample values."
            ),
            "series_count": total_series_count,
            "series_limit": MAX_EXPORT_SERIES,
            "series_truncated": total_series_count > len(series),
            "serialization_limit_bytes": MAX_EXPORT_JSON_BYTES,
        },
        "run": view_model.get("run", {}),
        "capture": capture,
        "filters": filters,
        "overview": overview,
        "series_summaries": series,
        "node_comparison": node_comparison,
        "node_balance": node_balance,
        "job_balance": job_balance,
        "nic_detail": nic_detail,
        "phase_analysis": phase_analysis,
        "job_profile": job_profile,
        "question_capabilities": question_rows,
        "methodology": {
            "formulas": formulas,
            "counter_semantic_registry_version": COUNTER_SEMANTIC_REGISTRY_VERSION,
            "counter_semantics": (
                "network-frame octets; TX/RX and OK/BAD counters are separate; decreases are conservatively excluded and wrap is unverified"
            ),
            "counter_reset_behavior": "new accumulation epoch after hardware reset",
            "counter_wrap_modulus": 2**56,
            "units": {
                "traffic_totals": "bytes",
                "rates": "bytes/s",
                "spatial_quantiles": "bytes",
                "durations": "nanoseconds",
                "coverage": "fraction",
                "coefficient_of_variation": "dimensionless",
                "flow_control_pause_cycles": "cycles",
                "flow_control_events": "events",
                "matching_fractions": "fraction",
                "operation_commands": "commands",
                "operation_rates": "commands/s",
            },
            "windows": {
                "selected": {
                    "start_ns": filters.get("start_ns"),
                    "end_ns": filters.get("end_ns"),
                    "duration_ns": capture.get("selected_duration_ns"),
                },
                "acquisition": {
                    "start_ns": capture.get("acquisition_start_ns"),
                    "end_ns": capture.get("acquisition_end_ns"),
                    "duration_ns": capture.get("acquisition_duration_ns"),
                },
                "represented": {
                    "covered_duration_ns": overview.get("covered_duration_ns"),
                    "coverage_fraction": overview.get("coverage_fraction"),
                },
                "flow_control": {
                    "requested_window_ns": flow_control.get("requested_window_ns"),
                    "coverage_scope": flow_control.get(
                        "coverage_scope",
                        "Coverage is reported separately for each collector.",
                    ),
                    "collector_coverage": flow_control.get("collector_coverage", ()),
                },
                "matching": {
                    "requested_window_ns": matching.get("requested_window_ns"),
                    "collector_priority_fractions": matching.get(
                        "collector_priority_fractions", ()
                    ),
                    "priority_fraction_scope": matching.get(
                        "priority_fraction_scope",
                        "Collector-separated values; collectors are not merged.",
                    ),
                },
                "operations": {
                    "requested_window_ns": operations.get("requested_window_ns"),
                    "coverage_scope": operations.get(
                        "coverage_scope",
                        "Collector-separated coverage; DMA and IDC mix requires common coverage.",
                    ),
                    "collector_coverage": operations.get("collector_coverage", ()),
                },
                "frames": {
                    "requested_window_ns": frames.get("requested_window_ns"),
                    "coverage_scope": frames.get(
                        "coverage_scope",
                        "Collector-separated coverage; bins combine only on exact boundaries.",
                    ),
                    "collector_coverage": frames.get("collector_coverage", ()),
                    "bin_edges_source": frames.get("bin_edges_source"),
                    "population_compatibility": frames.get(
                        "population_compatibility", ()
                    ),
                },
            },
            "populations": {
                "selected_node_count": capture.get("node_count"),
                "selected_nic_count": capture.get("nic_count"),
                "directions": sorted(
                    {record.get("direction") for record in series if record.get("direction")}
                ),
                "series_count": len(series),
                "valid_interval_count": overview.get("valid_interval_count", 0),
                "invalid_interval_count": overview.get("invalid_interval_count", 0),
                "aggregation_populations": sorted(
                    {
                        record["aggregation_population"]
                        for record in series
                        if record.get("aggregation_population") is not None
                    }
                ),
                "supported_question_ids": supported_question_ids,
                "job_directions": [
                    record.get("direction")
                    for record in job_balance.get("directions", [])
                ],
                "phase_count": phase_analysis.get("phase_count", 0),
                "flow_control_collector_count": flow_control.get(
                    "collector_count", 0
                ),
                "flow_control_counter_count": flow_control.get(
                    "counter_count", 0
                ),
                "flow_control_priorities": flow_control.get(
                    "expected_priorities", ()
                ),
                "matching_collector_count": matching.get("collector_count", 0),
                "matching_priority_result_count": matching.get(
                    "priority_count", 0
                ),
                "operation_collector_count": operations.get("collector_count", 0),
                "operation_opcode_result_count": operations.get("opcode_count", 0),
                "operation_total_result_count": operations.get("total_count", 0),
                "frame_collector_count": frames.get("collector_count", 0),
                "frame_bin_result_count": frames.get("bin_count", 0),
                "frame_quantile_result_count": frames.get("quantile_count", 0),
            },
            "quality_flags": sorted(quality_flags),
            "limitations": limitations,
        },
    }
    return _sanitize_export_value(document)


def serialize_export_document(
    document: dict[str, Any], *, max_bytes: int = MAX_EXPORT_JSON_BYTES
) -> str:
    """Serialize with finite JSON, stable key order, and a byte ceiling."""

    def dump(candidate: dict[str, Any]) -> str:
        return json.dumps(
            candidate,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    payload = dump(document)
    if len(payload.encode("utf-8")) <= max_bytes:
        return payload

    total_series = len(document.get("series_summaries", []))
    low, high = 0, total_series
    best: str | None = None
    while low <= high:
        series_count = (low + high) // 2
        candidate = dict(document)
        series = document["series_summaries"][:series_count]
        candidate["series_summaries"] = series
        nic_detail = dict(candidate.get("nic_detail", {}))
        nic_detail["series"] = nic_detail.get("series", [])[:series_count]
        candidate["nic_detail"] = nic_detail
        export_metadata = dict(candidate.get("export", {}))
        export_metadata.update(
            {
                "series_count": total_series,
                "series_truncated": series_count < total_series,
                "serialization_truncated": True,
                "serialization_limit_bytes": max_bytes,
            }
        )
        candidate["export"] = export_metadata
        payload = dump(candidate)
        if len(payload.encode("utf-8")) <= max_bytes:
            best = payload
            low = series_count + 1
        else:
            high = series_count - 1
    if best is not None:
        return best
    raise ValueError("The filtered view-model export exceeds the JSON byte limit.")


def build_export_payload(view_model: dict[str, Any] | None) -> dict[str, Any]:
    """Return a Dash Download payload for the current filtered view model."""

    try:
        payload = serialize_export_document(build_export_document(view_model))
    except (TypeError, ValueError, OverflowError):
        payload = serialize_export_document(
            _invalid_export_document(
                "The filtered view model could not be serialized within the export limit."
            )
        )
    return {
        "content": payload,
        "filename": "aurora-nic-filtered-view-model.json",
        "type": "application/json",
    }


@dataclass(frozen=True)
class AnalysisFilterState:
    """The complete, shareable UI filter state."""

    node_id: int | None = None
    nic_id: int | None = None
    direction: Literal["all", "tx", "rx"] = "all"
    start_ns: int | None = None
    end_ns: int | None = None


_QUESTION_MILESTONES = {
    "NIC-01": "M2/M3/M4",
    **{question_id: "M5" for question_id in (
        "NIC-02", "NIC-03", "NIC-04", "NIC-05", "NIC-06", "NIC-07", "NIC-08",
        "NIC-09",
    )},
    "NODE-01": "M6",
    "NODE-02": "M6",
    "JOB-01": "M6",
    "JOB-02": "M6",
    "JOB-03": "M4/M6",
    "JOB-04": "M6",
}


def _question_rows(store: TraceStore) -> list[dict[str, object]]:
    rows = []
    for capability in build_question_capability_matrix(store):
        if capability.status == "supported":
            status = "Supported"
        else:
            status = "Pending / unsupported"
        reasons = " ".join(capability.reasons)
        if len(reasons) > 600:
            reasons = reasons[:597].rstrip() + "..."
        rows.append(
            {
                "id": capability.question_id,
                "question": capability.question,
                "milestone": _QUESTION_MILESTONES[capability.question_id],
                "status": status,
                "state": capability.status,
                "reason": reasons,
                "required_counter_count": len(capability.required_counters),
                "available_required_count": len(
                    capability.available_required_counters
                ),
                "missing_required_count": len(capability.missing_required_counters),
                "unreadable_required_count": len(
                    capability.unreadable_required_counters
                ),
                "implemented_analyses": ", ".join(
                    capability.implemented_analysis_ids
                )
                or "None",
                "limitations": "; ".join(capability.scientific_limitations)
                or "None",
            }
        )
    return rows


def default_filters(store: TraceStore) -> AnalysisFilterState:
    """Return the full-store window and an unfiltered node/NIC/direction state."""

    if store.coverage.acquisition_start_ns is not None and store.coverage.acquisition_end_ns is not None:
        start_ns = int(store.coverage.acquisition_start_ns)
        end_ns = int(store.coverage.acquisition_end_ns)
    else:
        start_ns = int(np.min(store.timestamps_ns))
        end_ns = int(np.max(store.timestamps_ns))
    max_timestamp_ns = int(np.max(store.timestamps_ns))
    if end_ns <= max_timestamp_ns:
        end_ns = max_timestamp_ns + 1
    return AnalysisFilterState(start_ns=start_ns, end_ns=end_ns)


def downsample_series(
    timestamps_ns: np.ndarray | tuple[int, ...], values: np.ndarray | tuple[int | float, ...]
) -> tuple[list[int], list[int | float]]:
    """Return at most 500 display points while preserving first and last samples."""

    if len(timestamps_ns) != len(values):
        raise ValueError("timestamps and values must have equal length")
    sample_count = len(timestamps_ns)
    if sample_count == 0:
        return [], []
    if sample_count <= MAX_DISPLAY_POINTS_PER_SERIES:
        indices = np.arange(sample_count, dtype=np.int64)
    else:
        indices = np.linspace(
            0,
            sample_count - 1,
            num=MAX_DISPLAY_POINTS_PER_SERIES,
            dtype=np.int64,
        )
        indices = np.unique(indices)
    return (
        [int(timestamps_ns[index]) for index in indices],
        [values[index].item() for index in indices],
    )


def _parse_identity_value(value: Any, prefix: str) -> int | None:
    if value in (None, "", "all"):
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str) and value.startswith(f"{prefix}:"):
        try:
            return int(value.split(":", 1)[1])
        except ValueError as error:
            raise ValueError(f"invalid {prefix} filter value {value!r}") from error
    raise ValueError(f"invalid {prefix} filter value {value!r}")


def _directions(direction: str) -> tuple[TrafficDirection, ...]:
    normalized = direction.lower()
    if normalized == "all":
        return (TrafficDirection.TX, TrafficDirection.RX)
    if normalized == "tx":
        return (TrafficDirection.TX,)
    if normalized == "rx":
        return (TrafficDirection.RX,)
    raise ValueError("direction must be 'all', 'tx', or 'rx'")


def _selected_counter(
    counters: tuple[CounterMetadata, ...], direction: TrafficDirection
) -> tuple[CounterMetadata | None, tuple[CounterMetadata, ...]]:
    candidates = []
    for counter in counters:
        definition = counter_semantic_definition(counter.name)
        if (
            counter.unit != "bytes"
            or counter.semantics is not CounterSemantics.CUMULATIVE
            or definition is None
            or definition.quality is not CounterQuality.OK
        ):
            continue
        try:
            if infer_traffic_direction(counter.name) is direction:
                candidates.append(counter)
        except ValueError:
            continue
    preferred_names = (
        ("hni_sts_tx_ok_octets",)
        if direction is TrafficDirection.TX
        else ("hni_sts_rx_ok_octets",)
    )
    for preferred_name in preferred_names:
        for counter in candidates:
            if counter.name == preferred_name:
                return counter, tuple(candidates)
    return (candidates[0] if candidates else None), tuple(candidates)


def _requested_counter_name_for_direction(
    store: TraceStore, direction: TrafficDirection
) -> str | None:
    for name in store.coverage.requested_counter_names:
        try:
            if infer_traffic_direction(name) is direction:
                return name
        except ValueError:
            continue
    return None


def _analysis_state(status: AnalysisStatus, total_bytes: int | float | None) -> tuple[str, str]:
    if status is AnalysisStatus.ZERO_TRAFFIC or (
        status is AnalysisStatus.OK and total_bytes == 0
    ):
        return "zero", "Zero traffic (observed)"
    if status is AnalysisStatus.OK:
        return "ready", "Ready"
    if status is AnalysisStatus.PARTIAL_INVALID_DATA:
        return "partial", "Partial data"
    if status in (AnalysisStatus.MISSING_DATA, AnalysisStatus.INSUFFICIENT_DATA):
        return "missing", "Missing in selected window"
    if status is AnalysisStatus.INVALID_WINDOW:
        return "invalid", "Invalid window"
    if status is AnalysisStatus.INVALID_DATA:
        return "invalid", "Invalid data"
    return "unsupported", "Unsupported series"


def _finite_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) if np.isfinite(value) else None


def _series_record(
    *,
    node_label: str,
    nic_label: str,
    node_id: int,
    nic_id: int,
    direction: TrafficDirection,
    counter_name: str | None,
    analysis=None,
    stream_count: int = 1,
    collector_identities: tuple[dict[str, int | str], ...] = (),
    collector_policy: str = "no collector observation",
    state: str | None = None,
    status: str | None = None,
    reasons: tuple[str, ...] = (),
    include_display: bool = False,
    timestamps_ns=None,
    values=None,
) -> dict[str, Any]:
    collector_id = (
        collector_identities[0]["collector_id"]
        if len(collector_identities) == 1
        else None
    )
    if analysis is not None:
        resolved_state, resolved_status = _analysis_state(analysis.status, analysis.total_bytes)
        record: dict[str, Any] = {
            "node": node_label,
            "nic": nic_label,
            "node_id": node_id,
            "nic_id": nic_id,
            "direction": direction.value,
            "counter": analysis.source_counter,
            "state": resolved_state,
            "status": resolved_status,
            "stream_count": stream_count,
            "collector_id": collector_id,
            "collector_identities": collector_identities,
            "collector_policy": collector_policy,
            "point_count": analysis.interval_count + 1,
            "total_bytes": analysis.total_bytes,
            "covered_duration_ns": analysis.covered_duration_ns,
            "requested_duration_ns": analysis.requested_duration_ns,
            "coverage_fraction": _finite_float(analysis.coverage_fraction),
            "mean_rate_bytes_per_s": _finite_float(analysis.mean_rate_bytes_per_s),
            "p50_rate_bytes_per_s": _finite_float(analysis.p50_rate_bytes_per_s),
            "p99_rate_bytes_per_s": _finite_float(analysis.p99_rate_bytes_per_s),
            "coefficient_of_variation": _finite_float(analysis.coefficient_of_variation),
            "valid_interval_count": analysis.valid_interval_count,
            "invalid_interval_count": analysis.invalid_interval_count,
            "quality_flags": sorted(
                flag.value for flag in analysis.quality_flags
            ),
            "status_reasons": tuple(analysis.status_reasons),
            "rate_formula": analysis.rate_formula,
            "mean_rate_formula": analysis.mean_rate_formula,
            "aggregation_population": analysis.aggregation_population,
            "baseline_relative": analysis.baseline_relative,
            "counter_semantic_registry_version": analysis.counter_semantic_registry_version,
            "counter_semantics": analysis.counter_semantics,
            "counter_quality": analysis.counter_quality,
            "counter_scope": analysis.counter_scope,
            "counter_width_bits": analysis.counter_width_bits,
            "counter_reset_behavior": analysis.counter_reset_behavior,
            "counter_caveats": analysis.counter_caveats,
            "accumulation_epoch_count": analysis.accumulation_epoch_count,
            "wrap_suspect_interval_count": analysis.wrap_suspect_interval_count,
            "reset_epoch_boundary_count": analysis.reset_epoch_boundary_count,
        }
    else:
        definition = counter_semantic_definition(counter_name or "")
        record = {
            "node": node_label,
            "nic": nic_label,
            "node_id": node_id,
            "nic_id": nic_id,
            "direction": direction.value,
            "counter": counter_name,
            "state": state or "missing",
            "status": status or "Missing",
            "stream_count": stream_count,
            "collector_id": collector_id,
            "collector_identities": collector_identities,
            "collector_policy": collector_policy,
            "point_count": 0,
            "total_bytes": None,
            "covered_duration_ns": 0,
            "requested_duration_ns": 0,
            "coverage_fraction": None,
            "mean_rate_bytes_per_s": None,
            "p50_rate_bytes_per_s": None,
            "p99_rate_bytes_per_s": None,
            "coefficient_of_variation": None,
            "valid_interval_count": 0,
            "invalid_interval_count": 0,
            "quality_flags": [],
            "status_reasons": reasons,
            "rate_formula": "delta_bytes / duration_seconds",
            "mean_rate_formula": "unavailable",
            "aggregation_population": "0 valid complete intervals",
            "baseline_relative": False,
            "counter_semantic_registry_version": (
                definition.registry_version if definition is not None else None
            ),
            "counter_semantics": definition.measurement if definition is not None else None,
            "counter_quality": definition.quality.value if definition is not None else None,
            "counter_scope": definition.scope.value if definition is not None else None,
            "counter_width_bits": definition.counter_width_bits if definition is not None else None,
            "counter_reset_behavior": (
                definition.reset_behavior.value if definition is not None else None
            ),
            "counter_caveats": definition.caveats if definition is not None else (),
            "accumulation_epoch_count": 1,
            "wrap_suspect_interval_count": 0,
            "reset_epoch_boundary_count": 0,
        }
    if include_display and timestamps_ns is not None and values is not None:
        display_timestamps, display_values = downsample_series(timestamps_ns, values)
        record["display_timestamps_ns"] = display_timestamps
        record["display_values"] = display_values
        record["display_point_count"] = len(display_values)
    else:
        record["display_point_count"] = 0
    return record


def _reliability_quality_flags(flags: int) -> tuple[ReliabilityQualityFlag, ...]:
    mappings = {
        QualityFlag.REORDERED: ReliabilityQualityFlag.REORDERED,
        QualityFlag.NONMONOTONIC_TIMESTAMP: ReliabilityQualityFlag.CONFLICTING_TIMESTAMP,
        QualityFlag.DUPLICATE_REMOVED: ReliabilityQualityFlag.DUPLICATE,
        QualityFlag.BASELINE_RELATIVE: ReliabilityQualityFlag.BASELINE_RELATIVE,
        QualityFlag.PRECISION_LOSS_POSSIBLE: ReliabilityQualityFlag.UNSUPPORTED_SERIES,
        QualityFlag.RESET_SUSPECT: ReliabilityQualityFlag.DECREASE_OR_RESET,
        QualityFlag.WRAP_SUSPECT: ReliabilityQualityFlag.DECREASE_OR_RESET,
        QualityFlag.GAP: ReliabilityQualityFlag.PARTIAL_WINDOW,
        QualityFlag.UNKNOWN_METADATA: ReliabilityQualityFlag.UNSUPPORTED_SERIES,
    }
    return tuple(
        dict.fromkeys(
            mapped
            for source_flag, mapped in mappings.items()
            if int(flags) & source_flag
        )
    )


def _reliability_series(
    result,
    spec: ReliabilityCounterSpec,
) -> ReliabilityCounterSeries | None:
    values: list[int] = []
    for value in result.values:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            if not math.isfinite(value) or not value.is_integer():
                return None
            value = int(value)
        if type(value) is not int or not 0 <= value <= 2**64 - 1:
            return None
        values.append(value)

    quality_flags = set(
        _reliability_quality_flags(int(result.counter.quality_flags))
    )
    quality_flags.update(_reliability_quality_flags(int(np.bitwise_or.reduce(result.quality_flags))))
    collector = result.collector
    return ReliabilityCounterSeries(
        spec=spec,
        identity=CounterIdentity(
            run_id=result.run_identity.run_id,
            node_id=str(result.node.node_id),
            nic_id=str(result.nic.nic_id),
            collector_id=f"{collector.name} (ID {collector.collector_id})",
            source_counter_id=result.counter.name,
        ),
        timestamps_ns=tuple(int(timestamp) for timestamp in result.timestamps_ns),
        values=tuple(values),
        value_kind=(
            ReliabilityValueKind.CUMULATIVE
            if result.counter.value_kind is ValueKind.EXACT_UINT64
            else ReliabilityValueKind.FLOAT64
        ),
        semantics=(
            "cumulative counter"
            if result.counter.semantics is CounterSemantics.CUMULATIVE
            else result.counter.semantics.value
        ),
        baseline_relative=result.counter.baseline_relative,
        quality_flags=tuple(quality_flags),
    )


def _reliability_counter_record(result: Any) -> dict[str, Any]:
    identity = result.source_identity
    return {
        "counter": result.spec.counter_id,
        "role": result.spec.role.value,
        "unit": result.unit,
        "rate_unit": result.rate_unit,
        "state": result.status.value,
        "status": result.status.value.replace("_", " ").title(),
        "status_reasons": result.status_reasons,
        "quality_flags": sorted(flag.value for flag in result.quality_flags),
        "run_id": identity.run_id if identity else None,
        "node_id": int(identity.node_id) if identity else None,
        "nic_id": int(identity.nic_id) if identity else None,
        "collector_id": identity.collector_id if identity else None,
        "source_counter_id": identity.source_counter_id if identity else result.spec.counter_id,
        "requested_window_ns": result.requested_window_ns,
        "represented_window_ns": result.represented_window_ns,
        "requested_duration_ns": result.requested_duration_ns,
        "covered_duration_ns": result.covered_duration_ns,
        "coverage_fraction": _finite_float(result.coverage_fraction),
        "sample_count": result.sample_count,
        "valid_interval_count": result.valid_interval_count,
        "invalid_interval_count": result.invalid_interval_count,
        "outside_window_interval_count": result.outside_window_interval_count,
        "total": result.total,
        "rate_per_second": _finite_float(result.rate_per_second),
        "first_nonzero_sample_ns": result.first_nonzero_sample_ns,
        "first_nonzero_increment_ns": result.first_nonzero_increment_ns,
        "persistent_zero": result.persistent_zero,
        "zero_observed": result.zero_observed,
        "baseline_relative": "baseline_relative" in result.quality_flags,
    }


def _reliability_record(result: Any, collector: Any) -> dict[str, Any]:
    counter_records = [
        _reliability_counter_record(counter)
        for counter in result.counter_results
    ]
    return {
        "schema_version": 1,
        "state": result.status.value,
        "status": result.status.value.replace("_", " ").title(),
        "status_reasons": result.status_reasons,
        "quality_flags": sorted(flag.value for flag in result.quality_flags),
        "population": {
            "run_id": result.population.run_id,
            "node_id": int(result.population.node_id),
            "nic_id": int(result.population.nic_id),
            "collector_id": result.population.collector_id,
            "collector_name": collector.name,
            "collector_registry_id": collector.collector_id,
            "collector_rank": collector.rank,
            "collector_process": collector.process,
            "collector_type": collector.collector_type,
        },
        "requested_window_ns": result.requested_window_ns,
        "represented_window_ns": result.represented_window_ns,
        "requested_duration_ns": result.requested_duration_ns,
        "covered_duration_ns": result.covered_duration_ns,
        "coverage_fraction": _finite_float(result.coverage_fraction),
        "counter_results": counter_records[:MAX_RELIABILITY_COUNTER_RESULTS],
        "counter_count": len(counter_records),
        "counter_truncated": len(counter_records) > MAX_RELIABILITY_COUNTER_RESULTS,
        "error_total": result.error_total,
        "good_total": result.good_total,
        "aggregate_error_rate_per_second": _finite_float(
            result.aggregate_error_rate_per_second
        ),
        "aggregate_rate_unit": result.aggregate_rate_unit,
        "compatible_counter_ids": result.compatible_counter_ids,
        "incompatible_counter_ids": result.incompatible_counter_ids,
        "first_nonzero_error_sample_ns": result.first_nonzero_error_sample_ns,
        "first_nonzero_error_increment_ns": result.first_nonzero_error_increment_ns,
        "persistent_zero": result.persistent_zero,
        "zero_observed": result.zero_observed,
        "rate_formula": result.rate_formula,
        "aggregation_population": result.aggregation_population,
        "historical_zero_note": result.historical_zero_note,
    }


def _build_reliability_analysis(
    store: TraceStore,
    node: Any,
    nic: Any,
    start_ns: int,
    end_ns: int,
) -> dict[str, Any]:
    counter_by_name = {counter.name: counter for counter in store.counters}
    specs = tuple(
        ReliabilityCounterSpec(
            counter_id=name,
            role=role,
            unit=counter_by_name[name].unit if name in counter_by_name else "unknown",
            aggregation_group=f"nic09:{name}",
        )
        for name, role in _RELIABILITY_COUNTER_ROLES.items()
    )
    if not specs:
        return _empty_reliability_analysis()

    series_by_collector: dict[int, list[ReliabilityCounterSeries]] = {}
    collectors: dict[int, Any] = {}
    unsupported_series: list[str] = []
    duplicate_stream_count = 0
    for spec in specs:
        if spec.counter_id not in counter_by_name:
            continue
        counter = counter_by_name[spec.counter_id]
        try:
            query_result = store.query_series(
                node,
                nic,
                counter,
                start_ns=start_ns,
                end_ns=end_ns,
            )
        except TraceValidationError:
            continue
        for result in query_result.results:
            collectors[result.collector.collector_id] = result.collector
            duplicate_stream_count = max(
                duplicate_stream_count,
                result.observation_stream_count,
            )
            series = _reliability_series(result, spec)
            if series is None:
                unsupported_series.append(spec.counter_id)
                continue
            series_by_collector.setdefault(result.collector.collector_id, []).append(series)

    collector_ids = sorted(series_by_collector)
    records = []
    for collector_id in collector_ids[:MAX_RELIABILITY_COLLECTORS]:
        collector = collectors[collector_id]
        reference = series_by_collector[collector_id][0]
        analysis = analyze_reliability(
            ReliabilityDataset(
                population=ReliabilityPopulation(
                    run_id=store.identity.run_id,
                    node_id=str(node.node_id),
                    nic_id=str(nic.nic_id),
                    collector_id=reference.identity.collector_id,
                ),
                requested_counters=specs,
                series=tuple(series_by_collector[collector_id]),
                window_ns=(start_ns, end_ns - 1),
            )
        )
        record = _reliability_record(analysis, collector)
        if unsupported_series:
            record["unsupported_series"] = sorted(set(unsupported_series))
            record["quality_flags"] = sorted(
                set(record["quality_flags"])
                | {ReliabilityQualityFlag.UNSUPPORTED_SERIES.value}
            )
            record["status_reasons"] = tuple(
                dict.fromkeys(
                    (
                        *record["status_reasons"],
                        "At least one queried value could not be represented as an exact uint64 observation.",
                    )
                )
            )
        records.append(record)

    first_nonzero_events = [
        (
            record["first_nonzero_error_sample_ns"],
            record["first_nonzero_error_increment_ns"],
        )
        for record in records
        if record["first_nonzero_error_sample_ns"] is not None
    ]
    first_nonzero_event = min(
        first_nonzero_events,
        key=lambda event: (
            event[0],
            event[1] if event[1] is not None else float("inf"),
        ),
        default=None,
    )

    if not records:
        return {
            **_empty_reliability_analysis(),
            "requested_window_ns": (start_ns, end_ns - 1),
        }

    reasons = tuple(
        dict.fromkeys(
            reason for record in records for reason in record["status_reasons"]
        )
    )
    states = {record["state"] for record in records}
    state = (
        "partial"
        if any(
            value in states
            for value in (
                "partial_missing_data",
                "partial_invalid_data",
                "partial_unsupported_data",
                "missing_data",
                "insufficient_data",
                "unsupported",
            )
        )
        else next(iter(states))
    )
    return {
        "schema_version": 1,
        "question_id": "NIC-09",
        "state": state,
        "status": state.replace("_", " ").title(),
        "status_reasons": reasons,
        "quality_flags": sorted(
            {
                flag
                for record in records
                for flag in record["quality_flags"]
            }
        ),
        "collector_analyses": records,
        "collector_count": len(collector_ids),
        "collector_limit": MAX_RELIABILITY_COLLECTORS,
        "counter_limit": MAX_RELIABILITY_COUNTER_RESULTS,
        "collector_truncated": len(collector_ids) > MAX_RELIABILITY_COLLECTORS,
        "duplicate_stream_count": duplicate_stream_count,
        "first_nonzero_error_sample_ns": (
            first_nonzero_event[0] if first_nonzero_event is not None else None
        ),
        "first_nonzero_error_increment_ns": (
            first_nonzero_event[1] if first_nonzero_event is not None else None
        ),
        "persistent_zero": all(
            record["persistent_zero"] for record in records
        ),
        "zero_observed": any(record["zero_observed"] for record in records),
        "unsupported_series": sorted(set(unsupported_series)),
        "limitations": (
            "Reliability counters are correlational evidence, not a causal diagnosis.",
            "Missing counters and samples are not zero-valued observations.",
            "Decrease or reset intervals are excluded from totals and rate denominators.",
            "Good codewords are not errors; only explicitly role-mapped counters are used.",
            "Duplicate collector streams are analyzed separately and never merged.",
            "The traffic direction filter is not applied because NIC-09 counters have no verified direction mapping.",
            "Corrected and uncorrected codeword categories remain counter definitions, not equivalent failure severities.",
        ),
    }


def _flow_control_quality_flags(flags: int) -> tuple[SeriesQualityFlag, ...]:
    mappings = {
        QualityFlag.REORDERED: SeriesQualityFlag.REORDERED,
        QualityFlag.NONMONOTONIC_TIMESTAMP: SeriesQualityFlag.CONFLICTING_TIMESTAMP,
        QualityFlag.DUPLICATE_REMOVED: SeriesQualityFlag.DUPLICATE,
        QualityFlag.BASELINE_RELATIVE: SeriesQualityFlag.BASELINE_RELATIVE,
        QualityFlag.PRECISION_LOSS_POSSIBLE: SeriesQualityFlag.PRECISION_LOSS_POSSIBLE,
        QualityFlag.RESET_SUSPECT: SeriesQualityFlag.DECREASE_OR_RESET,
        QualityFlag.WRAP_SUSPECT: SeriesQualityFlag.WRAP_SUSPECT,
        QualityFlag.GAP: SeriesQualityFlag.GAP,
        QualityFlag.UNKNOWN_METADATA: SeriesQualityFlag.UNSUPPORTED_SERIES,
    }
    return tuple(
        dict.fromkeys(
            mapped
            for source_flag, mapped in mappings.items()
            if int(flags) & source_flag
        )
    )


def _flow_control_series(result) -> MetricSeries | None:
    direction = _FLOW_CONTROL_DIRECTION_BY_COUNTER.get(result.counter.name)
    if direction is None and result.counter.name not in _FLOW_CONTROL_UNDIRECTED_COUNTERS:
        return None
    value_dtype = (
        "uint64"
        if result.counter.value_kind is ValueKind.EXACT_UINT64
        else "float64"
    )
    if value_dtype == "float64" and not bool(np.all(np.isfinite(result.values))):
        return None
    values = (
        tuple(int(value) for value in result.values)
        if value_dtype == "uint64"
        else tuple(float(value) for value in result.values)
    )
    semantic_kind = {
        CounterSemantics.CUMULATIVE: TrafficValueKind.CUMULATIVE,
        CounterSemantics.DELTA: TrafficValueKind.DELTA,
        CounterSemantics.GAUGE: TrafficValueKind.GAUGE,
        CounterSemantics.UNKNOWN: TrafficValueKind.UNKNOWN,
    }[result.counter.semantics]
    quality_flags = set(
        _flow_control_quality_flags(int(result.counter.quality_flags))
    )
    if result.quality_flags.size:
        quality_flags.update(
            _flow_control_quality_flags(
                int(np.bitwise_or.reduce(result.quality_flags))
            )
        )
    if result.counter.baseline_relative:
        quality_flags.add(SeriesQualityFlag.BASELINE_RELATIVE)
    if result.observation_stream_count > 1:
        quality_flags.add(SeriesQualityFlag.MULTIPLE_OBSERVATION_STREAMS)
    return MetricSeries(
        identity=SeriesIdentity(
            run_id=result.run_identity.run_id,
            node_id=result.node.label,
            nic_id=result.nic.label,
            collector_id=result.collector.name,
            counter_id=result.counter.name,
            direction=direction,
        ),
        timestamps_ns=tuple(int(timestamp) for timestamp in result.timestamps_ns),
        values=values,
        value_dtype=value_dtype,
        value_kind=semantic_kind,
        semantics=result.counter.unit,
        unit=result.counter.unit,
        baseline_relative=result.counter.baseline_relative,
        quality_flags=tuple(
            sorted(quality_flags, key=lambda flag: flag.value)
        ),
    )


def _flow_control_counter_record(
    result: Any,
    *,
    collector: Any,
    node: Any,
    nic: Any,
    run_id: str,
) -> dict[str, Any]:
    return {
        "counter": result.counter_name,
        "metric_kind": result.metric_kind.value,
        "direction": result.direction.value if result.direction else "unsupported",
        "priority": result.priority,
        "state": result.status.value,
        "status": result.status.value.replace("_", " ").title(),
        "status_reasons": result.status_reasons,
        "quality_flags": sorted(flag.value for flag in result.quality_flags),
        "run_id": run_id,
        "node_id": node.node_id,
        "nic_id": nic.nic_id,
        "collector_id": f"{collector.name} (ID {collector.collector_id})",
        "source_counter": result.source_counter,
        "unit": result.unit,
        "semantics": result.semantics,
        "value_kind": result.value_kind.value,
        "baseline_relative": result.baseline_relative,
        "requested_window_ns": result.requested_window_ns,
        "represented_window_ns": result.represented_window_ns,
        "requested_duration_ns": result.requested_duration_ns,
        "covered_duration_ns": result.covered_duration_ns,
        "uncovered_requested_duration_ns": result.uncovered_requested_duration_ns,
        "coverage_fraction": _finite_float(result.coverage_fraction),
        "sample_count": result.sample_count,
        "interval_count": result.interval_count,
        "valid_interval_count": result.valid_interval_count,
        "invalid_interval_count": result.invalid_interval_count,
        "boundary_excluded_interval_count": result.boundary_excluded_interval_count,
        "total": result.total,
        "mean_rate_per_second": _finite_float(result.mean_rate_per_second),
        "first_nonzero_ns": result.first_nonzero_ns,
        "first_nonzero_delta": result.first_nonzero_delta,
        "first_nonzero_value": result.first_nonzero_value,
        "delta_formula": result.delta_formula,
        "interval_rate_formula": result.interval_rate_formula,
        "mean_rate_formula": result.mean_rate_formula,
        "aggregation_population": result.aggregation_population,
    }


def _flow_control_record(
    result: Any,
    *,
    collector: Any,
    node: Any,
    nic: Any,
    run_id: str,
    unsupported_series: tuple[str, ...],
) -> dict[str, Any]:
    counter_records = [
        _flow_control_counter_record(
            counter,
            collector=collector,
            node=node,
            nic=nic,
            run_id=run_id,
        )
        for counter in result.series_results
    ]
    first_nonzero_events = [
        (counter["first_nonzero_ns"], counter["counter"])
        for counter in counter_records
        if counter["first_nonzero_ns"] is not None
    ]
    first_nonzero = min(first_nonzero_events, default=None)
    return {
        "schema_version": FLOW_CONTROL_ANALYSIS_VERSION,
        "state": result.status.value,
        "status": result.status.value.replace("_", " ").title(),
        "status_reasons": result.status_reasons,
        "quality_flags": sorted(flag.value for flag in result.quality_flags),
        "population": {
            "run_id": run_id,
            "node_id": node.node_id,
            "nic_id": nic.nic_id,
            "collector_id": f"{collector.name} (ID {collector.collector_id})",
            "collector_name": collector.name,
            "collector_registry_id": collector.collector_id,
            "collector_rank": collector.rank,
            "collector_process": collector.process,
            "collector_type": collector.collector_type,
        },
        "requested_window_ns": result.requested_window_ns,
        "represented_window_ns": result.represented_window_ns,
        "requested_duration_ns": result.requested_duration_ns,
        "covered_duration_ns": result.covered_duration_ns,
        "uncovered_requested_duration_ns": result.uncovered_requested_duration_ns,
        "coverage_fraction": _finite_float(result.coverage_fraction),
        "counter_results": counter_records[:MAX_FLOW_CONTROL_COUNTER_RESULTS],
        "counter_count": len(counter_records),
        "counter_truncated": len(counter_records) > MAX_FLOW_CONTROL_COUNTER_RESULTS,
        "observed_counter_names": result.observed_counter_names,
        "missing_counter_names": result.missing_counter_names,
        "unsupported_counter_names": tuple(
            dict.fromkeys((*result.unsupported_counter_names, *unsupported_series))
        ),
        "zero_observed_counter_names": result.zero_observed_counter_names,
        "first_nonzero_counter_names": result.first_nonzero_counter_names,
        "first_nonzero_ns": first_nonzero[0] if first_nonzero else None,
        "pause_cycle_results_count": len(result.pause_cycle_results),
        "event_results_count": len(result.event_results),
        "xoff_event_results_count": len(result.xoff_event_results),
        "fgfc_event_results_count": len(result.fgfc_event_results),
        "delta_formula": next(
            (record["delta_formula"] for record in counter_records), None
        ),
        "interval_rate_formula": next(
            (record["interval_rate_formula"] for record in counter_records), None
        ),
        "mean_rate_formula": next(
            (record["mean_rate_formula"] for record in counter_records), None
        ),
    }


def _build_flow_control_analysis(
    store: TraceStore,
    node: Any,
    nic: Any,
    start_ns: int,
    end_ns: int,
    direction_filter: str,
) -> dict[str, Any]:
    if direction_filter == "tx":
        expected_counter_names = _FLOW_CONTROL_TX_COUNTERS
    elif direction_filter == "rx":
        expected_counter_names = _FLOW_CONTROL_RX_COUNTERS
    else:
        expected_counter_names = _FLOW_CONTROL_COUNTERS

    counter_by_name = {
        counter.name: counter
        for counter in store.counters
        if counter.name in expected_counter_names
    }
    series_by_collector: dict[int, list[MetricSeries]] = {}
    collectors: dict[int, Any] = {}
    unsupported_by_collector: dict[int, set[str]] = {}
    duplicate_stream_count = 0
    for counter_name in expected_counter_names:
        counter = counter_by_name.get(counter_name)
        if counter is None:
            continue
        try:
            query_result = store.query_series(
                node,
                nic,
                counter,
                start_ns=start_ns,
                end_ns=end_ns,
            )
        except TraceValidationError:
            continue
        for result in query_result.results:
            collector_id = result.collector.collector_id
            collectors[collector_id] = result.collector
            duplicate_stream_count = max(
                duplicate_stream_count,
                result.observation_stream_count
            )
            series = _flow_control_series(result)
            if series is None:
                unsupported_by_collector.setdefault(collector_id, set()).add(
                    counter_name
                )
                continue
            series_by_collector.setdefault(collector_id, []).append(series)

    records: list[dict[str, Any]] = []
    for collector_id in sorted(series_by_collector)[:MAX_FLOW_CONTROL_COLLECTORS]:
        analysis = analyze_flow_control(
            series_by_collector[collector_id],
            window_start_ns=start_ns,
            window_end_ns=end_ns - 1,
            expected_counter_names=expected_counter_names,
        )
        record = _flow_control_record(
            analysis,
            collector=collectors[collector_id],
            node=node,
            nic=nic,
            run_id=store.identity.run_id,
            unsupported_series=tuple(
                sorted(unsupported_by_collector.get(collector_id, ()))
            ),
        )
        records.append(record)

    if not records:
        return {
            **_empty_flow_control_analysis(),
            "requested_window_ns": (start_ns, end_ns - 1),
            "expected_counter_names": expected_counter_names,
            "duplicate_stream_count": duplicate_stream_count,
        }

    states = {record["state"] for record in records}
    first_record = records[0]
    first_nonzero_events = [
        (record["first_nonzero_ns"], record["population"]["collector_registry_id"])
        for record in records
        if record["first_nonzero_ns"] is not None
    ]
    first_nonzero_event = min(first_nonzero_events, default=None)
    collector_coverage = [
        {
            "collector_id": record["population"]["collector_id"],
            "collector_registry_id": record["population"][
                "collector_registry_id"
            ],
            "covered_duration_ns": record["covered_duration_ns"],
            "coverage_fraction": record["coverage_fraction"],
        }
        for record in records
    ]
    return {
        "schema_version": FLOW_CONTROL_ANALYSIS_VERSION,
        "question_id": "NIC-02",
        "state": next(iter(states)) if len(states) == 1 else "partial",
        "status": (
            next(iter(states)).replace("_", " ").title()
            if len(states) == 1
            else "Partial flow-control evidence"
        ),
        "status_reasons": tuple(
            dict.fromkeys(
                reason for record in records for reason in record["status_reasons"]
            )
        ),
        "quality_flags": sorted(
            {flag for record in records for flag in record["quality_flags"]}
        ),
        "collector_analyses": records,
        "collector_count": len(series_by_collector),
        "collector_limit": MAX_FLOW_CONTROL_COLLECTORS,
        "counter_limit": MAX_FLOW_CONTROL_COUNTER_RESULTS,
        "collector_truncated": len(series_by_collector) > MAX_FLOW_CONTROL_COLLECTORS,
        "counter_truncated": any(
            record["counter_truncated"] for record in records
        ),
        "duplicate_stream_count": duplicate_stream_count,
        "first_nonzero_ns": (
            first_nonzero_event[0] if first_nonzero_event is not None else None
        ),
        "requested_window_ns": first_record["requested_window_ns"],
        "collector_coverage": collector_coverage,
        "coverage_scope": (
            "Coverage is reported separately for each collector; no "
            "collector-merged or job-wide coverage value is reported."
        ),
        "delta_formula": first_record["delta_formula"],
        "mean_rate_formula": first_record["mean_rate_formula"],
        "expected_counter_names": expected_counter_names,
        "expected_priorities": DEFAULT_FLOW_CONTROL_PRIORITIES,
        "direction_source": "explicit flow-control counter catalog",
        "priority_source": "explicit flow-control counter catalog",
        "clock_conversion": "unsupported without verified clock evidence",
        "limitations": FLOW_CONTROL_LIMITATIONS,
    }


def _matching_quality_flags(flags: int) -> tuple[MatchingQualityFlag, ...]:
    mappings = {
        QualityFlag.REORDERED: MatchingQualityFlag.REORDERED,
        QualityFlag.NONMONOTONIC_TIMESTAMP: MatchingQualityFlag.CONFLICTING_TIMESTAMP,
        QualityFlag.DUPLICATE_REMOVED: MatchingQualityFlag.DUPLICATE,
        QualityFlag.BASELINE_RELATIVE: MatchingQualityFlag.BASELINE_RELATIVE,
        QualityFlag.PRECISION_LOSS_POSSIBLE: MatchingQualityFlag.UNSUPPORTED_SERIES,
        QualityFlag.RESET_SUSPECT: MatchingQualityFlag.DECREASE_OR_RESET,
        QualityFlag.WRAP_SUSPECT: MatchingQualityFlag.DECREASE_OR_RESET,
        QualityFlag.GAP: MatchingQualityFlag.PARTIAL_WINDOW,
        QualityFlag.UNKNOWN_METADATA: MatchingQualityFlag.UNSUPPORTED_SERIES,
    }
    return tuple(
        dict.fromkeys(
            mapped
            for source_flag, mapped in mappings.items()
            if int(flags) & source_flag
        )
    )


def _matching_series(result, spec: MatchingCounterSpec) -> MatchingCounterSeries | None:
    if result.counter.value_kind is not ValueKind.EXACT_UINT64:
        return None
    values: list[int] = []
    for value in result.values:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            if not math.isfinite(value) or not value.is_integer():
                return None
            value = int(value)
        if type(value) is not int or not 0 <= value <= 2**64 - 1:
            return None
        values.append(value)

    quality_flags = set(_matching_quality_flags(int(result.counter.quality_flags)))
    if result.quality_flags.size:
        quality_flags.update(
            _matching_quality_flags(int(np.bitwise_or.reduce(result.quality_flags)))
        )
    if result.counter.baseline_relative:
        quality_flags.add(MatchingQualityFlag.BASELINE_RELATIVE)
    return MatchingCounterSeries(
        spec=spec,
        identity=MatchingCounterIdentity(
            run_id=result.run_identity.run_id,
            node_id=str(result.node.node_id),
            nic_id=str(result.nic.nic_id),
            collector_id=f"{result.collector.name} (ID {result.collector.collector_id})",
            source_counter_id=result.counter.name,
        ),
        timestamps_ns=tuple(int(timestamp) for timestamp in result.timestamps_ns),
        values=tuple(values),
        value_kind=MatchingValueKind.CUMULATIVE,
        semantics=(
            "cumulative counter"
            if result.counter.semantics is CounterSemantics.CUMULATIVE
            else result.counter.semantics.value
        ),
        baseline_relative=result.counter.baseline_relative,
        quality_flags=tuple(quality_flags),
    )


def _matching_rate(total: int | None, covered_duration_ns: int | None) -> float | None:
    if total is None or covered_duration_ns is None or covered_duration_ns <= 0:
        return None
    return _finite_float(total / (covered_duration_ns / 1_000_000_000))


def _matching_counter_record(result: Any) -> dict[str, Any]:
    identity = result.source_identity
    return {
        "counter": result.spec.counter_id,
        "role": result.spec.role.value,
        "priority": result.spec.priority,
        "unit": result.unit,
        "state": result.status.value,
        "status": result.status.value.replace("_", " ").title(),
        "status_reasons": result.status_reasons,
        "quality_flags": sorted(flag.value for flag in result.quality_flags),
        "run_id": identity.run_id if identity else None,
        "node_id": int(identity.node_id) if identity else None,
        "nic_id": int(identity.nic_id) if identity else None,
        "collector_id": identity.collector_id if identity else None,
        "source_counter_id": (
            identity.source_counter_id if identity else result.spec.counter_id
        ),
        "population_id": result.spec.population_id,
        "requested_window_ns": result.requested_window_ns,
        "represented_window_ns": result.represented_window_ns,
        "requested_duration_ns": result.requested_duration_ns,
        "covered_duration_ns": result.covered_duration_ns,
        "coverage_fraction": _finite_float(result.coverage_fraction),
        "sample_count": result.sample_count,
        "valid_interval_count": result.valid_interval_count,
        "invalid_interval_count": result.invalid_interval_count,
        "outside_window_interval_count": result.outside_window_interval_count,
        "total": result.total,
        "rate_per_second": _matching_rate(
            result.total, result.covered_duration_ns
        ),
        "first_nonzero_observed": result.first_nonzero_sample_ns is not None,
        "persistent_zero": result.persistent_zero,
        "zero_observed": result.zero_observed,
        "baseline_relative": MatchingQualityFlag.BASELINE_RELATIVE in result.quality_flags,
    }


def _matching_priority_record(
    result: Any,
    covered_duration_ns: int | None,
) -> dict[str, Any]:
    return {
        "priority": result.priority,
        "priority_counter_id": result.priority_counter_id,
        "overflow_counter_id": result.overflow_counter_id,
        "priority_total": result.priority_total,
        "overflow_total": result.overflow_total,
        "total": result.total,
        "priority_fraction": _finite_float(result.priority_fraction),
        "overflow_fraction": _finite_float(result.overflow_fraction),
        "state": result.status.value,
        "status": result.status.value.replace("_", " ").title(),
        "status_reasons": result.status_reasons,
        "quality_flags": sorted(flag.value for flag in result.quality_flags),
        "rate_per_second": _matching_rate(result.total, covered_duration_ns),
        "priority_rate_per_second": _matching_rate(
            result.priority_total, covered_duration_ns
        ),
        "overflow_rate_per_second": _matching_rate(
            result.overflow_total, covered_duration_ns
        ),
    }


def _matching_record(
    result: Any,
    *,
    collector: Any,
    node: Any,
    nic: Any,
    run_id: str,
    unsupported_series: tuple[str, ...],
) -> dict[str, Any]:
    counter_records = [
        _matching_counter_record(counter) for counter in result.counter_results
    ]
    priority_records = [
        _matching_priority_record(priority, result.covered_duration_ns)
        for priority in result.priority_results
    ]
    return {
        "schema_version": MATCHING_ANALYSIS_VERSION,
        "state": result.status.value,
        "status": result.status.value.replace("_", " ").title(),
        "status_reasons": result.status_reasons,
        "quality_flags": sorted(flag.value for flag in result.quality_flags),
        "population": {
            "run_id": run_id,
            "node_id": node.node_id,
            "nic_id": nic.nic_id,
            "collector_id": f"{collector.name} (ID {collector.collector_id})",
            "collector_name": collector.name,
            "collector_registry_id": collector.collector_id,
            "collector_rank": collector.rank,
            "collector_process": collector.process,
            "collector_type": collector.collector_type,
        },
        "requested_window_ns": result.requested_window_ns,
        "represented_window_ns": result.represented_window_ns,
        "requested_duration_ns": result.requested_duration_ns,
        "covered_duration_ns": result.covered_duration_ns,
        "coverage_fraction": _finite_float(result.coverage_fraction),
        "counter_results": counter_records[:MAX_MATCHING_COUNTER_RESULTS],
        "counter_count": len(counter_records),
        "counter_truncated": len(counter_records) > MAX_MATCHING_COUNTER_RESULTS,
        "priority_results": priority_records[:MAX_MATCHING_PRIORITY_RESULTS],
        "priority_count": len(priority_records),
        "priority_truncated": len(priority_records) > MAX_MATCHING_PRIORITY_RESULTS,
        "priority_total": result.priority_total,
        "overflow_total": result.overflow_total,
        "total": result.total,
        "priority_fraction": _finite_float(result.priority_fraction),
        "overflow_fraction": _finite_float(result.overflow_fraction),
        "population_compatible": result.population_compatible,
        "compatible_counter_ids": result.compatible_counter_ids,
        "incompatible_counter_ids": result.incompatible_counter_ids,
        "missing_counter_ids": result.missing_counter_ids,
        "unsupported_counter_names": unsupported_series,
        "temporal_interval_count": len(result.temporal_intervals),
        "first_nonzero_observed": any(
            counter["first_nonzero_observed"] for counter in counter_records
        ),
        "persistent_zero": bool(counter_records)
        and all(counter["persistent_zero"] for counter in counter_records),
        "zero_observed": any(counter["zero_observed"] for counter in counter_records),
        "delta_formula": result.delta_formula,
        "priority_fraction_formula": result.priority_fraction_formula,
        "overflow_fraction_formula": result.overflow_fraction_formula,
        "aggregation_population": result.aggregation_population,
    }


def _build_matching_analysis(
    store: TraceStore,
    node: Any,
    nic: Any,
    start_ns: int,
    end_ns: int,
    direction_filter: str,
) -> dict[str, Any]:
    if direction_filter == "tx":
        return {
            **_empty_matching_analysis(),
            "state": "missing",
            "status": "No receive-path matching evidence",
            "status_reasons": (
                "NIC-03 matching counters are receive-path evidence; select RX or all directions.",
            ),
            "requested_window_ns": (start_ns, end_ns - 1),
            "direction_filter": direction_filter,
        }

    counter_by_name = {counter.name: counter for counter in store.counters}
    specs = tuple(
        MatchingCounterSpec(
            counter_id=counter_name,
            role=(
                MatchingRole.PRIORITY
                if counter_name.startswith("lpe_net_match_priority_")
                else MatchingRole.OVERFLOW
            ),
            priority=int(counter_name.rsplit("_", 1)[1]),
            unit=counter_by_name[counter_name].unit
            if counter_name in counter_by_name
            else "matches",
            population_id=_MATCHING_POPULATION_ID,
        )
        for counter_name in _MATCHING_COUNTERS
    )
    series_by_collector: dict[int, list[MatchingCounterSeries]] = {}
    collectors: dict[int, Any] = {}
    unsupported_by_collector: dict[int, set[str]] = {}
    duplicate_stream_count = 0
    for spec in specs:
        counter = counter_by_name.get(spec.counter_id)
        if counter is None:
            continue
        try:
            query_result = store.query_series(
                node,
                nic,
                counter,
                start_ns=start_ns,
                end_ns=end_ns,
            )
        except TraceValidationError:
            continue
        for result in query_result.results:
            collector_id = result.collector.collector_id
            collectors[collector_id] = result.collector
            duplicate_stream_count = max(
                duplicate_stream_count, result.observation_stream_count
            )
            series = _matching_series(result, spec)
            if series is None:
                unsupported_by_collector.setdefault(collector_id, set()).add(
                    spec.counter_id
                )
                continue
            series_by_collector.setdefault(collector_id, []).append(series)

    collector_ids = sorted(series_by_collector)
    records = []
    for collector_id in collector_ids[:MAX_MATCHING_COLLECTORS]:
        analysis = analyze_matching(
            MatchingDataset(
                population=MatchingPopulation(
                    run_id=store.identity.run_id,
                    node_id=str(node.node_id),
                    nic_id=str(nic.nic_id),
                    collector_id=series_by_collector[collector_id][0].identity.collector_id,
                ),
                requested_counters=specs,
                series=tuple(series_by_collector[collector_id]),
                window_ns=(start_ns, end_ns - 1),
            )
        )
        records.append(
            _matching_record(
                analysis,
                collector=collectors[collector_id],
                node=node,
                nic=nic,
                run_id=store.identity.run_id,
                unsupported_series=tuple(
                    sorted(unsupported_by_collector.get(collector_id, ()))
                ),
            )
        )

    if not records:
        return {
            **_empty_matching_analysis(),
            "requested_window_ns": (start_ns, end_ns - 1),
            "duplicate_stream_count": duplicate_stream_count,
            "direction_filter": direction_filter,
        }

    states = {record["state"] for record in records}
    state = next(iter(states)) if len(states) == 1 else "partial"
    collector_priority_fractions = [
        {
            "collector_id": record["population"]["collector_id"],
            "collector_registry_id": record["population"][
                "collector_registry_id"
            ],
            "state": record["state"],
            "population_compatible": record["population_compatible"],
            "priority_fraction": record["priority_fraction"],
            "overflow_fraction": record["overflow_fraction"],
            "fraction_status": (
                "undefined"
                if record["priority_fraction"] is None
                else record["state"]
            ),
        }
        for record in records[:MAX_MATCHING_COLLECTORS]
    ]
    return {
        "schema_version": MATCHING_ANALYSIS_VERSION,
        "question_id": "NIC-03",
        "state": state,
        "status": state.replace("_", " ").title(),
        "status_reasons": tuple(
            dict.fromkeys(
                reason for record in records for reason in record["status_reasons"]
            )
        ),
        "quality_flags": sorted(
            {flag for record in records for flag in record["quality_flags"]}
        ),
        "collector_analyses": records,
        "collector_count": len(collector_ids),
        "collector_limit": MAX_MATCHING_COLLECTORS,
        "counter_limit": MAX_MATCHING_COUNTER_RESULTS,
        "collector_truncated": len(collector_ids) > MAX_MATCHING_COLLECTORS,
        "counter_truncated": any(
            record["counter_truncated"] for record in records
        ),
        "priority_limit": MAX_MATCHING_PRIORITY_RESULTS,
        "priority_count": sum(record["priority_count"] for record in records),
        "priority_truncated": any(
            record["priority_truncated"] for record in records
        ),
        "collector_priority_fractions": collector_priority_fractions,
        "priority_fraction_formula": records[0]["priority_fraction_formula"],
        "overflow_fraction_formula": records[0]["overflow_fraction_formula"],
        "priority_fraction_scope": (
            "Collector-separated values; incompatible or heterogeneous collectors are not merged."
        ),
        "duplicate_stream_count": duplicate_stream_count,
        "requested_window_ns": records[0]["requested_window_ns"],
        "population_id": _MATCHING_POPULATION_ID,
        "population_compatibility_source": (
            "Explicit catalog pairing of the four priority and overflow receive-match "
            "subcounters; duplicate collectors remain separate."
        ),
        "direction_filter": direction_filter,
        "collector_coverage": [
            {
                "collector_id": record["population"]["collector_id"],
                "collector_registry_id": record["population"][
                    "collector_registry_id"
                ],
                "covered_duration_ns": record["covered_duration_ns"],
                "coverage_fraction": record["coverage_fraction"],
            }
            for record in records
        ],
        "coverage_scope": (
            "Coverage is reported separately for each collector; no "
            "collector-merged or job-wide coverage value is reported."
        ),
        "expected_counter_names": _MATCHING_COUNTERS,
        "interpretation": MATCHING_INTERPRETATION,
        "limitations": MATCHING_LIMITATIONS,
    }


def _empty_host_pressure_analysis() -> dict[str, Any]:
    return {
        "schema_version": HOST_PRESSURE_ANALYSIS_VERSION,
        "question_id": "NIC-04",
        "state": "missing",
        "status": "Missing",
        "status_reasons": ("No host-pressure counters are available for this NIC.",),
        "quality_flags": [],
        "collector_analyses": (),
        "collector_count": 0,
        "collector_limit": MAX_HOST_PRESSURE_COLLECTORS,
        "path_limit": MAX_HOST_PRESSURE_PATH_RESULTS,
        "stall_limit": MAX_HOST_PRESSURE_STALL_RESULTS,
        "collector_truncated": False,
        "path_truncated": False,
        "stall_truncated": False,
        "duplicate_stream_count": 0,
        "requested_window_ns": None,
        "collector_coverage": (),
        "coverage_scope": (
            "Coverage is reported separately for each collector; no collector-merged "
            "or job-wide coverage value is reported."
        ),
        "expected_counter_names": (
            *_HOST_PRESSURE_PACKET_COUNTERS.values(),
            *_HOST_PRESSURE_BLOCKED_COUNTERS.values(),
            *(name for name, _ in _HOST_PRESSURE_STALL_COUNTERS),
        ),
        "missing_counter_ids": (),
        "unsupported_counter_names": (),
        "blocked_unit": "unresolved",
        "blocked_unit_resolution": None,
        "blocked_per_packet": None,
        "blocked_per_packet_unit": None,
        "blocked_per_packet_status": (
            "Unsupported until authoritative blocked-counter units are supplied."
        ),
        "limitations": HOST_PRESSURE_LIMITATIONS,
    }


def _host_pressure_quality_flags(flags: int) -> tuple[HostPressureQualityFlag, ...]:
    mappings = {
        QualityFlag.REORDERED: HostPressureQualityFlag.REORDERED,
        QualityFlag.NONMONOTONIC_TIMESTAMP: HostPressureQualityFlag.CONFLICTING_TIMESTAMP,
        QualityFlag.DUPLICATE_REMOVED: HostPressureQualityFlag.DUPLICATE,
        QualityFlag.BASELINE_RELATIVE: HostPressureQualityFlag.BASELINE_RELATIVE,
        QualityFlag.PRECISION_LOSS_POSSIBLE: HostPressureQualityFlag.UNSUPPORTED_SERIES,
        QualityFlag.RESET_SUSPECT: HostPressureQualityFlag.DECREASE_OR_RESET,
        QualityFlag.WRAP_SUSPECT: HostPressureQualityFlag.DECREASE_OR_RESET,
        QualityFlag.GAP: HostPressureQualityFlag.PARTIAL_WINDOW,
        QualityFlag.UNKNOWN_METADATA: HostPressureQualityFlag.UNSUPPORTED_SERIES,
    }
    return tuple(
        dict.fromkeys(
            mapped
            for source_flag, mapped in mappings.items()
            if int(flags) & source_flag
        )
    )


def _host_pressure_series(
    result,
    spec: HostPressureSeriesSpec,
) -> HostPressureSeries | None:
    values: list[int] = []
    for value in result.values:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            if not math.isfinite(value) or not value.is_integer():
                return None
            value = int(value)
        if type(value) is not int or not 0 <= value <= 2**64 - 1:
            return None
        values.append(value)

    quality_flags = set(
        _host_pressure_quality_flags(int(result.counter.quality_flags))
    )
    quality_flags.update(
        _host_pressure_quality_flags(int(np.bitwise_or.reduce(result.quality_flags)))
    )
    collector = result.collector
    return HostPressureSeries(
        spec=spec,
        identity=HostPressureSeriesIdentity(
            run_id=result.run_identity.run_id,
            node_id=str(result.node.node_id),
            nic_id=str(result.nic.nic_id),
            collector_id=f"{collector.name} (ID {collector.collector_id})",
            source_counter_id=result.counter.name,
        ),
        timestamps_ns=tuple(int(timestamp) for timestamp in result.timestamps_ns),
        values=tuple(values),
        value_kind=(
            HostPressureValueKind.CUMULATIVE
            if result.counter.value_kind is ValueKind.EXACT_UINT64
            else HostPressureValueKind.UNKNOWN
        ),
        semantics=(
            "cumulative counter"
            if result.counter.semantics is CounterSemantics.CUMULATIVE
            else result.counter.semantics.value
        ),
        baseline_relative=result.counter.baseline_relative,
        quality_flags=tuple(quality_flags),
    )


def _host_pressure_counter_record(result: Any) -> dict[str, Any]:
    identity = result.source_identity
    return {
        "counter": result.spec.counter_id,
        "counter_kind": result.spec.counter_kind.value,
        "path": result.spec.path.value if result.spec.path else None,
        "stall_kind": result.spec.stall_kind.value if result.spec.stall_kind else None,
        "unit": result.unit,
        "rate_unit": result.rate_unit,
        "state": result.status.value,
        "status": result.status.value.replace("_", " ").title(),
        "status_reasons": result.status_reasons,
        "quality_flags": sorted(flag.value for flag in result.quality_flags),
        "run_id": identity.run_id if identity else None,
        "node_id": int(identity.node_id) if identity else None,
        "nic_id": int(identity.nic_id) if identity else None,
        "collector_id": identity.collector_id if identity else None,
        "source_counter_id": (
            identity.source_counter_id if identity else result.spec.counter_id
        ),
        "requested_window_ns": result.requested_window_ns,
        "represented_window_ns": result.represented_window_ns,
        "requested_duration_ns": result.requested_duration_ns,
        "covered_duration_ns": result.covered_duration_ns,
        "coverage_fraction": _finite_float(result.coverage_fraction),
        "sample_count": result.sample_count,
        "valid_interval_count": result.valid_interval_count,
        "invalid_interval_count": result.invalid_interval_count,
        "outside_window_interval_count": result.outside_window_interval_count,
        "total": result.total,
        "rate_per_second": _finite_float(result.rate_per_second),
        "zero_observed": result.zero_observed,
        "baseline_relative": (
            HostPressureQualityFlag.BASELINE_RELATIVE in result.quality_flags
        ),
    }


def _host_pressure_path_record(result: Any) -> dict[str, Any]:
    return {
        "path": result.path.value,
        "state": result.status.value,
        "status": result.status.value.replace("_", " ").title(),
        "status_reasons": result.status_reasons,
        "quality_flags": sorted(flag.value for flag in result.quality_flags),
        "requested_window_ns": result.requested_window_ns,
        "represented_window_ns": result.represented_window_ns,
        "requested_duration_ns": result.requested_duration_ns,
        "covered_duration_ns": result.covered_duration_ns,
        "coverage_fraction": _finite_float(result.coverage_fraction),
        "packet_result": _host_pressure_counter_record(result.packet_result),
        "blocked_result": _host_pressure_counter_record(result.blocked_result),
        "packet_total": result.packet_total,
        "blocked_total": result.blocked_total,
        "packet_rate_per_second": _finite_float(result.packet_rate_per_second),
        "blocked_rate_per_second": _finite_float(result.blocked_rate_per_second),
        "blocked_per_packet": _finite_float(result.blocked_per_packet),
        "blocked_per_packet_unit": result.blocked_per_packet_unit,
        "denominator_state": result.denominator_state.value,
        "population_compatible": result.common_valid_interval_count > 0,
        "common_valid_interval_count": result.common_valid_interval_count,
        "reset_excluded_interval_count": result.reset_excluded_interval_count,
        "packet_delta_formula": result.packet_delta_formula,
        "rate_formula": result.rate_formula,
        "blocked_per_packet_formula": result.blocked_per_packet_formula,
    }


def _build_host_pressure_analysis(
    store: TraceStore,
    node: Any,
    nic: Any,
    start_ns: int,
    end_ns: int,
) -> dict[str, Any]:
    specs: list[HostPressureSeriesSpec] = []
    counter_by_name = {counter.name: counter for counter in store.counters}
    for path, counter_name in _HOST_PRESSURE_PACKET_COUNTERS.items():
        counter = counter_by_name.get(counter_name)
        specs.append(
            HostPressureSeriesSpec(
                counter_id=counter_name,
                counter_kind=HostPressureCounterKind.PACKET,
                unit=counter.unit if counter else "packets",
                path=path,
            )
        )
    for path, counter_name in _HOST_PRESSURE_BLOCKED_COUNTERS.items():
        counter = counter_by_name.get(counter_name)
        specs.append(
            HostPressureSeriesSpec(
                counter_id=counter_name,
                counter_kind=HostPressureCounterKind.BLOCKED,
                unit=counter.unit if counter else "unresolved",
                path=path,
            )
        )
    for counter_name, stall_kind in _HOST_PRESSURE_STALL_COUNTERS:
        counter = counter_by_name.get(counter_name)
        specs.append(
            HostPressureSeriesSpec(
                counter_id=counter_name,
                counter_kind=HostPressureCounterKind.STALL,
                unit=counter.unit if counter else "unresolved",
                stall_kind=stall_kind,
            )
        )

    series_by_collector: dict[int, list[HostPressureSeries]] = {}
    collectors: dict[int, Any] = {}
    unsupported_by_collector: dict[int, set[str]] = {}
    duplicate_stream_count = 0
    for spec in specs:
        counter = counter_by_name.get(spec.counter_id)
        if counter is None:
            continue
        try:
            query_result = store.query_series(
                node,
                nic,
                counter,
                start_ns=start_ns,
                end_ns=end_ns,
            )
        except TraceValidationError:
            continue
        for result in query_result.results:
            collector_id = result.collector.collector_id
            collectors[collector_id] = result.collector
            duplicate_stream_count = max(
                duplicate_stream_count, result.observation_stream_count
            )
            series = _host_pressure_series(result, spec)
            if series is None:
                unsupported_by_collector.setdefault(collector_id, set()).add(
                    spec.counter_id
                )
                continue
            series_by_collector.setdefault(collector_id, []).append(series)

    collector_ids = sorted(series_by_collector)
    records = []
    for collector_id in collector_ids[:MAX_HOST_PRESSURE_COLLECTORS]:
        analysis = analyze_host_pressure(
            HostPressureDataset(
                population=HostPressurePopulation(
                    run_id=store.identity.run_id,
                    node_id=str(node.node_id),
                    nic_id=str(nic.nic_id),
                    collector_id=series_by_collector[collector_id][0].identity.collector_id,
                ),
                series=tuple(series_by_collector[collector_id]),
                window_ns=(start_ns, end_ns - 1),
            )
        )
        records.append(
            {
                "schema_version": HOST_PRESSURE_ANALYSIS_VERSION,
                "state": analysis.status.value,
                "status": analysis.status.value.replace("_", " ").title(),
                "status_reasons": analysis.status_reasons,
                "quality_flags": sorted(flag.value for flag in analysis.quality_flags),
                "population": {
                    "run_id": store.identity.run_id,
                    "node_id": node.node_id,
                    "nic_id": nic.nic_id,
                    "collector_id": f"{collectors[collector_id].name} (ID {collector_id})",
                    "collector_name": collectors[collector_id].name,
                    "collector_registry_id": collector_id,
                    "collector_rank": collectors[collector_id].rank,
                    "collector_process": collectors[collector_id].process,
                    "collector_type": collectors[collector_id].collector_type,
                },
                "requested_window_ns": analysis.requested_window_ns,
                "represented_window_ns": analysis.represented_window_ns,
                "requested_duration_ns": (
                    analysis.requested_window_ns[1] - analysis.requested_window_ns[0]
                ),
                "covered_duration_ns": None,
                "coverage_fraction": _finite_float(analysis.coverage_fraction),
                "path_results": [
                    _host_pressure_path_record(result) for result in analysis.path_results
                ],
                "stall_results": [
                    _host_pressure_counter_record(result)
                    for result in analysis.stall_results
                ],
                "missing_counter_ids": analysis.missing_counter_ids,
                "unsupported_counter_names": tuple(
                    sorted(unsupported_by_collector.get(collector_id, ()))
                ),
                "blocked_unit": analysis.blocked_unit,
                "blocked_unit_resolution": (
                    {
                        "unit": analysis.blocked_unit_resolution.unit.value,
                        "evidence": analysis.blocked_unit_resolution.evidence,
                    }
                    if analysis.blocked_unit_resolution
                    else None
                ),
                "blocked_per_packet": None,
                "blocked_per_packet_unit": None,
            }
        )

    if not records:
        missing = {
            name
            for name in (
                *_HOST_PRESSURE_PACKET_COUNTERS.values(),
                *_HOST_PRESSURE_BLOCKED_COUNTERS.values(),
            )
            if name not in counter_by_name
        }
        unsupported_names = {
            name
            for names in unsupported_by_collector.values()
            for name in names
        }
        return {
            **_empty_host_pressure_analysis(),
            "status": (
                "Unsupported series"
                if unsupported_names and not missing
                else "No host-pressure evidence"
            ),
            "status_reasons": (
                "One or more host-pressure series have unsupported value types."
                if unsupported_names
                else "No host-pressure counters are available for this NIC.",
            ),
            "requested_window_ns": (start_ns, end_ns - 1),
            "missing_counter_ids": tuple(sorted(missing)),
            "unsupported_counter_names": tuple(sorted(unsupported_names)),
            "duplicate_stream_count": duplicate_stream_count,
        }

    selected_records = records[:MAX_HOST_PRESSURE_COLLECTORS]
    missing_stall_counter_ids = {
        counter_name
        for counter_name, _ in _HOST_PRESSURE_STALL_COUNTERS
        if counter_name not in counter_by_name
    }
    path_count = 0
    stall_count = 0
    for record in selected_records:
        remaining_paths = MAX_HOST_PRESSURE_PATH_RESULTS - path_count
        path_results = record["path_results"][: max(remaining_paths, 0)]
        path_count += len(path_results)
        record["path_count"] = len(record["path_results"])
        record["path_truncated"] = len(path_results) < record["path_count"]
        record["path_results"] = path_results
        remaining_stalls = MAX_HOST_PRESSURE_STALL_RESULTS - stall_count
        stall_results = record["stall_results"][: max(remaining_stalls, 0)]
        stall_count += len(stall_results)
        record["stall_count"] = len(record["stall_results"])
        record["stall_truncated"] = len(stall_results) < record["stall_count"]
        record["stall_results"] = stall_results

    states = {record["state"] for record in selected_records}
    state = next(iter(states)) if len(states) == 1 else "partial"
    total_path_count = sum(record["path_count"] for record in records)
    total_stall_count = sum(record["stall_count"] for record in records)
    return {
        "schema_version": HOST_PRESSURE_ANALYSIS_VERSION,
        "question_id": "NIC-04",
        "state": state,
        "status": state.replace("_", " ").title(),
        "status_reasons": tuple(
            dict.fromkeys(
                reason for record in selected_records for reason in record["status_reasons"]
            )
        ),
        "quality_flags": sorted(
            {flag for record in selected_records for flag in record["quality_flags"]}
        ),
        "collector_analyses": selected_records,
        "collector_count": len(series_by_collector),
        "collector_limit": MAX_HOST_PRESSURE_COLLECTORS,
        "path_limit": MAX_HOST_PRESSURE_PATH_RESULTS,
        "stall_limit": MAX_HOST_PRESSURE_STALL_RESULTS,
        "collector_truncated": len(series_by_collector) > MAX_HOST_PRESSURE_COLLECTORS,
        "path_truncated": total_path_count > MAX_HOST_PRESSURE_PATH_RESULTS,
        "stall_truncated": total_stall_count > MAX_HOST_PRESSURE_STALL_RESULTS,
        "duplicate_stream_count": duplicate_stream_count,
        "requested_window_ns": selected_records[0]["requested_window_ns"],
        "collector_coverage": [
            {
                "collector_id": record["population"]["collector_id"],
                "collector_registry_id": record["population"]["collector_registry_id"],
                "covered_duration_ns": record["covered_duration_ns"],
                "coverage_fraction": record["coverage_fraction"],
            }
            for record in selected_records
        ],
        "coverage_scope": (
            "Coverage is reported separately for each collector; no collector-merged "
            "or job-wide coverage value is reported."
        ),
        "expected_counter_names": (
            *_HOST_PRESSURE_PACKET_COUNTERS.values(),
            *_HOST_PRESSURE_BLOCKED_COUNTERS.values(),
            *(name for name, _ in _HOST_PRESSURE_STALL_COUNTERS),
        ),
            "missing_counter_ids": tuple(
                dict.fromkeys(
                    (
                        *missing_stall_counter_ids,
                        *(
                            name
                            for record in selected_records
                            for name in record["missing_counter_ids"]
                        ),
                    ),
                )
            ),
        "unsupported_counter_names": tuple(
            dict.fromkeys(
                name
                for record in selected_records
                for name in record["unsupported_counter_names"]
            )
        ),
        "blocked_unit": "unresolved",
        "blocked_unit_resolution": None,
        "blocked_per_packet": None,
        "blocked_per_packet_unit": None,
        "blocked_per_packet_status": (
            "Unsupported until authoritative blocked-counter units are supplied."
        ),
        "packet_delta_formula": "end_value - start_value",
        "rate_formula": (
            "sum(valid interval deltas) / sum(valid interval durations)"
        ),
        "blocked_per_packet_formula": (
            "sum(blocked deltas) / sum(packet deltas)"
        ),
        "population_compatibility_scope": (
            "Posted and nonposted populations remain separate; packet and blocked "
            "series are paired only on exact valid interval boundaries within each collector."
        ),
        "limitations": HOST_PRESSURE_LIMITATIONS,
    }


def _empty_view_model(
    store: TraceStore,
    filters: AnalysisFilterState,
    *,
    traffic_state: str,
    traffic_status: str,
    reasons: tuple[str, ...],
) -> dict[str, Any]:
    base = _base_view_model(store, filters)
    base["overview"]["traffic_state"] = traffic_state
    base["overview"]["traffic_status"] = traffic_status
    base["overview"]["traffic_status_reasons"] = reasons
    base["overview"]["tx_total_bytes"] = None
    base["overview"]["rx_total_bytes"] = None
    base["overview"]["zero_observation_count"] = 0
    base["overview"]["missing_observation_count"] = 0
    base["overview"]["partial_observation_count"] = 0
    base["overview"]["unsupported_observation_count"] = 0
    base["node_comparison"] = {
        "rows": [],
        "status": traffic_status,
        "state": traffic_state,
        "truncated": False,
        "coverage_definition": "Not applicable; no analyzed series.",
    }
    base["job_balance"] = _empty_job_balance()
    base["nic_detail"] = {
        "selected_node": None,
        "selected_nic": None,
        "selection_note": "No NIC matches the current filters.",
        "series": [],
        "reliability": _empty_reliability_analysis(),
        "matching": _empty_matching_analysis(),
        "flow_control": _empty_flow_control_analysis(),
        "frames": _empty_frame_analysis(),
        "translation": _empty_translation_analysis(),
        "host_pressure": _empty_host_pressure_analysis(),
        "operations": _empty_operation_analysis(),
        "latency": _empty_latency_analysis(),
    }
    base["phase_analysis"] = _empty_phase_analysis()
    base["job_profile"] = _empty_job_profile()
    return base


def _base_view_model(store: TraceStore, filters: AnalysisFilterState) -> dict[str, Any]:
    coverage = store.coverage
    requested = set(coverage.requested_counter_names)
    available = set(coverage.available_counter_names)
    readable = set(coverage.readable_counter_names)
    observed = set(coverage.observed_counter_names)
    unreadable = set(coverage.unreadable_counter_names)
    acquisition_duration = (
        coverage.acquisition_end_ns - coverage.acquisition_start_ns
        if coverage.acquisition_start_ns is not None
        and coverage.acquisition_end_ns is not None
        else None
    )
    selected_nodes = tuple(
        node for node in store.nodes if filters.node_id is None or node.node_id == filters.node_id
    )
    selected_nics = tuple(
        nic
        for nic in store.nics
        if (filters.nic_id is None or nic.nic_id == filters.nic_id)
        and any(node.node_id == nic.node_id for node in selected_nodes)
    )
    return {
        "schema_version": 1,
        "run": {
            "run_id": store.identity.run_id,
            "source_format": store.identity.source_format,
            "source_fingerprint": store.identity.source_fingerprint,
            "reader_version": store.identity.reader_version,
            "clock_domain": store.identity.clock_domain,
            "synchronization_evidence": store.identity.synchronization_evidence,
        },
        "capture": {
            "acquisition_start_ns": coverage.acquisition_start_ns,
            "acquisition_end_ns": coverage.acquisition_end_ns,
            "acquisition_duration_ns": acquisition_duration,
            "selected_start_ns": filters.start_ns,
            "selected_end_ns": filters.end_ns,
            "selected_duration_ns": (
                filters.end_ns - filters.start_ns
                if filters.start_ns is not None and filters.end_ns is not None
                else None
            ),
            "node_count": len(selected_nodes),
            "nic_count": len(selected_nics),
            "counter_count": len(store.counters),
            "row_count": store.row_count,
            "requested_counter_count": len(requested),
            "available_counter_count": len(available),
            "readable_counter_count": len(readable),
            "observed_counter_count": len(observed),
            "missing_counter_names": tuple(sorted(requested - observed)),
            "unreadable_counter_names": tuple(sorted(unreadable)),
            "available_not_observed_counter_names": tuple(sorted(readable - observed)),
            "lost_record_count": coverage.lost_record_count,
            "gap_count": coverage.gap_count,
            "notes": coverage.notes,
        },
        "filters": {
            "node_id": filters.node_id,
            "nic_id": filters.nic_id,
            "direction": filters.direction,
            "start_ns": filters.start_ns,
            "end_ns": filters.end_ns,
        },
        "overview": {
            "traffic_state": "missing",
            "traffic_status": "Missing",
            "traffic_status_reasons": (),
        },
        "series_summaries": [],
        "question_coverage": _question_rows(store),
    }


def _phase_error(
    status: str, reasons: tuple[str, ...]
) -> dict[str, Any]:
    return {
        **_empty_phase_analysis(),
        "state": "invalid",
        "status": status,
        "status_reasons": reasons,
    }


def _normalize_phase_windows(phase_windows: Any) -> list[PhaseWindow]:
    if isinstance(phase_windows, (str, bytes)) or not isinstance(
        phase_windows, (list, tuple)
    ):
        raise ValueError("phase windows must be a list of objects")
    if not phase_windows:
        return []
    if len(phase_windows) > MAX_PHASE_WINDOWS:
        raise ValueError(f"at most {MAX_PHASE_WINDOWS} phase windows are supported")

    normalized: list[PhaseWindow] = []
    for item in phase_windows:
        if isinstance(item, PhaseWindow):
            normalized.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError("every phase window must be an object")
        name = item.get("name")
        start_ns = item.get("start_ns")
        end_ns = item.get("end_ns")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("every phase name must be a nonempty string")
        if isinstance(start_ns, bool) or not isinstance(start_ns, int):
            raise ValueError("phase start must be an integer nanosecond value")
        if isinstance(end_ns, bool) or not isinstance(end_ns, int):
            raise ValueError("phase end must be an integer nanosecond value")
        normalized.append(
            PhaseWindow(name=name.strip(), start_ns=start_ns, end_ns=end_ns)
        )
    normalized.sort(key=lambda phase: (phase.start_ns, phase.end_ns, phase.name))
    return normalized


def build_phase_analysis(
    store: TraceStore,
    filters: AnalysisFilterState,
    phase_windows: Any = (),
) -> dict[str, Any]:
    """Run full-detail phase analysis for one selected node/NIC/direction."""

    if filters.start_ns is None or filters.end_ns is None:
        defaults = default_filters(store)
        filters = AnalysisFilterState(
            node_id=filters.node_id,
            nic_id=filters.nic_id,
            direction=filters.direction,
            start_ns=(
                filters.start_ns
                if filters.start_ns is not None
                else defaults.start_ns
            ),
            end_ns=filters.end_ns if filters.end_ns is not None else defaults.end_ns,
        )
    try:
        phases = _normalize_phase_windows(phase_windows)
    except ValueError as error:
        return _phase_error("Invalid phase input", (str(error),))
    if not phases:
        return _empty_phase_analysis()
    if filters.start_ns is None or filters.end_ns is None or filters.start_ns >= filters.end_ns:
        return _phase_error(
            "Invalid time window",
            ("The selected time window must have a positive duration.",),
        )

    directions = _directions(filters.direction)
    if len(directions) != 1:
        return _phase_error(
            "Direction required",
            ("Select TX or RX before analyzing a manual phase.",),
        )
    direction = directions[0]
    selected_nodes = tuple(
        node
        for node in store.nodes
        if filters.node_id is None or node.node_id == filters.node_id
    )
    selected_nics = tuple(
        nic
        for nic in store.nics
        if (filters.nic_id is None or nic.nic_id == filters.nic_id)
        and any(node.node_id == nic.node_id for node in selected_nodes)
    )
    detail_nic = selected_nics[0] if selected_nics else None
    detail_node = next(
        (
            node
            for node in selected_nodes
            if detail_nic and node.node_id == detail_nic.node_id
        ),
        None,
    )
    if detail_nic is None or detail_node is None:
        return {
            **_empty_phase_analysis(),
            "state": "missing",
            "status": "No selected series",
            "status_reasons": ("No node/NIC series matches the current filters.",),
        }

    counter, _ = _selected_counter(store.counters, direction)
    if counter is None:
        requested_name = _requested_counter_name_for_direction(store, direction)
        if requested_name is not None:
            return {
                **_empty_phase_analysis(),
                "state": "missing",
                "status": "Missing phase data",
                "status_reasons": (
                    "The requested TX/RX byte counter is not present in this dataset.",
                ),
            }
        return {
            **_empty_phase_analysis(),
            "state": "unsupported",
            "status": "Unsupported direction",
            "status_reasons": (
                "No cumulative byte counter is available for the selected direction.",
            ),
        }
    try:
        query_result = store.query_series(
            detail_node,
            detail_nic,
            counter,
            start_ns=filters.start_ns,
            end_ns=filters.end_ns,
        )
    except TraceValidationError:
        return {
            **_empty_phase_analysis(),
            "state": "missing",
            "status": "Missing phase data",
            "status_reasons": (
                "The selected node/NIC has no counter series for this phase context.",
            ),
        }
    if len(query_result.results) != 1:
        return {
            **_empty_phase_analysis(),
            "state": "partial",
            "status": "Ambiguous series",
            "status_reasons": (
                "The selected physical NIC has multiple observation streams; they are not merged.",
            ),
        }

    result = query_result.results[0]
    metric_series = metric_series_from_series_result(
        result,
        run_id=store.identity.run_id,
        direction=direction,
    )
    detailed_analysis = analyze_traffic_series(
        metric_series,
        window_start_ns=filters.start_ns,
        window_end_ns=filters.end_ns,
        include_intervals=True,
    )
    records = {
        phase.name: PhaseTrafficRecord(
            clock_id=store.identity.clock_domain,
            analysis=detailed_analysis,
        )
        for phase in phases
    }
    try:
        phase_result = analyze_manual_phase_windows(
            phases,
            records,
            gap_policy=PhaseGapPolicy.REJECT_GAPS,
        )
    except ValueError as error:
        return _phase_error("Invalid phase windows", (str(error),))
    return _phase_analysis_record(phase_result)


def _summary_status(state: str, value: int | float | None) -> SummaryStatus:
    if value == 0:
        return SummaryStatus.OBSERVED_ZERO
    if state in ("ready", "ok"):
        return SummaryStatus.OBSERVED
    if state in ("partial", "zero", "invalid"):
        return SummaryStatus.PARTIAL
    if state == "missing":
        return SummaryStatus.MISSING
    return SummaryStatus.UNSUPPORTED


def _profile_summaries(
    view: dict[str, Any],
    run_id: str,
    window_ns: tuple[int, int],
) -> tuple[list[ProfileSummary], list[ProfileSummary], list[ProfileSummary]]:
    traffic: list[ProfileSummary] = []
    job: list[ProfileSummary] = []
    phase: list[ProfileSummary] = []

    for index, record in enumerate(view.get("series_summaries", ())[:MAX_EXPORT_SERIES]):
        total = record.get("total_bytes")
        if total is None:
            continue
        traffic.append(
            ProfileSummary(
                analysis_id=(
                    f"traffic.{record.get('node')}.{record.get('nic')}."
                    f"{record.get('direction')}.{index}"
                ),
                summary_kind=SummaryKind.TRAFFIC,
                dimension_id="traffic_intensity",
                status=_summary_status(str(record.get("state", "missing")), total),
                run_id=run_id,
                value=total,
                unit="bytes",
                window_ns=window_ns,
                coverage_fraction=record.get("coverage_fraction"),
                zero_observation=total == 0,
            )
        )

    for record in view.get("job_balance", {}).get("directions", ()):
        direction = record.get("direction", "unknown")
        total = record.get("job_total_bytes")
        if total is None:
            continue
        status = _summary_status(str(record.get("state", "missing")), total)
        job.append(
            ProfileSummary(
                analysis_id=f"job.spatial_balance.{direction}",
                summary_kind=SummaryKind.JOB,
                dimension_id="spatial_balance",
                status=status,
                run_id=run_id,
                value=total,
                unit="bytes",
                window_ns=window_ns,
                coverage_fraction=record.get("coverage_fraction"),
                zero_observation=total == 0,
            )
        )
        hotspot = record.get("hotspot_concentration")
        if hotspot is not None:
            job.append(
                ProfileSummary(
                    analysis_id=f"job.topology_hotspots.{direction}",
                    summary_kind=SummaryKind.JOB,
                    dimension_id="topology_hotspots",
                    status=SummaryStatus.OBSERVED,
                    run_id=run_id,
                    value=hotspot,
                    unit="fraction",
                    window_ns=window_ns,
                    coverage_fraction=record.get("coverage_fraction"),
                    topology_supported=False,
                    communication_edges_supported=False,
                )
            )

    for record in view.get("phase_analysis", {}).get("phases", ()):
        total = record.get("total_bytes")
        if total is None:
            continue
        phase.append(
            ProfileSummary(
                analysis_id=f"phase.{record.get('name', 'unnamed')}",
                summary_kind=SummaryKind.PHASE,
                dimension_id="phase_change",
                status=_summary_status(str(record.get("state", "missing")), total),
                run_id=run_id,
                value=total,
                unit="bytes",
                window_ns=(
                    record.get("window_ns", (None, None))[0],
                    record.get("window_ns", (None, None))[1],
                ),
                coverage_fraction=record.get("coverage_fraction"),
                zero_observation=total == 0,
            )
        )
    return traffic, job, phase


def _job_profile_record(profile: Any) -> dict[str, Any]:
    return {
        "version": profile.version,
        "status": profile.status.value,
        "summary": profile.summary,
        "provenance": {
            "run_id": profile.provenance.run_id,
            "capture_id": profile.provenance.capture_id,
            "source": profile.provenance.source,
            "reader_version": profile.provenance.reader_version,
            "clock_id": profile.provenance.clock_id,
            "selected_window_ns": profile.provenance.selected_window_ns,
        },
        "thresholds": {
            "low_intensity_bytes_per_s": profile.thresholds.low_intensity_bytes_per_s,
            "high_intensity_bytes_per_s": profile.thresholds.high_intensity_bytes_per_s,
            "high_node_max_min_ratio": profile.thresholds.high_node_max_min_ratio,
            "high_hotspot_concentration": profile.thresholds.high_hotspot_concentration,
        },
        "dimensions": [
            {
                "dimension_id": dimension.dimension_id,
                "title": dimension.title,
                "status": dimension.status.value,
                "question_ids": dimension.question_ids,
                "evidence_ids": dimension.evidence_ids,
                "observed_evidence_count": dimension.observed_evidence_count,
                "zero_observation_count": dimension.zero_observation_count,
                "missing_observation_count": dimension.missing_observation_count,
                "coverage_fraction": _finite_float(dimension.coverage_fraction),
                "limitations": dimension.limitations,
                "observation": dimension.observation,
            }
            for dimension in profile.dimensions
        ],
        "questions": [
            {
                "question_id": question.question_id,
                "status": question.status,
                "reasons": question.reasons,
                "scientific_limitations": question.scientific_limitations,
            }
            for question in profile.questions
        ],
        "evidence": [
            {
                "analysis_id": summary.analysis_id,
                "summary_kind": summary.summary_kind.value,
                "dimension_id": summary.dimension_id,
                "status": summary.status.value,
                "run_id": summary.run_id,
                "value": summary.value,
                "unit": summary.unit,
                "window_ns": summary.window_ns,
                "coverage_fraction": _finite_float(summary.coverage_fraction),
                "zero_observation": summary.zero_observation,
                "missing_observation": summary.missing_observation,
                "topology_supported": summary.topology_supported,
                "communication_edges_supported": summary.communication_edges_supported,
                "reasons": summary.reasons,
            }
            for summary in profile.evidence
        ],
        "observed_summary_count": profile.observed_summary_count,
        "zero_observation_count": profile.zero_observation_count,
        "missing_observation_count": profile.missing_observation_count,
        "coverage_fraction": _finite_float(profile.coverage_fraction),
        "supported_dimensions": profile.supported_dimensions,
        "partial_dimensions": profile.partial_dimensions,
        "unsupported_dimensions": profile.unsupported_dimensions,
        "topology_status": profile.topology_status,
        "topology_reasons": profile.topology_reasons,
        "communication_edge_status": profile.communication_edge_status,
        "communication_edge_reasons": profile.communication_edge_reasons,
        "formulas": [
            {
                "dimension_id": formula.dimension_id,
                "formula": formula.formula,
                "threshold_rule": formula.threshold_rule,
            }
            for formula in profile.formulas
        ],
        "interpretation_limits": profile.interpretation_limits,
    }


def _empty_operation_analysis() -> dict[str, Any]:
    return {
        "schema_version": OPERATION_ANALYSIS_VERSION,
        "question_id": "NIC-05",
        "state": "missing",
        "status": "No command-queue evidence",
        "status_reasons": (
            "No cataloged command-operation counters are available for this NIC.",
        ),
        "quality_flags": [],
        "collector_analyses": (),
        "collector_count": 0,
        "collector_limit": MAX_OPERATION_COLLECTORS,
        "opcode_limit": MAX_OPERATION_OPCODE_RESULTS,
        "total_limit": MAX_OPERATION_TOTAL_RESULTS,
        "collector_truncated": False,
        "opcode_truncated": False,
        "total_truncated": False,
        "duplicate_stream_count": 0,
        "requested_window_ns": None,
        "collector_coverage": (),
        "coverage_scope": (
            "Coverage is reported separately for each collector; DMA and IDC "
            "populations combine only on exact common valid interval boundaries."
        ),
        "expected_counter_names": tuple(spec.counter_id for spec in _OPERATION_SPECS),
        "missing_counter_ids": tuple(
            spec.counter_id for spec in _OPERATION_SPECS
        ),
        "unsupported_counter_names": (),
        "opcode_mapping_status": "missing_mapping",
        "opcode_mapping_version": None,
        "opcode_mapping_evidence": (
            "No authoritative opcode mapping was supplied; bins remain explicit and unlabeled."
        ),
        "mapped_opcode_counter_ids": (),
        "unmapped_opcode_counter_ids": (),
        "missing_opcode_counter_ids": tuple(
            spec.counter_id for spec in _OPERATION_SPECS
            if spec.counter_kind is OperationCounterKind.DMA_OPCODE_BIN
        ),
        "limitations": OPERATION_LIMITATIONS,
    }


def _frame_quality_flags(flags: int) -> tuple[Any, ...]:
    mappings = {
        QualityFlag.REORDERED: FrameQualityFlag.REORDERED,
        QualityFlag.NONMONOTONIC_TIMESTAMP: FrameQualityFlag.CONFLICTING_TIMESTAMP,
        QualityFlag.DUPLICATE_REMOVED: FrameQualityFlag.DUPLICATE,
        QualityFlag.BASELINE_RELATIVE: FrameQualityFlag.BASELINE_RELATIVE,
        QualityFlag.PRECISION_LOSS_POSSIBLE: FrameQualityFlag.UNSUPPORTED_SERIES,
        QualityFlag.RESET_SUSPECT: FrameQualityFlag.DECREASE_OR_RESET,
        QualityFlag.WRAP_SUSPECT: FrameQualityFlag.DECREASE_OR_RESET,
        QualityFlag.GAP: FrameQualityFlag.PARTIAL_WINDOW,
        QualityFlag.UNKNOWN_METADATA: FrameQualityFlag.UNSUPPORTED_SERIES,
    }
    return tuple(
        dict.fromkeys(
            mapped
            for source_flag, mapped in mappings.items()
            if int(flags) & source_flag
        )
    )


def _frame_series(result, spec, source_unit: str) -> FrameCounterSeries | None:
    values: list[int] = []
    for value in result.values:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            if not math.isfinite(value) or not value.is_integer():
                return None
            value = int(value)
        if type(value) is not int or not 0 <= value <= 2**64 - 1:
            return None
        values.append(value)
    quality_flags = set(_frame_quality_flags(int(result.counter.quality_flags)))
    quality_flags.update(
        _frame_quality_flags(int(np.bitwise_or.reduce(result.quality_flags)))
    )
    collector = result.collector
    return FrameCounterSeries(
        spec=spec,
        identity=FrameCounterIdentity(
            run_id=result.run_identity.run_id,
            node_id=str(result.node.node_id),
            nic_id=str(result.nic.nic_id),
            collector_id=f"{collector.name} (ID {collector.collector_id})",
            source_counter_id=result.counter.name,
        ),
        timestamps_ns=tuple(int(timestamp) for timestamp in result.timestamps_ns),
        values=tuple(values),
        value_kind=(
            FrameValueKind.CUMULATIVE
            if result.counter.value_kind is ValueKind.EXACT_UINT64
            and result.counter.semantics is CounterSemantics.CUMULATIVE
            and source_unit == spec.unit
            else FrameValueKind.UNKNOWN
        ),
        semantics=(
            result.counter.semantics.value
            if result.counter.semantics is not CounterSemantics.CUMULATIVE
            else "cumulative counter"
        ),
        baseline_relative=result.counter.baseline_relative,
        quality_flags=tuple(quality_flags),
    )


def _frame_counter_record(result: Any) -> dict[str, Any]:
    identity = result.source_identity
    return {
        "counter": result.spec.counter_id,
        "bin_id": result.spec.bin_id,
        "counter_kind": result.spec.counter_kind.value,
        "direction": result.spec.direction.value,
        "state": result.status.value,
        "status": result.status.value.replace("_", " ").title(),
        "status_reasons": result.status_reasons,
        "quality_flags": sorted(flag.value for flag in result.quality_flags),
        "run_id": identity.run_id if identity else None,
        "node_id": int(identity.node_id) if identity else None,
        "nic_id": int(identity.nic_id) if identity else None,
        "collector_id": identity.collector_id if identity else None,
        "source_counter_id": identity.source_counter_id if identity else result.spec.counter_id,
        "requested_window_ns": result.requested_window_ns,
        "represented_window_ns": result.represented_window_ns,
        "requested_duration_ns": result.requested_duration_ns,
        "covered_duration_ns": result.covered_duration_ns,
        "coverage_fraction": _finite_float(result.coverage_fraction),
        "sample_count": result.sample_count,
        "valid_interval_count": result.valid_interval_count,
        "invalid_interval_count": result.invalid_interval_count,
        "outside_window_interval_count": result.outside_window_interval_count,
        "total": result.total,
        "unit": result.unit,
        "rate_per_second": _finite_float(result.rate_per_second),
        "zero_observed": result.zero_observed,
        "persistent_zero": result.persistent_zero,
        "bin_edge": (
            {
                "lower_bytes": result.bin_edge.lower_bytes,
                "upper_bytes": result.bin_edge.upper_bytes,
                "open_ended": result.bin_edge.open_ended,
            }
            if result.bin_edge is not None
            else None
        ),
        "bin_share": _finite_float(result.bin_share),
    }


def _frame_quantile_record(result: Any) -> dict[str, Any]:
    return {
        "quantile": result.quantile,
        "state": result.status.value,
        "lower_bytes": result.lower_bytes,
        "upper_bytes": result.upper_bytes,
        "status_reasons": result.status_reasons,
    }


def _frame_direction_record(result: Any) -> dict[str, Any]:
    return {
        "direction": result.direction.value,
        "state": result.status.value,
        "status": result.status.value.replace("_", " ").title(),
        "status_reasons": result.status_reasons,
        "quality_flags": sorted(flag.value for flag in result.quality_flags),
        "requested_window_ns": result.requested_window_ns,
        "represented_window_ns": result.represented_window_ns,
        "requested_duration_ns": result.requested_duration_ns,
        "covered_duration_ns": result.covered_duration_ns,
        "coverage_fraction": _finite_float(result.coverage_fraction),
        "bin_results": [
            _frame_counter_record(bin_result) for bin_result in result.bin_results
        ],
        "byte_result": (
            _frame_counter_record(result.byte_result)
            if result.byte_result is not None
            else None
        ),
        "bin_total": result.bin_total,
        "byte_total": result.byte_total,
        "bin_edges_source": result.bin_edges_source,
        "bin_edges_complete": result.bin_edges_complete,
        "open_ended_tail": result.open_ended_tail,
        "mean_wire_frame_size_bytes": _finite_float(
            result.mean_wire_frame_size_bytes
        ),
        "mean_state": result.mean_status.value,
        "mean_status_reasons": result.mean_status_reasons,
        "mean_coverage_fraction": _finite_float(result.mean_coverage_fraction),
        "approximate_mean_wire_frame_size_bytes": _finite_float(
            result.approximate_bin_midpoint_mean_wire_frame_size_bytes
        ),
        "quantile_results": [
            _frame_quantile_record(quantile)
            for quantile in result.quantile_results
        ],
    }


def _frame_collector_record(analysis: Any, collector: Any) -> dict[str, Any]:
    return {
        "schema_version": FRAME_ANALYSIS_VERSION,
        "state": analysis.status.value,
        "status": analysis.status.value.replace("_", " ").title(),
        "status_reasons": analysis.status_reasons,
        "quality_flags": sorted(flag.value for flag in analysis.quality_flags),
        "population": {
            "run_id": analysis.population.run_id,
            "node_id": int(analysis.population.node_id),
            "nic_id": int(analysis.population.nic_id),
            "collector_id": analysis.population.collector_id,
            "collector_name": collector.name,
            "collector_registry_id": collector.collector_id,
            "collector_rank": collector.rank,
            "collector_process": collector.process,
            "collector_type": collector.collector_type,
        },
        "requested_window_ns": analysis.requested_window_ns,
        "direction_results": [
            _frame_direction_record(result)
            for result in analysis.direction_results
        ],
        "missing_counter_ids": tuple(
            dict.fromkeys(
                counter_id
                for direction in analysis.direction_results
                for result in direction.bin_results
                if result.status is not None
                and result.status.value == "missing_data"
                for counter_id in (result.spec.counter_id,)
            )
        ),
    }


def _build_frame_analysis(
    store: TraceStore,
    node: Any,
    nic: Any,
    start_ns: int,
    end_ns: int,
    direction: Any,
) -> dict[str, Any]:
    selected_directions = _directions(direction)
    frame_direction_names = {
        direction.value.lower() for direction in selected_directions
    }
    specs = tuple(
        spec for spec in _FRAME_SPECS if spec.direction.value in frame_direction_names
    )
    counter_by_name = {counter.name: counter for counter in store.counters}
    series_by_collector: dict[int, list[FrameCounterSeries]] = {}
    collectors: dict[int, Any] = {}
    unsupported_by_collector: dict[int, set[str]] = {}
    duplicate_stream_count = 0
    for spec in specs:
        counter = counter_by_name.get(spec.counter_id)
        if counter is None:
            continue
        try:
            query_result = store.query_series(
                node,
                nic,
                counter,
                start_ns=start_ns,
                end_ns=end_ns,
            )
        except TraceValidationError:
            continue
        for result in query_result.results:
            collector_id = result.collector.collector_id
            collectors[collector_id] = result.collector
            duplicate_stream_count = max(
                duplicate_stream_count, result.observation_stream_count
            )
            series = _frame_series(result, spec, counter.unit)
            if series is None:
                unsupported_by_collector.setdefault(collector_id, set()).add(
                    spec.counter_id
                )
                continue
            series_by_collector.setdefault(collector_id, []).append(series)

    collector_ids = sorted(series_by_collector)
    records = [
        _frame_collector_record(
            analyze_frames(
                FrameDataset(
                    population=FramePopulation(
                        run_id=store.identity.run_id,
                        node_id=str(node.node_id),
                        nic_id=str(nic.nic_id),
                        collector_id=series_by_collector[collector_id][0].identity.collector_id,
                    ),
                    requested_counters=specs,
                    series=tuple(series_by_collector[collector_id]),
                    bin_edges=_FRAME_BIN_EDGE_SETS,
                    byte_count_compatibility=_FRAME_BYTE_COMPATIBILITY,
                    window_ns=(start_ns, end_ns - 1),
                )
            ),
            collectors[collector_id],
        )
        for collector_id in collector_ids[:MAX_FRAME_COLLECTORS]
    ]
    if not records:
        unsupported_names = {
            name for names in unsupported_by_collector.values() for name in names
        }
        return {
            **_empty_frame_analysis(),
            "status": (
                "Unsupported series"
                if unsupported_names
                else "No wire-frame evidence"
            ),
            "status_reasons": (
                "One or more frame series have unsupported precision, value type, semantics, or unit."
                if unsupported_names
                else "No cataloged frame-bin counter series is available for this NIC."
            ),
            "requested_window_ns": (start_ns, end_ns - 1),
            "missing_counter_ids": tuple(
                spec.counter_id
                for spec in specs
                if spec.counter_kind is FrameCounterKind.FRAME_BIN
                and counter_by_name.get(spec.counter_id) is None
            ),
            "unsupported_counter_names": tuple(sorted(unsupported_names)),
            "duplicate_stream_count": duplicate_stream_count,
        }

    selected_records = records[:MAX_FRAME_COLLECTORS]
    states = {record["state"] for record in selected_records}
    state = next(iter(states)) if len(states) == 1 else "partial"
    bin_count = sum(
        len(direction["bin_results"])
        for record in selected_records
        for direction in record["direction_results"]
    )
    quantile_count = sum(
        len(direction["quantile_results"])
        for record in selected_records
        for direction in record["direction_results"]
    )
    selected_counter_names = tuple(spec.counter_id for spec in specs)
    return {
        "schema_version": FRAME_ANALYSIS_VERSION,
        "question_id": "NIC-06",
        "state": state,
        "status": state.replace("_", " ").title(),
        "status_reasons": tuple(
            dict.fromkeys(
                reason
                for record in selected_records
                for reason in record["status_reasons"]
            )
        ),
        "quality_flags": sorted(
            {flag for record in selected_records for flag in record["quality_flags"]}
        ),
        "collector_analyses": selected_records,
        "collector_count": len(series_by_collector),
        "collector_limit": MAX_FRAME_COLLECTORS,
        "bin_limit": MAX_FRAME_BIN_RESULTS,
        "quantile_limit": MAX_FRAME_QUANTILE_RESULTS,
        "collector_truncated": len(series_by_collector) > MAX_FRAME_COLLECTORS,
        "bin_truncated": bin_count > MAX_FRAME_BIN_RESULTS,
        "quantile_truncated": quantile_count > MAX_FRAME_QUANTILE_RESULTS,
        "duplicate_stream_count": duplicate_stream_count,
        "requested_window_ns": selected_records[0]["requested_window_ns"],
        "collector_coverage": [
            {
                "collector_id": record["population"]["collector_id"],
                "collector_registry_id": record["population"]["collector_registry_id"],
                "directions": [
                    {
                        "direction": direction["direction"],
                        "covered_duration_ns": direction["covered_duration_ns"],
                        "coverage_fraction": direction["coverage_fraction"],
                    }
                    for direction in record["direction_results"]
                ],
            }
            for record in selected_records
        ],
        "coverage_scope": (
            "Coverage is reported separately for each collector and direction; "
            "bins combine only on identical valid interval boundaries."
        ),
        "expected_counter_names": selected_counter_names,
        "missing_counter_ids": tuple(
            dict.fromkeys(
                counter_id
                for record in selected_records
                for counter_id in record["missing_counter_ids"]
            )
        ),
        "unsupported_counter_names": tuple(
            dict.fromkeys(
                name
                for collector_id, names in unsupported_by_collector.items()
                if collector_id in {
                    record["population"]["collector_registry_id"]
                    for record in selected_records
                }
                for name in names
            )
        ),
        "bin_edges_source": next(
            (
                direction["bin_edges_source"]
                for record in selected_records
                for direction in record["direction_results"]
                if direction["bin_edges_source"] is not None
            ),
            None,
        ),
        "bin_edges_complete": all(
            direction["bin_edges_complete"]
            for record in selected_records
            for direction in record["direction_results"]
        ),
        "population_compatibility": [
            {
                "direction": compatibility.direction.value,
                "byte_counter_id": compatibility.byte_counter_id,
                "frame_population_id": compatibility.frame_population_id,
                "byte_population_id": compatibility.byte_population_id,
                "compatible": compatibility.compatible,
                "definition_evidence": compatibility.definition_evidence,
            }
            for compatibility in _FRAME_BYTE_COMPATIBILITY
        ],
        "delta_formula": "end_value - start_value",
        "rate_formula": "sum(valid interval deltas) / sum(valid interval durations)",
        "bin_share_formula": (
            "bin frame delta total / direction frame delta total"
        ),
        "mean_formula": (
            "compatible direction byte delta total / direction frame delta total"
        ),
        "approximate_mean_formula": (
            "sum(bin frame delta total * bin midpoint bytes) / direction frame delta total"
        ),
        "interpretation": FRAME_INTERPRETATION,
        "limitations": FRAME_LIMITATIONS,
    }


def _operation_quality_flags(flags: int) -> tuple[Any, ...]:
    mappings = {
        QualityFlag.REORDERED: OperationQualityFlag.REORDERED,
        QualityFlag.NONMONOTONIC_TIMESTAMP: OperationQualityFlag.CONFLICTING_TIMESTAMP,
        QualityFlag.DUPLICATE_REMOVED: OperationQualityFlag.DUPLICATE,
        QualityFlag.BASELINE_RELATIVE: OperationQualityFlag.BASELINE_RELATIVE,
        QualityFlag.PRECISION_LOSS_POSSIBLE: OperationQualityFlag.UNSUPPORTED_SERIES,
        QualityFlag.RESET_SUSPECT: OperationQualityFlag.DECREASE_OR_RESET,
        QualityFlag.WRAP_SUSPECT: OperationQualityFlag.DECREASE_OR_RESET,
        QualityFlag.GAP: OperationQualityFlag.PARTIAL_WINDOW,
        QualityFlag.UNKNOWN_METADATA: OperationQualityFlag.UNSUPPORTED_SERIES,
    }
    return tuple(
        dict.fromkeys(
            mapped
            for source_flag, mapped in mappings.items()
            if int(flags) & source_flag
        )
    )


def _operation_series(
    result,
    spec,
    source_unit: str,
) -> OperationCounterSeries | None:
    values: list[int] = []
    for value in result.values:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            if not math.isfinite(value) or not value.is_integer():
                return None
            value = int(value)
        if type(value) is not int or not 0 <= value <= 2**64 - 1:
            return None
        values.append(value)

    quality_flags = set(_operation_quality_flags(int(result.counter.quality_flags)))
    quality_flags.update(
        _operation_quality_flags(int(np.bitwise_or.reduce(result.quality_flags)))
    )
    collector = result.collector
    return OperationCounterSeries(
        spec=spec,
        identity=OperationSeriesIdentity(
            run_id=result.run_identity.run_id,
            node_id=str(result.node.node_id),
            nic_id=str(result.nic.nic_id),
            collector_id=f"{collector.name} (ID {collector.collector_id})",
            source_counter_id=result.counter.name,
        ),
        timestamps_ns=tuple(int(timestamp) for timestamp in result.timestamps_ns),
        values=tuple(values),
        value_kind=(
            OperationValueKind.CUMULATIVE
            if result.counter.value_kind is ValueKind.EXACT_UINT64
            and result.counter.semantics is CounterSemantics.CUMULATIVE
            and source_unit == spec.unit
            else OperationValueKind.UNKNOWN
        ),
        semantics=(
            result.counter.semantics.value
            if result.counter.semantics is not CounterSemantics.CUMULATIVE
            else "cumulative counter"
        ),
        baseline_relative=result.counter.baseline_relative,
        quality_flags=tuple(quality_flags),
    )


def _operation_counter_record(result: Any) -> dict[str, Any]:
    identity = result.source_identity
    return {
        "counter": result.spec.counter_id,
        "counter_kind": result.spec.counter_kind.value,
        "bin_index": result.spec.bin_index,
        "queue_index": result.spec.queue_index,
        "state": result.status.value,
        "status": result.status.value.replace("_", " ").title(),
        "status_reasons": result.status_reasons,
        "quality_flags": sorted(flag.value for flag in result.quality_flags),
        "run_id": identity.run_id if identity else None,
        "node_id": int(identity.node_id) if identity else None,
        "nic_id": int(identity.nic_id) if identity else None,
        "collector_id": identity.collector_id if identity else None,
        "source_counter_id": identity.source_counter_id if identity else result.spec.counter_id,
        "requested_window_ns": result.requested_window_ns,
        "represented_window_ns": result.represented_window_ns,
        "requested_duration_ns": result.requested_duration_ns,
        "covered_duration_ns": result.covered_duration_ns,
        "coverage_fraction": _finite_float(result.coverage_fraction),
        "sample_count": result.sample_count,
        "valid_interval_count": result.valid_interval_count,
        "invalid_interval_count": result.invalid_interval_count,
        "outside_window_interval_count": result.outside_window_interval_count,
        "total": result.total,
        "unit": result.unit,
        "rate_per_second": _finite_float(result.rate_per_second),
        "rate_unit": result.rate_unit,
        "zero_observed": result.zero_observed,
    }


def _operation_aggregate_record(result: Any) -> dict[str, Any]:
    return {
        "counter_kind": result.counter_kind.value,
        "state": result.status.value,
        "status": result.status.value.replace("_", " ").title(),
        "status_reasons": result.status_reasons,
        "quality_flags": sorted(flag.value for flag in result.quality_flags),
        "requested_window_ns": result.requested_window_ns,
        "represented_window_ns": result.represented_window_ns,
        "requested_duration_ns": result.requested_duration_ns,
        "covered_duration_ns": result.covered_duration_ns,
        "coverage_fraction": _finite_float(result.coverage_fraction),
        "source_counter_ids": result.source_counter_ids,
        "compatible_counter_ids": result.compatible_counter_ids,
        "incompatible_counter_ids": result.incompatible_counter_ids,
        "missing_counter_ids": result.missing_counter_ids,
        "common_valid_interval_count": result.common_valid_interval_count,
        "total": result.total,
        "unit": result.unit,
        "rate_per_second": _finite_float(result.rate_per_second),
        "rate_unit": result.rate_unit,
        "zero_observed": result.zero_observed,
    }


def _operation_opcode_record(result: Any) -> dict[str, Any]:
    return {
        "bin_index": result.bin_index,
        "counter": result.counter_id,
        "opcode_label": result.opcode_label,
        "opcode_mapped": result.opcode_mapped,
        "state": result.counter_result.status.value,
        "status": result.counter_result.status.value.replace("_", " ").title(),
        "quality_flags": sorted(
            flag.value for flag in result.counter_result.quality_flags
        ),
        "collector_id": (
            result.counter_result.source_identity.collector_id
            if result.counter_result.source_identity
            else None
        ),
        "total": result.total,
        "unit": result.counter_result.unit,
        "rate_per_second": _finite_float(result.rate_per_second),
        "fraction_of_dma_opcode_commands": _finite_float(
            result.fraction_of_dma_opcode_commands
        ),
        "coverage_fraction": _finite_float(result.counter_result.coverage_fraction),
        "valid_interval_count": result.counter_result.valid_interval_count,
        "invalid_interval_count": result.counter_result.invalid_interval_count,
        "zero_observed": result.zero_observed,
    }


def _operation_family_record(result: Any) -> dict[str, Any]:
    return {
        "state": result.status.value,
        "status": result.status.value.replace("_", " ").title(),
        "status_reasons": result.status_reasons,
        "quality_flags": sorted(flag.value for flag in result.quality_flags),
        "dma_command_total": result.dma_command_total,
        "idc_command_total": result.idc_command_total,
        "total_commands": result.total_commands,
        "dma_fraction": _finite_float(result.dma_fraction),
        "idc_fraction": _finite_float(result.idc_fraction),
        "covered_duration_ns": result.covered_duration_ns,
        "coverage_fraction": _finite_float(result.coverage_fraction),
        "represented_window_ns": result.represented_window_ns,
        "formula": result.formula,
    }


def _operation_collector_record(analysis: Any, collector: Any) -> dict[str, Any]:
    return {
        "schema_version": OPERATION_ANALYSIS_VERSION,
        "state": analysis.status.value,
        "status": analysis.status.value.replace("_", " ").title(),
        "status_reasons": analysis.status_reasons,
        "quality_flags": sorted(flag.value for flag in analysis.quality_flags),
        "population": {
            "run_id": analysis.population.run_id,
            "node_id": int(analysis.population.node_id),
            "nic_id": int(analysis.population.nic_id),
            "collector_id": analysis.population.collector_id,
            "collector_name": collector.name,
            "collector_registry_id": collector.collector_id,
            "collector_rank": collector.rank,
            "collector_process": collector.process,
            "collector_type": collector.collector_type,
        },
        "requested_window_ns": analysis.requested_window_ns,
        "requested_duration_ns": analysis.requested_duration_ns,
        "covered_duration_ns": analysis.dma_idc_mix.covered_duration_ns,
        "coverage_fraction": _finite_float(analysis.dma_idc_mix.coverage_fraction),
        "dma_opcode_bin_results": [
            _operation_counter_record(result)
            for result in analysis.dma_opcode_bin_results
        ],
        "dma_total_results": [
            _operation_counter_record(result)
            for result in analysis.dma_total_results
        ],
        "idc_total_results": [
            _operation_counter_record(result)
            for result in analysis.idc_total_results
        ],
        "aggregate_dma_commands": _operation_aggregate_record(
            analysis.aggregate_dma_commands
        ),
        "aggregate_idc_commands": _operation_aggregate_record(
            analysis.aggregate_idc_commands
        ),
        "opcode_results": [
            _operation_opcode_record(result)
            for result in analysis.opcode_results
        ],
        "opcode_mapping_status": analysis.opcode_mapping_status.value,
        "opcode_mapping_version": analysis.opcode_mapping_version,
        "mapped_opcode_counter_ids": analysis.mapped_opcode_counter_ids,
        "unmapped_opcode_counter_ids": analysis.unmapped_opcode_counter_ids,
        "missing_opcode_counter_ids": analysis.missing_opcode_counter_ids,
        "mapped_opcode_total": analysis.mapped_opcode_total,
        "unknown_opcode_total": analysis.unknown_opcode_total,
        "dma_opcode_command_total": analysis.dma_opcode_command_total,
        "dma_idc_mix": _operation_family_record(analysis.dma_idc_mix),
        "missing_counter_ids": analysis.aggregate_dma_commands.missing_counter_ids
        + analysis.aggregate_idc_commands.missing_counter_ids,
    }


def _build_operation_analysis(
    store: TraceStore,
    node: Any,
    nic: Any,
    start_ns: int,
    end_ns: int,
) -> dict[str, Any]:
    counter_by_name = {counter.name: counter for counter in store.counters}
    series_by_collector: dict[int, list[OperationCounterSeries]] = {}
    collectors: dict[int, Any] = {}
    unsupported_by_collector: dict[int, set[str]] = {}
    duplicate_stream_count = 0

    for spec in _OPERATION_SPECS:
        counter = counter_by_name.get(spec.counter_id)
        if counter is None:
            continue
        try:
            query_result = store.query_series(
                node,
                nic,
                counter,
                start_ns=start_ns,
                end_ns=end_ns,
            )
        except TraceValidationError:
            continue
        for result in query_result.results:
            collector_id = result.collector.collector_id
            collectors[collector_id] = result.collector
            duplicate_stream_count = max(
                duplicate_stream_count, result.observation_stream_count
            )
            series = _operation_series(result, spec, counter.unit)
            if series is None:
                unsupported_by_collector.setdefault(collector_id, set()).add(
                    spec.counter_id
                )
                continue
            series_by_collector.setdefault(collector_id, []).append(series)

    collector_ids = sorted(series_by_collector)
    records = [
        _operation_collector_record(
            analyze_operations(
                OperationDataset(
                    population=OperationPopulation(
                        run_id=store.identity.run_id,
                        node_id=str(node.node_id),
                        nic_id=str(nic.nic_id),
                        collector_id=series_by_collector[collector_id][0].identity.collector_id,
                    ),
                    requested_counters=_OPERATION_SPECS,
                    series=tuple(series_by_collector[collector_id]),
                    window_ns=(start_ns, end_ns - 1),
                    opcode_mapping=_OPERATION_OPCODE_MAPPING,
                )
            ),
            collectors[collector_id],
        )
        for collector_id in collector_ids[:MAX_OPERATION_COLLECTORS]
    ]
    if not records:
        unsupported_names = {
            name for names in unsupported_by_collector.values() for name in names
        }
        return {
            **_empty_operation_analysis(),
            "status": (
                "Unsupported series"
                if unsupported_names
                else "No command-queue evidence"
            ),
            "status_reasons": (
                "One or more command-operation series have unsupported precision, value type, semantics, or unit."
                if unsupported_names
                else "No cataloged command-operation counter series is available for this NIC."
            ),
            "requested_window_ns": (start_ns, end_ns - 1),
            "unsupported_counter_names": tuple(sorted(unsupported_names)),
            "duplicate_stream_count": duplicate_stream_count,
        }

    selected_records = records[:MAX_OPERATION_COLLECTORS]
    opcode_count = 0
    total_count = 0
    for record in selected_records:
        remaining_opcodes = MAX_OPERATION_OPCODE_RESULTS - opcode_count
        opcodes = record["opcode_results"][: max(remaining_opcodes, 0)]
        opcode_count += len(opcodes)
        record["opcode_count"] = len(record["opcode_results"])
        record["opcode_truncated"] = len(opcodes) < record["opcode_count"]
        record["opcode_results"] = opcodes
        available_totals = (
            *record["dma_total_results"],
            *record["idc_total_results"],
        )
        totals = available_totals[
            : max(MAX_OPERATION_TOTAL_RESULTS - total_count, 0)
        ]
        total_count += len(totals)
        record["total_count"] = len(available_totals)
        record["total_truncated"] = len(totals) < record["total_count"]
        dma_count = min(len(record["dma_total_results"]), len(totals))
        record["dma_total_results"] = tuple(totals[:dma_count])
        record["idc_total_results"] = tuple(totals[dma_count:])
        record["dma_total_results"] = tuple(totals[: len(record["dma_total_results"])])
        record["idc_total_results"] = tuple(totals[len(record["dma_total_results"]) :])

    states = {record["state"] for record in selected_records}
    state = next(iter(states)) if len(states) == 1 else "partial"
    return {
        "schema_version": OPERATION_ANALYSIS_VERSION,
        "question_id": "NIC-05",
        "state": state,
        "status": state.replace("_", " ").title(),
        "status_reasons": tuple(
            dict.fromkeys(
                reason
                for record in selected_records
                for reason in record["status_reasons"]
            )
        ),
        "quality_flags": sorted(
            {
                flag
                for record in selected_records
                for flag in record["quality_flags"]
            }
        ),
        "collector_analyses": selected_records,
        "collector_count": len(series_by_collector),
        "collector_limit": MAX_OPERATION_COLLECTORS,
        "opcode_limit": MAX_OPERATION_OPCODE_RESULTS,
        "total_limit": MAX_OPERATION_TOTAL_RESULTS,
        "collector_truncated": len(series_by_collector) > MAX_OPERATION_COLLECTORS,
        "opcode_truncated": any(record["opcode_truncated"] for record in selected_records),
        "total_truncated": any(record["total_truncated"] for record in selected_records),
        "duplicate_stream_count": duplicate_stream_count,
        "requested_window_ns": selected_records[0]["requested_window_ns"],
        "collector_coverage": [
            {
                "collector_id": record["population"]["collector_id"],
                "collector_registry_id": record["population"]["collector_registry_id"],
                "covered_duration_ns": record["covered_duration_ns"],
                "coverage_fraction": record["coverage_fraction"],
            }
            for record in selected_records
        ],
        "coverage_scope": (
            "Coverage is reported separately for each collector; DMA and IDC "
            "populations combine only on exact common valid interval boundaries."
        ),
        "expected_counter_names": tuple(spec.counter_id for spec in _OPERATION_SPECS),
        "missing_counter_ids": tuple(
            dict.fromkeys(
                name
                for record in selected_records
                for name in record["missing_counter_ids"]
            )
        ),
        "unsupported_counter_names": tuple(
            dict.fromkeys(
                name
                for collector_id, names in unsupported_by_collector.items()
                if collector_id in {
                    record["population"]["collector_registry_id"]
                    for record in selected_records
                }
                for name in names
            )
        ),
        "opcode_mapping_status": selected_records[0]["opcode_mapping_status"],
        "opcode_mapping_version": (
            _OPERATION_OPCODE_MAPPING.mapping_version
            if _OPERATION_OPCODE_MAPPING
            else None
        ),
        "opcode_mapping_evidence": (
            _OPERATION_OPCODE_MAPPING.evidence
            if _OPERATION_OPCODE_MAPPING
            else "No authoritative opcode mapping was supplied; bins remain explicit and unlabeled."
        ),
        "mapped_opcode_counter_ids": selected_records[0]["mapped_opcode_counter_ids"],
        "unmapped_opcode_counter_ids": selected_records[0][
            "unmapped_opcode_counter_ids"
        ],
        "missing_opcode_counter_ids": selected_records[0][
            "missing_opcode_counter_ids"
        ],
        "delta_formula": "end_value - start_value",
        "rate_formula": (
            "sum(valid interval deltas) / sum(valid interval durations)"
        ),
        "opcode_fraction_formula": (
            "opcode bin delta total / sum(DMA opcode bin delta totals)"
        ),
        "dma_idc_fraction_formula": (
            "family command total / sum(DMA and IDC command totals)"
        ),
        "limitations": OPERATION_LIMITATIONS,
    }
def build_view_model(
    store: TraceStore,
    filters: AnalysisFilterState | None = None,
    phase_windows: Any = (),
) -> dict[str, Any]:
    """Build a bounded, browser-safe view model from server-owned trace data.

    The raw :class:`TraceStore`, value columns, and full-resolution samples are
    intentionally not placed in this structure. Only one selected NIC receives
    bounded display samples; every scientific summary comes from TrafficAnalysis.
    Repeated identical requests use a bounded process-local cache.
    """

    resolved_filters = filters or default_filters(store)
    if resolved_filters.start_ns is None or resolved_filters.end_ns is None:
        defaults = default_filters(store)
        start_ns = resolved_filters.start_ns if resolved_filters.start_ns is not None else defaults.start_ns
        end_ns = resolved_filters.end_ns if resolved_filters.end_ns is not None else defaults.end_ns
        resolved_filters = AnalysisFilterState(
            node_id=resolved_filters.node_id,
            nic_id=resolved_filters.nic_id,
            direction=resolved_filters.direction,
            start_ns=start_ns,
            end_ns=end_ns,
        )
    if resolved_filters.start_ns >= resolved_filters.end_ns:
        return _empty_view_model(
            store,
            resolved_filters,
            traffic_state="invalid",
            traffic_status="Invalid time window",
            reasons=("The selected time window must have a positive duration.",),
        )

    phase_cache_key = _phase_windows_cache_key(phase_windows)
    cache_key = (
        id(store),
        store.identity,
        resolved_filters,
        phase_cache_key,
    )
    if phase_cache_key is not None:
        with _VIEW_MODEL_CACHE_LOCK:
            cached = _VIEW_MODEL_CACHE.get(cache_key)
            if cached is not None:
                cached_store = cached[0]()
                if cached_store is store:
                    _VIEW_MODEL_CACHE.move_to_end(cache_key)
                    return cached[1]
                _VIEW_MODEL_CACHE.pop(cache_key, None)

    view = _base_view_model(store, resolved_filters)
    selected_nodes = tuple(
        node
        for node in store.nodes
        if resolved_filters.node_id is None or node.node_id == resolved_filters.node_id
    )
    selected_nics = tuple(
        nic
        for nic in store.nics
        if (resolved_filters.nic_id is None or nic.nic_id == resolved_filters.nic_id)
        and any(node.node_id == nic.node_id for node in selected_nodes)
    )
    directions = _directions(resolved_filters.direction)
    counter_plan: dict[TrafficDirection, tuple[CounterMetadata | None, tuple[CounterMetadata, ...]]] = {
        direction: _selected_counter(store.counters, direction) for direction in directions
    }
    detail_nic = selected_nics[0] if selected_nics else None
    detail_node = next(
        (node for node in selected_nodes if detail_nic and node.node_id == detail_nic.node_id),
        None,
    )
    reliability_analysis = (
        _build_reliability_analysis(
            store,
            detail_node,
            detail_nic,
            resolved_filters.start_ns,
            resolved_filters.end_ns,
        )
        if detail_node is not None and detail_nic is not None
        else _empty_reliability_analysis()
    )
    flow_control_analysis = (
        _build_flow_control_analysis(
            store,
            detail_node,
            detail_nic,
            resolved_filters.start_ns,
            resolved_filters.end_ns,
            resolved_filters.direction,
        )
        if detail_node is not None and detail_nic is not None
        else _empty_flow_control_analysis()
    )
    matching_analysis = (
        _build_matching_analysis(
            store,
            detail_node,
            detail_nic,
            resolved_filters.start_ns,
            resolved_filters.end_ns,
            resolved_filters.direction,
        )
        if detail_node is not None and detail_nic is not None
        else _empty_matching_analysis()
    )
    host_pressure_analysis = (
        _build_host_pressure_analysis(
            store,
            detail_node,
            detail_nic,
            resolved_filters.start_ns,
            resolved_filters.end_ns,
        )
        if detail_node is not None and detail_nic is not None
        else _empty_host_pressure_analysis()
    )
    operation_analysis = (
        _build_operation_analysis(
            store,
            detail_node,
            detail_nic,
            resolved_filters.start_ns,
            resolved_filters.end_ns,
        )
        if detail_node is not None and detail_nic is not None
        else _empty_operation_analysis()
    )
    frame_analysis = (
        _build_frame_analysis(
            store,
            detail_node,
            detail_nic,
            resolved_filters.start_ns,
            resolved_filters.end_ns,
            resolved_filters.direction,
        )
        if detail_node is not None and detail_nic is not None
        else _empty_frame_analysis()
    )
    translation_analysis = (
        _build_translation_analysis(
            store,
            detail_node,
            detail_nic,
            resolved_filters.start_ns,
            resolved_filters.end_ns,
            resolved_filters.direction,
        )
        if detail_node is not None and detail_nic is not None
        else _empty_translation_analysis()
    )
    latency_analysis = (
        _build_latency_analysis(
            store,
            detail_node,
            detail_nic,
            resolved_filters.start_ns,
            resolved_filters.end_ns,
        )
        if detail_node is not None and detail_nic is not None
        else _empty_latency_analysis()
    )

    series_summaries: list[dict[str, Any]] = []
    analyses_by_node_direction: dict[
        tuple[str, TrafficDirection], dict[str, object]
    ] = {}
    for nic in selected_nics:
        node = next(node for node in selected_nodes if node.node_id == nic.node_id)
        for direction in directions:
            counter, candidates = counter_plan[direction]
            if counter is None:
                requested_name = _requested_counter_name_for_direction(store, direction)
                if requested_name is not None:
                    series_summaries.append(
                        _series_record(
                            node_label=node.label,
                            nic_label=nic.label,
                            node_id=node.node_id,
                            nic_id=nic.nic_id,
                            direction=direction,
                            counter_name=requested_name,
                            state="missing",
                            status="Missing",
                            reasons=(
                                "The requested TX/RX byte counter is not present in this dataset.",
                            ),
                        )
                    )
                    continue
                series_summaries.append(
                    _series_record(
                        node_label=node.label,
                        nic_label=nic.label,
                        node_id=node.node_id,
                        nic_id=nic.nic_id,
                        direction=direction,
                        counter_name=None,
                        state="unsupported",
                        status="Unsupported",
                        reasons=("No cumulative TX/RX byte counter is available for this direction.",),
                    )
                )
                continue
            try:
                query_result = store.query_series(
                    node,
                    nic,
                    counter,
                    start_ns=resolved_filters.start_ns,
                    end_ns=resolved_filters.end_ns,
                )
            except TraceValidationError:
                series_summaries.append(
                    _series_record(
                        node_label=node.label,
                        nic_label=nic.label,
                        node_id=node.node_id,
                        nic_id=nic.nic_id,
                        direction=direction,
                        counter_name=counter.name,
                        state="missing",
                        status="Missing",
                        reasons=("This node/NIC does not have the selected counter series.",),
                    )
                )
                continue
            if len(query_result.results) > 1:
                collector_identities = tuple(
                    {
                        "collector_id": result.collector.collector_id,
                        "name": result.collector.name,
                        "rank": result.collector.rank,
                    }
                    for result in query_result.results
                )
                series_summaries.append(
                    _series_record(
                        node_label=node.label,
                        nic_label=nic.label,
                        node_id=node.node_id,
                        nic_id=nic.nic_id,
                        direction=direction,
                        counter_name=counter.name,
                        stream_count=len(query_result.results),
                        collector_identities=collector_identities,
                        collector_policy=(
                            "ambiguous: explicit collector selection required; "
                            + ", ".join(
                                f"{identity['name']} (ID {identity['collector_id']})"
                                for identity in collector_identities
                            )
                        ),
                        state="partial",
                        status="Ambiguous: multiple collector streams",
                        reasons=(
                            "Multiple collector streams observe this physical NIC; they are not merged.",
                        ),
                    )
                )
                continue
            result = query_result.results[0]
            metric_series = metric_series_from_series_result(
                result,
                run_id=store.identity.run_id,
                direction=direction,
            )
            analysis = analyze_traffic_series(
                metric_series,
                window_start_ns=resolved_filters.start_ns,
                window_end_ns=resolved_filters.end_ns,
                include_intervals=False,
            )
            analyses_by_node_direction.setdefault(
                (node.label, direction), {}
            )[nic.label] = analysis
            include_display = detail_nic is not None and nic.nic_id == detail_nic.nic_id
            series_summaries.append(
                _series_record(
                    node_label=node.label,
                    nic_label=nic.label,
                    node_id=node.node_id,
                    nic_id=nic.nic_id,
                    direction=direction,
                    counter_name=counter.name,
                    analysis=analysis,
                    stream_count=len(query_result.results),
                    collector_identities=(
                        {
                            "collector_id": result.collector.collector_id,
                            "name": result.collector.name,
                            "rank": result.collector.rank,
                        },
                    ),
                    collector_policy=(
                        f"single explicit collector stream: {result.collector.name} "
                        f"(ID {result.collector.collector_id})"
                    ),
                    include_display=include_display,
                    timestamps_ns=result.timestamps_ns,
                    values=result.values,
                )
            )

    view["series_summaries"] = series_summaries
    node_rows: list[dict[str, Any]] = []
    node_balances: dict[str, dict[Any, Any]] = {}
    for node in selected_nodes:
        known_nic_labels = tuple(
            nic.label for nic in store.nics if nic.node_id == node.node_id
        )
        for direction in directions:
            balance = analyze_node_nic_balance(
                node.label,
                analyses_by_node_direction.get((node.label, direction), {}),
                known_nic_labels,
            )
            node_balances.setdefault(direction, {})[node.label] = balance
            total_bytes = balance.node_total_bytes
            mean_rate = (
                total_bytes / (balance.covered_duration_ns / 1_000_000_000)
                if total_bytes is not None and balance.covered_duration_ns > 0
                else None
            )
            node_rows.append(
                {
                    "rank": 0,
                    "node": node.label,
                    "direction": direction.value,
                    "nic_count": balance.known_nic_count,
                    "observed_nic_count": balance.observed_nic_count,
                    "missing_nic_count": len(balance.missing_nic_ids),
                    "total_bytes": total_bytes,
                    "unit": "bytes",
                    "mean_rate_bytes_per_s": mean_rate,
                    "rate_unit": "bytes/s",
                    "effective_nic_count": balance.effective_nic_count,
                    "active_nic_fraction": balance.active_nic_fraction,
                    "max_min_ratio": balance.max_min_ratio,
                    "spatial_p50_bytes": balance.spatial_p50_bytes,
                    "spatial_p99_bytes": balance.spatial_p99_bytes,
                    "cross_nic_cv": balance.cross_nic_coefficient_of_variation,
                    "coverage_fraction": balance.coverage_fraction,
                    "coverage_definition": balance.coverage_aggregation,
                    "status": balance.status.value.replace("_", " ").title(),
                    "state": balance.status.value,
                    "status_reasons": balance.status_reasons,
                }
            )
    node_rows.sort(
        key=lambda row: (
            row["total_bytes"] is None,
            -(row["total_bytes"] or 0),
            row["node"],
            row["direction"],
        )
    )
    for rank, row in enumerate(node_rows, start=1):
        row["rank"] = rank
    view["node_comparison"] = {
        "rows": node_rows[:MAX_NODE_COMPARISON_ROWS],
        "state": _population_state(series_summaries),
        "status": _population_status(series_summaries),
        "truncated": len(node_rows) > MAX_NODE_COMPARISON_ROWS,
        "coverage_definition": "minimum coverage fraction across selected NIC/direction series",
    }
    view["node_balance"] = {
        "rows": node_rows[:MAX_NODE_COMPARISON_ROWS],
        "traffic_share_formula": "nic_total_bytes / node_total_bytes",
        "effective_nic_count_formula": "1 / sum((nic_total_bytes / node_total_bytes)^2)",
        "active_nic_rule": "mean_rate_bytes_per_s > 0 bytes/s",
        "spatial_quantile_interpolation": (
            "nearest-rank across NIC total bytes; temporal P99s are not averaged"
        ),
    }


    selected_node_labels = tuple(node.label for node in selected_nodes)
    view["job_balance"] = {
        **_empty_job_balance(),
        "directions": [
            _job_balance_record(
                analyze_job_node_balance(
                    node_balances.get(direction, {}),
                    known_node_ids=selected_node_labels,
                )
            )
            for direction in directions
        ],
    }
    view["nic_detail"] = {
        "selected_node": detail_node.label if detail_node else None,
        "selected_nic": detail_nic.label if detail_nic else None,
        "selection_note": (
            "Showing the first NIC selected by the current filters; choose a specific NIC for another detail view."
            if resolved_filters.nic_id is None and detail_nic is not None
            else "No NIC matches the current filters."
            if detail_nic is None
            else "Showing one physical node/NIC identity; ranks are not NICs."
        ),
        "series": [
            record
            for record in series_summaries
            if detail_nic is not None
            and record["node_id"] == detail_node.node_id
            and record["nic_id"] == detail_nic.nic_id
        ],
        "collector_policy": _collector_policy(
            record
            for record in series_summaries
            if detail_nic is not None
            and record["node_id"] == detail_node.node_id
            and record["nic_id"] == detail_nic.nic_id
        ),
        "reliability": reliability_analysis,
        "matching": matching_analysis,
        "flow_control": flow_control_analysis,
        "host_pressure": host_pressure_analysis,
        "operations": operation_analysis,
        "frames": frame_analysis,
        "translation": translation_analysis,
        "latency": latency_analysis,
    }

    analyzed_records = [record for record in series_summaries if record["total_bytes"] is not None]
    zero_count = sum(record["state"] == "zero" for record in series_summaries)
    missing_count = sum(record["state"] == "missing" for record in series_summaries)
    partial_count = sum(record["state"] == "partial" for record in series_summaries)
    unsupported_count = sum(record["state"] == "unsupported" for record in series_summaries)
    state = _population_state(series_summaries)
    view["overview"].update(
        {
            "traffic_state": state,
            "traffic_status": _population_status(series_summaries),
            "traffic_status_reasons": _population_reasons(series_summaries),
            "tx_total_bytes": _direction_total(series_summaries, TrafficDirection.TX),
            "rx_total_bytes": _direction_total(series_summaries, TrafficDirection.RX),
            "zero_observation_count": zero_count,
            "missing_observation_count": missing_count,
            "partial_observation_count": partial_count,
            "unsupported_observation_count": unsupported_count,
            "coverage_fraction": (
                min(record["coverage_fraction"] for record in analyzed_records)
                if analyzed_records and all(
                    record["coverage_fraction"] is not None for record in analyzed_records
                )
                else None
            ),
            "covered_duration_ns": (
                min(record["covered_duration_ns"] for record in analyzed_records)
                if analyzed_records
                else None
            ),
            "valid_interval_count": sum(record["valid_interval_count"] for record in analyzed_records),
            "invalid_interval_count": sum(record["invalid_interval_count"] for record in analyzed_records),
            "quality_flags": tuple(
                dict.fromkeys(
                    flag for record in series_summaries for flag in record["quality_flags"]
                )
            ),
            "counter_candidates": {
                direction.value: [counter.name for counter in candidates]
                for direction, (_, candidates) in counter_plan.items()
            },
            "counter_semantic_registry_version": COUNTER_SEMANTIC_REGISTRY_VERSION,
            "counter_semantics": (
                "network-frame octets; TX/RX and OK/BAD counters are separate; decreases are conservatively excluded and wrap is unverified"
            ),
        }
    )
    view["phase_analysis"] = build_phase_analysis(store, resolved_filters, phase_windows)
    traffic_summaries, job_summaries, phase_summaries = _profile_summaries(
        view,
        store.identity.run_id,
        (resolved_filters.start_ns, resolved_filters.end_ns),
    )
    view["job_profile"] = _job_profile_record(
        build_job_communication_profile(
            build_question_capability_matrix(store),
            provenance=JobProfileProvenance(
                run_id=store.identity.run_id,
                capture_id=store.identity.source_fingerprint,
                source=store.identity.source_format,
                reader_version=store.identity.reader_version,
                clock_id=store.identity.clock_domain,
                selected_window_ns=(
                    resolved_filters.start_ns,
                    resolved_filters.end_ns,
                ),
            ),
            traffic_summaries=traffic_summaries,
            job_summaries=job_summaries,
            phase_summaries=phase_summaries,
        )
    )
    if phase_cache_key is not None:
        with _VIEW_MODEL_CACHE_LOCK:
            _VIEW_MODEL_CACHE[cache_key] = (weakref.ref(store), view)
            _VIEW_MODEL_CACHE.move_to_end(cache_key)
            while len(_VIEW_MODEL_CACHE) > MAX_VIEW_MODEL_CACHE_ENTRIES:
                _VIEW_MODEL_CACHE.popitem(last=False)
    return view


def clear_view_model_cache() -> None:
    with _VIEW_MODEL_CACHE_LOCK:
        _VIEW_MODEL_CACHE.clear()


def _phase_windows_cache_key(phase_windows: Any) -> tuple[tuple[str, int, int], ...] | None:
    if phase_windows in ((), None):
        return ()
    if not isinstance(phase_windows, (list, tuple)):
        return None

    key: list[tuple[str, int, int]] = []
    for item in phase_windows:
        if isinstance(item, PhaseWindow):
            name, start_ns, end_ns = item.name, item.start_ns, item.end_ns
        elif isinstance(item, dict):
            name, start_ns, end_ns = (
                item.get("name"),
                item.get("start_ns"),
                item.get("end_ns"),
            )
        else:
            return None
        if (
            not isinstance(name, str)
            or type(start_ns) is not int
            or type(end_ns) is not int
        ):
            return None
        key.append((name, start_ns, end_ns))
    return tuple(key)


def _direction_total(
    records: list[dict[str, Any]], direction: TrafficDirection
) -> int | float | None:
    matching = [
        record["total_bytes"]
        for record in records
        if record["direction"] == direction.value and record["total_bytes"] is not None
    ]
    return sum(matching) if matching else None


def _collector_policy(records: Any) -> str:
    policies = {
        record["collector_policy"]
        for record in records
        if record.get("collector_policy") is not None
    }
    if not policies:
        return "no collector observation"
    if len(policies) == 1:
        return next(iter(policies))
    return "mixed collector policies: " + " | ".join(sorted(policies))


def _population_state(records: list[dict[str, Any]]) -> str:
    if not records:
        return "missing"
    states = {record["state"] for record in records}
    if "unsupported" in states:
        return "unsupported"
    if "partial" in states or "invalid" in states or "missing" in states:
        return "partial"
    if states == {"zero"}:
        return "zero"
    return "ready"


def _population_status(records: list[dict[str, Any]]) -> str:
    return {
        "ready": "Ready",
        "zero": "Zero traffic (observed)",
        "partial": "Partial traffic evidence",
        "missing": "Missing traffic evidence",
        "unsupported": "Unsupported traffic evidence",
        "invalid": "Invalid traffic evidence",
    }[_population_state(records)]


def _population_reasons(records: list[dict[str, Any]]) -> tuple[str, ...]:
    reasons: list[str] = []
    for record in records:
        if record["state"] not in ("ready", "zero"):
            reasons.extend(record["status_reasons"])
    return tuple(dict.fromkeys(reasons))


def _format_bytes(value: int | float | None, suffix: str = "bytes") -> str:
    if value is None:
        return "Missing"
    if value == 0:
        return f"0 {suffix} (observed)"
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    number = float(value)
    unit_index = 0
    while abs(number) >= 1000 and unit_index < len(units) - 1:
        number /= 1000
        unit_index += 1
    rendered = f"{number:.3f}".rstrip("0").rstrip(".") if unit_index else f"{int(number)}"
    return f"{rendered} {units[unit_index]}{'' if suffix == 'bytes' else '/' + suffix} ({suffix})"


def _format_rate(value: float | None) -> str:
    if value is None:
        return "Missing"
    return _format_bytes(value, suffix="bytes/s")


def _format_percent(value: float | None) -> str:
    return "Missing" if value is None else f"{value * 100:.1f}%"


def _format_duration(value: int | None) -> str:
    return "Missing" if value is None else f"{value / 1_000_000_000:.3f} s"


def _status_class(state: str) -> str:
    if state.startswith("partial_invalid"):
        return "status status-invalid"
    if state.startswith("partial_"):
        return "status status-partial"
    return f"status status-{state}"


def _stat_card(label: str, value: str, detail: str = "") -> html.Div:
    return html.Div(
        className="stat-card",
        children=[
            html.P(label, className="stat-label"),
            html.P(value, className="stat-value"),
            html.P(detail, className="stat-detail") if detail else None,
        ],
    )


def _render_overview(view: dict[str, Any]) -> html.Section:
    capture = view["capture"]
    overview = view["overview"]
    run = view["run"]
    missing = ", ".join(capture["missing_counter_names"]) or "None"
    unreadable = ", ".join(capture["unreadable_counter_names"]) or "None"
    return html.Section(
        className="panel",
        **{"aria-labelledby": "overview-heading"},
        children=[
            html.H2("Job overview", id="overview-heading"),
            html.Div(
                className="status-line",
                children=[
                    html.Span(
                        overview["traffic_status"],
                        className=_status_class(overview["traffic_state"]),
                        role="status",
                    ),
                    html.Span("Missing is not zero; zero is not missing.", className="status-note"),
                ],
            ),
            html.Div(
                className="card-grid",
                children=[
                    _stat_card("Run", run["run_id"], run["source_format"]),
                    _stat_card("Clock", run["clock_domain"], run["synchronization_evidence"]),
                    _stat_card("Capture window", _format_duration(capture["selected_duration_ns"])),
                    _stat_card("Nodes / NICs", f"{capture['node_count']} / {capture['nic_count']}"),
                    _stat_card("TX total", _format_bytes(overview.get("tx_total_bytes"))),
                    _stat_card("RX total", _format_bytes(overview.get("rx_total_bytes"))),
                    _stat_card("Minimum traffic coverage", _format_percent(overview.get("coverage_fraction"))),
                    _stat_card(
                        "Observed states",
                        (
                            f"zero {overview['zero_observation_count']}; "
                            f"missing {overview['missing_observation_count']}; "
                            f"partial {overview['partial_observation_count']}; "
                            f"unsupported {overview['unsupported_observation_count']}"
                        ),
                    ),
                ],
            ),
            _render_job_summary(view),
            _render_job_profile(view),
            html.Div(
                className="quality-grid",
                children=[
                    html.Div(
                        className="quality-card",
                        children=[
                            html.H3("Capture and coverage"),
                            html.Ul(
                                children=[
                                    html.Li(f"Source fingerprint: {run['source_fingerprint']}"),
                                    html.Li(f"Reader: {run['reader_version']}"),
                                    html.Li(
                                        "Counters requested / available / readable / observed: "
                                        f"{capture['requested_counter_count']} / "
                                        f"{capture['available_counter_count']} / "
                                        f"{capture['readable_counter_count']} / "
                                        f"{capture['observed_counter_count']}"
                                    ),
                                    html.Li(f"Missing counters: {missing}"),
                                    html.Li(f"Unreadable counters: {unreadable}"),
                                    html.Li(f"Lost records: {capture['lost_record_count']}"),
                                    html.Li(f"Detected gaps: {capture['gap_count']}"),
                                    html.Li(f"Indexed rows: {capture['row_count']}"),
                                ]
                            ),
                        ],
                    ),
                    html.Div(
                        className="quality-card",
                        children=[
                            html.H3("Quality and limitations"),
                            html.Ul(
                                children=[
                                    html.Li(f"Traffic status: {overview['traffic_status']}"),
                                    html.Li(f"Valid intervals: {overview.get('valid_interval_count', 0)}"),
                                    html.Li(f"Invalid intervals: {overview.get('invalid_interval_count', 0)}"),
                                    html.Li(f"Quality flags: {', '.join(overview.get('quality_flags', ())) or 'None'}"),
                                    html.Li("Full-resolution analysis remains on the server."),
                                    html.Li("Raw trace rows and value columns are not sent to the browser."),
                                ]
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def _render_filter_status(view: dict[str, Any]) -> html.Div:
    filters = view["filters"]
    capture = view["capture"]
    node = filters["node_id"] if filters["node_id"] is not None else "all"
    nic = filters["nic_id"] if filters["nic_id"] is not None else "all"
    return html.Div(
        className="filter-status",
        role="status",
        **{"aria-live": "polite"},
        children=[
            html.Strong("Active filters: "),
            html.Span(
                f"node {node}; NIC {nic}; direction {filters['direction']}; "
                f"window {capture['selected_start_ns']}–{capture['selected_end_ns']} ns"
            ),
        ],
    )


def _render_node_actions(view: dict[str, Any]) -> list[html.Button]:
    identities: list[tuple[int, int, str, str]] = []
    seen: set[tuple[int, int]] = set()
    for record in view.get("series_summaries", ()):
        identity = (record["node_id"], record["nic_id"])
        if identity in seen:
            continue
        seen.add(identity)
        identities.append(
            (record["node_id"], record["nic_id"], record["node"], record["nic"])
        )

    return [
        html.Button(
            f"open {node_label} {nic_label}",
            id={
                "section": NODE_OPEN_SECTION,
                "node": node_id,
                "nic": nic_id,
            },
            **{"aria-label": f"open .*{node_label}.*{nic_label}"},
            type="button",
            className="node-open-button",
        )
        for node_id, nic_id, node_label, nic_label in identities
    ]


def _render_node_table(view: dict[str, Any]) -> html.Div:
    rows = view["node_comparison"]["rows"]
    return html.Div(
        className="node-table-wrapper",
        children=[
            dash_table.DataTable(
                id=NODE_TABLE_ID,
                columns=[
                    {"name": "Rank", "id": "rank", "type": "numeric"},
                    {"name": "Node", "id": "node"},
                    {"name": "Direction", "id": "direction"},
                    {"name": "NICs", "id": "nic_count", "type": "numeric"},
                    {"name": "Observed NICs", "id": "observed_nic_count", "type": "numeric"},
                    {"name": "Missing NICs", "id": "missing_nic_count", "type": "numeric"},
                    {"name": "Total", "id": "total_bytes", "type": "numeric", "format": None},
                    {"name": "Unit", "id": "unit"},
                    {"name": "Mean rate", "id": "mean_rate_bytes_per_s", "type": "numeric"},
                    {"name": "Rate unit", "id": "rate_unit"},
                    {"name": "Coverage", "id": "coverage_fraction", "type": "numeric"},
                    {"name": "Effective NICs", "id": "effective_nic_count", "type": "numeric"},
                    {"name": "Active fraction", "id": "active_nic_fraction", "type": "numeric"},
                    {"name": "Max/min", "id": "max_min_ratio", "type": "numeric"},
                    {"name": "Spatial P99 (bytes)", "id": "spatial_p99_bytes", "type": "numeric"},
                    {"name": "Cross-NIC CV", "id": "cross_nic_cv", "type": "numeric"},
                    {"name": "Status", "id": "status"},
                ],
                data=rows,
                sort_action="native",
                filter_action="native",
                page_action="native",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "minWidth": "90px"},
                markdown_options={"link_target": "_self"},
            ),
            html.Div(
                id=NODE_ACTIONS_ID,
                className="node-actions",
                role="list",
                children=_render_node_actions(view),
            ),
        ],
    )


def _render_node_graph(view: dict[str, Any]) -> dcc.Graph:
    figure = go.Figure()
    rows = view["node_comparison"]["rows"][:MAX_NODE_COMPARISON_BARS]
    node_labels = list(dict.fromkeys(row["node"] for row in rows))
    missing_directions = []
    for direction in ("TX", "RX"):
        direction_rows = [row for row in rows if row["direction"] == direction]
        has_values = any(row["total_bytes"] is not None for row in direction_rows)
        if not has_values:
            missing_directions.append(direction)
        x_values = (
            [row["node"] for row in direction_rows]
            if direction_rows
            else node_labels
        )
        y_values = (
            [row["total_bytes"] for row in direction_rows]
            if direction_rows
            else [None] * len(node_labels)
        )
        figure.add_trace(
            go.Bar(
                x=x_values,
                y=y_values,
                name=direction,
                customdata=[
                    [row["status"], _format_percent(row["coverage_fraction"])]
                    for row in direction_rows
                ],
                hovertemplate=(
                    "%{x}<br>" + direction + " total: %{y} bytes<br>"
                    "Status: %{customdata[0]}<br>Coverage: %{customdata[1]}<extra></extra>"
                ),
            )
        )
    if missing_directions:
        figure.add_annotation(
            text=", ".join(missing_directions)
            + " missing (not observed as zero)",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.96,
            showarrow=False,
        )
    figure.update_layout(
        barmode="group",
        xaxis_title="Node",
        yaxis_title="Total bytes",
        legend_title="Direction",
        template="plotly_white",
        margin={"l": 56, "r": 16, "t": 24, "b": 48},
    )
    return dcc.Graph(
        id=NODE_GRAPH_ID,
        figure=figure,
        config={"displayModeBar": True, "responsive": True},
        className="analysis-graph",
    )


def _render_nic_detail(view: dict[str, Any]) -> html.Section:
    detail = view["nic_detail"]
    start_ns = view["capture"]["selected_start_ns"]
    figure = go.Figure()
    for record in detail["series"]:
        if record.get("display_timestamps_ns"):
            x_seconds = [
                (timestamp - start_ns) / 1_000_000_000
                for timestamp in record["display_timestamps_ns"]
            ]
            figure.add_trace(
                go.Scatter(
                    x=x_seconds,
                    y=record["display_values"],
                    mode="lines+markers",
                    name=f"{record['direction']} {record['counter']}",
                    hovertemplate="Time: %{x:.3f} s<br>Counter: %{y} bytes<extra></extra>",
                )
            )
    figure.update_layout(
        xaxis_title="Time from window start (s)",
        yaxis_title="Cumulative bytes",
        legend_title="Direction / counter",
        template="plotly_white",
        margin={"l": 64, "r": 16, "t": 24, "b": 48},
    )

    summary_cards: list[html.Div] = []
    for record in detail["series"]:
        summary_cards.extend(
            [
                _stat_card(
                    f"{record['direction']} total",
                    _format_bytes(record["total_bytes"]),
                    record["status"],
                ),
                _stat_card(
                    f"{record['direction']} mean rate",
                    _format_rate(record["mean_rate_bytes_per_s"]),
                    record["mean_rate_formula"],
                ),
                _stat_card(
                    f"{record['direction']} P99 rate",
                    _format_rate(record["p99_rate_bytes_per_s"]),
                    "Duration-weighted; no averaging of per-NIC P99 values.",
                ),
                _stat_card(
                    f"{record['direction']} coverage",
                    _format_percent(record["coverage_fraction"]),
                    record["aggregation_population"],
                ),
            ]
        )

    reliability = detail.get("reliability") or _empty_reliability_analysis()
    reliability_rows: list[dict[str, Any]] = []
    for collector in reliability.get("collector_analyses", ()):
        for counter in collector.get("counter_results", ()):
            reliability_rows.append(
                {
                    "collector": counter["collector_id"],
                    "counter": counter["counter"],
                    "role": counter["role"],
                    "status": counter["status"],
                    "total": counter["total"],
                    "unit": counter["unit"],
                    "rate": counter["rate_per_second"],
                    "coverage": counter["coverage_fraction"],
                    "first_nonzero_s": (
                        (counter["first_nonzero_sample_ns"] - start_ns) / 1_000_000_000
                        if counter["first_nonzero_sample_ns"] is not None
                        else None
                    ),
                    "quality": ", ".join(counter["quality_flags"]) or "none",
                }
            )
            if len(reliability_rows) >= MAX_RELIABILITY_COUNTER_RESULTS:
                break
        if len(reliability_rows) >= MAX_RELIABILITY_COUNTER_RESULTS:
            break

    first_nonzero = reliability.get("first_nonzero_error_sample_ns")
    first_collector = next(iter(reliability.get("collector_analyses", ())), {})
    reliability_panel = html.Div(
        className="reliability-panel",
        children=[
            html.H3("NIC-09 reliability", id="nic-09-reliability-heading"),
            html.Div(
                className="status-line",
                children=[
                    html.Span(
                        str(reliability.get("status", "Missing")),
                        className=_status_class(str(reliability.get("state", "missing"))),
                        role="status",
                    ),
                    html.Span(
                        "Counter evidence only; no causal reliability claim.",
                        className="status-note",
                    ),
                ],
            ),
            html.Div(
                className="card-grid",
                children=[
                    _stat_card(
                        "NIC-09 status",
                        str(reliability.get("status", "Missing")),
                        "Collector streams are analyzed separately and are never merged.",
                    ),
                    _stat_card(
                        "Collectors",
                        str(reliability.get("collector_count", 0)),
                        f"Bounded to {reliability.get('collector_limit', MAX_RELIABILITY_COLLECTORS)} streams; "
                        f"{reliability.get('duplicate_stream_count', 0)} maximum streams per counter.",
                    ),
                    _stat_card(
                        "First nonzero error",
                        (
                            f"{(first_nonzero - start_ns) / 1_000_000_000:.3f} s"
                            if first_nonzero is not None
                            else "Not observed"
                        ),
                        "First sample timestamp in the selected window; missing is not zero.",
                    ),
                    _stat_card(
                        "Reliability coverage",
                        _format_percent(first_collector.get("coverage_fraction")),
                        "Valid complete intervals divided by the requested window duration.",
                    ),
                ],
            ),
            dash_table.DataTable(
                columns=[
                    {"name": "Collector", "id": "collector"},
                    {"name": "Counter", "id": "counter"},
                    {"name": "Role", "id": "role"},
                    {"name": "Status", "id": "status"},
                    {"name": "Total", "id": "total"},
                    {"name": "Unit", "id": "unit"},
                    {"name": "Rate", "id": "rate"},
                    {"name": "Coverage", "id": "coverage"},
                    {"name": "First nonzero (s)", "id": "first_nonzero_s"},
                    {"name": "Quality", "id": "quality"},
                ],
                data=reliability_rows,
                filter_action="native",
                sort_action="native",
                page_action="native",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "minWidth": "120px"},
            ),
            html.P(
                "Zero is an observed value; missing and reset-affected intervals are "
                "reported separately. Good codewords are not errors. "
                f"At most {MAX_RELIABILITY_COUNTER_RESULTS} counter rows are rendered.",
                className="payload-note",
            ),
            html.Ul(
                children=[
                    html.Li(limitation)
                    for limitation in reliability.get("limitations", ())
                ],
                className="reliability-limitations",
            ),
        ],
    )

    matching = detail.get("matching") or _empty_matching_analysis()
    matching_rows: list[dict[str, Any]] = []
    for collector in matching.get("collector_analyses", ()):
        for counter in collector.get("counter_results", ()):
            matching_rows.append(
                {
                    "collector": counter["collector_id"],
                    "counter": counter["counter"],
                    "role": counter["role"],
                    "priority": counter["priority"],
                    "status": counter["status"],
                    "total": counter["total"],
                    "unit": counter["unit"],
                    "rate": counter["rate_per_second"],
                    "coverage": counter["coverage_fraction"],
                    "zero_observed": counter["zero_observed"],
                    "quality": ", ".join(counter["quality_flags"]) or "none",
                }
            )
            if len(matching_rows) >= MAX_MATCHING_COUNTER_RESULTS:
                break
        if len(matching_rows) >= MAX_MATCHING_COUNTER_RESULTS:
            break

    matching_priority_rows: list[dict[str, Any]] = []
    for collector in matching.get("collector_analyses", ()):
        for priority in collector.get("priority_results", ()):
            matching_priority_rows.append(
                {
                    "collector": collector["population"]["collector_id"],
                    "priority": priority["priority"],
                    "status": priority["status"],
                    "priority_total": priority["priority_total"],
                    "overflow_total": priority["overflow_total"],
                    "total": priority["total"],
                    "priority_fraction": priority["priority_fraction"],
                    "overflow_fraction": priority["overflow_fraction"],
                    "quality": ", ".join(priority["quality_flags"]) or "none",
                }
            )
            if len(matching_priority_rows) >= MAX_MATCHING_PRIORITY_RESULTS:
                break
        if len(matching_priority_rows) >= MAX_MATCHING_PRIORITY_RESULTS:
            break

    collector_fractions = tuple(
        matching.get("collector_priority_fractions", ())
    )
    fraction_values = tuple(
        (
            _format_percent(record.get("priority_fraction"))
            if record.get("priority_fraction") is not None
            else "Undefined"
        )
        for record in collector_fractions[:MAX_MATCHING_COLLECTORS]
    )
    matching_panel = html.Div(
        className="matching-panel",
        children=[
            html.H3("NIC-03 matching", id="nic-03-matching-heading"),
            html.Div(
                className="status-line",
                children=[
                    html.Span(
                        str(matching.get("status", "Missing")),
                        className=_status_class(str(matching.get("state", "missing"))),
                        role="status",
                    ),
                    html.Span(
                        "Receive-path evidence only; no copy-cost, latency, or causal claim.",
                        className="status-note",
                    ),
                ],
            ),
            html.Div(
                className="card-grid",
                children=[
                    _stat_card(
                        "NIC-03 status",
                        str(matching.get("status", "Missing")),
                        "Priority and overflow populations are combined only when compatible.",
                    ),
                    _stat_card(
                        "Collectors",
                        str(matching.get("collector_count", 0)),
                        f"Bounded to {matching.get('collector_limit', MAX_MATCHING_COLLECTORS)} streams; "
                        f"{matching.get('duplicate_stream_count', 0)} maximum streams per counter.",
                    ),
                    _stat_card(
                        "Priority fraction",
                        ", ".join(fraction_values) if fraction_values else "No collector",
                        "Collector-separated P / (P + O); undefined, missing, and incompatible values are not merged.",
                    ),
                    _stat_card(
                        "Matching coverage",
                        "Per collector",
                        "Each collector has its own valid-interval coverage; no collector-merged value is implied.",
                    ),
                ],
            ),
            dash_table.DataTable(
                columns=[
                    {"name": "Collector", "id": "collector"},
                    {"name": "Counter", "id": "counter"},
                    {"name": "Role", "id": "role"},
                    {"name": "Priority", "id": "priority"},
                    {"name": "Status", "id": "status"},
                    {"name": "Total", "id": "total"},
                    {"name": "Unit", "id": "unit"},
                    {"name": "Rate/s", "id": "rate"},
                    {"name": "Coverage", "id": "coverage"},
                    {"name": "Zero observed", "id": "zero_observed"},
                    {"name": "Quality", "id": "quality"},
                ],
                data=matching_rows,
                filter_action="native",
                sort_action="native",
                page_action="native",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "minWidth": "120px"},
            ),
            dash_table.DataTable(
                columns=[
                    {"name": "Collector", "id": "collector"},
                    {"name": "Priority", "id": "priority"},
                    {"name": "Status", "id": "status"},
                    {"name": "Priority total", "id": "priority_total"},
                    {"name": "Overflow total", "id": "overflow_total"},
                    {"name": "Total", "id": "total"},
                    {"name": "Priority fraction", "id": "priority_fraction"},
                    {"name": "Overflow fraction", "id": "overflow_fraction"},
                    {"name": "Quality", "id": "quality"},
                ],
                data=matching_priority_rows,
                filter_action="native",
                sort_action="native",
                page_action="native",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "minWidth": "120px"},
            ),
            html.P(
                "Zero and missing remain distinct; reset-affected intervals are excluded "
                "from totals, rates, and fractions. "
                f"At most {MAX_MATCHING_COUNTER_RESULTS} counter rows are rendered.",
                className="payload-note",
            ),
            html.Ul(
                children=[
                    html.Li(limitation)
                    for limitation in matching.get("limitations", ())
                ],
                className="matching-limitations",
            ),
        ],
    )

    flow_control = detail.get("flow_control") or _empty_flow_control_analysis()
    flow_control_rows: list[dict[str, Any]] = []
    for collector in flow_control.get("collector_analyses", ()):
        for counter in collector.get("counter_results", ()):
            flow_control_rows.append(
                {
                    "collector": counter["collector_id"],
                    "counter": counter["counter"],
                    "metric": counter["metric_kind"],
                    "direction": counter["direction"],
                    "priority": counter["priority"],
                    "status": counter["status"],
                    "total": counter["total"],
                    "unit": counter["unit"],
                    "rate": counter["mean_rate_per_second"],
                    "coverage": counter["coverage_fraction"],
                    "first_nonzero_s": (
                        (counter["first_nonzero_ns"] - start_ns) / 1_000_000_000
                        if counter["first_nonzero_ns"] is not None
                        else None
                    ),
                    "quality": ", ".join(counter["quality_flags"]) or "none",
                }
            )
            if len(flow_control_rows) >= MAX_FLOW_CONTROL_COUNTER_RESULTS:
                break
        if len(flow_control_rows) >= MAX_FLOW_CONTROL_COUNTER_RESULTS:
            break

    first_flow_nonzero = flow_control.get("first_nonzero_ns")
    flow_control_panel = html.Div(
        className="flow-control-panel",
        children=[
            html.H3("NIC-02 flow control", id="nic-02-flow-control-heading"),
            html.Div(
                className="status-line",
                children=[
                    html.Span(
                        str(flow_control.get("status", "Missing")),
                        className=_status_class(
                            str(flow_control.get("state", "missing"))
                        ),
                        role="status",
                    ),
                    html.Span(
                        "Pressure evidence only; no congestion, bottleneck, or ECN claim.",
                        className="status-note",
                    ),
                ],
            ),
            html.Div(
                className="card-grid",
                children=[
                    _stat_card(
                        "NIC-02 status",
                        str(flow_control.get("status", "Missing")),
                        "Collector streams are analyzed separately and are never merged.",
                    ),
                    _stat_card(
                        "Collectors",
                        str(flow_control.get("collector_count", 0)),
                        f"Bounded to {flow_control.get('collector_limit', MAX_FLOW_CONTROL_COLLECTORS)} streams; "
                        f"{flow_control.get('duplicate_stream_count', 0)} maximum streams per counter.",
                    ),
                    _stat_card(
                        "First nonzero",
                        (
                            f"{(first_flow_nonzero - start_ns) / 1_000_000_000:.3f} s"
                            if first_flow_nonzero is not None
                            else "Not observed"
                        ),
                        "First complete valid interval endpoint; missing is not zero.",
                    ),
                    _stat_card(
                        "Flow-control coverage",
                        "Per collector",
                        "Each collector has its own valid-interval coverage; no "
                        "job-wide or collector-merged value is implied.",
                    ),
                ],
            ),
            dash_table.DataTable(
                columns=[
                    {"name": "Collector", "id": "collector"},
                    {"name": "Counter", "id": "counter"},
                    {"name": "Metric", "id": "metric"},
                    {"name": "Direction", "id": "direction"},
                    {"name": "Priority", "id": "priority"},
                    {"name": "Status", "id": "status"},
                    {"name": "Total", "id": "total"},
                    {"name": "Unit", "id": "unit"},
                    {"name": "Rate/s", "id": "rate"},
                    {"name": "Coverage", "id": "coverage"},
                    {"name": "First nonzero (s)", "id": "first_nonzero_s"},
                    {"name": "Quality", "id": "quality"},
                ],
                data=flow_control_rows,
                filter_action="native",
                sort_action="native",
                page_action="native",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "minWidth": "120px"},
            ),
            html.P(
                "Pause cycles remain cycles because clock frequency is not inferred. "
                "Zero, missing, unsupported, and reset-affected results remain distinct. "
                f"At most {MAX_FLOW_CONTROL_COUNTER_RESULTS} counter rows are rendered.",
                className="payload-note",
            ),
            html.Ul(
                children=[
                    html.Li(limitation)
                    for limitation in flow_control.get("limitations", ())
                ],
                className="flow-control-limitations",
            ),
        ],
    )

    host_pressure = detail.get("host_pressure") or _empty_host_pressure_analysis()
    host_pressure_rows: list[dict[str, Any]] = []
    for collector in host_pressure.get("collector_analyses", ()):
        for path in collector.get("path_results", ()):
            host_pressure_rows.append(
                {
                    "collector": collector["population"]["collector_id"],
                    "path": path["path"],
                    "status": path["status"],
                    "packets": path["packet_total"],
                    "blocked": path["blocked_total"],
                    "packet_unit": path["packet_result"]["unit"],
                    "blocked_unit": path["blocked_result"]["unit"],
                    "packet_rate": path["packet_rate_per_second"],
                    "blocked_rate": path["blocked_rate_per_second"],
                    "blocked_per_packet": (
                        "Unsupported units"
                        if path["blocked_per_packet"] is None
                        else path["blocked_per_packet"]
                    ),
                    "population": (
                        "compatible" if path["population_compatible"] else "incompatible"
                    ),
                    "resets": path["reset_excluded_interval_count"],
                    "coverage": path["coverage_fraction"],
                    "quality": ", ".join(path["quality_flags"]) or "none",
                }
            )
            if len(host_pressure_rows) >= MAX_HOST_PRESSURE_PATH_RESULTS:
                break
        if len(host_pressure_rows) >= MAX_HOST_PRESSURE_PATH_RESULTS:
            break

    host_stall_rows: list[dict[str, Any]] = []
    for collector in host_pressure.get("collector_analyses", ()):
        for stall in collector.get("stall_results", ()):
            host_stall_rows.append(
                {
                    "collector": collector["population"]["collector_id"],
                    "counter": stall["counter"],
                    "kind": stall["stall_kind"],
                    "status": stall["status"],
                    "total": stall["total"],
                    "unit": stall["unit"],
                    "rate": stall["rate_per_second"],
                    "coverage": stall["coverage_fraction"],
                    "zero_observed": stall["zero_observed"],
                    "quality": ", ".join(stall["quality_flags"]) or "none",
                }
            )
            if len(host_stall_rows) >= MAX_HOST_PRESSURE_STALL_RESULTS:
                break
        if len(host_stall_rows) >= MAX_HOST_PRESSURE_STALL_RESULTS:
            break

    first_host_collector = next(iter(host_pressure.get("collector_analyses", ())), {})
    host_pressure_panel = html.Div(
        className="host-pressure-panel",
        children=[
            html.H3("NIC-04 host pressure", id="nic-04-host-pressure-heading"),
            html.Div(
                className="status-line",
                children=[
                    html.Span(
                        str(host_pressure.get("status", "Missing")),
                        className=_status_class(
                            str(host_pressure.get("state", "missing"))
                        ),
                        role="status",
                    ),
                    html.Span(
                        "Correlation evidence only; no DMA, dispatcher, or PCIe causal claim.",
                        className="status-note",
                    ),
                ],
            ),
            html.Div(
                className="card-grid",
                children=[
                    _stat_card(
                        "NIC-04 status",
                        str(host_pressure.get("status", "Missing")),
                        "Posted and nonposted populations are never merged.",
                    ),
                    _stat_card(
                        "Collectors",
                        str(host_pressure.get("collector_count", 0)),
                        f"Bounded to {host_pressure.get('collector_limit', MAX_HOST_PRESSURE_COLLECTORS)} streams; "
                        f"{host_pressure.get('duplicate_stream_count', 0)} maximum streams per counter.",
                    ),
                    _stat_card(
                        "Blocked per packet",
                        "Unsupported",
                        str(host_pressure.get("blocked_per_packet_status", "")),
                    ),
                    _stat_card(
                        "Host-pressure coverage",
                        _format_percent(first_host_collector.get("coverage_fraction")),
                        "Collector-separated common valid-interval coverage; no merged value.",
                    ),
                ],
            ),
            dash_table.DataTable(
                columns=[
                    {"name": "Collector", "id": "collector"},
                    {"name": "Path", "id": "path"},
                    {"name": "Status", "id": "status"},
                    {"name": "Packets", "id": "packets"},
                    {"name": "Blocked", "id": "blocked"},
                    {"name": "Packet unit", "id": "packet_unit"},
                    {"name": "Blocked unit", "id": "blocked_unit"},
                    {"name": "Packet rate/s", "id": "packet_rate"},
                    {"name": "Blocked rate/s", "id": "blocked_rate"},
                    {"name": "Blocked/packet", "id": "blocked_per_packet"},
                    {"name": "Population", "id": "population"},
                    {"name": "Resets", "id": "resets"},
                    {"name": "Coverage", "id": "coverage"},
                    {"name": "Quality", "id": "quality"},
                ],
                data=host_pressure_rows,
                filter_action="native",
                sort_action="native",
                page_action="native",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "minWidth": "120px"},
            ),
            dash_table.DataTable(
                columns=[
                    {"name": "Collector", "id": "collector"},
                    {"name": "Counter", "id": "counter"},
                    {"name": "Stall kind", "id": "kind"},
                    {"name": "Status", "id": "status"},
                    {"name": "Total", "id": "total"},
                    {"name": "Unit", "id": "unit"},
                    {"name": "Rate/s", "id": "rate"},
                    {"name": "Coverage", "id": "coverage"},
                    {"name": "Zero observed", "id": "zero_observed"},
                    {"name": "Quality", "id": "quality"},
                ],
                data=host_stall_rows,
                filter_action="native",
                sort_action="native",
                page_action="native",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "minWidth": "120px"},
            ),
            html.P(
                "Blocked-counter units remain unresolved, so blocked-per-packet is "
                "not reported. Zero, missing, insufficient, unsupported, and "
                "reset-affected states remain distinct. "
                f"At most {MAX_HOST_PRESSURE_PATH_RESULTS} path rows and "
                f"{MAX_HOST_PRESSURE_STALL_RESULTS} stall rows are rendered.",
                className="payload-note",
            ),
            html.Ul(
                children=[
                    html.Li(limitation)
                    for limitation in host_pressure.get("limitations", ())
                ],
                className="host-pressure-limitations",
            ),
        ],
    )

    operations = detail.get("operations") or _empty_operation_analysis()
    opcode_rows: list[dict[str, Any]] = []
    for collector in operations.get("collector_analyses", ()):
        for opcode in collector.get("opcode_results", ()):
            opcode_rows.append(
                {
                    "collector": collector["population"]["collector_id"],
                    "bin": opcode["bin_index"],
                    "label": (
                        opcode["opcode_label"]
                        if opcode["opcode_label"] is not None
                        else "Unknown"
                    ),
                    "mapped": opcode["opcode_mapped"],
                    "status": opcode["status"],
                    "total": opcode["total"],
                    "unit": opcode["unit"],
                    "rate": opcode["rate_per_second"],
                    "fraction": opcode["fraction_of_dma_opcode_commands"],
                    "coverage": opcode["coverage_fraction"],
                    "resets": opcode["invalid_interval_count"],
                    "zero_observed": opcode["zero_observed"],
                    "quality": ", ".join(opcode["quality_flags"]) or "none",
                }
            )
            if len(opcode_rows) >= MAX_OPERATION_OPCODE_RESULTS:
                break
        if len(opcode_rows) >= MAX_OPERATION_OPCODE_RESULTS:
            break

    total_rows: list[dict[str, Any]] = []
    for collector in operations.get("collector_analyses", ()):
        for result in (
            *collector.get("dma_total_results", ()),
            *collector.get("idc_total_results", ()),
        ):
            total_rows.append(
                {
                    "collector": collector["population"]["collector_id"],
                    "family": result["counter_kind"],
                    "queue": result["queue_index"],
                    "status": result["status"],
                    "total": result["total"],
                    "unit": result["unit"],
                    "rate": result["rate_per_second"],
                    "coverage": result["coverage_fraction"],
                    "resets": result["invalid_interval_count"],
                    "zero_observed": result["zero_observed"],
                    "quality": ", ".join(result["quality_flags"]) or "none",
                }
            )
            if len(total_rows) >= MAX_OPERATION_TOTAL_RESULTS:
                break
        if len(total_rows) >= MAX_OPERATION_TOTAL_RESULTS:
            break

    collector_mix_values = []
    for collector in operations.get("collector_analyses", ()):
        mix = collector.get("dma_idc_mix", {})
        if mix.get("dma_fraction") is None or mix.get("idc_fraction") is None:
            collector_mix_values.append(
                f"{collector['population']['collector_id']}: unsupported"
            )
            continue
        collector_mix_values.append(
            f"{collector['population']['collector_id']}: "
            f"{_format_percent(mix.get('dma_fraction'))} / "
            f"{_format_percent(mix.get('idc_fraction'))}"
        )
    operations_panel = html.Div(
        className="operations-panel",
        children=[
            html.H3("NIC-05 command operations", id="nic-05-operations-heading"),
            html.Div(
                className="status-line",
                children=[
                    html.Span(
                        str(operations.get("status", "Missing")),
                        className=_status_class(
                            str(operations.get("state", "missing"))
                        ),
                        role="status",
                    ),
                    html.Span(
                        "Control-path command counts; no wire-byte, MPI-payload, "
                        "eager/rendezvous, or causal claim.",
                        className="status-note",
                    ),
                ],
            ),
            html.Div(
                className="card-grid",
                children=[
                    _stat_card(
                        "NIC-05 status",
                        str(operations.get("status", "Missing")),
                        "DMA and IDC populations are combined only on exact common coverage.",
                    ),
                    _stat_card(
                        "Collectors",
                        str(operations.get("collector_count", 0)),
                        f"Bounded to {operations.get('collector_limit', MAX_OPERATION_COLLECTORS)} streams; "
                        f"{operations.get('duplicate_stream_count', 0)} maximum streams per counter.",
                    ),
                    _stat_card(
                        "Opcode mapping",
                        str(operations.get("opcode_mapping_status", "missing_mapping")),
                        str(operations.get("opcode_mapping_evidence", "")),
                    ),
                    _stat_card(
                        "DMA / IDC mix",
                        "; ".join(collector_mix_values) or "Unsupported",
                        "Collector-separated mixes; no merged value is shown.",
                    ),
                ],
            ),
            dash_table.DataTable(
                columns=[
                    {"name": "Collector", "id": "collector"},
                    {"name": "Opcode bin", "id": "bin"},
                    {"name": "Label", "id": "label"},
                    {"name": "Mapped", "id": "mapped"},
                    {"name": "Status", "id": "status"},
                    {"name": "Total", "id": "total"},
                    {"name": "Unit", "id": "unit"},
                    {"name": "Rate/s", "id": "rate"},
                    {"name": "Fraction", "id": "fraction"},
                    {"name": "Coverage", "id": "coverage"},
                    {"name": "Resets", "id": "resets"},
                    {"name": "Zero observed", "id": "zero_observed"},
                    {"name": "Quality", "id": "quality"},
                ],
                data=opcode_rows,
                filter_action="native",
                sort_action="native",
                page_action="native",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "minWidth": "120px"},
            ),
            dash_table.DataTable(
                columns=[
                    {"name": "Collector", "id": "collector"},
                    {"name": "Family", "id": "family"},
                    {"name": "Queue", "id": "queue"},
                    {"name": "Status", "id": "status"},
                    {"name": "Total", "id": "total"},
                    {"name": "Unit", "id": "unit"},
                    {"name": "Rate/s", "id": "rate"},
                    {"name": "Coverage", "id": "coverage"},
                    {"name": "Resets", "id": "resets"},
                    {"name": "Zero observed", "id": "zero_observed"},
                    {"name": "Quality", "id": "quality"},
                ],
                data=total_rows,
                filter_action="native",
                sort_action="native",
                page_action="native",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "minWidth": "120px"},
            ),
            html.P(
                "Unknown opcodes remain explicitly unlabeled. Zero, missing, "
                "insufficient, unsupported, and reset-affected states remain distinct. "
                f"At most {MAX_OPERATION_OPCODE_RESULTS} opcode rows and "
                f"{MAX_OPERATION_TOTAL_RESULTS} total rows are rendered.",
                className="payload-note",
            ),
            html.Ul(
                children=[
                    html.Li(limitation)
                    for limitation in operations.get("limitations", ())
                ],
                className="operations-limitations",
            ),
        ],
    )

    frames = detail.get("frames") or _empty_frame_analysis()
    frame_rows: list[dict[str, Any]] = []
    for collector in frames.get("collector_analyses", ()):
        for direction in collector.get("direction_results", ()):
            for bin_result in direction.get("bin_results", ()):
                frame_rows.append(
                    {
                        "collector": collector["population"]["collector_id"],
                        "direction": direction["direction"],
                        "bin": bin_result["bin_id"],
                        "counter": bin_result["counter"],
                        "status": bin_result["status"],
                        "frames": bin_result["total"],
                        "rate/s": bin_result["rate_per_second"],
                        "share": bin_result["bin_share"],
                        "coverage": bin_result["coverage_fraction"],
                        "resets": bin_result["invalid_interval_count"],
                        "zero": bin_result["zero_observed"],
                        "quality": ", ".join(bin_result["quality_flags"]) or "none",
                    }
                )
                if len(frame_rows) >= MAX_FRAME_BIN_RESULTS:
                    break
            if len(frame_rows) >= MAX_FRAME_BIN_RESULTS:
                break
        if len(frame_rows) >= MAX_FRAME_BIN_RESULTS:
            break

    frame_direction_values = []
    for collector in frames.get("collector_analyses", ()):
        for direction in collector.get("direction_results", ()):
            mean = direction.get("mean_wire_frame_size_bytes")
            value = (
                f"{direction['direction']}: {_format_number(mean)} B"
                if mean is not None
                else f"{direction['direction']}: {direction.get('mean_state', 'unsupported')}"
            )
            frame_direction_values.append(
                f"{collector['population']['collector_id']} / {value}"
            )

    frame_quantile_rows = []
    for collector in frames.get("collector_analyses", ()):
        for direction in collector.get("direction_results", ()):
            for quantile in direction.get("quantile_results", ()):
                frame_quantile_rows.append(
                    {
                        "collector": collector["population"]["collector_id"],
                        "direction": direction["direction"],
                        "quantile": quantile["quantile"],
                        "state": quantile["state"],
                        "lower_bytes": quantile["lower_bytes"],
                        "upper_bytes": quantile["upper_bytes"],
                    }
                )
                if len(frame_quantile_rows) >= MAX_FRAME_QUANTILE_RESULTS:
                    break
            if len(frame_quantile_rows) >= MAX_FRAME_QUANTILE_RESULTS:
                break
        if len(frame_quantile_rows) >= MAX_FRAME_QUANTILE_RESULTS:
            break

    frames_panel = html.Div(
        className="frames-panel",
        children=[
            html.H3("NIC-06 wire-frame sizes", id="nic-06-frames-heading"),
            html.Div(
                className="status-line",
                children=[
                    html.Span(
                        str(frames.get("status", "Missing")),
                        className=_status_class(str(frames.get("state", "missing"))),
                        role="status",
                    ),
                    html.Span(
                        "Wire-frame evidence only; no MPI-message, payload, "
                        "allreduce-size, or causal claim.",
                        className="status-note",
                    ),
                ],
            ),
            html.Div(
                className="card-grid",
                children=[
                    _stat_card(
                        "NIC-06 status",
                        str(frames.get("status", "Missing")),
                        "Direction and frame/byte populations remain explicit.",
                    ),
                    _stat_card(
                        "Collectors",
                        str(frames.get("collector_count", 0)),
                        f"Bounded to {frames.get('collector_limit', MAX_FRAME_COLLECTORS)} streams; "
                        f"{frames.get('duplicate_stream_count', 0)} maximum streams per counter.",
                    ),
                    _stat_card(
                        "Bin-edge provenance",
                        str(frames.get("bin_edges_source") or "Missing"),
                        "Means and quantiles are unsupported without sourced edges.",
                    ),
                    _stat_card(
                        "Mean frame size",
                        "; ".join(frame_direction_values) or "Unsupported",
                        "Requires compatible byte and frame populations plus exact coverage.",
                    ),
                ],
            ),
            dash_table.DataTable(
                columns=[
                    {"name": "Collector", "id": "collector"},
                    {"name": "Direction", "id": "direction"},
                    {"name": "Bin", "id": "bin"},
                    {"name": "Counter", "id": "counter"},
                    {"name": "Status", "id": "status"},
                    {"name": "Frames", "id": "frames"},
                    {"name": "Rate/s", "id": "rate/s"},
                    {"name": "Share", "id": "share"},
                    {"name": "Coverage", "id": "coverage"},
                    {"name": "Resets", "id": "resets"},
                    {"name": "Zero", "id": "zero"},
                    {"name": "Quality", "id": "quality"},
                ],
                data=frame_rows,
                filter_action="native",
                sort_action="native",
                page_action="native",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "minWidth": "120px"},
            ),
            dash_table.DataTable(
                columns=[
                    {"name": "Collector", "id": "collector"},
                    {"name": "Direction", "id": "direction"},
                    {"name": "Quantile", "id": "quantile"},
                    {"name": "State", "id": "state"},
                    {"name": "Lower bytes", "id": "lower_bytes"},
                    {"name": "Upper bytes", "id": "upper_bytes"},
                ],
                data=frame_quantile_rows,
                filter_action="native",
                sort_action="native",
                page_action="native",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "minWidth": "120px"},
            ),
            html.P(
                "Zero, missing, insufficient, unsupported, and reset-affected "
                "states remain distinct. Duplicate collectors are not merged. "
                f"At most {MAX_FRAME_BIN_RESULTS} bin rows and "
                f"{MAX_FRAME_QUANTILE_RESULTS} quantile rows are rendered.",
                className="payload-note",
            ),
            html.Ul(
                children=[
                    html.Li(limitation)
                    for limitation in frames.get("limitations", ())
                ],
                className="frames-limitations",
            ),
        ],
    )

    translation = detail.get("translation") or _empty_translation_analysis()
    translation_rows: list[dict[str, Any]] = []
    for collector in translation.get("collector_analyses", ()):
        for counter in collector.get("counter_results", ()):
            translation_rows.append(
                {
                    "collector": collector["population"]["collector_id"],
                    "counter": counter["counter"],
                    "role": counter["role"],
                    "miss_view": counter["miss_view"] or "—",
                    "status": counter["status"],
                    "total": counter["total"],
                    "unit": counter["unit"],
                    "rate/s": counter["rate_per_second"],
                    "coverage": counter["coverage_fraction"],
                    "resets": counter["invalid_interval_count"],
                    "zero": counter["zero_observed"],
                    "quality": ", ".join(counter["quality_flags"]) or "none",
                }
            )
            if len(translation_rows) >= MAX_TRANSLATION_COUNTER_RESULTS:
                break
        if len(translation_rows) >= MAX_TRANSLATION_COUNTER_RESULTS:
            break

    translation_view_rows: list[dict[str, Any]] = []
    for collector in translation.get("collector_analyses", ()):
        for view in collector.get("view_results", ()):
            translation_view_rows.append(
                {
                    "collector": collector["population"]["collector_id"],
                    "miss_view": view["miss_view"],
                    "status": view["status"],
                    "requests": view["request_total"],
                    "hits": view["hit_total"],
                    "misses": view["miss_total"],
                    "hit_ratio": view["hit_ratio"],
                    "misses/request": view["misses_per_request"],
                    "misses/GB": view["misses_per_gb"],
                    "traffic": view["traffic_status"],
                    "coverage": view["coverage_fraction"],
                    "quality": ", ".join(view["quality_flags"]) or "none",
                }
            )
            if len(translation_view_rows) >= MAX_TRANSLATION_VIEW_RESULTS:
                break
        if len(translation_view_rows) >= MAX_TRANSLATION_VIEW_RESULTS:
            break

    translation_view_values = []
    for collector in translation.get("collector_analyses", ()):
        for view in collector.get("view_results", ()):
            hit_ratio = view.get("hit_ratio")
            value = (
                f"{view['miss_view']}: {_format_percent(hit_ratio)}"
                if hit_ratio is not None
                else f"{view['miss_view']}: {view.get('status', 'unsupported')}"
            )
            translation_view_values.append(
                f"{collector['population']['collector_id']} / {value}"
            )

    translation_panel = html.Div(
        className="translation-panel",
        children=[
            html.H3(
                "NIC-07 address translation",
                id="nic-07-translation-heading",
            ),
            html.Div(
                className="status-line",
                children=[
                    html.Span(
                        str(translation.get("status", "Missing")),
                        className=_status_class(
                            str(translation.get("state", "missing"))
                        ),
                        role="status",
                    ),
                    html.Span(
                        "Counter evidence only; no ATU latency or causal claim.",
                        className="status-note",
                    ),
                ],
            ),
            html.Div(
                className="card-grid",
                children=[
                    _stat_card(
                        "NIC-07 status",
                        str(translation.get("status", "Missing")),
                        "Pool and client miss views are never merged.",
                    ),
                    _stat_card(
                        "Collectors",
                        str(translation.get("collector_count", 0)),
                        f"Bounded to {translation.get('collector_limit', MAX_TRANSLATION_COLLECTORS)} streams; "
                        f"{translation.get('duplicate_stream_count', 0)} maximum streams per counter.",
                    ),
                    _stat_card(
                        "Hit ratios",
                        "; ".join(translation_view_values) or "Unsupported",
                        "Requires compatible roles, units, populations, and exact intervals.",
                    ),
                    _stat_card(
                        "Traffic normalization",
                        str(translation.get("traffic_status", "not_supplied")),
                        "Traffic totals require the same population and exact interval window.",
                    ),
                ],
            ),
            dash_table.DataTable(
                columns=[
                    {"name": "Collector", "id": "collector"},
                    {"name": "Counter", "id": "counter"},
                    {"name": "Role", "id": "role"},
                    {"name": "Miss view", "id": "miss_view"},
                    {"name": "Status", "id": "status"},
                    {"name": "Total", "id": "total"},
                    {"name": "Unit", "id": "unit"},
                    {"name": "Rate/s", "id": "rate/s"},
                    {"name": "Coverage", "id": "coverage"},
                    {"name": "Resets", "id": "resets"},
                    {"name": "Zero", "id": "zero"},
                    {"name": "Quality", "id": "quality"},
                ],
                data=translation_rows,
                filter_action="native",
                sort_action="native",
                page_action="native",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "minWidth": "120px"},
            ),
            dash_table.DataTable(
                columns=[
                    {"name": "Collector", "id": "collector"},
                    {"name": "Miss view", "id": "miss_view"},
                    {"name": "Status", "id": "status"},
                    {"name": "Requests", "id": "requests"},
                    {"name": "Hits", "id": "hits"},
                    {"name": "Misses", "id": "misses"},
                    {"name": "Hit ratio", "id": "hit_ratio"},
                    {"name": "Misses/request", "id": "misses/request"},
                    {"name": "Misses/GB", "id": "misses/GB"},
                    {"name": "Traffic", "id": "traffic"},
                    {"name": "Coverage", "id": "coverage"},
                    {"name": "Quality", "id": "quality"},
                ],
                data=translation_view_rows,
                filter_action="native",
                sort_action="native",
                page_action="native",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "minWidth": "120px"},
            ),
            html.P(
                "Zero, missing, insufficient, unsupported, and reset-affected states "
                "remain distinct. Duplicate collectors are not merged. "
                f"At most {MAX_TRANSLATION_COUNTER_RESULTS} counter rows and "
                f"{MAX_TRANSLATION_VIEW_RESULTS} miss-view rows are rendered.",
                className="payload-note",
            ),
            html.Ul(
                children=[
                    html.Li(limitation)
                    for limitation in translation.get("limitations", ())
                ],
                className="translation-limitations",
            ),
        ],
    )

    latency = detail.get("latency") or _empty_latency_analysis()
    latency_rows: list[dict[str, Any]] = []
    for collector in latency.get("collector_analyses", ()):
        for bin_result in collector.get("bin_results", ()):
            latency_rows.append(
                {
                    "collector": collector["population"]["collector_id"],
                    "group": collector["histogram_group"],
                    "bin": bin_result["bin"],
                    "counter": bin_result["source_counter"],
                    "status": collector["status"],
                    "count": bin_result["count"],
                    "kind": bin_result["count_kind"],
                    "share": bin_result["normalized_fraction"],
                    "unit": collector["bin_unit"] or "unsourced",
                    "coverage": collector["coverage_fraction"],
                    "resets": collector["reset_excluded_interval_count"],
                    "quality": ", ".join(collector["quality_flags"]) or "none",
                }
            )
            if len(latency_rows) >= MAX_LATENCY_BIN_RESULTS:
                break
        if len(latency_rows) >= MAX_LATENCY_BIN_RESULTS:
            break

    latency_quantile_rows: list[dict[str, Any]] = []
    for collector in latency.get("collector_analyses", ()):
        for quantile in collector.get("quantile_results", ()):
            latency_quantile_rows.append(
                {
                    "collector": collector["population"]["collector_id"],
                    "group": collector["histogram_group"],
                    "quantile": quantile["quantile"],
                    "state": quantile["state"],
                    "bin": quantile["bin"],
                    "lower": quantile["lower_edge"],
                    "upper": quantile["upper_edge"],
                    "reason": quantile["reason"],
                }
            )
            if len(latency_quantile_rows) >= MAX_LATENCY_QUANTILE_RESULTS:
                break
        if len(latency_quantile_rows) >= MAX_LATENCY_QUANTILE_RESULTS:
            break

    latency_total_values = [
        f"{collector['histogram_group']}: {collector['total_observations']}"
        for collector in latency.get("collector_analyses", ())
        if collector.get("total_observations") is not None
    ]
    latency_panel = html.Div(
        className="latency-panel",
        children=[
            html.H3(
                "NIC-08 communication latency",
                id="nic-08-latency-heading",
            ),
            html.Div(
                className="status-line",
                children=[
                    html.Span(
                        str(latency.get("status", "Missing")),
                        className=_status_class(
                            str(latency.get("state", "missing"))
                        ),
                        role="status",
                    ),
                    html.Span(
                        "Histogram evidence only; no ATU-latency or MPI-duration substitution.",
                        className="status-note",
                    ),
                ],
            ),
            html.Div(
                className="card-grid",
                children=[
                    _stat_card(
                        "NIC-08 status",
                        str(latency.get("status", "Missing")),
                        "Missing, zero, partial, unsupported, and reset states stay distinct.",
                    ),
                    _stat_card(
                        "Collectors",
                        str(latency.get("collector_count", 0)),
                        f"Bounded to {latency.get('collector_limit', MAX_LATENCY_COLLECTORS)} streams; "
                        f"{latency.get('duplicate_stream_count', 0)} maximum streams per counter.",
                    ),
                    _stat_card(
                        "Observation totals",
                        "; ".join(latency_total_values) or "Unsupported",
                        "Histogram groups and collectors are not merged.",
                    ),
                    _stat_card(
                        "Bin metadata",
                        str(latency.get("bin_metadata_status", "unsourced")),
                        "Quantiles and means require sourced edges and units.",
                    ),
                ],
            ),
            dash_table.DataTable(
                columns=[
                    {"name": "Collector", "id": "collector"},
                    {"name": "Group", "id": "group"},
                    {"name": "Bin", "id": "bin"},
                    {"name": "Counter", "id": "counter"},
                    {"name": "Status", "id": "status"},
                    {"name": "Count", "id": "count"},
                    {"name": "Kind", "id": "kind"},
                    {"name": "Share", "id": "share"},
                    {"name": "Unit", "id": "unit"},
                    {"name": "Coverage", "id": "coverage"},
                    {"name": "Resets", "id": "resets"},
                    {"name": "Quality", "id": "quality"},
                ],
                data=latency_rows,
                filter_action="native",
                sort_action="native",
                page_action="native",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "minWidth": "120px"},
            ),
            dash_table.DataTable(
                columns=[
                    {"name": "Collector", "id": "collector"},
                    {"name": "Group", "id": "group"},
                    {"name": "Quantile", "id": "quantile"},
                    {"name": "State", "id": "state"},
                    {"name": "Bin", "id": "bin"},
                    {"name": "Lower", "id": "lower"},
                    {"name": "Upper", "id": "upper"},
                    {"name": "Reason", "id": "reason"},
                ],
                data=latency_quantile_rows,
                filter_action="native",
                sort_action="native",
                page_action="native",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "minWidth": "120px"},
            ),
            html.P(
                "Named-bin counts remain visible when encoding or edges are "
                "unsupported; they are not normalized as a distribution. "
                f"At most {MAX_LATENCY_BIN_RESULTS} bin rows and "
                f"{MAX_LATENCY_QUANTILE_RESULTS} quantile rows are rendered.",
                className="payload-note",
            ),
            html.Ul(
                children=[
                    html.Li(limitation)
                    for limitation in latency.get("limitations", ())
                ],
                className="latency-limitations",
            ),
        ],
    )

    return html.Section(
        className="panel",
        **{"aria-labelledby": "nic-detail-heading"},
        children=[
            html.H2("NIC detail", id="nic-detail-heading"),
            html.P(
                "Job / "
                f"{detail['selected_node'] if detail['selected_node'] else 'no selected node'} / "
                f"{detail['selected_nic'] if detail['selected_nic'] else 'no selected NIC'}",
                className="breadcrumb",
            ),
            html.P(detail["selection_note"], className="selection-note"),
            dcc.Graph(
                id="analysis-nic-graph",
                figure=figure,
                config={"displayModeBar": True, "responsive": True},
                className="analysis-graph",
            ),
            html.Div(className="card-grid", children=summary_cards),
            reliability_panel,
            matching_panel,
            flow_control_panel,
            host_pressure_panel,
            operations_panel,
            frames_panel,
            translation_panel,
            latency_panel,
            html.P(
                f"Display payload: at most {MAX_DISPLAY_POINTS_PER_SERIES} points per series. "
                "Summary statistics use full-resolution server-side analysis.",
                className="payload-note",
            ),
        ],
    )


def _render_job_summary(view: dict[str, Any]) -> html.Div:
    job_balance = view.get("job_balance") or _empty_job_balance()
    directions = job_balance.get("directions", ())
    cards: list[html.Div] = []
    quality_items: list[html.Li] = []
    for record in directions:
        direction = record.get("direction") or "unknown"
        cards.extend(
            [
                _stat_card(
                    f"{direction} job total",
                    _format_bytes(record.get("job_total_bytes")),
                    record.get("traffic_share_formula"),
                ),
                _stat_card(
                    f"{direction} effective nodes",
                    (
                        f"{record.get('effective_node_count'):.3f}"
                        if record.get("effective_node_count") is not None
                        else "Unknown"
                    ),
                    record.get("effective_node_count_formula"),
                ),
                _stat_card(
                    f"{direction} max / min node",
                    (
                        f"{record.get('max_min_ratio'):.2f}"
                        if record.get("max_min_ratio") is not None
                        else "Undefined / missing"
                    ),
                    f"max {_format_bytes(record.get('max_traffic_bytes'))}; "
                    f"min {_format_bytes(record.get('min_traffic_bytes'))}",
                ),
                _stat_card(
                    f"{direction} spatial node P99",
                    _format_bytes(record.get("spatial_p99_bytes")),
                    record.get("spatial_quantile_interpolation"),
                ),
                _stat_card(
                    f"{direction} top-1% hotspot share",
                    _format_percent(record.get("hotspot_concentration")),
                    f"largest {record.get('hotspot_node_count') or 0} node share(s)",
                ),
                _stat_card(
                    f"{direction} node coverage",
                    _format_percent(record.get("coverage_fraction")),
                    record.get("coverage_aggregation"),
                ),
            ]
        )
        quality_items.extend(
            [
                html.Li(
                    f"{direction} status: {record.get('status')}; "
                    f"state flags: {', '.join(record.get('state_flags', ())) or 'none'}"
                ),
                html.Li(
                    f"{direction} topology: {record.get('topology_state')}; "
                    f"missing nodes: {', '.join(record.get('missing_node_ids', ())) or 'none'}; "
                    f"unknown nodes: {', '.join(record.get('unknown_observed_node_ids', ())) or 'none'}"
                ),
                html.Li(
                    f"{direction} peer/near-far: "
                    f"{'supported' if record.get('peer_near_far_supported') else 'unsupported'}; "
                    f"{record.get('peer_near_far_reason')}"
                ),
            ]
        )

    return html.Div(
        className="job-summary",
        children=[
            html.Div(
                className="status-line",
                children=[
                    html.Span(
                        " / ".join(
                            record.get("status", "Missing")
                            for record in directions
                        ) or "Missing",
                        className=_status_class(
                            directions[0].get("state", "missing") if directions else "missing"
                        ),
                        role="status",
                    ),
                    html.Span(
                        "Node totals use compatible per-node/direction analyses.",
                        className="status-note",
                    ),
                ],
            ),
            html.P(
                "Node totals use the selected shared time window and compatible "
                "per-node/direction analyses.",
                className="selection-note",
            ),
            html.Div(className="card-grid", children=cards),
            html.Div(
                className="quality-card job-limitations",
                children=[
                    html.H3("Topology and interpretation limits"),
                    html.Ul(
                        children=quality_items
                        or [html.Li("No job-level direction analysis is available.")]
                    ),
                ],
            ),
        ],
    )


def _render_job_profile(view: dict[str, Any]) -> html.Section:
    profile = view.get("job_profile") or _empty_job_profile()
    dimension_rows = [
        {
            "dimension": row.get("dimension_id"),
            "title": row.get("title"),
            "status": row.get("status"),
            "evidence": row.get("observed_evidence_count", 0),
            "zero": row.get("zero_observation_count", 0),
            "missing": row.get("missing_observation_count", 0),
            "coverage": row.get("coverage_fraction"),
            "limitations": "; ".join(row.get("limitations", ())) or "None",
        }
        for row in profile.get("dimensions", ())
    ]
    return html.Section(
        id=JOB_PROFILE_ID,
        className="panel",
        **{"aria-labelledby": "job-profile-heading"},
        children=[
            html.H2("Job communication profile", id="job-profile-heading"),
            html.P(profile.get("summary"), className="panel-note"),
            html.Div(
                className="card-grid",
                children=[
                    _stat_card(
                        "Profile status",
                        str(profile.get("status", "unavailable")).title(),
                        "Descriptive only; no aggregate score or causal diagnosis.",
                    ),
                    _stat_card(
                        "Dimensions",
                        (
                            f"{len(profile.get('supported_dimensions', ()))} supported; "
                            f"{len(profile.get('partial_dimensions', ()))} partial; "
                            f"{len(profile.get('unsupported_dimensions', ()))} unsupported"
                        ),
                    ),
                    _stat_card(
                        "Question coverage",
                        f"{len(profile.get('questions', ()))} / 15",
                    ),
                    _stat_card(
                        "Topology",
                        str(profile.get("topology_status", "unsupported")).title(),
                        "; ".join(profile.get("topology_reasons", ()))
                        or "No mapping supplied.",
                    ),
                ],
            ),
            dash_table.DataTable(
                id="analysis-job-profile-table",
                columns=[
                    {"name": "Dimension", "id": "dimension"},
                    {"name": "Title", "id": "title"},
                    {"name": "Status", "id": "status"},
                    {"name": "Evidence", "id": "evidence", "type": "numeric"},
                    {"name": "Zero", "id": "zero", "type": "numeric"},
                    {"name": "Missing", "id": "missing", "type": "numeric"},
                    {"name": "Coverage", "id": "coverage", "type": "numeric"},
                    {"name": "Limitations", "id": "limitations"},
                ],
                data=dimension_rows,
                sort_action="native",
                filter_action="native",
                page_action="native",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "minWidth": "120px"},
            ),
        ],
    )


def _render_phase_analysis(view: dict[str, Any]) -> html.Div:
    phase = view.get("phase_analysis") or _empty_phase_analysis()
    cards: list[html.Div] = []
    for metric in phase.get("phases", ()):
        state_flags = ", ".join(metric.get("state_flags", ())) or "none"
        cards.extend(
            [
                _stat_card(
                    f"{metric.get('name')} total",
                    _format_bytes(metric.get("total_bytes")),
                    metric.get("status_reasons") or "Complete valid intervals only.",
                ),
                _stat_card(
                    f"{metric.get('name')} mean rate",
                    _format_rate(metric.get("mean_rate_bytes_per_s")),
                    f"coverage {_format_percent(metric.get('coverage_fraction'))}",
                ),
            ]
        )
        cards.append(
            html.Div(
                className="stat-card phase-state-card",
                children=[
                    html.P(f"{metric.get('name')} states", className="stat-label"),
                    html.P(state_flags, className="stat-value"),
                    html.P(
                        "reset observed"
                        if metric.get("reset_observed")
                        else "no reset observed in selected intervals",
                        className="stat-detail",
                    ),
                ],
            )
        )

    changes = phase.get("changes", ())
    change_items = [
        html.Li(
            f"{change.get('from_phase')} → {change.get('to_phase')}: "
            f"{change.get('comparison_status')}; "
            f"gap {change.get('inter_phase_gap_duration_ns')} ns; "
            f"{' / '.join(change.get('uncertainty_reasons', ())) or 'no additional uncertainty'}"
        )
        for change in changes
    ]
    return html.Div(
        id=PHASE_RESULT_ID,
        role="status",
        **{"aria-live": "polite"},
        children=[
            html.Div(
                className="status-line",
                children=[
                    html.Span(
                        phase.get("status", "No phase selected"),
                        className=_status_class(phase.get("state", "missing")),
                    ),
                    html.Span(
                        " / ".join(phase.get("status_reasons", ())) or "No status reason",
                        className="status-note",
                    ),
                ],
            ),
            html.Div(className="card-grid", children=cards),
            html.Div(
                className="quality-card",
                children=[
                    html.H3("Phase changes and uncertainty"),
                    html.Ul(
                        children=change_items
                        or [
                            html.Li(
                                "Add at least two ordered, non-overlapping phases for a change comparison."
                            )
                        ]
                    ),
                    html.P(phase.get("interpretation_limits"), className="panel-note"),
                ],
            ),
        ],
    )


def _counter_rows(store: TraceStore) -> list[dict[str, Any]]:
    observed = set(store.coverage.observed_counter_names)
    requested = set(store.coverage.requested_counter_names)
    unreadable = set(store.coverage.unreadable_counter_names)
    rows: list[dict[str, Any]] = []
    known_names: set[str] = set()
    for counter in store.counters:
        known_names.add(counter.name)
        try:
            direction = infer_traffic_direction(counter.name).value
        except ValueError:
            direction = "N/A"
        rows.append(
            {
                "counter": counter.name,
                "direction": direction,
                "unit": counter.unit,
                "semantics": counter.semantics.value,
                "value_kind": counter.value_kind.value,
                "state": "unreadable" if counter.name in unreadable else "available",
                "status": "Unreadable" if counter.name in unreadable else "Available",
            }
        )
    for name in sorted((requested | unreadable) - known_names):
        rows.append(
            {
                "counter": name,
                "direction": "Unknown",
                "unit": "unknown",
                "semantics": "unknown",
                "value_kind": "unknown",
                "state": "missing" if name not in unreadable else "unreadable",
                "status": "Missing" if name not in unreadable else "Unreadable",
            }
        )
    return rows


def _question_table(store: TraceStore) -> dash_table.DataTable:
    return dash_table.DataTable(
        id=QUESTION_TABLE_ID,
        columns=[
            {"name": "ID", "id": "id"},
            {"name": "Question", "id": "question"},
            {"name": "Milestone", "id": "milestone"},
            {"name": "Status", "id": "status"},
            {"name": "Reason / limitation", "id": "reason"},
            {"name": "Required", "id": "required_counter_count"},
            {"name": "Available", "id": "available_required_count"},
            {"name": "Missing", "id": "missing_required_count"},
            {"name": "Unreadable", "id": "unreadable_required_count"},
            {"name": "Implemented analyses", "id": "implemented_analyses"},
            {"name": "Scientific limitations", "id": "limitations"},
        ],
        data=_question_rows(store),
        filter_action="native",
        sort_action="native",
        page_action="native",
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "minWidth": "120px"},
    )


def _counter_table(store: TraceStore) -> dash_table.DataTable:
    return dash_table.DataTable(
        id=COUNTER_TABLE_ID,
        columns=[
            {"name": "Counter", "id": "counter"},
            {"name": "Direction", "id": "direction"},
            {"name": "Unit", "id": "unit"},
            {"name": "Semantics", "id": "semantics"},
            {"name": "Value kind", "id": "value_kind"},
            {"name": "Status", "id": "status"},
        ],
        data=_counter_rows(store),
        filter_action="native",
        sort_action="native",
        page_action="native",
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "minWidth": "100px"},
    )


def _trace_app_readiness_report(store: TraceStore, app: Dash) -> dict:
    checks = {
        "store_present": store is not None,
        "store_schema_supported": (
            store is not None and store.schema_version == SCHEMA_VERSION
        ),
        "indexed_rows_present": store is not None and store.row_count > 0,
        "trace_dimensions_present": (
            store is not None
            and bool(store.nodes)
            and bool(store.nics)
            and bool(store.counters)
            and bool(store.collectors)
        ),
        "observed_counters_present": (
            store is not None and bool(store.coverage.observed_counter_names)
        ),
        "layout_initialized": app.layout is not None,
        "callbacks_registered": bool(getattr(app, "callback_map", {})),
    }
    return {
        "status": "ready" if all(checks.values()) else "not_ready",
        "checks": checks,
    }


class _CallbackOutputId(str):
    def split(self, sep=None, maxsplit=-1):
        if sep == "..":
            return [
                part.strip(".")
                for part in super().split("...")
                if part
            ]
        return super().split(sep, maxsplit)


def create_analysis_app(
    store: TraceStore,
    *,
    path: str = "/",
    dataset_manager: Any | None = None,
    performance_recorder: PerformanceMetricsRecorder | None = None,
) -> Dash:
    """Create the legacy synthetic Dash application for one server-owned TraceStore."""

    if dataset_manager is not None and dataset_manager.current() is not None:
        store = dataset_manager.current().store

    initial_filters = default_filters(store)
    time_max = max(1, initial_filters.end_ns - initial_filters.start_ns)
    node_options = [
        {"label": f"All nodes ({len(store.nodes)})", "value": "all"},
        *[
            {"label": f"{node.label} (ID {node.node_id})", "value": f"node:{node.node_id}"}
            for node in store.nodes
        ],
    ]
    nic_options = [
        {"label": f"All NICs ({len(store.nics)})", "value": "all"},
        *[
            {"label": f"{nic.label} on node {nic.node_id} (ID {nic.nic_id})", "value": f"nic:{nic.nic_id}"}
            for nic in store.nics
        ],
    ]

    performance_recorder = performance_recorder or PerformanceMetricsRecorder()


    app = Dash(
        __name__,
        url_base_pathname=path,
        assets_folder=str(Path(__file__).parent / "assets"),
        title="Aurora NIC Metrics",
    )
    app.server.config["AURORA_PERFORMANCE_RECORDER"] = performance_recorder
    @app.server.get("/healthz")
    def healthz():
        return {"status": "ok"}, 200

    @app.server.get("/readyz")
    def readyz():
        report = _trace_app_readiness_report(store_provider(), app)
        return report, 200 if report["status"] == "ready" else 503

    app.layout = html.Div(
        className="app-shell",
        children=[
            html.Header(
                className="app-header",
                children=[
                    html.H1("Aurora NIC Metrics"),
                    html.P(f"Run: {store.identity.run_id}", className="run-id"),
                    html.Button(
                        "Export JSON",
                        id=EXPORT_BUTTON_ID,
                        className="export-button",
                        type="button",
                        n_clicks=0,
                        **{
                            "aria-label": "Export the currently filtered view model as JSON"
                        },
                    ),
                ],
            ),
            dcc.Download(id=EXPORT_DOWNLOAD_ID),
            html.Fieldset(
                className="filter-bar",
                children=[
                    html.Legend("Shared filters"),
                    html.Div(
                        className="filter-grid",
                        children=[
                            html.Label(
                                [
                                    html.Span("Node", className="control-label"),
                                    dcc.Dropdown(
                                        id=NODE_FILTER_ID,
                                        options=node_options,
                                        value="all",
                                        placeholder=f"All nodes ({len(store.nodes)})",
                                        clearable=False,
                                        searchable=True,
                                    ),
                                ],
                                className="control",
                            ),
                            html.Label(
                                [
                                    html.Span("NIC", className="control-label"),
                                    dcc.Dropdown(
                                        id=NIC_FILTER_ID,
                                        options=nic_options,
                                        value="all",
                                        placeholder=f"All NICs ({len(store.nics)})",
                                        clearable=False,
                                        searchable=True,
                                    ),
                                ],
                                className="control",
                            ),
                            html.Label(
                                [
                                    html.Span("Direction", className="control-label"),
                                    dcc.RadioItems(
                                        id=DIRECTION_FILTER_ID,
                                        options=[
                                            {"label": "All", "value": "all"},
                                            {"label": "TX", "value": "tx"},
                                            {"label": "RX", "value": "rx"},
                                        ],
                                        value="all",
                                        inline=True,
                                    ),
                                ],
                                className="control",
                            ),
                            html.Div(
                                className="control filter-reset-control",
                                children=[
                                    html.Button(
                                        "Reset filters",
                                        id=FILTER_RESET_ID,
                                        className="export-button",
                                        type="button",
                                        n_clicks=0,
                                        **{
                                            "aria-label": (
                                                "Reset node, NIC, direction, and time filters"
                                            )
                                        },
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Label(
                        [
                            html.Span("Time window", className="control-label"),
                            dcc.RangeSlider(
                                id=TIME_FILTER_ID,
                                min=0,
                                max=time_max,
                                step=1,
                                value=[0, time_max],
                                allowCross=False,
                                marks={0: "start", time_max: "end"},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                        className="time-control",
                    ),
                    html.Div(
                        id=FILTER_STATUS_ID,
                        children=_render_filter_status(
                            build_view_model(store, initial_filters)
                        ),
                    ),
                ],
            ),
            dcc.Store(
                id=TIME_ORIGIN_ID,
                data={"acquisition_start_ns": initial_filters.start_ns},
                storage_type="memory",
            ),
            html.Div(
                className="chart-mode-bar",
                children=[
                    html.Label("NIC chart mode", className="control-label"),
                    dcc.RadioItems(
                        id=investigation_shell.CHART_MODE_ID,
                        options=[
                            {"label": "Rate", "value": "rate"},
                            {"label": "Cumulative", "value": "cumulative"},
                        ],
                        value="rate",
                        inline=True,
                    ),
                ],
            ),
            html.Main(
                className="app-main",
                children=dcc.Tabs(
                    id="analysis-tabs",
                    value="overview",
                    children=[
                        dcc.Tab(
                            label="Overview and nodes",
                            value="overview",
                            children=html.Div(
                                className="tab-panel",
                                children=[
                                    html.Div(
                                        id=OVERVIEW_ID,
                                        children=_render_overview(
                                            build_view_model(store, initial_filters)
                                        ),
                                    ),
                                    html.Section(
                                        className="panel",
                                        **{"aria-labelledby": "node-heading"},
                                        children=[
                                            html.H2("Top node comparison", id="node-heading"),
                                            html.P(
                                                "Ranked by TX/RX totals in bytes. Coverage is the minimum "
                                                "fraction across selected NIC/direction series; missing series "
                                                "are not treated as zero.",
                                                className="panel-note",
                                            ),
                                            _render_node_graph(
                                                build_view_model(store, initial_filters)
                                            ),
                                            _render_node_table(
                                                build_view_model(store, initial_filters)
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ),
                        dcc.Tab(
                            label="NIC detail",
                            value="nic",
                            children=html.Div(
                                className="tab-panel",
                                children=[
                                    html.Div(id=NIC_DETAIL_ID),
                                    html.Section(
                                        id=PHASE_PANEL_ID,
                                        className="panel",
                                        **{"aria-labelledby": "phase-heading"},
                                        children=[
                                            html.H2(
                                                "Manual phase analysis",
                                                id="phase-heading",
                                            ),
                                            html.P(
                                                "Annotate the currently selected node, NIC, and TX/RX "
                                                "direction. Boundaries are trace-clock nanoseconds; "
                                                "phases must not overlap, and gaps remain unassigned.",
                                                className="panel-note",
                                            ),
                                            html.Div(
                                                className="phase-form",
                                                children=[
                                                    html.Label(
                                                        [
                                                            html.Span(
                                                                "Phase name",
                                                                className="control-label",
                                                            ),
                                                            dcc.Input(
                                                                id=PHASE_NAME_ID,
                                                                type="text",
                                                                value="",
                                                                placeholder="steady-state",
                                                                debounce=True,
                                                            ),
                                                        ],
                                                        className="control",
                                                    ),
                                                    html.Label(
                                                        [
                                                            html.Span(
                                                                "Start (ns)",
                                                                className="control-label",
                                                            ),
                                                            dcc.Input(
                                                                id=PHASE_START_ID,
                                                                type="number",
                                                                value=None,
                                                                min=initial_filters.start_ns,
                                                                max=initial_filters.end_ns,
                                                                step=1,
                                                                debounce=True,
                                                            ),
                                                        ],
                                                        className="control",
                                                    ),
                                                    html.Label(
                                                        [
                                                            html.Span(
                                                                "End (ns)",
                                                                className="control-label",
                                                            ),
                                                            dcc.Input(
                                                                id=PHASE_END_ID,
                                                                type="number",
                                                                value=None,
                                                                min=initial_filters.start_ns,
                                                                max=initial_filters.end_ns,
                                                                step=1,
                                                                debounce=True,
                                                            ),
                                                        ],
                                                        className="control",
                                                    ),
                                                    html.Button(
                                                        "Add phase",
                                                        id=PHASE_ADD_ID,
                                                        className="export-button",
                                                        type="button",
                                                        n_clicks=0,
                                                        **{
                                                            "aria-label": (
                                                                "Add a manual phase window to the analysis"
                                                            )
                                                        },
                                                    ),
                                                ],
                                            ),
                                            dcc.Store(
                                                id=PHASE_STORE_ID,
                                                data=[],
                                                storage_type="memory",
                                            ),
                                            _render_phase_analysis(
                                                build_view_model(store, initial_filters)
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ),
                        dcc.Tab(
                            label="Questions and counters",
                            value="questions",
                            children=html.Div(
                                className="tab-panel",
                                children=[
                                    html.Section(
                                        className="panel",
                                        **{"aria-labelledby": "question-heading"},
                                        children=[
                                            html.H2("Question navigation", id="question-heading"),
                                            html.P(
                                                "Search all 15 PDF-derived question families. NIC-01 is "
                                                "the only supported family in this synthetic slice; other "
                                                "families remain explicitly pending or unsupported. NIC-01 "
                                                "uses the HPE-sourced network-frame octet registry, and "
                                                "BAD-octet evidence stays separate.",
                                                className="panel-note",
                                            ),
                                            _question_table(store),
                                        ],
                                    ),
                                    html.Section(
                                        className="panel",
                                        **{"aria-labelledby": "counter-heading"},
                                        children=[
                                            html.H2("Counter navigation", id="counter-heading"),
                                            html.P(
                                                "Counter metadata and availability only; source values "
                                                "remain server-side.",
                                                className="panel-note",
                                            ),
                                            _counter_table(store),
                                        ],
                                    ),
                                ],
                            ),
                        ),
                    ],
                ),
            ),
        ],
    )
    initial_view = build_view_model(store, initial_filters)
    investigation_state = initial_state(
        run_id=store.identity.run_id,
        source_fingerprint=store.identity.source_fingerprint,
        schema_version=store.schema_version,
        start_ns=initial_filters.start_ns,
        end_ns=initial_filters.end_ns,
    )
    component_ids = {
        "state_store": investigation_shell.STATE_STORE_ID,
        "tabs": investigation_shell.TABS_ID,
        "back": investigation_shell.BACK_BUTTON_ID,
        "reset_selection": investigation_shell.RESET_SELECTION_ID,
        "reset_interval": investigation_shell.RESET_INTERVAL_ID,
        "chart_mode": investigation_shell.CHART_MODE_ID,
        "node_filter": NODE_FILTER_ID,
        "nic_filter": NIC_FILTER_ID,
        "direction_filter": DIRECTION_FILTER_ID,
        "time_filter": TIME_FILTER_ID,
        "time_origin": TIME_ORIGIN_ID,
        "nic_detail": NIC_DETAIL_ID,
        "node_view": "analysis-node-view",
        "overview": OVERVIEW_ID,
        "filter_status": FILTER_STATUS_ID,
        "filter_bar": "analysis-filter-bar",
        "question_select": investigation_question_index.QUESTION_SELECT_ID,
        "question_result": investigation_question_index.QUESTION_RESULT_ID,
        "question_list": investigation_question_index.QUESTION_LIST_ID,
        "interval_store": investigation_question_index.INTERVAL_STORE_ID,
        "interval_result": investigation_question_index.INTERVAL_RESULT_ID,
    }
    nodes_by_id = {node.node_id: node for node in store.nodes}
    node_action_buttons = []
    seen_nic_ids = set()
    for nic in store.nics:
        if nic.nic_id in seen_nic_ids:
            continue
        seen_nic_ids.add(nic.nic_id)
        node = nodes_by_id[nic.node_id]
        node_action_buttons.append(
            investigation_overview.open_action(
                {
                    "node": node.label,
                    "nic": nic.label,
                    "node_id": node.node_id,
                    "nic_id": nic.nic_id,
                    "direction": "TX",
                },
                section="node-nic-open",
            )
        )
    dataset_controls = (
        DatasetControls(dataset_manager.dataset_ids)
        if dataset_manager is not None
        else None
    )

    def store_provider():
        if dataset_manager is not None and dataset_manager.current() is not None:
            return dataset_manager.current().store
        return store

    recent_selection_tokens: set[tuple[str, str]] = set()
    selection_token_lock = threading.Lock()
    original_build_view_model = build_view_model

    def _performance_selection_token(filters, phase_windows) -> str:
        resolved = filters or default_filters(store)
        return (
            f"node={resolved.node_id};nic={resolved.nic_id};"
            f"direction={resolved.direction};start={resolved.start_ns};"
            f"end={resolved.end_ns};phases={phase_windows!r}"
        )

    def performance_build_view_model(
        current_store: TraceStore,
        filters: AnalysisFilterState | None = None,
        phase_windows: Any = (),
    ) -> dict[str, Any]:
        dataset_token = current_store.identity.source_fingerprint
        selection_token = _performance_selection_token(filters, phase_windows)
        request_token = uuid.uuid4().hex
        identity_key = (dataset_token, selection_token)
        with selection_token_lock:
            stage = (
                "warm_repeat"
                if identity_key in recent_selection_tokens
                else "warm_new_query"
            )
            recent_selection_tokens.add(identity_key)

        operation = performance_recorder.begin(
            stage,
            dataset_token=dataset_token,
            request_token=request_token,
            selection_token=selection_token,
        )
        try:
            view = original_build_view_model(
                current_store,
                filters,
                phase_windows=phase_windows,
            )
            try:
                payload_bytes = len(json.dumps(view, default=str).encode("utf-8"))
            except (TypeError, ValueError):
                payload_bytes = len(str(view).encode("utf-8"))
            performance_recorder.finish(
                operation,
                result_identity=MetricIdentity(
                    dataset_token=dataset_token,
                    request_token=request_token,
                    selection_token=selection_token,
                ),
                payload_bytes=payload_bytes,
            )
            return view
        except Exception as error:
            performance_recorder.record_exception(operation, error)
            raise

    def current_time_max():
        current_filters = default_filters(store_provider())
        return max(1, current_filters.end_ns - current_filters.start_ns)

    investigation_layout_template = html.Div(
        className="investigation-app",
        children=[
            investigation_shell.render_shell(
                initial_view,
                investigation_state,
                export_button_id=EXPORT_BUTTON_ID,
                export_download_id=EXPORT_DOWNLOAD_ID,
                filter_status_id=FILTER_STATUS_ID,
            ),
            *(
                [dataset_controls.render()]
                if dataset_controls is not None
                else []
            ),
            dcc.Store(
                id=DATASET_LOADED_STORE_ID,
                data=(
                    dataset_manager.current().descriptor.dataset_id
                    if dataset_manager is not None and dataset_manager.current() is not None
                    else store.identity.run_id
                ),
                storage_type="memory",
            ),
            html.Fieldset(
                id=component_ids["filter_bar"],
                className="filter-bar compact-filter-bar",
                style={"display": "none"},
                children=[
                    html.Legend("Shared selection"),
                    html.Div(
                        className="filter-grid",
                        children=[
                            html.Label(
                                [
                                    html.Span("Node", className="control-label"),
                                    dcc.Dropdown(
                                        id=NODE_FILTER_ID,
                                        options=node_options,
                                        value=None,
                                        placeholder=f"All nodes ({len(store.nodes)})",
                                        clearable=False,
                                        searchable=True,
                                    ),
                                ],
                                className="control",
                            ),
                            html.Label(
                                [
                                    html.Span("NIC", className="control-label"),
                                    dcc.Dropdown(
                                        id=NIC_FILTER_ID,
                                        options=nic_options,
                                        value=None,
                                        placeholder=f"All NICs ({len(store.nics)})",
                                        clearable=False,
                                        searchable=True,
                                    ),
                                ],
                                className="control",
                            ),
                            html.Label(
                                [
                                    html.Span("Direction", className="control-label"),
                                    dcc.RadioItems(
                                        id=DIRECTION_FILTER_ID,
                                        options=[
                                            {"label": "All", "value": "all"},
                                            {"label": "TX", "value": "tx"},
                                            {"label": "RX", "value": "rx"},
                                        ],
                                        value="all",
                                        inline=True,
                                    ),
                                ],
                                className="control",
                            ),
                            html.Div(
                                className="control filter-reset-control",
                                children=[
                                    html.Button(
                                        "Reset filters",
                                        id=FILTER_RESET_ID,
                                        className="secondary-button",
                                        type="button",
                                        n_clicks=0,
                                        **{
                                            "aria-label": (
                                                "Reset node, NIC, direction, and time filters"
                                            )
                                        },
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Label(
                        [
                            html.Span("Shared interval", className="control-label"),
                            dcc.RangeSlider(
                                id=TIME_FILTER_ID,
                                min=0,
                                max=time_max,
                                step=1,
                                value=[0, time_max],
                                allowCross=False,
                                marks={0: "start", time_max: "end"},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                        className="time-control",
                    ),
                ],
            ),
            dcc.Store(
                id=TIME_ORIGIN_ID,
                data={"acquisition_start_ns": initial_filters.start_ns},
                storage_type="memory",
            ),
            html.Main(
                className="investigation-main",
                children=dcc.Tabs(
                    id=investigation_shell.TABS_ID,
                    value="overview",
                    children=[
                        dcc.Tab(
                            label="Overview",
                            value="overview",
                            children=html.Div(
                                className="overview-layout",
                                children=[
                                    investigation_overview.render_timeline(
                                        initial_view, graph_id=NODE_GRAPH_ID
                                    ),
                                    investigation_overview.render_ranked_table(
                                        initial_view,
                                        table_id=NODE_TABLE_ID,
                                        actions_id=NODE_ACTIONS_ID,
                                    ),
                                    html.Div(
                                        id=OVERVIEW_ID,
                                        style={
                                            "display": "none",
                                            "visibility": "hidden",
                                            "transition": (
                                                "visibility 0s linear 500ms"
                                            ),
                                        },
                                        children=investigation_overview.render_overview(
                                            initial_view
                                        ),
                                    ),
                                ],
                            ),
                        ),
                        dcc.Tab(
                            label="Node",
                            value="node",
                            children=html.Div(
                                className="node-screen",
                                children=[
                                    html.Div(
                                        className="node-action-bar",
                                        children=node_action_buttons,
                                    ),
                                    html.Div(
                                        id=component_ids["node_view"],
                                        children=investigation_node_view.render_node_view(
                                            initial_view
                                        ),
                                    ),
                                ],
                            ),
                        ),
                        dcc.Tab(
                            label="NIC detail",
                            value="nic",
                            children=html.Div(
                                id=NIC_DETAIL_ID,
                                children=investigation_nic_view.render_nic_detail(
                                    initial_view,
                                    chart_mode_id=investigation_shell.CHART_MODE_ID,
                                ),
                            ),
                        ),
                        dcc.Tab(
                            label="Questions",
                            value=investigation_shell.QUESTION_TAB_VALUE,
                            children=html.Div(
                                className="question-screen",
                                children=[
                                    investigation_question_index.render_question_index(
                                        initial_view,
                                        investigation_diagnostics.payload_by_question(
                                            initial_view
                                        ),
                                    ),
                                    investigation_question_index.render_interval_controls(),
                                    dcc.Store(
                                        id=investigation_question_index.INTERVAL_STORE_ID,
                                        data=[],
                                        storage_type="memory",
                                    ),
                                    html.Div(
                                        id=investigation_question_index.INTERVAL_RESULT_ID,
                                        children=[
                                            html.P(
                                                "No named intervals selected.",
                                                className="empty-state",
                                            )
                                        ],
                                    ),
                                    html.Div(
                                        id=investigation_question_index.QUESTION_RESULT_ID,
                                        children=[
                                            html.P(
                                                "No diagnostic requested yet.",
                                                className="empty-state",
                                            )
                                        ],
                                    ),
                                ],
                            ),
                        ),
                    ],
                ),
            ),
            html.Details(
                className="legacy-report",
                children=[
                    html.Summary(
                        "Temporary legacy diagnostic report — closed by default"
                    ),
                    html.P(
                        "Compatibility-only counter tables, manual phases, and diagnostic panels. "
                        "Traffic analysis uses the compact workspace above."
                    ),
                    _render_job_profile(initial_view),
                    _question_table(store),
                    _counter_table(store),
                    html.Section(
                        id=PHASE_PANEL_ID,
                        children=[
                            html.H2("Manual phase analysis", id="phase-heading"),
                            html.Div(
                                className="phase-form",
                                children=[
                                    dcc.Input(
                                        id=PHASE_NAME_ID,
                                        type="text",
                                        value="",
                                        placeholder="steady-state",
                                        debounce=True,
                                    ),
                                    dcc.Input(
                                        id=PHASE_START_ID,
                                        type="number",
                                        value=None,
                                        min=initial_filters.start_ns,
                                        max=initial_filters.end_ns,
                                        step=1,
                                        debounce=True,
                                    ),
                                    dcc.Input(
                                        id=PHASE_END_ID,
                                        type="number",
                                        value=None,
                                        min=initial_filters.start_ns,
                                        max=initial_filters.end_ns,
                                        step=1,
                                        debounce=True,
                                    ),
                                    html.Button(
                                        "Add phase",
                                        id=PHASE_ADD_ID,
                                        className="secondary-button",
                                        type="button",
                                        n_clicks=0,
                                        **{
                                            "aria-label": (
                                                "Add a manual phase window to the analysis"
                                            )
                                        },
                                    ),
                                ],
                            ),
                            dcc.Store(id=PHASE_STORE_ID, data=[], storage_type="memory"),
                            _render_phase_analysis(initial_view),
                        ],
                    ),
                ],
            ),
        ],
    )
    app.layout = investigation_layout_template

    @app.server.before_request
    def reset_investigation_layout():
        if request.method == "GET" and request.path in {
            path,
            path + "_dash-layout",
        }:
            app.layout = deepcopy(investigation_layout_template)
    @app.callback(
        Output(PHASE_STORE_ID, "data"),
        Output(PHASE_RESULT_ID, "children"),
        Input(PHASE_ADD_ID, "n_clicks"),
        Input(NODE_FILTER_ID, "value"),
        Input(NIC_FILTER_ID, "value"),
        Input(DIRECTION_FILTER_ID, "value"),
        Input(TIME_FILTER_ID, "value"),
        State(PHASE_NAME_ID, "value"),
        State(PHASE_START_ID, "value"),
        State(PHASE_END_ID, "value"),
        State(PHASE_STORE_ID, "data"),
        State(TIME_ORIGIN_ID, "data"),
        prevent_initial_call=True,
    )
    def update_manual_phase(
        add_clicks,
        node_value,
        nic_value,
        direction_value,
        time_value,
        phase_name,
        phase_start,
        phase_end,
        phase_store,
        time_origin,
    ):
        current_store = store_provider()
        if not add_clicks:
            return no_update, no_update
        node_id = _parse_identity_value(node_value, "node")
        nic_id = _parse_identity_value(nic_value, "nic")
        if node_id is not None and not any(
            nic.node_id == node_id and nic.nic_id == nic_id
            for nic in current_store.nics
        ):
            nic_id = None
            selected_nic_value = "all"
        else:
            selected_nic_value = nic_value
        origin = int(time_origin["acquisition_start_ns"])
        window = time_value if time_value else [0, current_time_max()]
        filters = AnalysisFilterState(
            node_id=node_id,
            nic_id=nic_id,
            direction=direction_value,
            start_ns=origin + int(window[0]),
            end_ns=origin + int(window[-1]),
        )
        current = list(phase_store or [])

        if callback_context.triggered_id == PHASE_ADD_ID:
            name = phase_name.strip() if isinstance(phase_name, str) else ""
            if not name:
                return no_update, _render_phase_analysis(
                    {"phase_analysis": _phase_error(
                        "Invalid phase input",
                        ("The phase name must be a nonempty string.",),
                    )}
                )
            if isinstance(phase_start, bool) or not isinstance(phase_start, (int, float)):
                return no_update, _render_phase_analysis(
                    {"phase_analysis": _phase_error(
                        "Invalid phase input",
                        ("The phase start must be an integer nanosecond value.",),
                    )}
                )
            if isinstance(phase_end, bool) or not isinstance(phase_end, (int, float)):
                return no_update, _render_phase_analysis(
                    {"phase_analysis": _phase_error(
                        "Invalid phase input",
                        ("The phase end must be an integer nanosecond value.",),
                    )}
                )
            start_ns = int(phase_start)
            end_ns = int(phase_end)
            if start_ns != phase_start or end_ns != phase_end:
                return no_update, _render_phase_analysis(
                    {"phase_analysis": _phase_error(
                        "Invalid phase input",
                        ("Phase boundaries must be integer nanosecond values.",),
                    )}
                )
            if start_ns >= end_ns:
                return no_update, _render_phase_analysis(
                    {"phase_analysis": _phase_error(
                        "Invalid phase input",
                        ("The phase window must have positive duration.",),
                    )}
                )
            if start_ns < filters.start_ns or end_ns > filters.end_ns:
                return no_update, _render_phase_analysis(
                    {"phase_analysis": _phase_error(
                        "Phase outside selected window",
                        (f"The phase must lie within {filters.start_ns}–{filters.end_ns} ns.",),
                    )}
                )
            if any(item.get("name") == name for item in current):
                return no_update, _render_phase_analysis(
                    {"phase_analysis": _phase_error(
                        "Invalid phase input",
                        (f"Phase names must be unique; {name!r} is already used.",),
                    )}
                )
            if len(current) >= MAX_PHASE_WINDOWS:
                return no_update, _render_phase_analysis(
                    {"phase_analysis": _phase_error(
                        "Phase limit reached",
                        (f"At most {MAX_PHASE_WINDOWS} phase windows are supported.",),
                    )}
                )

            candidate = [*current, {"name": name, "start_ns": start_ns, "end_ns": end_ns}]
            try:
                ordered = _normalize_phase_windows(candidate)
                for previous, next_phase in zip(ordered, ordered[1:]):
                    if next_phase.start_ns < previous.end_ns:
                        raise ValueError(
                            f"phase windows {previous.name!r} and {next_phase.name!r} overlap"
                        )
                    gap = next_phase.start_ns - previous.end_ns
                    if gap > 0:
                        raise ValueError(
                            f"phase windows {previous.name!r} and {next_phase.name!r} "
                            f"have a {gap} ns gap; adjacent phases are required"
                        )
            except ValueError as error:
                return no_update, _render_phase_analysis(
                    {"phase_analysis": _phase_error("Invalid phase windows", (str(error),))}
                )
            current = [
                {"name": phase.name, "start_ns": phase.start_ns, "end_ns": phase.end_ns}
                for phase in ordered
            ]

        phase = build_phase_analysis(current_store, filters, current)
        return current, _render_phase_analysis({"phase_analysis": phase})

    @app.callback(
        Output(NODE_FILTER_ID, "value"),
        Output(NIC_FILTER_ID, "value"),
        Output(DIRECTION_FILTER_ID, "value"),
        Output(TIME_FILTER_ID, "value"),
        Input(FILTER_RESET_ID, "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_shared_filters(n_clicks):
        if not n_clicks:
            return no_update, no_update, no_update, no_update
        return "all", "all", "all", [0, current_time_max()]

    @app.callback(
        Output(NODE_FILTER_ID, "value"),
        Output(NIC_FILTER_ID, "value"),
        Input(
            {"section": ALL, "node": ALL, "nic": ALL},
            "n_clicks",
        ),
        prevent_initial_call=True,
    )
    def open_ranked_node_nic(clicks):
        identity = callback_context.triggered_id
        if not clicks or not any(clicks):
            return no_update, no_update
        if (
            not isinstance(identity, dict)
            or identity.get("section") not in {NODE_OPEN_SECTION, "node-nic-open"}
        ):
            return no_update, no_update
        return (
            f"node:{identity['node']}",
            f"nic:{identity['nic']}",
        )

    @app.callback(
        Output(OVERVIEW_ID, "children"),
        Output(FILTER_STATUS_ID, "children"),
        Output(NODE_TABLE_ID, "data"),
        Output(NODE_ACTIONS_ID, "children"),
        Output(NODE_GRAPH_ID, "figure"),
        Output(NIC_DETAIL_ID, "children"),
        Output(NIC_FILTER_ID, "options"),
        Output(NIC_FILTER_ID, "value"),
        Input(NODE_FILTER_ID, "value"),
        State(NIC_FILTER_ID, "value"),
        Input(DIRECTION_FILTER_ID, "value"),
        Input(TIME_FILTER_ID, "value"),
        State(TIME_ORIGIN_ID, "data"),
    )
    def update_analysis(
        node_value,
        nic_value,
        direction_value,
        time_value,
        time_origin,
    ):
        current_store = store_provider()
        node_id = _parse_identity_value(node_value, "node")
        nic_id = _parse_identity_value(nic_value, "nic")
        if node_id is not None and not any(
            nic.node_id == node_id and nic.nic_id == nic_id
            for nic in current_store.nics
        ):
            nic_id = None
            selected_nic_value = "all"
        else:
            selected_nic_value = nic_value
        origin = int(time_origin["acquisition_start_ns"])
        window = time_value if time_value else [0, current_time_max()]
        filters = AnalysisFilterState(
            node_id=node_id,
            nic_id=nic_id,
            direction=direction_value,
            start_ns=origin + int(window[0]),
            end_ns=origin + int(window[-1]),
        )
        view = performance_build_view_model(current_store, filters)
        if node_id is None:
            updated_nic_options = [
                {"label": f"All NICs ({len(current_store.nics)})", "value": "all"},
                *[
                    {
                        "label": f"{nic.label} on node {nic.node_id} (ID {nic.nic_id})",
                        "value": f"nic:{nic.nic_id}",
                    }
                    for nic in current_store.nics
                ],
            ]
        else:
            node_nics = [nic for nic in current_store.nics if nic.node_id == node_id]
            updated_nic_options = [
                {"label": f"All NICs ({len(node_nics)})", "value": "all"},
                *[
                    {
                        "label": f"{nic.label} on node {nic.node_id} (ID {nic.nic_id})",
                        "value": f"nic:{nic.nic_id}",
                    }
                    for nic in node_nics
                ],
            ]
        return (
            investigation_overview.render_overview(view),
            investigation_shell.render_filter_status(view),
            investigation_overview.ranked_rows(view),
            no_update,
            investigation_overview.render_timeline_graph(
                view, graph_id=NODE_GRAPH_ID
            ).figure,
            investigation_nic_view.render_nic_detail(
                view, chart_mode_id=investigation_shell.CHART_MODE_ID
            ),
            updated_nic_options,
            selected_nic_value,
        )

    @app.callback(
        Output(OVERVIEW_ID, "children", allow_duplicate=True),
        Output(FILTER_STATUS_ID, "children", allow_duplicate=True),
        Output(NODE_TABLE_ID, "data", allow_duplicate=True),
        Output(NODE_ACTIONS_ID, "children", allow_duplicate=True),
        Output(NODE_GRAPH_ID, "figure", allow_duplicate=True),
        Output(NIC_DETAIL_ID, "children", allow_duplicate=True),
        Input(NIC_FILTER_ID, "value"),
        State(NODE_FILTER_ID, "value"),
        State(DIRECTION_FILTER_ID, "value"),
        State(TIME_FILTER_ID, "value"),
        State(TIME_ORIGIN_ID, "data"),
        prevent_initial_call=True,
    )
    def update_analysis_for_nic(
        nic_value,
        node_value,
        direction_value,
        time_value,
        time_origin,
    ):
        current_store = store_provider()
        origin = int(time_origin["acquisition_start_ns"])
        window = time_value if time_value else [0, current_time_max()]
        filters = AnalysisFilterState(
            node_id=_parse_identity_value(node_value, "node"),
            nic_id=_parse_identity_value(nic_value, "nic"),
            direction=direction_value,
            start_ns=origin + int(window[0]),
            end_ns=origin + int(window[-1]),
        )
        view = performance_build_view_model(current_store, filters)
        return (
            investigation_overview.render_overview(view),
            investigation_shell.render_filter_status(view),
            investigation_overview.ranked_rows(view),
            no_update,
            investigation_overview.render_timeline_graph(
                view, graph_id=NODE_GRAPH_ID
            ).figure,
            investigation_nic_view.render_nic_detail(
                view, chart_mode_id=investigation_shell.CHART_MODE_ID
            ),
        )

    @app.callback(
        Output(EXPORT_DOWNLOAD_ID, "data"),
        Input(EXPORT_BUTTON_ID, "n_clicks"),
        State(NODE_FILTER_ID, "value"),
        State(NIC_FILTER_ID, "value"),
        State(DIRECTION_FILTER_ID, "value"),
        State(TIME_FILTER_ID, "value"),
        State(TIME_ORIGIN_ID, "data"),
        State(PHASE_STORE_ID, "data"),
        prevent_initial_call=True,
    )
    def export_analysis(
        n_clicks,
        node_value,
        nic_value,
        direction_value,
        time_value,
        time_origin,
        phase_store,
    ):
        if not n_clicks:
            return no_update

        node_id = _parse_identity_value(node_value, "node")
        nic_id = _parse_identity_value(nic_value, "nic")
        origin = int(time_origin["acquisition_start_ns"])
        window = time_value if time_value else [0, current_time_max()]
        filters = AnalysisFilterState(
            node_id=node_id,
            nic_id=nic_id,
            direction=direction_value,
            start_ns=origin + int(window[0]),
            end_ns=origin + int(window[-1]),
        )
        resolved_investigation_state = resolve_state(
            investigation_state,
            node_id=node_id,
            nic_id=nic_id,
            direction=direction_value or "all",
            start_ns=filters.start_ns,
            end_ns=filters.end_ns,
            screen="nic" if node_id is not None and nic_id is not None else "overview",
        )
        view = performance_build_view_model(
            store_provider(), filters, phase_windows=phase_store or []
        )
        return investigation_export.build_investigation_export_payload(
            view,
            resolved_investigation_state,
            base_document_builder=build_export_document,
            serialize_document=serialize_export_document,
        )

    investigation_callbacks.register_investigation_callbacks(
        app,
        store_provider=store_provider,
        build_view_model=performance_build_view_model,
        filter_state_factory=AnalysisFilterState,
        initial_state=investigation_state,
        time_max_provider=current_time_max,
        component_ids=component_ids,
    )

    if dataset_controls is not None and dataset_manager is not None:
        dataset_controls.register_callbacks(app, dataset_manager)

        @app.callback(
            Output(NODE_FILTER_ID, "options"),
            Output(NODE_FILTER_ID, "value", allow_duplicate=True),
            Output(NIC_FILTER_ID, "options", allow_duplicate=True),
            Output(NIC_FILTER_ID, "value", allow_duplicate=True),
            Output(DIRECTION_FILTER_ID, "value", allow_duplicate=True),
            Output(TIME_FILTER_ID, "min"),
            Output(TIME_FILTER_ID, "max"),
            Output(TIME_FILTER_ID, "value", allow_duplicate=True),
            Output(TIME_FILTER_ID, "marks"),
            Output(TIME_ORIGIN_ID, "data"),
            Output(investigation_shell.STATE_STORE_ID, "data", allow_duplicate=True),
            Output(investigation_shell.SHELL_FACTS_ID, "children"),
            Output(DATASET_LOADED_STORE_ID, "data"),
            Input(dataset_controls.ready_store_id, "data"),
            Input(dataset_controls.current_dataset_store_id, "data"),
            State(DATASET_LOADED_STORE_ID, "data"),
            prevent_initial_call=True,
        )
        def apply_loaded_dataset(ready, current_dataset_id, loaded_dataset_id):
            if (
                not ready
                or current_dataset_id is None
                or current_dataset_id == loaded_dataset_id
            ):
                raise PreventUpdate

            current_handle = dataset_manager.current()
            if current_handle is None:
                raise PreventUpdate
            if current_handle.descriptor.dataset_id != current_dataset_id:
                raise PreventUpdate

            current_store = current_handle.store
            current_filters = default_filters(current_store)
            current_time_max = max(
                1, current_filters.end_ns - current_filters.start_ns
            )
            current_node_options = [
                {"label": f"All nodes ({len(current_store.nodes)})", "value": "all"},
                *[
                    {
                        "label": f"{node.label} (ID {node.node_id})",
                        "value": f"node:{node.node_id}",
                    }
                    for node in current_store.nodes
                ],
            ]
            current_nic_options = [
                {"label": f"All NICs ({len(current_store.nics)})", "value": "all"},
                *[
                    {
                        "label": f"{nic.label} on node {nic.node_id} (ID {nic.nic_id})",
                        "value": f"nic:{nic.nic_id}",
                    }
                    for nic in current_store.nics
                ],
            ]
            current_view = performance_build_view_model(
                current_store, current_filters
            )
            current_state = initial_state(
                run_id=current_store.identity.run_id,
                source_fingerprint=current_store.identity.source_fingerprint,
                schema_version=current_store.schema_version,
                start_ns=current_filters.start_ns,
                end_ns=current_filters.end_ns,
            )

            return (
                current_node_options,
                "all",
                current_nic_options,
                "all",
                "all",
                0,
                current_time_max,
                [0, current_time_max],
                {0: "start", current_time_max: "end"},
                {"acquisition_start_ns": current_filters.start_ns},
                current_state.to_dict(),
                investigation_shell.render_shell_facts(current_view),
                current_dataset_id,
            )

    for callback_spec in app._callback_list:
        callback_spec["output"] = _CallbackOutputId(callback_spec["output"])
    return app
