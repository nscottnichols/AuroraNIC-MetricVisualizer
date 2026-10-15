import json
import math
from datetime import datetime, timezone

import numpy as np
import pytest
import plotly.graph_objects as go
from dash import Dash, dcc, html
import analysis_app

from analysis_fixture import (
    NIC_A,
    NODE_A,
    NODE_B,
    make_store,
)
from trace_store import TraceStore
from host_pressure_analysis import HostPressureQualityFlag
from operation_analysis import OpcodeMapping, OpcodeMappingEntry
from matching_analysis import MatchingStatus
from frame_analysis import (
    FrameBinEdge,
    FrameBinEdgeSet,
    FrameBytePopulationCompatibility,
    FrameDirection,
    FrameMeanStatus,
    FrameQualityFlag,
    FrameQuantileStatus,
    FrameStatus,
)
from reliability_analysis import ReliabilityQualityFlag, ReliabilityStatus
from translation_analysis import TranslationQualityFlag, TranslationStatus
from latency_analysis import (
    LatencyBinDefinition,
    LatencyBinEdge,
    LatencyBinEncoding,
    LatencyHistogramGroup,
    LatencyQualityFlag,
    LatencyStatus,
)
from trace_model import (
    CollectorIdentity,
    CounterMetadata,
    CounterSemantics,
    Coverage,
    NicIdentity,
    NodeIdentity,
    NormalizedBatch,
    RunIdentity,
    ValueKind,
)
from analysis_app import (
    AnalysisFilterState,
    EXPORT_BUTTON_ID,
    EXPORT_DOWNLOAD_ID,
    MAX_EXPORT_JSON_BYTES,
    MAX_EXPORT_SERIES,
    MAX_DISPLAY_POINTS_PER_SERIES,
    MAX_RELIABILITY_COUNTER_RESULTS,
    MAX_FLOW_CONTROL_COLLECTORS,
    MAX_FLOW_CONTROL_COUNTER_RESULTS,
    MAX_MATCHING_COLLECTORS,
    MAX_MATCHING_COUNTER_RESULTS,
    MAX_HOST_PRESSURE_COLLECTORS,
    MAX_HOST_PRESSURE_PATH_RESULTS,
    MAX_HOST_PRESSURE_STALL_RESULTS,
MAX_OPERATION_COLLECTORS,
MAX_OPERATION_OPCODE_RESULTS,
MAX_OPERATION_TOTAL_RESULTS,
MAX_FRAME_BIN_RESULTS,
MAX_FRAME_COLLECTORS,
MAX_FRAME_QUANTILE_RESULTS,
MAX_TRANSLATION_COLLECTORS,
MAX_TRANSLATION_COUNTER_RESULTS,
MAX_TRANSLATION_VIEW_RESULTS,
MAX_LATENCY_BIN_RESULTS,
MAX_LATENCY_COLLECTORS,
MAX_LATENCY_QUANTILE_RESULTS,
    MAX_PHASE_WINDOWS,
    NODE_FILTER_ID,
    PHASE_ADD_ID,
    PHASE_END_ID,
    PHASE_NAME_ID,
    PHASE_PANEL_ID,
    PHASE_RESULT_ID,
    PHASE_START_ID,
    PHASE_STORE_ID,
    FILTER_RESET_ID,
    _base_view_model,
    build_export_document,
    build_export_payload,
    build_phase_analysis,
    build_view_model,
    clear_view_model_cache,
    create_analysis_app,
    default_filters,
    downsample_series,
    serialize_export_document,
    _render_overview,
    _render_nic_detail,
)

FLOW_CONTROL_RUN = RunIdentity(
    run_id="run-flow-control",
    source_fingerprint="sha256:flow-control",
    source_format="synthetic-ctf",
    reader_version="test-reader",
    conversion_lineage="direct",
    clock_domain="monotonic-test",
    synchronization_evidence="single-clock",
)
FLOW_CONTROL_NODE = NodeIdentity(201, "node-fc", "node-fc")
FLOW_CONTROL_NIC = NicIdentity(210, FLOW_CONTROL_NODE.node_id, "cxi-fc", "cxi-fc")
FLOW_CONTROL_COLLECTOR_A = CollectorIdentity(220, "collector-fc-a", rank=0)
FLOW_CONTROL_COLLECTOR_B = CollectorIdentity(221, "collector-fc-b", rank=1)


def _flow_control_counter(counter_id, name, unit):
    return CounterMetadata(
        counter_id,
        name,
        ValueKind.EXACT_UINT64,
        CounterSemantics.CUMULATIVE,
        unit,
        baseline_relative=False,
        registry_version="test",
    )


FLOW_TX_P0 = _flow_control_counter(230, "hni_tx_paused_0", "cycles")
FLOW_RX_P2 = _flow_control_counter(231, "hni_rx_paused_2", "cycles")
FLOW_TX_P3 = _flow_control_counter(232, "hni_tx_paused_3", "cycles")
FLOW_TX_XOFF_P4 = _flow_control_counter(233, "hni_pause_xoff_sent_4", "events")
FLOW_RX_PAUSE_P5 = _flow_control_counter(234, "hni_pause_recv_5", "events")
FLOW_FGFC_DISCARD = _flow_control_counter(235, "hni_fgfc_discard", "events")
FLOW_TX_INVALID_METADATA = _flow_control_counter(236, "hni_tx_paused_7", "events")


MATCHING_RUN = RunIdentity(
    run_id="run-matching",
    source_fingerprint="sha256:matching",
    source_format="synthetic-ctf",
    reader_version="test-reader",
    conversion_lineage="direct",
    clock_domain="monotonic-test",
    synchronization_evidence="single-clock",
)
MATCHING_NODE = NodeIdentity(301, "node-m", "node-m")
MATCHING_NIC = NicIdentity(310, MATCHING_NODE.node_id, "cxi-m", "cxi-m")
MATCHING_COLLECTOR_A = CollectorIdentity(320, "collector-m-a", rank=0)
MATCHING_COLLECTOR_B = CollectorIdentity(321, "collector-m-b", rank=1)
MATCHING_PRIORITY_0 = _flow_control_counter(
    330, "lpe_net_match_priority_0", "matches"
)
MATCHING_OVERFLOW_0 = _flow_control_counter(
    331, "lpe_net_match_overflow_0", "matches"
)
MATCHING_PRIORITY_1 = _flow_control_counter(
    332, "lpe_net_match_priority_1", "matches"
)
MATCHING_OVERFLOW_1 = _flow_control_counter(
    333, "lpe_net_match_overflow_1", "matches"
)
MATCHING_PRIORITY_2 = _flow_control_counter(
    334, "lpe_net_match_priority_2", "matches"
)
MATCHING_OVERFLOW_2 = _flow_control_counter(
    335, "lpe_net_match_overflow_2", "matches"
)
MATCHING_PRIORITY_3 = _flow_control_counter(
    336, "lpe_net_match_priority_3", "matches"
)
MATCHING_OVERFLOW_3 = _flow_control_counter(
    337, "lpe_net_match_overflow_3", "matches"
)

HOST_RUN = RunIdentity(
    run_id="run-host-pressure",
    source_fingerprint="sha256:host-pressure",
    source_format="synthetic-ctf",
    reader_version="test-reader",
    conversion_lineage="direct",
    clock_domain="monotonic-test",
    synchronization_evidence="single-clock",
)
HOST_NODE = NodeIdentity(401, "node-h", "node-h")
HOST_NIC = NicIdentity(410, HOST_NODE.node_id, "cxi-h", "cxi-h")
HOST_COLLECTOR_A = CollectorIdentity(420, "collector-h-a", rank=0)
HOST_COLLECTOR_B = CollectorIdentity(421, "collector-h-b", rank=1)
HOST_POSTED_PACKET = _flow_control_counter(
    430, "parbs_tarb_pi_posted_pkts", "packets"
)
HOST_POSTED_BLOCKED = _flow_control_counter(
    431, "parbs_tarb_pi_posted_blocked_cnt", "cycles"
)
HOST_NONPOSTED_PACKET = _flow_control_counter(
    432, "parbs_tarb_pi_non_posted_pkts", "packets"
)
HOST_NONPOSTED_BLOCKED = _flow_control_counter(
    433, "parbs_tarb_pi_non_posted_blocked_cnt", "cycles"
)
HOST_DMA_STALL = _flow_control_counter(
    434, "ixe_dmawr_stall_p_cdt", "cycles"
)
HOST_DUMMY = _flow_control_counter(435, "other_counter", "events")

OPERATION_RUN = RunIdentity(
    run_id="run-operations",
    source_fingerprint="sha256:operations",
    source_format="synthetic-ctf",
    reader_version="test-reader",
    conversion_lineage="direct",
    clock_domain="monotonic-test",
    synchronization_evidence="single-clock",
)
OPERATION_NODE = NodeIdentity(501, "node-op", "node-op")
OPERATION_NIC = NicIdentity(510, OPERATION_NODE.node_id, "cxi-op", "cxi-op")
OPERATION_COLLECTOR_A = CollectorIdentity(520, "collector-op-a", rank=0)
OPERATION_COLLECTOR_B = CollectorIdentity(521, "collector-op-b", rank=1)
OPERATION_DUMMY = _flow_control_counter(522, "operation_other_counter", "events")
OPERATION_BIN_0 = _flow_control_counter(
    523, "cq_dma_cmd_counts_0", "commands"
)
OPERATION_BIN_1 = _flow_control_counter(
    524, "cq_dma_cmd_counts_1", "commands"
)
OPERATION_DMA_0 = _flow_control_counter(
    525, "cq_num_dma_cmds_0", "commands"
)
OPERATION_IDC_0 = _flow_control_counter(
    526, "cq_num_idc_cmds_0", "commands"
)
OPERATION_COUNTERS = {
    spec.counter_id: _flow_control_counter(523 + index, spec.counter_id, "commands")
    for index, spec in enumerate(analysis_app._OPERATION_SPECS)
}

FRAME_RUN = RunIdentity(
    run_id="run-frames",
    source_fingerprint="sha256:frames",
    source_format="synthetic-ctf",
    reader_version="test-reader",
    conversion_lineage="direct",
    clock_domain="monotonic-test",
    synchronization_evidence="single-clock",
)
FRAME_NODE = NodeIdentity(601, "node-frame", "node-frame")
FRAME_NIC = NicIdentity(610, FRAME_NODE.node_id, "cxi-frame", "cxi-frame")
FRAME_COLLECTOR_A = CollectorIdentity(620, "collector-frame-a", rank=0)
FRAME_COLLECTOR_B = CollectorIdentity(621, "collector-frame-b", rank=1)
FRAME_COUNTERS = {
    spec.counter_id: _flow_control_counter(
        630 + index, spec.counter_id, spec.unit
    )
    for index, spec in enumerate(analysis_app._FRAME_SPECS)
}
FRAME_DUMMY = _flow_control_counter(700, "frame_other_counter", "events")

TRANSLATION_RUN = RunIdentity(
    run_id="run-translation",
    source_fingerprint="sha256:translation",
    source_format="synthetic-ctf",
    reader_version="test-reader",
    conversion_lineage="direct",
    clock_domain="monotonic-test",
    synchronization_evidence="single-clock",
)
TRANSLATION_NODE = NodeIdentity(701, "node-translation", "node-translation")
TRANSLATION_NIC = NicIdentity(710, TRANSLATION_NODE.node_id, "cxi-translation", "cxi-translation")
TRANSLATION_COLLECTOR_A = CollectorIdentity(720, "collector-translation-a", rank=0)
TRANSLATION_COLLECTOR_B = CollectorIdentity(721, "collector-translation-b", rank=1)
TRANSLATION_TX_BYTES = _flow_control_counter(
    730, "hni_sts_tx_ok_octets", "bytes"
)
TRANSLATION_RX_BYTES = _flow_control_counter(
    731, "hni_sts_rx_ok_octets", "bytes"
)
TRANSLATION_COUNTERS = {
    spec.counter_id: _flow_control_counter(
        740 + index, spec.counter_id, spec.unit
    )
    for index, spec in enumerate(analysis_app._TRANSLATION_SPECS)
}
TRANSLATION_DUMMY = _flow_control_counter(800, "translation_other_counter", "events")
TRANSLATION_VALUES = {
    "atu_client_req_ee": (0, 10, 15),
    "atu_cache_hit_base_page_size_0": (0, 8, 12),
    "atu_cache_miss_0": (0, 2, 3),
    "atu_cache_miss_ee": (0, 4, 5),
    "atu_cache_evictions": (0, 1, 1),
}

LATENCY_RUN = RunIdentity(
    run_id="run-latency",
    source_fingerprint="sha256:latency",
    source_format="synthetic-ctf",
    reader_version="test-reader",
    conversion_lineage="direct",
    clock_domain="monotonic-test",
    synchronization_evidence="single-clock",
)
LATENCY_NODE = NodeIdentity(801, "node-latency", "node-latency")
LATENCY_NIC = NicIdentity(810, LATENCY_NODE.node_id, "cxi-latency", "cxi-latency")
LATENCY_COLLECTOR_A = CollectorIdentity(820, "collector-latency-a", rank=0)
LATENCY_COLLECTOR_B = CollectorIdentity(821, "collector-latency-b", rank=1)
LATENCY_COUNTERS = {
    counter_name: _flow_control_counter(
        830 + index, counter_name, "observations"
    )
    for index, counter_name in enumerate(
        f"pct_req_rsp_latency_{bin_index}" for bin_index in range(32)
    )
}
LATENCY_VALUES = {
    "pct_req_rsp_latency_0": (0, 10, 15),
    "pct_req_rsp_latency_1": (0, 5, 7),
}


def latency_filters():
    return AnalysisFilterState(
        node_id=LATENCY_NODE.node_id,
        nic_id=LATENCY_NIC.nic_id,
        direction="all",
        start_ns=0,
        end_ns=2_000_000_001,
    )


def make_latency_store(
    *,
    duplicate_collectors=False,
    collector_b_overrides=None,
    counter_names=None,
):
    timestamps = (0, 1_000_000_000, 2_000_000_000)
    collectors = [LATENCY_COLLECTOR_A]
    if duplicate_collectors:
        collectors.append(LATENCY_COLLECTOR_B)
    rows = []
    selected_counters = tuple(
        LATENCY_COUNTERS[name] for name in (counter_names or LATENCY_COUNTERS)
    )
    for counter in selected_counters:
        for collector in collectors:
            values = LATENCY_VALUES.get(counter.name, (0, 0, 0))
            if (
                collector is LATENCY_COLLECTOR_B
                and collector_b_overrides
                and counter.name in collector_b_overrides
            ):
                values = collector_b_overrides[counter.name]
            for timestamp, value in zip(timestamps, values, strict=True):
                rows.append((counter, timestamp, value, collector))

    requested_counter_names = tuple(counter.name for counter in selected_counters)
    return TraceStore.from_batches(
        [
            NormalizedBatch(
                identity=LATENCY_RUN,
                nodes=(LATENCY_NODE,),
                nics=(LATENCY_NIC,),
                counters=selected_counters,
                collectors=tuple(collectors),
                coverage=Coverage(
                    requested_counter_names=requested_counter_names,
                    available_counter_names=requested_counter_names,
                    readable_counter_names=requested_counter_names,
                    observed_counter_names=requested_counter_names,
                    acquisition_start_ns=0,
                    acquisition_end_ns=2_000_000_000,
                    lost_record_count=0,
                    gap_count=0,
                ),
                node_ids=np.full(len(rows), LATENCY_NODE.node_id, dtype=np.int64),
                nic_ids=np.full(len(rows), LATENCY_NIC.nic_id, dtype=np.int64),
                counter_ids=np.asarray(
                    [row[0].counter_id for row in rows], dtype=np.int64
                ),
                collector_ids=np.asarray(
                    [row[3].collector_id for row in rows], dtype=np.int64
                ),
                timestamps_ns=np.asarray([row[1] for row in rows], dtype=np.int64),
                record_ids=np.asarray(
                    [f"latency-{index}" for index in range(len(rows))]
                ),
                uint64_values=np.asarray(
                    [row[2] for row in rows], dtype=np.uint64
                ),
                float64_values=np.full(len(rows), np.nan, dtype=np.float64),
                quality_flags=np.zeros(len(rows), dtype=np.int64),
            )
        ]
    )


def translation_filters(direction="all"):
    return AnalysisFilterState(
        node_id=TRANSLATION_NODE.node_id,
        nic_id=TRANSLATION_NIC.nic_id,
        direction=direction,
        start_ns=0,
        end_ns=2_000_000_001,
    )


def make_translation_store(
    *,
    duplicate_collectors=False,
    collector_b_overrides=None,
    collector_a_overrides=None,
):
    timestamps = (0, 1_000_000_000, 2_000_000_000)
    counters = [
        TRANSLATION_DUMMY,
        TRANSLATION_TX_BYTES,
        TRANSLATION_RX_BYTES,
        *TRANSLATION_COUNTERS.values(),
    ]
    collectors = [TRANSLATION_COLLECTOR_A]
    if duplicate_collectors:
        collectors.append(TRANSLATION_COLLECTOR_B)
    rows = []
    for counter in counters:
        for collector in collectors:
            if counter is TRANSLATION_DUMMY:
                values = (0, 1, 2)
            elif counter is TRANSLATION_TX_BYTES:
                values = (0, 5, 10)
            elif counter is TRANSLATION_RX_BYTES:
                values = (0, 5, 10)
            else:
                values = TRANSLATION_VALUES.get(counter.name, (0, 0, 0))
            if (
                collector is TRANSLATION_COLLECTOR_A
                and collector_a_overrides
                and counter.name in collector_a_overrides
            ):
                values = collector_a_overrides[counter.name]
            if (
                collector is TRANSLATION_COLLECTOR_B
                and collector_b_overrides
                and counter.name in collector_b_overrides
            ):
                values = collector_b_overrides[counter.name]
            for timestamp, value in zip(timestamps, values, strict=True):
                rows.append((counter, timestamp, value, collector))

    requested = tuple(spec.counter_id for spec in analysis_app._TRANSLATION_SPECS)
    available = tuple(counter.name for counter in counters)
    return TraceStore.from_batches(
        [
            NormalizedBatch(
                identity=TRANSLATION_RUN,
                nodes=(TRANSLATION_NODE,),
                nics=(TRANSLATION_NIC,),
                counters=tuple(counters),
                collectors=tuple(collectors),
                coverage=Coverage(
                    requested_counter_names=requested,
                    available_counter_names=available,
                    readable_counter_names=available,
                    observed_counter_names=available,
                    acquisition_start_ns=0,
                    acquisition_end_ns=2_000_000_000,
                    lost_record_count=0,
                    gap_count=0,
                ),
                node_ids=np.full(
                    len(rows), TRANSLATION_NODE.node_id, dtype=np.int64
                ),
                nic_ids=np.full(
                    len(rows), TRANSLATION_NIC.nic_id, dtype=np.int64
                ),
                counter_ids=np.asarray(
                    [row[0].counter_id for row in rows], dtype=np.int64
                ),
                collector_ids=np.asarray(
                    [row[3].collector_id for row in rows], dtype=np.int64
                ),
                timestamps_ns=np.asarray(
                    [row[1] for row in rows], dtype=np.int64
                ),
                record_ids=np.asarray(
                    [f"translation-{index}" for index in range(len(rows))]
                ),
                uint64_values=np.asarray(
                    [row[2] for row in rows], dtype=np.uint64
                ),
                float64_values=np.full(
                    len(rows), np.nan, dtype=np.float64
                ),
                quality_flags=np.zeros(len(rows), dtype=np.int64),
            )
        ]
    )
FRAME_BIN_IDS = (
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


def frame_edges(direction, open_ended=False):
    lower_upper = {
        "27": (0, 27),
        "35": (27, 35),
        "36_to_63": (35, 64),
        "64": (64, 65),
        "65_to_127": (65, 128),
        "128_to_255": (128, 256),
        "256_to_511": (256, 512),
        "512_to_1023": (512, 1024),
        "1024_to_2047": (1024, 2048),
        "2048_to_4095": (2048, 4096),
        "4096_to_8191": (4096, 8192),
        "8192_to_max": (8192, 8193),
    }
    return FrameBinEdgeSet(
        direction=direction,
        source="fixture sourced NIC-06 histogram edges",
        edges=tuple(
            FrameBinEdge(
                bin_id=bin_id,
                lower_bytes=lower,
                upper_bytes=None if open_ended and bin_id == "8192_to_max" else upper,
                open_ended=open_ended and bin_id == "8192_to_max",
            )
            for bin_id, (lower, upper) in lower_upper.items()
        ),
    )


def frame_compatibility(direction, compatible=True):
    direction_name = direction.value
    return FrameBytePopulationCompatibility(
        direction=direction,
        byte_counter_id=f"hni_sts_{direction_name}_ok_octets",
        frame_population_id=f"nic06-{direction_name}-wire-frames-v1",
        byte_population_id=f"nic06-{direction_name}-wire-bytes-v1",
        compatible=compatible,
        definition_evidence=(
            "fixture: exact same directional wire population"
            if compatible
            else "fixture: byte and frame counters count different populations"
        ),
    )


def frame_filters(direction):
    return AnalysisFilterState(
        node_id=FRAME_NODE.node_id,
        nic_id=FRAME_NIC.nic_id,
        direction=direction,
        start_ns=0,
        end_ns=2_000_000_001,
    )


def make_frame_store(
    *,
    counter_names=tuple(FRAME_COUNTERS),
    duplicate_collectors=False,
    rx_values=(0, 1, 3),
    collector_b_overrides=None,
):
    timestamps = (0, 1_000_000_000, 2_000_000_000)
    counters = [FRAME_DUMMY, *(FRAME_COUNTERS[name] for name in counter_names)]
    collectors = [FRAME_COLLECTOR_A]
    if duplicate_collectors:
        collectors.append(FRAME_COLLECTOR_B)
    rows = []
    for counter in counters:
        for collector in collectors:
            if counter is FRAME_DUMMY:
                values = (0, 1, 2)
            elif counter.name.startswith("hni_tx_ok_"):
                values = (0, 1, 3)
            elif counter.name.startswith("hni_rx_ok_"):
                values = rx_values
            elif counter.name == "hni_sts_tx_ok_octets":
                values = (0, 4, 16)
            else:
                values = (0, 0, 0)
            if (
                collector is FRAME_COLLECTOR_B
                and collector_b_overrides
                and counter.name in collector_b_overrides
            ):
                values = collector_b_overrides[counter.name]
            for timestamp, value in zip(timestamps, values, strict=True):
                rows.append((counter, timestamp, value, collector))

    requested = tuple(spec.counter_id for spec in analysis_app._FRAME_SPECS)
    available = tuple(counter.name for counter in counters)
    return TraceStore.from_batches(
        [
            NormalizedBatch(
                identity=FRAME_RUN,
                nodes=(FRAME_NODE,),
                nics=(FRAME_NIC,),
                counters=tuple(counters),
                collectors=tuple(collectors),
                coverage=Coverage(
                    requested_counter_names=requested,
                    available_counter_names=available,
                    readable_counter_names=available,
                    observed_counter_names=available,
                    acquisition_start_ns=0,
                    acquisition_end_ns=2_000_000_000,
                    lost_record_count=0,
                    gap_count=0,
                ),
                node_ids=np.full(len(rows), FRAME_NODE.node_id, dtype=np.int64),
                nic_ids=np.full(len(rows), FRAME_NIC.nic_id, dtype=np.int64),
                counter_ids=np.asarray(
                    [row[0].counter_id for row in rows], dtype=np.int64
                ),
                collector_ids=np.asarray(
                    [row[3].collector_id for row in rows], dtype=np.int64
                ),
                timestamps_ns=np.asarray([row[1] for row in rows], dtype=np.int64),
                record_ids=np.asarray(
                    [f"frames-{index}" for index in range(len(rows))]
                ),
                uint64_values=np.asarray([row[2] for row in rows], dtype=np.uint64),
                float64_values=np.full(len(rows), np.nan, dtype=np.float64),
                quality_flags=np.zeros(len(rows), dtype=np.int64),
            )
        ]
    )


def make_operation_store(
    *,
    counter_names=tuple(OPERATION_COUNTERS),
    bin_0_values=(0, 2, 4),
    bin_1_values=(0, 1, 2),
    dma_0_values=(0, 3, 5),
    idc_0_values=(0, 4, 8),
    duplicate_collectors=False,
    collector_b_timestamps_by_counter=None,
    collector_b_overrides=None,
):
    timestamps = (0, 1_000_000_000, 2_000_000_000)
    values_by_counter = {name: (0, 0, 0) for name in OPERATION_COUNTERS}
    values_by_counter.update(
        {
            OPERATION_BIN_0.name: bin_0_values,
            OPERATION_BIN_1.name: bin_1_values,
            OPERATION_DMA_0.name: dma_0_values,
            OPERATION_IDC_0.name: idc_0_values,
        }
    )
    counters = [OPERATION_DUMMY, *(OPERATION_COUNTERS[name] for name in counter_names)]
    collectors = [OPERATION_COLLECTOR_A]
    if duplicate_collectors:
        collectors.append(OPERATION_COLLECTOR_B)
    rows = []
    for counter in counters:
        for collector in collectors:
            values = values_by_counter.get(counter.name, (0, 1, 2))
            series_timestamps = timestamps
            if collector is OPERATION_COLLECTOR_B:
                if collector_b_timestamps_by_counter is not None:
                    series_timestamps = collector_b_timestamps_by_counter.get(
                        counter.name, timestamps
                    )
                if collector_b_overrides and counter.name in collector_b_overrides:
                    values = collector_b_overrides[counter.name]
            for timestamp, value in zip(series_timestamps, values, strict=True):
                rows.append((counter, timestamp, value, collector))

    return TraceStore.from_batches(
        [
            NormalizedBatch(
                identity=OPERATION_RUN,
                nodes=(OPERATION_NODE,),
                nics=(OPERATION_NIC,),
                counters=tuple(dict.fromkeys(counters)),
                collectors=tuple(collectors),
                coverage=Coverage(
                    requested_counter_names=tuple(
                        spec.counter_id for spec in analysis_app._OPERATION_SPECS
                    ),
                    available_counter_names=tuple(counter.name for counter in counters),
                    readable_counter_names=tuple(counter.name for counter in counters),
                    observed_counter_names=tuple(counter.name for counter in counters),
                    acquisition_start_ns=0,
                    acquisition_end_ns=2_000_000_000,
                    lost_record_count=0,
                    gap_count=0,
                ),
                node_ids=np.full(len(rows), OPERATION_NODE.node_id, dtype=np.int64),
                nic_ids=np.full(len(rows), OPERATION_NIC.nic_id, dtype=np.int64),
                counter_ids=np.asarray(
                    [row[0].counter_id for row in rows], dtype=np.int64
                ),
                collector_ids=np.asarray(
                    [row[3].collector_id for row in rows], dtype=np.int64
                ),
                timestamps_ns=np.asarray([row[1] for row in rows], dtype=np.int64),
                record_ids=np.asarray(
                    [f"operation-{index}" for index in range(len(rows))]
                ),
                uint64_values=np.asarray([row[2] for row in rows], dtype=np.uint64),
                float64_values=np.full(len(rows), np.nan, dtype=np.float64),
                quality_flags=np.zeros(len(rows), dtype=np.int64),
            )
        ]
    )


def make_host_pressure_store(
    *,
    include_host=True,
    posted_packets=(0, 10, 22),
    posted_blocked=(0, 2, 5),
    nonposted_packets=(0, 3, 7),
    nonposted_blocked=(0, 1, 2),
    stall=(0, 4, 9),
    packet_unit="packets",
    duplicate_collectors=False,
    collector_b_timestamps=None,
    collector_b_timestamps_by_counter=None,
    collector_b_overrides=None,
):
    timestamps = (0, 1_000_000_000, 2_000_000_000)
    counters = [HOST_DUMMY]
    values_by_counter = {HOST_DUMMY.name: (0, 1, 2)}
    if include_host:
        posted_packet = CounterMetadata(
            HOST_POSTED_PACKET.counter_id,
            HOST_POSTED_PACKET.name,
            ValueKind.EXACT_UINT64,
            CounterSemantics.CUMULATIVE,
            packet_unit,
            baseline_relative=False,
            registry_version="test",
        )
        counters.extend(
            (
                posted_packet,
                HOST_POSTED_BLOCKED,
                HOST_NONPOSTED_PACKET,
                HOST_NONPOSTED_BLOCKED,
                HOST_DMA_STALL,
            )
        )
        values_by_counter.update(
            {
                posted_packet.name: posted_packets,
                HOST_POSTED_BLOCKED.name: posted_blocked,
                HOST_NONPOSTED_PACKET.name: nonposted_packets,
                HOST_NONPOSTED_BLOCKED.name: nonposted_blocked,
                HOST_DMA_STALL.name: stall,
            }
        )

    collectors = [HOST_COLLECTOR_A]
    if duplicate_collectors:
        collectors.append(HOST_COLLECTOR_B)
    rows = []
    for counter in dict.fromkeys(counters):
        for collector in collectors:
            values = values_by_counter[counter.name]
            series_timestamps = timestamps
            if collector is HOST_COLLECTOR_B:
                if collector_b_timestamps_by_counter is not None:
                    series_timestamps = collector_b_timestamps_by_counter.get(
                        counter.name, timestamps
                    )
                elif collector_b_timestamps is not None:
                    series_timestamps = collector_b_timestamps
            if (
                collector is HOST_COLLECTOR_B
                and collector_b_overrides
                and counter.name in collector_b_overrides
            ):
                values = collector_b_overrides[counter.name]
            for timestamp, value in zip(series_timestamps, values, strict=True):
                rows.append((counter, timestamp, value, collector))

    return TraceStore.from_batches(
        [
            NormalizedBatch(
                identity=HOST_RUN,
                nodes=(HOST_NODE,),
                nics=(HOST_NIC,),
                counters=tuple(dict.fromkeys(counters)),
                collectors=tuple(collectors),
                coverage=Coverage(
                    requested_counter_names=tuple(counter.name for counter in counters),
                    available_counter_names=tuple(counter.name for counter in counters),
                    readable_counter_names=tuple(counter.name for counter in counters),
                    observed_counter_names=tuple(counter.name for counter in counters),
                    acquisition_start_ns=0,
                    acquisition_end_ns=2_000_000_000,
                    lost_record_count=0,
                    gap_count=0,
                ),
                node_ids=np.full(len(rows), HOST_NODE.node_id, dtype=np.int64),
                nic_ids=np.full(len(rows), HOST_NIC.nic_id, dtype=np.int64),
                counter_ids=np.asarray(
                    [row[0].counter_id for row in rows], dtype=np.int64
                ),
                collector_ids=np.asarray(
                    [row[3].collector_id for row in rows], dtype=np.int64
                ),
                timestamps_ns=np.asarray([row[1] for row in rows], dtype=np.int64),
                record_ids=np.asarray(
                    [f"host-pressure-{index}" for index in range(len(rows))]
                ),
                uint64_values=np.asarray([row[2] for row in rows], dtype=np.uint64),
                float64_values=np.full(len(rows), np.nan, dtype=np.float64),
                quality_flags=np.zeros(len(rows), dtype=np.int64),
            )
        ]
    )


def make_matching_store(
    *,
    priority_0_values=(0, 3, 5),
    overflow_0_values=(0, 2, 4),
    priority_1_values=(0, 1, 3),
    overflow_1_values=(0, 2, 3),
    include_all_counters=True,
    duplicate_collectors=False,
    collector_b_overrides=None,
    collector_b_timestamps=None,
    collector_b_timestamps_by_counter=None,
):
    timestamps = (0, 1_000_000_000, 2_000_000_000)
    all_counters = (
        MATCHING_PRIORITY_0,
        MATCHING_OVERFLOW_0,
        MATCHING_PRIORITY_1,
        MATCHING_OVERFLOW_1,
        MATCHING_PRIORITY_2,
        MATCHING_OVERFLOW_2,
        MATCHING_PRIORITY_3,
        MATCHING_OVERFLOW_3,
    )
    counters = all_counters if include_all_counters else all_counters[:4]
    values_by_counter = {
        MATCHING_PRIORITY_0.name: priority_0_values,
        MATCHING_OVERFLOW_0.name: overflow_0_values,
        MATCHING_PRIORITY_1.name: priority_1_values,
        MATCHING_OVERFLOW_1.name: overflow_1_values,
        MATCHING_PRIORITY_2.name: (0, 0, 0),
        MATCHING_OVERFLOW_2.name: (0, 0, 0),
        MATCHING_PRIORITY_3.name: (0, 0, 0),
        MATCHING_OVERFLOW_3.name: (0, 0, 0),
    }
    collectors = [MATCHING_COLLECTOR_A]
    if duplicate_collectors:
        collectors.append(MATCHING_COLLECTOR_B)
    rows = []
    for counter in counters:
        for collector in collectors:
            values = values_by_counter[counter.name]
            series_timestamps = timestamps
            if collector is MATCHING_COLLECTOR_B:
                if collector_b_timestamps_by_counter is not None:
                    series_timestamps = collector_b_timestamps_by_counter.get(
                        counter.name, timestamps
                    )
                elif collector_b_timestamps is not None:
                    series_timestamps = collector_b_timestamps
                if collector_b_overrides and counter.name in collector_b_overrides:
                    values = collector_b_overrides[counter.name]
            for timestamp, value in zip(series_timestamps, values, strict=True):
                rows.append((counter, timestamp, value, collector))

    requested = tuple(
        f"lpe_net_match_{role}_{priority}"
        for role in ("priority", "overflow")
        for priority in range(4)
    )
    available = tuple(counter.name for counter in counters)
    return TraceStore.from_batches(
        [
            NormalizedBatch(
                identity=MATCHING_RUN,
                nodes=(MATCHING_NODE,),
                nics=(MATCHING_NIC,),
                counters=counters,
                collectors=tuple(collectors),
                coverage=Coverage(
                    requested_counter_names=requested,
                    available_counter_names=available,
                    readable_counter_names=available,
                    observed_counter_names=available,
                    acquisition_start_ns=0,
                    acquisition_end_ns=2_000_000_000,
                    lost_record_count=0,
                    gap_count=0,
                ),
                node_ids=np.full(len(rows), MATCHING_NODE.node_id, dtype=np.int64),
                nic_ids=np.full(len(rows), MATCHING_NIC.nic_id, dtype=np.int64),
                counter_ids=np.asarray(
                    [row[0].counter_id for row in rows], dtype=np.int64
                ),
                collector_ids=np.asarray(
                    [row[3].collector_id for row in rows], dtype=np.int64
                ),
                timestamps_ns=np.asarray([row[1] for row in rows], dtype=np.int64),
                record_ids=np.asarray(
                    [f"matching-{index}" for index in range(len(rows))]
                ),
                uint64_values=np.asarray([row[2] for row in rows], dtype=np.uint64),
                float64_values=np.full(len(rows), np.nan, dtype=np.float64),
                quality_flags=np.zeros(len(rows), dtype=np.int64),
            )
        ]
    )


def make_flow_control_store(
    *,
    tx_p0_values=(0, 0, 0),
    rx_p2_values=(0, 5, 7),
    tx_p3_values=(0, 10, 2),
    xoff_p4_values=(0, 3, 5),
    rx_pause_p5_values=(0, 2, 4),
    fgfc_values=(0, 1, 1),
    invalid_metadata=False,
    duplicate_collectors=False,
    collector_b_overrides=None,
):
    timestamps = (0, 1_000_000_000, 2_000_000_000)
    counters = [
        FLOW_TX_P0,
        FLOW_RX_P2,
        FLOW_TX_P3,
        FLOW_TX_XOFF_P4,
        FLOW_RX_PAUSE_P5,
        FLOW_FGFC_DISCARD,
    ]
    values_by_counter = {
        FLOW_TX_P0.name: tx_p0_values,
        FLOW_RX_P2.name: rx_p2_values,
        FLOW_TX_P3.name: tx_p3_values,
        FLOW_TX_XOFF_P4.name: xoff_p4_values,
        FLOW_RX_PAUSE_P5.name: rx_pause_p5_values,
        FLOW_FGFC_DISCARD.name: fgfc_values,
    }
    if invalid_metadata:
        counters.append(FLOW_TX_INVALID_METADATA)
        values_by_counter[FLOW_TX_INVALID_METADATA.name] = (0, 4, 8)

    collectors = [FLOW_CONTROL_COLLECTOR_A]
    if duplicate_collectors:
        collectors.append(FLOW_CONTROL_COLLECTOR_B)
    rows = []
    for counter in counters:
        for collector in collectors:
            values = values_by_counter[counter.name]
            if (
                collector_b_overrides
                and collector is FLOW_CONTROL_COLLECTOR_B
                and counter.name in collector_b_overrides
            ):
                values = collector_b_overrides[counter.name]
            for timestamp, value in zip(timestamps, values, strict=True):
                rows.append((counter, timestamp, value, collector))

    requested = tuple(counter.name for counter in counters)
    return TraceStore.from_batches(
        [
            NormalizedBatch(
                identity=FLOW_CONTROL_RUN,
                nodes=(FLOW_CONTROL_NODE,),
                nics=(FLOW_CONTROL_NIC,),
                counters=tuple(counters),
                collectors=tuple(collectors),
                coverage=Coverage(
                    requested_counter_names=requested,
                    available_counter_names=requested,
                    readable_counter_names=requested,
                    observed_counter_names=requested,
                    acquisition_start_ns=0,
                    acquisition_end_ns=2_000_000_000,
                    lost_record_count=0,
                    gap_count=0,
                ),
                node_ids=np.full(len(rows), FLOW_CONTROL_NODE.node_id, dtype=np.int64),
                nic_ids=np.full(len(rows), FLOW_CONTROL_NIC.nic_id, dtype=np.int64),
                counter_ids=np.asarray(
                    [row[0].counter_id for row in rows], dtype=np.int64
                ),
                collector_ids=np.asarray(
                    [row[3].collector_id for row in rows], dtype=np.int64
                ),
                timestamps_ns=np.asarray([row[1] for row in rows], dtype=np.int64),
                record_ids=np.asarray(
                    [f"flow-control-{index}" for index in range(len(rows))]
                ),
                uint64_values=np.asarray([row[2] for row in rows], dtype=np.uint64),
                float64_values=np.full(len(rows), np.nan, dtype=np.float64),
                quality_flags=np.zeros(len(rows), dtype=np.int64),
            )
        ]
    )


RELIABILITY_RUN = RunIdentity(
    run_id="run-reliability",
    source_fingerprint="sha256:reliability",
    source_format="synthetic-ctf",
    reader_version="test-reader",
    conversion_lineage="direct",
    clock_domain="monotonic-test",
    synchronization_evidence="single-clock",
)
RELIABILITY_NODE = NodeIdentity(101, "node-r", "node-r")
RELIABILITY_NIC = NicIdentity(110, RELIABILITY_NODE.node_id, "cxi-r", "cxi-r")
RELIABILITY_COLLECTOR_A = CollectorIdentity(120, "collector-a", rank=0)
RELIABILITY_COLLECTOR_B = CollectorIdentity(121, "collector-b", rank=1)
RELIABILITY_ERROR = CounterMetadata(
    130,
    "pct_sct_timeouts",
    ValueKind.EXACT_UINT64,
    CounterSemantics.CUMULATIVE,
    "events",
    baseline_relative=False,
    registry_version="test",
)
RELIABILITY_GOOD = CounterMetadata(
    131,
    "hni_pcs_good_cw",
    ValueKind.EXACT_UINT64,
    CounterSemantics.CUMULATIVE,
    "codewords",
    baseline_relative=False,
    registry_version="test",
)
RELIABILITY_RESET = CounterMetadata(
    132,
    "hni_llr_rx_replay_event",
    ValueKind.EXACT_UINT64,
    CounterSemantics.CUMULATIVE,
    "events",
    baseline_relative=False,
    registry_version="test",
)


def make_reliability_store(
    *,
    error_values=(0, 2, 5),
    good_values=(0, 0, 0),
    include_reset=False,
    duplicate_collectors=False,
    collector_b_error_values=None,
):
    timestamps = (0, 1_000_000_000, 2_000_000_000)
    counters = [RELIABILITY_ERROR, RELIABILITY_GOOD]
    if include_reset:
        counters.append(RELIABILITY_RESET)
    values_by_counter = {
        RELIABILITY_ERROR.name: error_values,
        RELIABILITY_GOOD.name: good_values,
    }
    if include_reset:
        values_by_counter[RELIABILITY_RESET.name] = (0, 3, 1)

    collectors = [RELIABILITY_COLLECTOR_A]
    if duplicate_collectors:
        collectors.append(RELIABILITY_COLLECTOR_B)
    rows = []
    for counter in counters:
        for collector in collectors:
            values = (
                collector_b_error_values
                if collector is RELIABILITY_COLLECTOR_B
                and counter is RELIABILITY_ERROR
                and collector_b_error_values is not None
                else values_by_counter[counter.name]
            )
            for timestamp, value in zip(timestamps, values, strict=True):
                rows.append((counter, timestamp, value, collector))

    return TraceStore.from_batches(
        [
            NormalizedBatch(
                identity=RELIABILITY_RUN,
                nodes=(RELIABILITY_NODE,),
                nics=(RELIABILITY_NIC,),
                counters=tuple(counters),
                collectors=tuple(collectors),
                coverage=Coverage(
                    requested_counter_names=(
                        RELIABILITY_ERROR.name,
                        RELIABILITY_GOOD.name,
                        "pct_spt_timeouts",
                    ),
                    available_counter_names=tuple(counter.name for counter in counters),
                    readable_counter_names=tuple(counter.name for counter in counters),
                    observed_counter_names=tuple(counter.name for counter in counters),
                    acquisition_start_ns=0,
                    acquisition_end_ns=2_000_000_000,
                    lost_record_count=0,
                    gap_count=0,
                ),
                node_ids=np.full(len(rows), RELIABILITY_NODE.node_id, dtype=np.int64),
                nic_ids=np.full(len(rows), RELIABILITY_NIC.nic_id, dtype=np.int64),
                counter_ids=np.asarray([row[0].counter_id for row in rows], dtype=np.int64),
                collector_ids=np.asarray(
                    [row[3].collector_id for row in rows], dtype=np.int64
                ),
                timestamps_ns=np.asarray([row[1] for row in rows], dtype=np.int64),
                record_ids=np.asarray(
                    [f"reliability-{index}" for index in range(len(rows))]
                ),
                uint64_values=np.asarray([row[2] for row in rows], dtype=np.uint64),
                float64_values=np.full(len(rows), np.nan, dtype=np.float64),
                quality_flags=np.zeros(len(rows), dtype=np.int64),
            )
        ]
    )
def component_ids(component):
    ids = set()
    try:
        if getattr(component, "id", None) is not None:
            ids.add(component.id)
    except TypeError:
        pass
    children = getattr(component, "children", None)
    if children is None:
        children = ()
    elif not isinstance(children, (list, tuple)):
        children = (children,)
    for child in children:
        if isinstance(child, (list, tuple)):
            for nested in child:
                ids.update(component_ids(nested))
        else:
            ids.update(component_ids(child))
    return ids


def find_component(component, component_id):
    if getattr(component, "id", None) == component_id:
        return component
    children = getattr(component, "children", None)
    if children is None:
        return None
    if not isinstance(children, (list, tuple)):
        children = (children,)
    for child in children:
        if isinstance(child, (list, tuple)):
            for nested in child:
                found = find_component(nested, component_id)
                if found is not None:
                    return found
        else:
            found = find_component(child, component_id)
            if found is not None:
                return found
    return None


def recursively_collect_values(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from recursively_collect_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursively_collect_values(child)
    else:
        yield value


def test_app_layout_and_filter_state_structure():
    store = make_store()
    app = create_analysis_app(store)
    ids = component_ids(app.layout)

    assert isinstance(app, Dash)
    assert app.title == "Aurora NIC Metrics"
    assert {
        "analysis-node-filter",
        "analysis-nic-filter",
        "analysis-direction-filter",
        "analysis-time-filter",
        "analysis-overview",
        "analysis-node-table",
        "analysis-node-graph",
        "analysis-nic-detail",
        "analysis-question-table",
        "analysis-counter-table",
    }.issubset(ids)

    filters = default_filters(store)
    assert filters.node_id is None
    assert filters.nic_id is None
    assert filters.direction == "all"
    assert filters.start_ns == 0
    assert filters.end_ns == 59_900_000_001


def test_shared_filters_have_accessible_one_action_reset():
    store = make_store()
    app = create_analysis_app(store)
    ids = component_ids(app.layout)

    assert FILTER_RESET_ID in ids
    reset_button = find_component(app.layout, FILTER_RESET_ID)
    assert isinstance(reset_button, html.Button)
    assert reset_button.type == "button"
    assert reset_button.to_plotly_json()["props"]["aria-label"] == (
        "Reset node, NIC, direction, and time filters"
    )

    callback = next(
        callback
        for key, callback in app.callback_map.items()
        if f"{NODE_FILTER_ID}.value" in key
    )
    assert callback["callback"].__wrapped__(1) == (
        "all",
        "all",
        "all",
        [0, 59_900_000_001],
    )


def test_filter_state_selects_node_direction_and_time_window():
    store = make_store()
    filters = AnalysisFilterState(
        node_id=NODE_B.node_id,
        direction="tx",
        start_ns=100_000_000,
        end_ns=200_000_001,
    )
    view = build_view_model(store, filters)

    assert view["filters"] == {
        "node_id": NODE_B.node_id,
        "nic_id": None,
        "direction": "tx",
        "start_ns": 100_000_000,
        "end_ns": 200_000_001,
    }
    assert view["capture"]["node_count"] == 1
    assert view["capture"]["nic_count"] == 1
    assert {record["direction"] for record in view["series_summaries"]} == {"TX"}
    assert {record["node"] for record in view["series_summaries"]} == {"node-b"}
    assert view["overview"]["tx_total_bytes"] == 1_000
    assert view["overview"]["rx_total_bytes"] is None
    row = view["node_comparison"]["rows"][0]
    assert row["node"] == "node-b"
    assert row["nic_count"] == 1
    assert row["observed_nic_count"] == 1
    assert row["missing_nic_count"] == 0
    assert row["effective_nic_count"] == 1.0
    assert row["active_nic_fraction"] == 1.0
    assert row["max_min_ratio"] == 1.0
    assert row["spatial_p50_bytes"] == row["spatial_p99_bytes"] == 1_000
    assert view["node_balance"]["effective_nic_count_formula"].startswith("1 / sum")


def test_job_summary_uses_node_balance_values_and_limits():
    view = build_view_model(
        make_store(sample_count=600),
        AnalysisFilterState(direction="all", end_ns=59_900_000_001),
    )
    directions = view["job_balance"]["directions"]

    assert [record["direction"] for record in directions] == ["TX", "RX"]
    tx = directions[0]
    rx = directions[1]
    assert tx["state"] == "partial"
    assert tx["status"] == "Partial Data"
    assert "partial_coverage" in tx["state_flags"]
    assert tx["job_total_bytes"] == 1_198_000
    assert tx["effective_node_count"] == 2.0
    assert tx["max_traffic_bytes"] == tx["min_traffic_bytes"] == 599_000
    assert tx["max_min_ratio"] == 1.0
    assert tx["spatial_p99_bytes"] == 599_000
    assert tx["hotspot_node_count"] == 1
    assert tx["hotspot_concentration"] == 0.5
    assert tx["topology_state"] == "absent"
    assert tx["topology_supported"] is False
    assert tx["peer_near_far_supported"] is False
    assert "topology" in tx["unsupported_dimensions"]
    assert {"peer", "near_far"}.issubset(tx["unsupported_dimensions"])
    assert rx["state"] == "zero"
    assert rx["job_total_bytes"] == 0
    assert rx["effective_node_count"] is None


def test_manual_phase_analysis_uses_full_detail_and_bounded_summary(monkeypatch):
    import analysis_app

    original = analysis_app.analyze_traffic_series
    include_intervals = []

    def analyze(series, **kwargs):
        include_intervals.append(kwargs.get("include_intervals"))
        return original(series, **kwargs)

    monkeypatch.setattr(analysis_app, "analyze_traffic_series", analyze)
    store = make_store(sample_count=600)
    filters = AnalysisFilterState(nic_id=NIC_A.nic_id, direction="tx")
    phases = [
        {"name": "first", "start_ns": 0, "end_ns": 29_900_000_000},
        {"name": "second", "start_ns": 29_900_000_000, "end_ns": 59_900_000_000},
    ]
    result = build_phase_analysis(store, filters, phases)

    assert include_intervals == [True]
    assert result["state"] == "ready"
    assert result["status"] == "Ok"
    assert result["node_id"] == "node-a"
    assert result["nic_id"] == "cxi-0"
    assert result["direction"] == "TX"
    assert result["clock_id"] == "monotonic-test"
    assert [(phase["name"], phase["total_bytes"]) for phase in result["phases"]] == [
        ("first", 299_000),
        ("second", 300_000),
    ]
    assert all(phase["coverage_fraction"] == 1.0 for phase in result["phases"])
    assert result["boundary_excluded_interval_count"] == 0
    assert result["inter_phase_gap_duration_ns"] == 0
    assert result["changes"][0]["total_change_bytes"] == 1_000
    assert result["changes"][0]["comparison_status"] == "comparable"
    assert result["changes"][0]["is_comparable"] is True
    assert "user-provided" in result["interpretation_limits"]


def test_manual_phase_zero_and_missing_states_remain_distinct():
    zero_store = make_store(include_rx=True, sample_count=600)
    zero_result = build_phase_analysis(
        zero_store,
        AnalysisFilterState(nic_id=NIC_A.nic_id, direction="rx"),
        [{"name": "idle", "start_ns": 0, "end_ns": 29_900_000_000}],
    )
    assert zero_result["state"] == "zero"
    assert zero_result["zero_phase_names"] == ("idle",)
    assert zero_result["phases"][0]["total_bytes"] == 0
    assert zero_result["phases"][0]["reset_observed"] is False

    missing_store = make_store(include_rx=False, sample_count=600)
    missing_result = build_phase_analysis(
        missing_store,
        AnalysisFilterState(nic_id=NIC_A.nic_id, direction="rx"),
        [{"name": "missing", "start_ns": 0, "end_ns": 29_900_000_000}],
    )
    assert missing_result["state"] == "missing"
    assert missing_result["status"] == "Missing phase data"
    assert "counter is not present" in " ".join(missing_result["status_reasons"]).lower()


def test_manual_phase_rejects_overlap_and_gap():
    filters = AnalysisFilterState(nic_id=NIC_A.nic_id, direction="tx")
    overlap = build_phase_analysis(
        make_store(sample_count=12),
        filters,
        [
            {"name": "first", "start_ns": 0, "end_ns": 600_000_000},
            {"name": "second", "start_ns": 500_000_000, "end_ns": 1_000_000_000},
        ],
    )
    gap = build_phase_analysis(
        make_store(sample_count=12),
        filters,
        [
            {"name": "first", "start_ns": 0, "end_ns": 400_000_000},
            {"name": "second", "start_ns": 500_000_000, "end_ns": 1_000_000_000},
        ],
    )

    assert overlap["state"] == "invalid"
    assert overlap["status"] == "Invalid phase windows"
    assert "overlap" in " ".join(overlap["status_reasons"])
    assert gap["state"] == "invalid"
    assert "gap" in " ".join(gap["status_reasons"])
    assert gap["phases"] == []


def test_nic_display_payload_is_bounded_and_summaries_use_full_resolution():
    store = make_store(sample_count=600)
    view = build_view_model(store, AnalysisFilterState(nic_id=NIC_A.nic_id, direction="all"))

    series = view["nic_detail"]["series"]
    assert len(series) == 2
    assert all(record["point_count"] == 600 for record in series)
    assert all(
        record["display_point_count"] == MAX_DISPLAY_POINTS_PER_SERIES
        for record in series
    )
    assert all(
        len(record["display_timestamps_ns"]) == MAX_DISPLAY_POINTS_PER_SERIES
        for record in series
    )
    assert all(
        len(record["display_values"]) == MAX_DISPLAY_POINTS_PER_SERIES
        for record in series
    )
    assert series[0]["display_timestamps_ns"][0] == 0
    assert series[0]["display_timestamps_ns"][-1] == 59_900_000_000

    timestamps, values = downsample_series(np.arange(1001), np.arange(1001))
    assert len(timestamps) == len(values) == 500
    assert timestamps[0] == 0
    assert timestamps[-1] == 1000


def test_view_model_uses_summary_analysis_without_interval_detail(monkeypatch):
    import analysis_app

    original = analysis_app.analyze_traffic_series
    interval_details = []

    def analyze(series, **kwargs):
        interval_details.append(kwargs.get("include_intervals"))
        return original(series, **kwargs)

    monkeypatch.setattr(analysis_app, "analyze_traffic_series", analyze)
    store = make_store(sample_count=12)
    view = build_view_model(store, AnalysisFilterState(nic_id=NIC_A.nic_id))

    assert interval_details == [False, False]
    assert all(record["point_count"] == 12 for record in view["nic_detail"]["series"])


def test_view_model_cache_repeats_identical_requests_and_respects_filters(monkeypatch):
    clear_view_model_cache()
    store = make_store()
    calls = []
    original_base_view_model = _base_view_model

    def counted_base_view_model(current_store, filters):
        calls.append(filters)
        return original_base_view_model(current_store, filters)

    monkeypatch.setattr("analysis_app._base_view_model", counted_base_view_model)

    first = build_view_model(store)
    second = build_view_model(store)
    assert second is first
    assert len(calls) == 1

    filtered = build_view_model(store, AnalysisFilterState(direction="tx"))
    assert filtered is not first
    assert len(calls) == 2
    clear_view_model_cache()


def test_view_model_cache_is_bounded(monkeypatch):
    import analysis_app

    clear_view_model_cache()
    monkeypatch.setattr(analysis_app, "MAX_VIEW_MODEL_CACHE_ENTRIES", 2)
    store = make_store()

    build_view_model(store, AnalysisFilterState(node_id=1))
    build_view_model(store, AnalysisFilterState(node_id=2))
    build_view_model(store, AnalysisFilterState(node_id=3))

    with analysis_app._VIEW_MODEL_CACHE_LOCK:
        assert len(analysis_app._VIEW_MODEL_CACHE) <= 2
    clear_view_model_cache()


def test_question_status_navigation_covers_nic_01_and_pending_families():
    view = build_view_model(make_store(), default_filters(make_store()))
    questions = view["question_coverage"]

    assert len(questions) == 15
    nic_01 = next(row for row in questions if row["id"] == "NIC-01")
    assert nic_01["status"] == "Supported"
    assert nic_01["state"] == "supported"
    assert all(row["status"] == "Pending / unsupported" for row in questions if row["id"] != "NIC-01")
    assert any(row["id"] == "JOB-03" for row in questions)
    assert all(row["implemented_analyses"] for row in questions if row["id"] != "JOB-04")
    assert all(row["limitations"] != "None" for row in questions)


def test_job_profile_covers_all_questions_without_aggregate_score():
    store = make_store()
    view = build_view_model(store, default_filters(store))
    profile = view["job_profile"]

    assert profile["status"] in {"partial", "complete", "unavailable"}
    assert len(profile["dimensions"]) == 13
    assert len(profile["questions"]) == 15
    assert {question["question_id"] for question in profile["questions"]} == {
        row["id"] for row in view["question_coverage"]
    }
    assert "score" not in profile
    assert profile["topology_status"] == "unsupported"
    assert profile["communication_edge_status"] == "unsupported"


def test_overview_rendering_with_job_profile_does_not_require_phase_context():
    store = make_store()
    view = build_view_model(store, default_filters(store))

    component = _render_overview(view)

    assert component is not None


def test_missing_and_zero_observations_remain_distinct():
    store = make_store(include_rx=False)
    view = build_view_model(store, AnalysisFilterState(nic_id=NIC_A.nic_id))
    series = view["nic_detail"]["series"]
    tx = next(record for record in series if record["direction"] == "TX")
    rx = next(record for record in series if record["direction"] == "RX")

    assert tx["state"] == "ready"
    assert tx["status"] == "Ready"
    assert rx["state"] == "missing"
    assert rx["status"] == "Missing"
    assert rx["total_bytes"] is None
    assert "hni_sts_rx_ok_octets" in view["capture"]["missing_counter_names"]

    zero_store = make_store(include_rx=True, sample_count=3)
    zero_view = build_view_model(
        zero_store, AnalysisFilterState(nic_id=NIC_A.nic_id, direction="rx")
    )
    zero_rx = zero_view["nic_detail"]["series"][0]
    assert zero_rx["state"] == "zero"
    assert zero_rx["status"] == "Zero traffic (observed)"
    assert zero_rx["total_bytes"] == 0


def test_serialized_view_model_contains_no_raw_store_columns():
    store = make_store(sample_count=550)
    view = build_view_model(store, AnalysisFilterState(nic_id=NIC_A.nic_id))
    serialized = json.dumps(view, allow_nan=False)

    forbidden_keys = {
        "trace_store",
        "raw_store",
        "uint64_values",
        "float64_values",
        "timestamps_ns",
        "record_ids",
        "quality_flags_array",
    }

    def keys(value):
        if isinstance(value, dict):
            yield from value.keys()
            for child in value.values():
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    assert not forbidden_keys.intersection(keys(view))
    assert "TraceStore" not in serialized
    assert "record-" not in serialized
    assert "display_timestamps_ns" in serialized


def test_export_includes_bounded_job_profile_without_raw_intervals():
    store = make_store()
    view = build_view_model(store, default_filters(store))
    document = build_export_document(view)

    assert document["job_profile"]["status"] == view["job_profile"]["status"]
    assert len(document["job_profile"]["questions"]) == 15
    assert len(document["job_profile"]["dimensions"]) == 13
    serialized = json.dumps(document, allow_nan=False)
    assert "raw_store" not in serialized
    assert "record-" not in serialized


def test_json_export_contains_filtered_view_model_contract():
    store = make_store(sample_count=600)
    filters = AnalysisFilterState(
        node_id=NODE_B.node_id,
        direction="tx",
        start_ns=100_000_000,
        end_ns=200_000_001,
    )
    view = build_view_model(store, filters)
    exported_at = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
    document = build_export_document(view, exported_at_utc=exported_at)
    payload = serialize_export_document(document)

    assert document["schema_version"] == 1
    assert document["export"]["state"] == "ready"
    assert document["export"]["exported_at_utc"] == exported_at.isoformat()
    assert document["export"]["content_scope"].startswith("Currently filtered")
    assert document["filters"] == {
        "node_id": NODE_B.node_id,
        "nic_id": None,
        "direction": "tx",
        "start_ns": 100_000_000,
        "end_ns": 200_000_001,
    }
    assert document["run"]["run_id"] == "run-synthetic"
    assert document["run"]["source_fingerprint"] == "sha256:synthetic"
    assert document["capture"]["selected_start_ns"] == 100_000_000
    assert document["overview"]["tx_total_bytes"] == 1_000
    assert len(document["series_summaries"]) == 1
    assert document["node_comparison"]["rows"][0]["node"] == "node-b"
    assert document["node_balance"]["effective_nic_count_formula"].startswith("1 / sum")
    assert document["job_balance"]["directions"][0]["effective_node_count"] == 1.0
    assert document["nic_detail"]["selected_nic"] == "cxi-0"
    assert len(document["nic_detail"]["series"]) == 1
    assert len(document["question_capabilities"]) == 15
    assert document["methodology"]["formulas"]["node_traffic_share"] == (
        "nic_total_bytes / node_total_bytes"
    )
    assert document["methodology"]["formulas"]["job_node_traffic_share"] == (
        "node_total_bytes / job_total_bytes"
    )
    assert document["methodology"]["units"]["rates"] == "bytes/s"
    assert document["methodology"]["windows"]["selected"]["start_ns"] == 100_000_000
    assert document["methodology"]["populations"]["series_count"] == 1
    assert "quality_flags" in document["methodology"]
    assert document["methodology"]["limitations"]

    assert payload == json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert json.loads(payload, parse_constant=lambda value: (_ for _ in ()).throw(AssertionError(value)))
    assert all(
        math.isfinite(value)
        for value in recursively_collect_values(json.loads(payload))
        if isinstance(value, float)
    )


def test_json_export_excludes_raw_columns_and_full_resolution_samples():
    store = make_store(sample_count=600)
    view = build_view_model(store, AnalysisFilterState(nic_id=NIC_A.nic_id))
    document = build_export_document(view)
    payload = serialize_export_document(document)

    forbidden_keys = {
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
        "display_point_count",
    }

    def keys(value):
        if isinstance(value, dict):
            yield from value.keys()
            for child in value.values():
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    assert not forbidden_keys.intersection(keys(document))
    assert "TraceStore" not in payload
    assert "record-" not in payload
    assert all(record["point_count"] == 600 for record in document["nic_detail"]["series"])


def test_json_export_is_series_and_byte_bounded():
    view = build_view_model(
        make_store(sample_count=12), AnalysisFilterState(nic_id=NIC_A.nic_id)
    )
    view["series_summaries"] = [view["series_summaries"][0]] * (MAX_EXPORT_SERIES + 25)
    document = build_export_document(view)

    assert len(document["series_summaries"]) == MAX_EXPORT_SERIES
    assert document["export"]["series_count"] == MAX_EXPORT_SERIES + 25
    assert document["export"]["series_truncated"] is True

    oversized_view = build_view_model(
        make_store(sample_count=12), AnalysisFilterState(nic_id=NIC_A.nic_id)
    )
    oversized_view["run"]["source_fingerprint"] = "x" * (MAX_EXPORT_JSON_BYTES + 1)
    with pytest.raises(ValueError, match="exceeds the JSON byte limit"):
        serialize_export_document(build_export_document(oversized_view))

    fallback = json.loads(build_export_payload(oversized_view)["content"])
    assert fallback["export"]["state"] == "invalid"
    assert fallback["export"]["status"] == "Invalid export"
    assert len(build_export_payload(oversized_view)["content"].encode()) <= MAX_EXPORT_JSON_BYTES


def test_json_export_has_useful_empty_and_invalid_states():
    invalid_document = build_export_document(None)

    assert invalid_document["export"]["state"] == "invalid"
    assert invalid_document["export"]["status"] == "Invalid export"
    assert invalid_document["export"]["reasons"]
    assert invalid_document["nic_detail"]["series"] == []
    assert invalid_document["methodology"]["limitations"]
    json.loads(serialize_export_document(invalid_document))
    assert build_export_payload(None)["type"] == "application/json"

    store = make_store()
    view = build_view_model(store, AnalysisFilterState(start_ns=2, end_ns=1))
    document = build_export_document(view)

    assert document["export"]["state"] == "invalid"
    assert document["export"]["status"] == "Invalid time window"
    assert document["export"]["reasons"] == [
        "The selected time window must have a positive duration.",
    ]
    assert document["filters"]["start_ns"] == 2
    assert document["filters"]["end_ns"] == 1
    assert document["nic_detail"]["series"] == []
    assert document["question_capabilities"]
    json.loads(serialize_export_document(document))


def test_json_export_button_is_keyboard_reachable_and_uses_current_filters():
    app = create_analysis_app(make_store())
    ids = component_ids(app.layout)
    button = find_component(app.layout, EXPORT_BUTTON_ID)

    assert {EXPORT_BUTTON_ID, EXPORT_DOWNLOAD_ID}.issubset(ids)
    assert isinstance(button, html.Button)
    assert button.type == "button"
    assert button.to_plotly_json()["props"]["aria-label"] == (
        "Export the currently filtered view model as JSON"
    )
    callback = app.callback_map[f"{EXPORT_DOWNLOAD_ID}.data"]
    assert callback["inputs"] == [
        {"id": EXPORT_BUTTON_ID, "property": "n_clicks"}
    ]
    assert {state["id"] for state in callback["state"]} == {
        "analysis-node-filter",
        "analysis-nic-filter",
        "analysis-direction-filter",
        "analysis-time-filter",
        "analysis-time-origin",
        "analysis-phase-store",
    }


def test_phase_controls_are_accessible_and_callback_wiring_uses_shared_filters():
    app = create_analysis_app(make_store(sample_count=12))
    ids = component_ids(app.layout)

    assert {
        PHASE_PANEL_ID,
        PHASE_RESULT_ID,
        PHASE_STORE_ID,
        PHASE_NAME_ID,
        PHASE_START_ID,
        PHASE_END_ID,
        PHASE_ADD_ID,
    }.issubset(ids)
    add_button = find_component(app.layout, PHASE_ADD_ID)
    assert isinstance(add_button, html.Button)
    assert add_button.type == "button"
    assert add_button.to_plotly_json()["props"]["aria-label"] == (
        "Add a manual phase window to the analysis"
    )

    callback = next(
        callback
        for key, callback in app.callback_map.items()
        if f"{PHASE_RESULT_ID}.children" in key
    )
    assert {input_["id"] for input_ in callback["inputs"]} == {
        PHASE_ADD_ID,
        "analysis-node-filter",
        "analysis-nic-filter",
        "analysis-direction-filter",
        "analysis-time-filter",
    }
    assert {state["id"] for state in callback["state"]} == {
        PHASE_NAME_ID,
        PHASE_START_ID,
        PHASE_END_ID,
        PHASE_STORE_ID,
        "analysis-time-origin",
    }
    assert any(
        f"{PHASE_STORE_ID}.data" in key for key in app.callback_map
    )


def test_json_export_includes_job_summary_and_phase_results():
    store = make_store(sample_count=600)
    filters = AnalysisFilterState(nic_id=NIC_A.nic_id, direction="tx")
    phases = [
        {"name": "first", "start_ns": 0, "end_ns": 29_900_000_000},
        {"name": "second", "start_ns": 29_900_000_000, "end_ns": 59_900_000_000},
    ]
    view = build_view_model(store, filters, phase_windows=phases)
    document = build_export_document(view)

    assert len(document["job_balance"]["directions"]) == 1
    assert document["job_balance"]["directions"][0]["job_total_bytes"] == 599_000
    assert document["phase_analysis"]["phase_count"] == 2
    assert [phase["name"] for phase in document["phase_analysis"]["phases"]] == [
        "first",
        "second",
    ]
    assert document["methodology"]["populations"]["phase_count"] == 2
    assert document["methodology"]["formulas"]["phase_total"] == (
        "sum(delta_bytes for valid intervals with both endpoints inside the phase)"
    )
    assert "Communication-edge metadata is required" in " ".join(
        document["methodology"]["limitations"]
    )
    assert len(phases) <= MAX_PHASE_WINDOWS


def test_nic09_reliability_distinguishes_missing_zero_nonzero_and_reset():
    missing_view = build_view_model(make_store())
    assert missing_view["nic_detail"]["reliability"]["state"] == "missing"
    assert missing_view["nic_detail"]["reliability"]["status_reasons"]

    nonzero_view = build_view_model(make_reliability_store())
    nonzero = nonzero_view["nic_detail"]["reliability"]
    collector = nonzero["collector_analyses"][0]
    by_counter = {
        row["counter"]: row for row in collector["counter_results"]
    }
    error = by_counter[RELIABILITY_ERROR.name]
    good = by_counter[RELIABILITY_GOOD.name]
    absent = by_counter["pct_spt_timeouts"]

    assert collector["population"] == {
        **collector["population"],
        "run_id": RELIABILITY_RUN.run_id,
        "node_id": RELIABILITY_NODE.node_id,
        "nic_id": RELIABILITY_NIC.nic_id,
        "collector_id": "collector-a (ID 120)",
        "collector_name": "collector-a",
        "collector_registry_id": RELIABILITY_COLLECTOR_A.collector_id,
        "collector_rank": 0,
    }
    assert error["state"] == ReliabilityStatus.OBSERVED_NONZERO.value
    assert error["total"] == 5
    assert error["rate_per_second"] == 2.5
    assert error["first_nonzero_sample_ns"] == 1_000_000_000
    assert error["zero_observed"] is False
    assert good["state"] == ReliabilityStatus.OBSERVED_ZERO.value
    assert good["total"] == 0
    assert good["zero_observed"] is True
    assert good["persistent_zero"] is True
    assert absent["state"] == ReliabilityStatus.MISSING_DATA.value
    assert "hni_pcs_corrected_cw" in by_counter
    assert by_counter["hni_pcs_corrected_cw"]["state"] == (
        ReliabilityStatus.MISSING_DATA.value
    )
    assert absent["total"] is None
    assert absent["zero_observed"] is False
    assert nonzero["state"] == "partial"

    zero_view = build_view_model(
        make_reliability_store(error_values=(0, 0, 0))
    )
    zero_collector = zero_view["nic_detail"]["reliability"]["collector_analyses"][0]
    zero_error = next(
        row
        for row in zero_collector["counter_results"]
        if row["counter"] == RELIABILITY_ERROR.name
    )
    assert zero_error["state"] == ReliabilityStatus.OBSERVED_ZERO.value
    assert zero_error["total"] == 0
    assert zero_error["zero_observed"] is True
    assert zero_error["persistent_zero"] is True

    reset_view = build_view_model(make_reliability_store(include_reset=True))
    reset_collector = reset_view["nic_detail"]["reliability"]["collector_analyses"][0]
    reset = next(
        row
        for row in reset_collector["counter_results"]
        if row["counter"] == RELIABILITY_RESET.name
    )
    assert reset["state"] == ReliabilityStatus.PARTIAL_INVALID_DATA.value
    assert reset["total"] == 3
    assert reset["invalid_interval_count"] == 1
    assert reset["covered_duration_ns"] == 1_000_000_000
    assert ReliabilityQualityFlag.DECREASE_OR_RESET.value in reset["quality_flags"]


def test_nic09_reliability_preserves_filters_and_duplicate_collectors():
    store = make_reliability_store(
        duplicate_collectors=True,
        good_values=(0, 10, 20),
    )
    filters = AnalysisFilterState(
        node_id=RELIABILITY_NODE.node_id,
        nic_id=RELIABILITY_NIC.nic_id,
        start_ns=0,
        end_ns=1_500_000_001,
    )
    view = build_view_model(store, filters)
    reliability = view["nic_detail"]["reliability"]

    assert reliability["collector_count"] == 2
    assert reliability["duplicate_stream_count"] == 2
    assert [record["state"] for record in reliability["collector_analyses"]] == [
        "partial_missing_data",
        "partial_missing_data",
    ]
    assert {
        record["population"]["collector_registry_id"]
        for record in reliability["collector_analyses"]
    } == {120, 121}
    first = reliability["collector_analyses"][0]
    assert first["requested_window_ns"] == (0, 1_500_000_000)
    error = next(
        row
        for row in first["counter_results"]
        if row["counter"] == RELIABILITY_ERROR.name
    )
    assert error["total"] == 2
    assert error["covered_duration_ns"] == 1_000_000_000
    assert error["coverage_fraction"] == 1_000_000_000 / 1_500_000_000
    assert first["population"]["node_id"] == RELIABILITY_NODE.node_id
    assert first["population"]["nic_id"] == RELIABILITY_NIC.nic_id


def test_nic09_first_nonzero_uses_earliest_duplicate_collector_event():
    store = make_reliability_store(
        duplicate_collectors=True,
        error_values=(0, 0, 5),
        collector_b_error_values=(0, 2, 5),
    )
    reliability = build_view_model(store)["nic_detail"]["reliability"]

    by_collector = {
        record["population"]["collector_registry_id"]: record
        for record in reliability["collector_analyses"]
    }
    assert by_collector[120]["first_nonzero_error_sample_ns"] == 2_000_000_000
    assert by_collector[120]["first_nonzero_error_increment_ns"] == 2_000_000_000
    assert by_collector[121]["first_nonzero_error_sample_ns"] == 1_000_000_000
    assert by_collector[121]["first_nonzero_error_increment_ns"] == 1_000_000_000
    assert reliability["first_nonzero_error_sample_ns"] == 1_000_000_000
    assert reliability["first_nonzero_error_increment_ns"] == 1_000_000_000


def test_nic09_reliability_renders_bounded_summary():
    view = build_view_model(make_reliability_store())
    rendered = _render_nic_detail(view)
    rendered_ids = component_ids(rendered)
    rendered_json = str(rendered.to_plotly_json())

    assert "nic-09-reliability-heading" in rendered_ids
    assert "NIC-09 reliability" in rendered_json
    assert "Zero is an observed value" in rendered_json
    assert "no causal reliability claim" in rendered_json


def test_nic09_export_is_bounded_summary_without_raw_samples():
    view = build_view_model(make_reliability_store())
    document = build_export_document(view)
    reliability = document["nic_detail"]["reliability"]

    assert reliability["counter_count"] <= MAX_RELIABILITY_COUNTER_RESULTS
    assert len(reliability["collector_analyses"]) <= 8
    assert all(
        collector["counter_count"] <= MAX_RELIABILITY_COUNTER_RESULTS
        for collector in reliability["collector_analyses"]
    )
    assert "reliability_error_rate" in document["methodology"]["formulas"]
    assert any(
        "not a causal" in limitation
        for limitation in document["methodology"]["limitations"]
    )
    payload = serialize_export_document(document)
    parsed = json.loads(payload)

    def keys(value):
        if isinstance(value, dict):
            yield from value.keys()
            for child in value.values():
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    assert not {
        "intervals",
        "values",
        "samples",
        "timestamps_ns",
        "display_values",
        "display_timestamps_ns",
    }.intersection(keys(parsed["nic_detail"]["reliability"]))
    assert len(payload.encode()) <= MAX_EXPORT_JSON_BYTES


def test_nic02_flow_control_distinguishes_missing_zero_nonzero_and_reset():
    missing_view = build_view_model(make_store())
    assert missing_view["nic_detail"]["flow_control"]["state"] == "missing"
    assert missing_view["nic_detail"]["flow_control"]["collector_count"] == 0

    view = build_view_model(make_flow_control_store())
    flow_control = view["nic_detail"]["flow_control"]
    collector = flow_control["collector_analyses"][0]
    by_counter = {row["counter"]: row for row in collector["counter_results"]}

    zero = by_counter[FLOW_TX_P0.name]
    nonzero = by_counter[FLOW_RX_P2.name]
    reset = by_counter[FLOW_TX_P3.name]
    assert zero["state"] == "zero_observations"
    assert zero["total"] == 0
    assert zero["first_nonzero_ns"] is None
    assert nonzero["state"] == "ok"
    assert nonzero["total"] == 7
    assert nonzero["mean_rate_per_second"] == 3.5
    assert nonzero["first_nonzero_ns"] == 1_000_000_000
    assert nonzero["first_nonzero_delta"] == 5
    assert reset["state"] == "partial_invalid_data"
    assert reset["total"] == 10
    assert reset["invalid_interval_count"] == 1
    assert "decrease_or_reset" in reset["quality_flags"]
    assert flow_control["state"] == "partial_invalid_data"
    assert flow_control["first_nonzero_ns"] == 1_000_000_000
    assert zero["total"] != nonzero["total"] or nonzero["total"] is not None


def test_nic02_flow_control_preserves_filters_identity_and_duplicate_collectors():
    store = make_flow_control_store(
        duplicate_collectors=True,
        collector_b_overrides={
            FLOW_RX_P2.name: (0, 2, 3),
            FLOW_TX_P0.name: (0, 0, 0),
            FLOW_TX_P3.name: (0, 4, 5),
        },
    )
    filters = AnalysisFilterState(
        node_id=FLOW_CONTROL_NODE.node_id,
        nic_id=FLOW_CONTROL_NIC.nic_id,
        direction="tx",
        start_ns=0,
        end_ns=1_500_000_001,
    )
    view = build_view_model(store, filters)
    flow_control = view["nic_detail"]["flow_control"]
    by_collector = {
        record["population"]["collector_registry_id"]: record
        for record in flow_control["collector_analyses"]
    }

    assert flow_control["collector_count"] == 2
    assert flow_control["duplicate_stream_count"] == 2
    assert len(flow_control["collector_analyses"]) <= MAX_FLOW_CONTROL_COLLECTORS
    assert flow_control["requested_window_ns"] == (0, 1_500_000_000)
    assert all(
        counter["direction"] == "TX"
        for record in flow_control["collector_analyses"]
        for counter in record["counter_results"]
    )
    first = by_collector[FLOW_CONTROL_COLLECTOR_A.collector_id]
    second = by_collector[FLOW_CONTROL_COLLECTOR_B.collector_id]
    assert first["population"] == {
        **first["population"],
        "run_id": FLOW_CONTROL_RUN.run_id,
        "node_id": FLOW_CONTROL_NODE.node_id,
        "nic_id": FLOW_CONTROL_NIC.nic_id,
        "collector_registry_id": FLOW_CONTROL_COLLECTOR_A.collector_id,
    }
    first_tx = next(
        row for row in first["counter_results"] if row["counter"] == FLOW_TX_P3.name
    )
    second_tx = next(
        row for row in second["counter_results"] if row["counter"] == FLOW_TX_P3.name
    )
    assert first_tx["total"] == 10
    assert second_tx["total"] == 4
    assert first_tx["requested_duration_ns"] == 1_500_000_000
    assert first_tx["coverage_fraction"] == 1_000_000_000 / 1_500_000_000


def test_nic02_duplicate_collector_coverage_remains_per_collector():
    store = make_flow_control_store(
        duplicate_collectors=True,
        collector_b_overrides={
            FLOW_TX_P0.name: (2, 1, 0),
            FLOW_TX_P3.name: (2, 1, 5),
            FLOW_TX_XOFF_P4.name: (2, 1, 5),
        },
    )
    filters = AnalysisFilterState(
        node_id=FLOW_CONTROL_NODE.node_id,
        nic_id=FLOW_CONTROL_NIC.nic_id,
        direction="tx",
        start_ns=0,
        end_ns=1_500_000_001,
    )
    flow_control = build_view_model(store, filters)["nic_detail"]["flow_control"]

    assert "covered_duration_ns" not in flow_control
    assert "coverage_fraction" not in flow_control
    assert flow_control["coverage_scope"] == (
        "Coverage is reported separately for each collector; no "
        "collector-merged or job-wide coverage value is reported."
    )
    coverage_by_collector = {
        record["collector_registry_id"]: record
        for record in flow_control["collector_coverage"]
    }
    assert coverage_by_collector[
        FLOW_CONTROL_COLLECTOR_A.collector_id
    ] == {
        "collector_id": "collector-fc-a (ID 220)",
        "collector_registry_id": FLOW_CONTROL_COLLECTOR_A.collector_id,
        "covered_duration_ns": 1_000_000_000,
        "coverage_fraction": 1_000_000_000 / 1_500_000_000,
    }
    assert coverage_by_collector[
        FLOW_CONTROL_COLLECTOR_B.collector_id
    ] == {
        "collector_id": "collector-fc-b (ID 221)",
        "collector_registry_id": FLOW_CONTROL_COLLECTOR_B.collector_id,
        "covered_duration_ns": 0,
        "coverage_fraction": 0.0,
    }


def test_nic02_flow_control_propagates_direction_priority_and_unsupported_metadata():
    view = build_view_model(make_flow_control_store(invalid_metadata=True))
    collector = view["nic_detail"]["flow_control"]["collector_analyses"][0]
    by_counter = {row["counter"]: row for row in collector["counter_results"]}

    assert by_counter[FLOW_RX_P2.name]["direction"] == "RX"
    assert by_counter[FLOW_RX_P2.name]["priority"] == 2
    assert by_counter[FLOW_RX_P2.name]["unit"] == "cycles"
    assert by_counter[FLOW_TX_XOFF_P4.name]["direction"] == "TX"
    assert by_counter[FLOW_TX_XOFF_P4.name]["priority"] == 4
    assert by_counter[FLOW_TX_XOFF_P4.name]["unit"] == "events"
    assert by_counter[FLOW_FGFC_DISCARD.name]["direction"] == "unsupported"
    assert by_counter[FLOW_FGFC_DISCARD.name]["priority"] is None
    invalid = by_counter[FLOW_TX_INVALID_METADATA.name]
    assert invalid["state"] == "unsupported_series"
    assert invalid["total"] is None
    assert "unsupported_series" in invalid["quality_flags"]
    assert "The counter must have cycles semantics and units." in invalid["status_reasons"]


def test_nic02_flow_control_renders_bounded_summary():
    view = build_view_model(make_flow_control_store())
    rendered = _render_nic_detail(view)
    rendered_ids = component_ids(rendered)
    rendered_json = str(rendered.to_plotly_json())

    assert "nic-02-flow-control-heading" in rendered_ids
    assert "NIC-02 flow control" in rendered_json
    assert "clock frequency is not inferred" in rendered_json
    assert "no congestion, bottleneck, or ECN claim" in rendered_json
    assert "Zero, missing, unsupported, and reset-affected" in rendered_json


def test_nic02_flow_control_export_is_bounded_summary_without_raw_samples():
    view = build_view_model(make_flow_control_store())
    document = build_export_document(view)
    flow_control = document["nic_detail"]["flow_control"]
    payload = serialize_export_document(document)
    parsed = json.loads(payload)

    assert flow_control["counter_count"] <= MAX_FLOW_CONTROL_COUNTER_RESULTS
    assert len(flow_control["collector_analyses"]) <= MAX_FLOW_CONTROL_COLLECTORS
    assert "covered_duration_ns" not in flow_control
    assert "coverage_fraction" not in flow_control
    assert len(flow_control["collector_coverage"]) <= MAX_FLOW_CONTROL_COLLECTORS
    assert all(
        collector["counter_count"] <= MAX_FLOW_CONTROL_COUNTER_RESULTS
        for collector in flow_control["collector_analyses"]
    )
    assert "flow_control_delta" in document["methodology"]["formulas"]
    assert "Pause-cycle totals remain in cycles" in " ".join(
        document["methodology"]["limitations"]
    )

    def keys(value):
        if isinstance(value, dict):
            yield from value.keys()
            for child in value.values():
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    flow_keys = set(keys(parsed["nic_detail"]["flow_control"]))
    assert not {
        "intervals",
        "values",
        "samples",
        "timestamps_ns",
        "display_values",
        "display_timestamps_ns",
        "raw_store",
        "uint64_values",
        "float64_values",
    }.intersection(flow_keys)
    assert len(payload.encode()) <= MAX_EXPORT_JSON_BYTES


def test_nic03_matching_distinguishes_missing_zero_nonzero_and_reset():
    missing_view = build_view_model(make_store())
    assert missing_view["nic_detail"]["matching"]["state"] == "missing"
    assert missing_view["nic_detail"]["matching"]["collector_count"] == 0

    view = build_view_model(make_matching_store())
    matching = view["nic_detail"]["matching"]
    collector = matching["collector_analyses"][0]
    by_counter = {row["counter"]: row for row in collector["counter_results"]}
    priority = by_counter[MATCHING_PRIORITY_0.name]
    overflow = by_counter[MATCHING_OVERFLOW_0.name]

    assert collector["population"] == {
        **collector["population"],
        "run_id": MATCHING_RUN.run_id,
        "node_id": MATCHING_NODE.node_id,
        "nic_id": MATCHING_NIC.nic_id,
        "collector_id": "collector-m-a (ID 320)",
        "collector_registry_id": MATCHING_COLLECTOR_A.collector_id,
    }
    assert priority["state"] == MatchingStatus.OBSERVED_NONZERO.value
    assert priority["total"] == 5
    assert priority["rate_per_second"] == 2.5
    assert priority["zero_observed"] is False
    assert overflow["total"] == 4
    assert overflow["rate_per_second"] == 2.0
    assert collector["priority_total"] == 8
    assert collector["overflow_total"] == 7
    assert collector["total"] == 15
    assert collector["priority_fraction"] == 8 / 15
    assert collector["population_compatible"] is True
    assert collector["temporal_interval_count"] == 2
    assert matching["collector_priority_fractions"] == [
        {
            "collector_id": "collector-m-a (ID 320)",
            "collector_registry_id": MATCHING_COLLECTOR_A.collector_id,
            "state": MatchingStatus.OBSERVED_NONZERO.value,
            "population_compatible": True,
            "priority_fraction": 8 / 15,
            "overflow_fraction": 7 / 15,
            "fraction_status": MatchingStatus.OBSERVED_NONZERO.value,
        }
    ]
    assert matching["priority_fraction_scope"] == (
        "Collector-separated values; incompatible or heterogeneous collectors are not merged."
    )
    assert matching["priority_fraction_formula"] == (
        "priority_total / (priority_total + overflow_total)"
    )
    assert matching["overflow_fraction_formula"] == (
        "overflow_total / (priority_total + overflow_total)"
    )
    assert matching["state"] == MatchingStatus.OBSERVED_NONZERO.value

    partial_missing = build_view_model(make_matching_store(include_all_counters=False))
    partial_collector = partial_missing["nic_detail"]["matching"]["collector_analyses"][0]
    assert partial_missing["nic_detail"]["matching"]["state"] == (
        MatchingStatus.PARTIAL_MISSING_DATA.value
    )
    assert "lpe_net_match_priority_2" in partial_collector["missing_counter_ids"]
    assert partial_collector["priority_total"] is None
    assert partial_collector["priority_fraction"] is None

    zero_view = build_view_model(
        make_matching_store(
            priority_0_values=(0, 0, 0),
            overflow_0_values=(0, 0, 0),
            priority_1_values=(0, 0, 0),
            overflow_1_values=(0, 0, 0),
        )
    )
    zero_collector = zero_view["nic_detail"]["matching"]["collector_analyses"][0]
    assert zero_collector["state"] == MatchingStatus.OBSERVED_ZERO.value
    assert zero_collector["total"] == 0
    assert zero_collector["priority_fraction"] is None
    assert zero_collector["persistent_zero"] is True

    reset_view = build_view_model(
        make_matching_store(priority_1_values=(0, 4, 2))
    )
    reset_collector = reset_view["nic_detail"]["matching"]["collector_analyses"][0]
    reset = next(
        row
        for row in reset_collector["counter_results"]
        if row["counter"] == MATCHING_PRIORITY_1.name
    )
    assert reset["state"] == MatchingStatus.PARTIAL_INVALID_DATA.value
    assert reset["total"] == 4
    assert reset["invalid_interval_count"] == 1
    assert "decrease_or_reset" in reset["quality_flags"]
    assert reset_collector["priority_fraction"] is None
    assert reset_collector["temporal_interval_count"] == 0


def test_nic03_matching_preserves_filters_identity_and_duplicate_collectors():
    store = make_matching_store(
        duplicate_collectors=True,
        collector_b_overrides={
            MATCHING_PRIORITY_0.name: (0, 10, 12),
            MATCHING_OVERFLOW_0.name: (0, 1, 1),
            MATCHING_PRIORITY_1.name: (0, 2, 4),
            MATCHING_OVERFLOW_1.name: (0, 3, 3),
        },
    )
    filters = AnalysisFilterState(
        node_id=MATCHING_NODE.node_id,
        nic_id=MATCHING_NIC.nic_id,
        direction="rx",
        start_ns=0,
        end_ns=1_500_000_001,
    )
    view = build_view_model(store, filters)
    matching = view["nic_detail"]["matching"]
    by_collector = {
        record["population"]["collector_registry_id"]: record
        for record in matching["collector_analyses"]
    }

    assert matching["collector_count"] == 2
    assert matching["duplicate_stream_count"] == 2
    assert matching["requested_window_ns"] == (0, 1_500_000_000)
    assert matching["direction_filter"] == "rx"
    first = by_collector[MATCHING_COLLECTOR_A.collector_id]
    second = by_collector[MATCHING_COLLECTOR_B.collector_id]
    assert first["population"]["collector_id"] == "collector-m-a (ID 320)"
    assert second["population"]["collector_id"] == "collector-m-b (ID 321)"
    assert first["total"] == 8
    assert second["total"] == 16
    assert first["coverage_fraction"] == 1_000_000_000 / 1_500_000_000
    assert second["coverage_fraction"] == 1_000_000_000 / 1_500_000_000
    assert len(matching["collector_priority_fractions"]) == 2
    assert {
        record["priority_fraction"]
        for record in matching["collector_priority_fractions"]
    } == {first["priority_fraction"], second["priority_fraction"]}

    tx_view = build_view_model(
        store,
        AnalysisFilterState(
            node_id=MATCHING_NODE.node_id,
            nic_id=MATCHING_NIC.nic_id,
            direction="tx",
        ),
    )
    tx_matching = tx_view["nic_detail"]["matching"]
    assert tx_matching["collector_count"] == 0
    assert "receive-path" in " ".join(tx_matching["status_reasons"])


def test_nic03_matching_requires_compatible_populations_and_boundaries():
    store = make_matching_store(
        duplicate_collectors=True,
        collector_b_timestamps_by_counter={
            MATCHING_PRIORITY_0.name: (0, 700_000_000, 2_000_000_000),
        },
    )
    matching = build_view_model(store)["nic_detail"]["matching"]
    by_collector = {
        record["population"]["collector_registry_id"]: record
        for record in matching["collector_analyses"]
    }

    first = by_collector[MATCHING_COLLECTOR_A.collector_id]
    second = by_collector[MATCHING_COLLECTOR_B.collector_id]
    assert first["population_compatible"] is True
    assert second["population_compatible"] is False
    assert second["priority_fraction"] is None
    assert second["total"] is None
    assert "incompatible_coverage" in second["quality_flags"]
    assert "Priority/overflow totals and fractions require" in " ".join(
        second["status_reasons"]
    )


def test_nic03_matching_renders_bounded_summary():
    rendered = _render_nic_detail(build_view_model(make_matching_store()))
    rendered_ids = component_ids(rendered)
    rendered_json = str(rendered.to_plotly_json())

    assert "nic-03-matching-heading" in rendered_ids
    assert "NIC-03 matching" in rendered_json
    assert "no copy-cost, latency, or causal claim" in rendered_json
    assert "Zero and missing remain distinct" in rendered_json
    assert "53.3%" in rendered_json
    assert "Priority fraction" in rendered_json


def test_nic03_matching_export_is_bounded_summary_without_raw_samples():
    view = build_view_model(make_matching_store())
    document = build_export_document(view)
    matching = document["nic_detail"]["matching"]
    payload = serialize_export_document(document)
    parsed = json.loads(payload)

    assert matching["counter_count"] <= MAX_MATCHING_COUNTER_RESULTS
    assert len(matching["collector_analyses"]) <= MAX_MATCHING_COLLECTORS
    assert all(
        collector["counter_count"] <= MAX_MATCHING_COUNTER_RESULTS
        for collector in matching["collector_analyses"]
    )
    assert all(
        collector["priority_count"] <= 4
        for collector in matching["collector_analyses"]
    )
    assert "matching_priority_fraction" in document["methodology"]["formulas"]
    exported_fractions = document["methodology"]["windows"]["matching"][
        "collector_priority_fractions"
    ]
    assert exported_fractions[0]["priority_fraction"] == 8 / 15
    assert len(exported_fractions) <= MAX_MATCHING_COLLECTORS
    assert document["methodology"]["windows"]["matching"][
        "priority_fraction_scope"
    ] == (
        "Collector-separated values; incompatible or heterogeneous collectors are not merged."
    )
    assert any(
        "copy cost" in limitation
        for limitation in document["methodology"]["limitations"]
    )

    def keys(value):
        if isinstance(value, dict):
            yield from value.keys()
            for child in value.values():
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    matching_keys = set(keys(parsed["nic_detail"]["matching"]))
    assert not {
        "intervals",
        "temporal_intervals",
        "values",
        "samples",
        "timestamps_ns",
        "display_values",
        "display_timestamps_ns",
        "raw_store",
        "uint64_values",
        "float64_values",
    }.intersection(matching_keys)
    assert len(payload.encode()) <= MAX_EXPORT_JSON_BYTES


def test_nic04_host_pressure_distinguishes_missing_zero_and_nonzero():
    missing = build_view_model(make_host_pressure_store(include_host=False))[
        "nic_detail"
    ]["host_pressure"]
    assert missing["state"] == "missing"
    assert missing["collector_count"] == 0
    assert "No host-pressure evidence" in missing["status"]

    zero = build_view_model(
        make_host_pressure_store(
            posted_packets=(0, 0, 0),
            posted_blocked=(0, 0, 0),
            nonposted_packets=(0, 0, 0),
            nonposted_blocked=(0, 0, 0),
            stall=(0, 0, 0),
        )
    )["nic_detail"]["host_pressure"]
    zero_collector = zero["collector_analyses"][0]
    zero_posted = zero_collector["path_results"][0]
    assert zero_posted["packet_result"]["state"] == "observed_zero"
    assert zero_posted["blocked_result"]["state"] == "observed_zero"
    assert zero_posted["packet_result"]["zero_observed"] is True
    assert zero_posted["packet_total"] == 0
    assert zero_posted["blocked_total"] == 0
    assert zero_posted["denominator_state"] == "unresolved_blocked_units"
    assert zero_posted["blocked_per_packet"] is None

    nonzero = build_view_model(make_host_pressure_store())["nic_detail"]["host_pressure"]
    nonzero_collector = nonzero["collector_analyses"][0]
    nonzero_posted = nonzero_collector["path_results"][0]
    assert nonzero_posted["packet_result"]["state"] == "observed_nonzero"
    assert nonzero_posted["blocked_result"]["state"] == "observed_nonzero"
    assert nonzero_posted["packet_total"] == 22
    assert nonzero_posted["blocked_total"] == 5
    assert nonzero_posted["packet_rate_per_second"] == 11.0
    assert nonzero_posted["population_compatible"] is True
    assert nonzero_posted["common_valid_interval_count"] == 2
    assert nonzero_posted["blocked_per_packet"] is None
    assert nonzero["blocked_per_packet_status"] == (
        "Unsupported until authoritative blocked-counter units are supplied."
    )


def test_nic04_host_pressure_reports_reset_without_hiding_zero_or_missing():
    store = make_host_pressure_store(
        posted_packets=(0, 10, 3),
        posted_blocked=(0, 2, 4),
        nonposted_packets=(0, 0, 0),
        nonposted_blocked=(0, 0, 0),
        stall=(0, 0, 0),
    )
    host = build_view_model(store)["nic_detail"]["host_pressure"]
    posted = host["collector_analyses"][0]["path_results"][0]
    nonposted = host["collector_analyses"][0]["path_results"][1]

    assert posted["packet_result"]["state"] == "partial_invalid_data"
    assert posted["packet_result"]["invalid_interval_count"] == 1
    assert posted["packet_result"]["valid_interval_count"] == 1
    assert posted["reset_excluded_interval_count"] == 1
    assert "decrease_or_reset" in posted["quality_flags"]
    assert posted["packet_total"] == 10
    assert nonposted["packet_result"]["state"] == "observed_zero"
    assert nonposted["packet_result"]["zero_observed"] is True
    main_counter_names = {
        HOST_POSTED_PACKET.name,
        HOST_POSTED_BLOCKED.name,
        HOST_NONPOSTED_PACKET.name,
        HOST_NONPOSTED_BLOCKED.name,
    }
    assert not main_counter_names.intersection(host["missing_counter_ids"])
    assert HOST_DMA_STALL.name not in host["missing_counter_ids"]
    assert len(host["missing_counter_ids"]) > 0


def test_nic04_host_pressure_preserves_duplicate_collectors_and_filters():
    store = make_host_pressure_store(
        duplicate_collectors=True,
        collector_b_overrides={
            HOST_POSTED_PACKET.name: (0, 20, 50),
            HOST_POSTED_BLOCKED.name: (0, 4, 9),
        },
    )
    filters = AnalysisFilterState(
        node_id=HOST_NODE.node_id,
        nic_id=HOST_NIC.nic_id,
        direction="all",
        start_ns=0,
        end_ns=1_500_000_000,
    )
    host = build_view_model(store, filters)["nic_detail"]["host_pressure"]

    assert host["collector_count"] == 2
    assert host["duplicate_stream_count"] == 2
    assert len(host["collector_analyses"]) <= MAX_HOST_PRESSURE_COLLECTORS
    assert host["requested_window_ns"] == (0, 1_499_999_999)
    by_collector = {
        record["population"]["collector_registry_id"]: record
        for record in host["collector_analyses"]
    }
    first = by_collector[HOST_COLLECTOR_A.collector_id]
    second = by_collector[HOST_COLLECTOR_B.collector_id]
    assert first["path_results"][0]["packet_total"] == 10
    assert second["path_results"][0]["packet_total"] == 20
    assert first["population"]["run_id"] == HOST_RUN.run_id
    assert first["population"]["node_id"] == HOST_NODE.node_id
    assert first["population"]["nic_id"] == HOST_NIC.nic_id
    assert first["path_results"][0]["packet_result"]["collector_id"].startswith(
        "collector-h-a"
    )
    assert second["path_results"][0]["packet_result"]["collector_id"].startswith(
        "collector-h-b"
    )


def test_nic04_host_pressure_requires_compatible_populations_and_units():
    boundary_store = make_host_pressure_store(
        duplicate_collectors=True,
        collector_b_timestamps_by_counter={
            HOST_POSTED_PACKET.name: (500_000_000, 1_500_000_000, 2_000_000_000)
        },
    )
    boundary_host = build_view_model(boundary_store)["nic_detail"]["host_pressure"]
    boundary_records = boundary_host["collector_analyses"]
    assert boundary_records[0]["path_results"][0]["population_compatible"] is True
    assert boundary_records[1]["path_results"][0]["population_compatible"] is False
    assert boundary_records[1]["path_results"][0]["common_valid_interval_count"] == 0
    assert boundary_records[1]["path_results"][0]["denominator_state"] == (
        "unresolved_blocked_units"
    )
    assert boundary_records[1]["path_results"][0]["packet_total"] is None
    assert boundary_records[1]["path_results"][0]["blocked_total"] is None

    unit_store = make_host_pressure_store(packet_unit="messages")
    unit_host = build_view_model(unit_store)["nic_detail"]["host_pressure"]
    unit_posted = unit_host["collector_analyses"][0]["path_results"][0]
    assert unit_posted["state"] == "unsupported_units"
    assert unit_posted["denominator_state"] == "incompatible_units"
    assert HostPressureQualityFlag.INCOMPATIBLE_UNITS.value in unit_posted["quality_flags"]
    assert unit_posted["blocked_per_packet"] is None
    assert unit_posted["blocked_rate_per_second"] is None
    assert "The packet counter unit must be 'packets'" in " ".join(
        unit_posted["status_reasons"]
    )
    assert any(
        "authoritative resolution" in reason
        for reason in unit_host["status_reasons"]
    )


def test_nic04_host_pressure_renders_bounded_summary():
    rendered = _render_nic_detail(build_view_model(make_host_pressure_store()))
    rendered_ids = component_ids(rendered)
    rendered_json = str(rendered.to_plotly_json())

    assert "nic-04-host-pressure-heading" in rendered_ids
    assert "NIC-04 host pressure" in rendered_json
    assert "no DMA, dispatcher, or PCIe causal claim" in rendered_json
    assert "Blocked per packet" in rendered_json
    assert "Unsupported" in rendered_json
    assert "Zero, missing, insufficient, unsupported" in rendered_json


def test_nic04_host_pressure_export_is_bounded_without_raw_columns():
    view = build_view_model(make_host_pressure_store(duplicate_collectors=True))
    document = build_export_document(view)
    host = document["nic_detail"]["host_pressure"]
    payload = serialize_export_document(document)
    parsed = json.loads(payload)

    assert len(host["collector_analyses"]) <= MAX_HOST_PRESSURE_COLLECTORS
    assert all(
        len(collector["path_results"]) <= MAX_HOST_PRESSURE_PATH_RESULTS
        for collector in host["collector_analyses"]
    )
    assert all(
        len(collector["stall_results"]) <= MAX_HOST_PRESSURE_STALL_RESULTS
        for collector in host["collector_analyses"]
    )
    assert host["blocked_per_packet"] is None
    assert host["blocked_unit_resolution"] is None
    assert host["population_compatibility_scope"] == (
        "Posted and nonposted populations remain separate; packet and blocked "
        "series are paired only on exact valid interval boundaries within each collector."
    )
    assert any(
        "does not prove that PCIe" in limitation
        for limitation in host["limitations"]
    )

    def keys(value):
        if isinstance(value, dict):
            yield from value.keys()
            for child in value.values():
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    host_keys = set(keys(parsed["nic_detail"]["host_pressure"]))
    assert not {
        "intervals",
        "values",
        "samples",
        "timestamps_ns",
        "display_values",
        "display_timestamps_ns",
        "raw_store",
        "uint64_values",
        "float64_values",
    }.intersection(host_keys)
    assert len(payload.encode()) <= MAX_EXPORT_JSON_BYTES


def test_nic05_operations_distinguishes_missing_zero_and_nonzero():
    missing = build_view_model(make_operation_store(counter_names=()))[
        "nic_detail"
    ]["operations"]
    assert missing["state"] == "missing"
    assert missing["collector_count"] == 0
    assert missing["opcode_mapping_status"] == "missing_mapping"
    assert len(missing["missing_opcode_counter_ids"]) == 16

    zero = build_view_model(
        make_operation_store(
            bin_0_values=(0, 0, 0),
            bin_1_values=(0, 0, 0),
            dma_0_values=(0, 1, 1),
            idc_0_values=(0, 2, 2),
        )
    )["nic_detail"]["operations"]
    zero_collector = zero["collector_analyses"][0]
    assert zero_collector["opcode_results"][0]["state"] == "observed_zero"
    assert zero_collector["opcode_results"][0]["zero_observed"] is True
    assert zero_collector["dma_idc_mix"]["total_commands"] == 3

    nonzero = build_view_model(make_operation_store())["nic_detail"]["operations"]
    collector = nonzero["collector_analyses"][0]
    assert collector["opcode_results"][0]["total"] == 4
    assert collector["opcode_results"][0]["rate_per_second"] == 2.0
    assert collector["aggregate_dma_commands"]["total"] == 5
    assert collector["aggregate_idc_commands"]["total"] == 8
    assert collector["dma_idc_mix"]["dma_fraction"] == 5 / 13
    assert collector["dma_idc_mix"]["idc_fraction"] == 8 / 13
    assert collector["opcode_results"][0]["fraction_of_dma_opcode_commands"] == 4 / 6
    assert collector["dma_idc_mix"]["state"] == "observed_nonzero"


def test_nic05_operations_reports_reset_without_converting_it_to_zero():
    view = build_view_model(
        make_operation_store(
            bin_0_values=(0, 4, 2),
            bin_1_values=(0, 0, 0),
            dma_0_values=(0, 3, 3),
            idc_0_values=(0, 0, 0),
        )
    )["nic_detail"]["operations"]
    collector = view["collector_analyses"][0]
    reset_opcode = collector["opcode_results"][0]

    assert reset_opcode["state"] == "partial_invalid_data"
    assert reset_opcode["invalid_interval_count"] == 1
    assert reset_opcode["total"] == 4
    assert "decrease_or_reset" in reset_opcode["quality_flags"]
    assert collector["opcode_results"][1]["state"] == "observed_zero"
    assert collector["opcode_results"][1]["zero_observed"] is True


def test_nic05_operations_preserves_duplicate_collectors_and_filters():
    store = make_operation_store(
        duplicate_collectors=True,
        collector_b_overrides={
            "cq_dma_cmd_counts_0": (0, 4, 8),
            "cq_dma_cmd_counts_1": (0, 2, 4),
            "cq_num_dma_cmds_0": (0, 6, 10),
            "cq_num_idc_cmds_0": (0, 2, 4),
        },
    )
    filters = AnalysisFilterState(
        node_id=OPERATION_NODE.node_id,
        nic_id=OPERATION_NIC.nic_id,
        direction="all",
        start_ns=0,
        end_ns=1_500_000_000,
    )
    clear_view_model_cache()
    operations = build_view_model(store, filters)["nic_detail"]["operations"]

    assert operations["collector_count"] == 2
    assert operations["duplicate_stream_count"] == 2
    assert operations["requested_window_ns"] == (0, 1_499_999_999)
    by_collector = {
        record["population"]["collector_registry_id"]: record
        for record in operations["collector_analyses"]
    }
    first = by_collector[OPERATION_COLLECTOR_A.collector_id]
    second = by_collector[OPERATION_COLLECTOR_B.collector_id]
    assert first["population"]["run_id"] == OPERATION_RUN.run_id
    assert first["population"]["node_id"] == OPERATION_NODE.node_id
    assert first["population"]["nic_id"] == OPERATION_NIC.nic_id
    assert first["opcode_results"][0]["collector_id"].startswith("collector-op-a")
    assert second["dma_opcode_bin_results"][0]["collector_id"].startswith(
        "collector-op-b"
    )
    assert first["opcode_results"][0]["total"] == 2
    assert second["dma_opcode_bin_results"][0]["total"] == 4


def test_nic05_operations_keep_mapped_and_unknown_opcodes_explicit(monkeypatch):
    store = make_operation_store()
    mapped = OpcodeMapping(
        mapping_version="authoritative-fixture-v1",
        evidence="test-only authoritative mapping fixture",
        counter_group="cq_dma_cmd_counts",
        entries=tuple(
            OpcodeMappingEntry(
                bin_index=index,
                label={0: "PUT", 1: "GET"}.get(index, f"OPCODE_{index}"),
            )
            for index in range(16)
        ),
    )
    monkeypatch.setattr(analysis_app, "_OPERATION_OPCODE_MAPPING", mapped)
    clear_view_model_cache()
    mapped_view = build_view_model(store)["nic_detail"]["operations"]
    assert mapped_view["opcode_mapping_status"] == "all_mapped"
    assert mapped_view["opcode_mapping_version"] == "authoritative-fixture-v1"
    labels = {
        record["bin_index"]: record["opcode_label"]
        for record in mapped_view["collector_analyses"][0]["opcode_results"]
    }
    assert labels[0] == "PUT"
    assert labels[1] == "GET"
    assert mapped_view["unmapped_opcode_counter_ids"] == ()

    partial_mapping = OpcodeMapping(
        mapping_version="authoritative-fixture-v2",
        evidence="test-only partially mapped fixture",
        counter_group="cq_dma_cmd_counts",
        entries=(OpcodeMappingEntry(bin_index=0, label="PUT"),),
    )
    monkeypatch.setattr(analysis_app, "_OPERATION_OPCODE_MAPPING", partial_mapping)
    clear_view_model_cache()
    unknown_view = build_view_model(store)["nic_detail"]["operations"]
    assert unknown_view["opcode_mapping_status"] == "partial_unknown_opcode"
    assert "cq_dma_cmd_counts_1" in unknown_view["unmapped_opcode_counter_ids"]
    assert unknown_view["collector_analyses"][0]["opcode_results"][1]["opcode_label"] is None


def test_nic05_operations_requires_compatible_dma_and_idc_population():
    store = make_operation_store(
        duplicate_collectors=True,
        collector_b_timestamps_by_counter={
            "cq_num_dma_cmds_0": (500_000_000, 1_500_000_000, 2_000_000_000),
        },
    )
    clear_view_model_cache()
    operations = build_view_model(store)["nic_detail"]["operations"]
    incompatible = next(
        record
        for record in operations["collector_analyses"]
        if record["population"]["collector_registry_id"]
        == OPERATION_COLLECTOR_B.collector_id
    )
    mix = incompatible["dma_idc_mix"]

    assert incompatible["aggregate_dma_commands"]["total"] is None
    assert (
        "cq_num_dma_cmds_0"
        in incompatible["aggregate_dma_commands"]["incompatible_counter_ids"]
    )
    assert incompatible["aggregate_idc_commands"]["total"] is not None
    assert mix["dma_fraction"] is None
    assert mix["idc_fraction"] is None
    assert "incompatible_coverage" in mix["quality_flags"]
    assert (
        "Both DMA and IDC totals are required for a family mix."
        in mix["status_reasons"]
    )


def test_nic05_operations_mix_display_is_collector_separated_for_duplicates():
    store = make_operation_store(
        duplicate_collectors=True,
        collector_b_timestamps_by_counter={
            "cq_num_dma_cmds_0": (500_000_000, 1_500_000_000, 2_000_000_000),
        },
    )
    rendered = _render_nic_detail(build_view_model(store))
    rendered_json = str(rendered.to_plotly_json())

    assert "collector-op-a (ID 520): 38.5% / 61.5%" in rendered_json
    assert "collector-op-b (ID 521): unsupported" in rendered_json
    assert "Collector-separated mixes; no merged value is shown." in rendered_json


def test_nic05_operations_renders_bounded_summary():
    rendered = _render_nic_detail(build_view_model(make_operation_store()))
    rendered_ids = component_ids(rendered)
    rendered_json = str(rendered.to_plotly_json())

    assert "nic-05-operations-heading" in rendered_ids
    assert "NIC-05 command operations" in rendered_json
    assert "no wire-byte, MPI-payload, eager/rendezvous, or causal claim" in rendered_json
    assert "Unknown opcodes remain explicitly unlabeled" in rendered_json
    assert "Zero, missing, insufficient, unsupported" in rendered_json
    assert str(MAX_OPERATION_OPCODE_RESULTS) in rendered_json


def test_nic05_operations_export_is_bounded_without_raw_columns():
    view = build_view_model(make_operation_store(duplicate_collectors=True))
    document = build_export_document(view)
    operations = document["nic_detail"]["operations"]
    payload = serialize_export_document(document)
    parsed = json.loads(payload)

    assert len(operations["collector_analyses"]) <= MAX_OPERATION_COLLECTORS
    assert all(
        len(collector["opcode_results"]) <= MAX_OPERATION_OPCODE_RESULTS
        for collector in operations["collector_analyses"]
    )
    assert sum(
        len(collector["dma_opcode_bin_results"])
        for collector in operations["collector_analyses"]
    ) <= MAX_OPERATION_OPCODE_RESULTS
    assert operations["dma_opcode_bin_count"] <= MAX_OPERATION_OPCODE_RESULTS
    assert operations["dma_opcode_bin_truncated"] is True
    assert all(
        len(collector["dma_total_results"])
        + len(collector["idc_total_results"])
        <= MAX_OPERATION_TOTAL_RESULTS
        for collector in operations["collector_analyses"]
    )
    assert operations["opcode_mapping_status"] == "missing_mapping"
    assert operations["opcode_mapping_evidence"]
    assert any(
        "not wire bytes" in limitation for limitation in operations["limitations"]
    )

    def keys(value):
        if isinstance(value, dict):
            yield from value.keys()
            for child in value.values():
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    operation_keys = set(keys(parsed["nic_detail"]["operations"]))
    assert not {
        "intervals",
        "values",
        "samples",
        "timestamps_ns",
        "display_values",
        "display_timestamps_ns",
        "raw_store",
        "uint64_values",
        "float64_values",
    }.intersection(operation_keys)
    assert len(payload.encode()) <= MAX_EXPORT_JSON_BYTES


def test_nic06_frames_distinguish_missing_zero_nonzero_and_reset():
    missing_view = build_view_model(make_store())
    assert missing_view["nic_detail"]["frames"]["state"] == "missing_data"
    assert missing_view["nic_detail"]["frames"]["status_reasons"]

    zero_safe_names = tuple(
        name for name in FRAME_COUNTERS if name != "hni_rx_ok_27"
    )
    view = build_view_model(make_frame_store(), frame_filters("tx"))
    frames = view["nic_detail"]["frames"]
    collector = frames["collector_analyses"][0]
    directions = {
        direction["direction"]: direction
        for direction in collector["direction_results"]
    }
    tx = directions["tx"]
    rx_view = build_view_model(
        make_frame_store(
            counter_names=zero_safe_names, rx_values=(0, 0, 0)
        ),
        frame_filters("rx"),
    )
    rx = rx_view["nic_detail"]["frames"]["collector_analyses"][0][
        "direction_results"
    ][0]

    assert frames["state"] == "observed_nonzero"
    assert tx["state"] == "observed_nonzero"
    assert tx["bin_total"] == 36
    assert tx["byte_total"] == 16
    assert rx["state"] == "partial_missing_data"
    assert rx["bin_total"] is None
    assert all(
        bin_result["total"] == 0
        for bin_result in rx["bin_results"]
        if bin_result["state"] != "missing_data"
    )
    assert all(
        bin_result["zero_observed"]
        for bin_result in rx["bin_results"]
        if bin_result["state"] != "missing_data"
    )

    reset_store = make_frame_store(
        duplicate_collectors=True,
        collector_b_overrides={"hni_tx_ok_27": (0, 5, 3)}
    )
    reset_view = build_view_model(reset_store, frame_filters("tx"))
    reset_direction = next(
        direction
        for direction in reset_view["nic_detail"]["frames"]["collector_analyses"][1][
            "direction_results"
        ]
        if direction["direction"] == "tx"
    )
    reset_bin = next(
        bin_result
        for bin_result in reset_direction["bin_results"]
        if bin_result["bin_id"] == "27"
    )

    assert reset_bin["state"] == "partial_invalid_data"
    assert reset_bin["total"] == 5
    assert reset_bin["invalid_interval_count"] == 1
    assert "decrease_or_reset" in reset_bin["quality_flags"]
    assert reset_direction["bin_total"] is None


def test_nic06_frames_preserve_direction_identity_and_duplicate_collectors():
    store = make_frame_store(
        duplicate_collectors=True,
        collector_b_overrides={"hni_tx_ok_27": (0, 2, 6)},
    )
    view = build_view_model(store, frame_filters("tx"))
    frames = view["nic_detail"]["frames"]
    collector_records = frames["collector_analyses"]
    first_tx_bin = next(
        bin_result
        for direction in collector_records[0]["direction_results"]
        if direction["direction"] == "tx"
        for bin_result in direction["bin_results"]
        if bin_result["bin_id"] == "27"
    )
    second_tx_bin = next(
        bin_result
        for direction in collector_records[1]["direction_results"]
        if direction["direction"] == "tx"
        for bin_result in direction["bin_results"]
        if bin_result["bin_id"] == "27"
    )

    assert frames["collector_count"] == 2
    assert frames["duplicate_stream_count"] == 2
    assert len(collector_records) <= MAX_FRAME_COLLECTORS
    assert {
        record["population"]["collector_registry_id"]
        for record in collector_records
    } == {620, 621}
    assert first_tx_bin["collector_id"] != second_tx_bin["collector_id"]
    assert first_tx_bin["source_counter_id"] == second_tx_bin["source_counter_id"]
    assert first_tx_bin["total"] == 3
    assert second_tx_bin["total"] == 6

    rx_only = build_view_model(
        store,
        AnalysisFilterState(
            node_id=FRAME_NODE.node_id,
            nic_id=FRAME_NIC.nic_id,
            direction="rx",
            start_ns=0,
            end_ns=2_000_000_000,
        ),
    )
    frame_directions = {
        direction["direction"]
        for collector in rx_only["nic_detail"]["frames"]["collector_analyses"]
        for direction in collector["direction_results"]
    }
    assert frame_directions == {"rx"}


def test_nic06_frames_require_sourced_edges_and_population_compatibility(monkeypatch):
    view = build_view_model(make_frame_store(), frame_filters("tx"))
    tx = next(
        direction
        for direction in view["nic_detail"]["frames"]["collector_analyses"][0][
            "direction_results"
        ]
        if direction["direction"] == "tx"
    )
    assert tx["bin_edges_source"] is None
    assert tx["bin_edges_complete"] is False
    assert tx["mean_state"] == "unsupported"
    assert all(
        quantile["state"] == "unsupported"
        for quantile in tx["quantile_results"]
    )

    monkeypatch.setattr(
        analysis_app,
        "_FRAME_BIN_EDGE_SETS",
        (frame_edges(FrameDirection.TX), frame_edges(FrameDirection.RX)),
    )
    monkeypatch.setattr(
        analysis_app,
        "_FRAME_BYTE_COMPATIBILITY",
        (frame_compatibility(FrameDirection.TX),),
    )
    compatible = build_view_model(make_frame_store(), frame_filters("tx"))
    compatible_tx = next(
        direction
        for direction in compatible["nic_detail"]["frames"]["collector_analyses"][0][
            "direction_results"
        ]
        if direction["direction"] == "tx"
    )
    assert compatible_tx["bin_edges_complete"] is True
    assert compatible_tx["mean_state"] == FrameMeanStatus.SUPPORTED.value
    assert compatible_tx["mean_wire_frame_size_bytes"] == 16 / 36
    assert compatible_tx["approximate_mean_wire_frame_size_bytes"] is not None
    assert compatible_tx["quantile_results"][1]["state"] == FrameQuantileStatus.BOUNDED.value

    monkeypatch.setattr(
        analysis_app,
        "_FRAME_BIN_EDGE_SETS",
        (
            frame_edges(FrameDirection.TX, open_ended=True),
            frame_edges(FrameDirection.RX, open_ended=True),
        ),
    )
    open_ended = build_view_model(make_frame_store(), frame_filters("tx"))
    open_tx = next(
        direction
        for direction in open_ended["nic_detail"]["frames"]["collector_analyses"][0][
            "direction_results"
        ]
        if direction["direction"] == "tx"
    )
    assert open_tx["open_ended_tail"] is True
    assert open_tx["approximate_mean_wire_frame_size_bytes"] is None
    assert open_tx["quantile_results"][1]["state"] == FrameQuantileStatus.OPEN_ENDED.value
    assert open_tx["quantile_results"][1]["upper_bytes"] is None

    monkeypatch.setattr(
        analysis_app,
        "_FRAME_BYTE_COMPATIBILITY",
        (frame_compatibility(FrameDirection.TX, compatible=False),),
    )
    incompatible = build_view_model(make_frame_store(), frame_filters("tx"))
    incompatible_tx = next(
        direction
        for direction in incompatible["nic_detail"]["frames"]["collector_analyses"][0][
            "direction_results"
        ]
        if direction["direction"] == "tx"
    )
    assert incompatible_tx["mean_state"] == FrameMeanStatus.UNSUPPORTED.value
    assert incompatible_tx["mean_wire_frame_size_bytes"] is None
    assert (
        FrameQualityFlag.INCOMPATIBLE_BYTE_COUNT_POPULATION.value
        in incompatible_tx["quality_flags"]
    )


def test_nic06_rejects_gapped_bin_edge_sets():
    with pytest.raises(ValueError, match="no size-domain gap"):
        FrameBinEdgeSet(
            direction=FrameDirection.TX,
            source="fixture with an internal gap",
            edges=(
                FrameBinEdge("small", 0, 100, False),
                FrameBinEdge("large", 200, 300, False),
            ),
        )

    with pytest.raises(ValueError, match="first bin lower edge must be zero"):
        FrameBinEdgeSet(
            direction=FrameDirection.TX,
            source="fixture with a missing initial edge",
            edges=(
                FrameBinEdge("medium", 100, 200, False),
                FrameBinEdge("large", 200, 300, False),
            ),
        )


def test_nic06_frames_renders_bounded_summary():
    rendered = _render_nic_detail(
        build_view_model(make_frame_store(), frame_filters("tx"))
    )
    rendered_ids = component_ids(rendered)
    rendered_json = str(rendered.to_plotly_json())

    assert "nic-06-frames-heading" in rendered_ids
    assert "NIC-06 wire-frame sizes" in rendered_json
    assert "no MPI-message, payload, allreduce-size, or causal claim" in rendered_json
    assert "Zero, missing, insufficient, unsupported" in rendered_json
    assert str(MAX_FRAME_BIN_RESULTS) in rendered_json


def test_nic06_frames_export_is_bounded_without_raw_samples():
    view = build_view_model(
        make_frame_store(duplicate_collectors=True), frame_filters("all")
    )
    document = build_export_document(view)
    frames = document["nic_detail"]["frames"]
    payload = serialize_export_document(document)
    parsed = json.loads(payload)

    assert len(frames["collector_analyses"]) <= MAX_FRAME_COLLECTORS
    assert sum(
        len(direction["bin_results"])
        for collector in frames["collector_analyses"]
        for direction in collector["direction_results"]
    ) <= MAX_FRAME_BIN_RESULTS
    assert sum(
        len(direction["quantile_results"])
        for collector in frames["collector_analyses"]
        for direction in collector["direction_results"]
    ) <= MAX_FRAME_QUANTILE_RESULTS
    assert frames["bin_truncated"] is True

    def keys(value):
        if isinstance(value, dict):
            yield from value.keys()
            for child in value.values():
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    frame_keys = set(keys(parsed["nic_detail"]["frames"]))
    assert not {
        "intervals",
        "temporal_intervals",
        "values",
        "samples",
        "timestamps_ns",
        "display_values",
        "display_timestamps_ns",
        "raw_store",
        "uint64_values",
        "float64_values",
    }.intersection(frame_keys)
    assert len(payload.encode()) <= MAX_EXPORT_JSON_BYTES


def test_nic07_translation_distinguishes_missing_zero_nonzero_and_reset():
    missing_view = build_view_model(make_store())
    assert missing_view["nic_detail"]["translation"]["state"] == "missing_data"
    assert missing_view["nic_detail"]["translation"]["status_reasons"]

    view = build_view_model(make_translation_store(), translation_filters())
    translation = view["nic_detail"]["translation"]
    collector = translation["collector_analyses"][0]
    views = {
        result["miss_view"]: result
        for result in collector["view_results"]
    }
    counter_by_name = {
        result["counter"]: result
        for result in collector["counter_results"]
    }

    assert translation["state"] == TranslationStatus.OBSERVED_NONZERO.value
    assert views["pool"]["request_total"] == 15
    assert views["pool"]["hit_total"] == 12
    assert views["pool"]["miss_total"] == 3
    assert views["pool"]["hit_ratio"] == 12 / 15
    assert views["pool"]["misses_per_request"] == 3 / 15
    assert views["pool"]["traffic_total_gb"] == 20 / 1_000_000_000
    assert views["pool"]["misses_per_gb"] == 3 / (20 / 1_000_000_000)
    assert views["client"]["request_total"] == 15
    assert views["client"]["hit_total"] == 12
    assert views["client"]["miss_total"] == 5
    assert views["client"]["hit_ratio"] == 12 / 17
    assert views["client"]["misses_per_request"] == 5 / 15
    assert counter_by_name["atu_cache_evictions"]["total"] == 1
    assert counter_by_name["atu_client_req_ee"]["rate_per_second"] == 7.5

    zero_values = {
        name: (0, 0, 0)
        for name in TRANSLATION_VALUES
    }
    zero_store = make_translation_store(
        collector_a_overrides=zero_values
    )
    zero_view = build_view_model(zero_store, translation_filters("tx"))
    zero_collector = zero_view["nic_detail"]["translation"]["collector_analyses"][0]
    zero_views = {
        result["miss_view"]: result
        for result in zero_collector["view_results"]
    }
    assert zero_views["pool"]["state"] == TranslationStatus.OBSERVED_ZERO.value
    assert zero_views["pool"]["zero_observed"] is True
    assert zero_views["pool"]["hit_ratio"] is None
    assert TranslationQualityFlag.ZERO_DENOMINATOR.value in zero_views["pool"]["quality_flags"]
    assert zero_collector["eviction_zero_observed"] is True

    reset_store = make_translation_store(
        duplicate_collectors=True,
        collector_b_overrides={"atu_client_req_ee": (0, 10, 4)},
    )
    reset_view = build_view_model(reset_store, translation_filters())
    reset_collector = reset_view["nic_detail"]["translation"]["collector_analyses"][1]
    reset_request = next(
        result
        for result in reset_collector["counter_results"]
        if result["counter"] == "atu_client_req_ee"
    )
    reset_pool = next(
        result
        for result in reset_collector["view_results"]
        if result["miss_view"] == "pool"
    )
    assert reset_request["state"] == TranslationStatus.PARTIAL_INVALID_DATA.value
    assert reset_request["invalid_interval_count"] == 1
    assert TranslationQualityFlag.DECREASE_OR_RESET.value in reset_request["quality_flags"]
    assert reset_pool["request_denominator"] is None
    assert reset_pool["misses_per_request"] is None
    assert TranslationQualityFlag.INCOMPATIBLE_COVERAGE.value in reset_pool["quality_flags"]


def test_nic07_translation_keeps_unsupported_series_distinct_from_missing(monkeypatch):
    clear_view_model_cache()
    incompatible_specs = []
    for spec in analysis_app._TRANSLATION_SPECS:
        if spec.counter_id == "atu_cache_miss_ee":
            incompatible_specs.append(
                type(spec)(
                    counter_id=spec.counter_id,
                    role=spec.role,
                    unit="misses",
                    population_id=spec.population_id,
                    miss_view=spec.miss_view,
                )
            )
        else:
            incompatible_specs.append(spec)
    monkeypatch.setattr(
        analysis_app, "_TRANSLATION_SPECS", tuple(incompatible_specs)
    )

    view = build_view_model(make_translation_store(), translation_filters())
    translation = view["nic_detail"]["translation"]

    assert "atu_cache_miss_ee" in translation["unsupported_counter_names"]
    assert "atu_cache_miss_ee" not in translation["missing_counter_ids"]
    unsupported_counter = next(
        result
        for collector in translation["collector_analyses"]
        for result in collector["counter_results"]
        if result["counter"] == "atu_cache_miss_ee"
    )
    assert unsupported_counter["state"] == "unsupported"
    assert unsupported_counter["total"] is None
    client = next(
        result
        for collector in translation["collector_analyses"]
        for result in collector["view_results"]
        if result["miss_view"] == "client"
    )
    assert client["miss_total"] == 0
    assert TranslationQualityFlag.UNSUPPORTED_SERIES.value in client["quality_flags"]


def test_nic07_translation_preserves_identity_pool_client_views_and_duplicates():
    store = make_translation_store(
        duplicate_collectors=True,
        collector_b_overrides={
            "atu_client_req_ee": (0, 12, 18),
            "atu_cache_hit_base_page_size_0": (0, 9, 14),
            "atu_cache_miss_0": (0, 3, 4),
            "atu_cache_miss_ee": (0, 5, 6),
            "atu_cache_evictions": (0, 0, 2),
        },
    )
    view = build_view_model(store, translation_filters("all"))
    translation = view["nic_detail"]["translation"]
    collector_records = translation["collector_analyses"]

    assert translation["collector_count"] == 2
    assert translation["duplicate_stream_count"] == 2
    assert {
        record["population"]["collector_registry_id"]
        for record in collector_records
    } == {720, 721}
    for record in collector_records:
        assert record["population"]["run_id"] == TRANSLATION_RUN.run_id
        assert record["population"]["node_id"] == TRANSLATION_NODE.node_id
        assert record["population"]["nic_id"] == TRANSLATION_NIC.nic_id
        view_names = [result["miss_view"] for result in record["view_results"]]
        assert view_names == ["pool", "client"]
        pool = next(
            result for result in record["view_results"] if result["miss_view"] == "pool"
        )
        client = next(
            result for result in record["view_results"] if result["miss_view"] == "client"
        )
        assert pool["miss_total"] != client["miss_total"]
        assert pool["hit_ratio"] != client["hit_ratio"]
        assert not hasattr(record, "combined_miss_total")
    first_pool = next(
        result
        for result in collector_records[0]["view_results"]
        if result["miss_view"] == "pool"
    )
    second_pool = next(
        result
        for result in collector_records[1]["view_results"]
        if result["miss_view"] == "pool"
    )
    assert first_pool["request_total"] == 15
    assert second_pool["request_total"] == 18
    assert first_pool["miss_total"] == 3
    assert second_pool["miss_total"] == 4

    tx_only = build_view_model(store, translation_filters("tx"))
    tx_traffic = next(
        result
        for collector in tx_only["nic_detail"]["translation"]["collector_analyses"]
        for result in collector["view_results"]
        if result["miss_view"] == "pool"
    )
    assert tx_traffic["traffic_direction"] == "tx"
    assert tx_traffic["traffic_total_gb"] == 10 / 1_000_000_000


def test_nic07_translation_requires_compatible_populations(monkeypatch):
    original_specs = analysis_app._TRANSLATION_SPECS
    incompatible_specs = []
    for spec in original_specs:
        if spec.counter_id == "atu_cache_miss_0":
            incompatible_specs.append(
                type(spec)(
                    counter_id=spec.counter_id,
                    role=spec.role,
                    unit=spec.unit,
                    population_id="incompatible-fixture-population",
                    miss_view=spec.miss_view,
                )
            )
        else:
            incompatible_specs.append(spec)
    monkeypatch.setattr(
        analysis_app,
        "_TRANSLATION_SPECS",
        tuple(incompatible_specs),
    )
    view = build_view_model(make_translation_store(), translation_filters())
    pool = next(
        result
        for collector in view["nic_detail"]["translation"]["collector_analyses"]
        for result in collector["view_results"]
        if result["miss_view"] == "pool"
    )
    assert pool["hit_ratio"] is None
    assert pool["misses_per_request"] is None
    assert TranslationQualityFlag.INCOMPATIBLE_POPULATION.value in pool["quality_flags"]
    assert any(
        "incompatible" in reason.lower() for reason in pool["status_reasons"]
    )


def test_nic07_translation_renders_bounded_summary():
    rendered = _render_nic_detail(
        build_view_model(make_translation_store(), translation_filters())
    )
    rendered_ids = component_ids(rendered)
    rendered_json = str(rendered.to_plotly_json())

    assert "nic-07-translation-heading" in rendered_ids
    assert "NIC-07 address translation" in rendered_json
    assert "no ATU latency or causal claim" in rendered_json
    assert "Zero, missing, insufficient, unsupported" in rendered_json
    assert str(MAX_TRANSLATION_COUNTER_RESULTS) in rendered_json


def test_nic07_translation_export_is_bounded_without_raw_samples():
    view = build_view_model(
        make_translation_store(duplicate_collectors=True),
        translation_filters("all"),
    )
    document = build_export_document(view)
    translation = document["nic_detail"]["translation"]
    payload = serialize_export_document(document)
    parsed = json.loads(payload)

    assert len(translation["collector_analyses"]) <= MAX_TRANSLATION_COLLECTORS
    assert sum(
        len(collector["counter_results"])
        for collector in translation["collector_analyses"]
    ) <= MAX_TRANSLATION_COUNTER_RESULTS
    assert sum(
        len(collector["view_results"])
        for collector in translation["collector_analyses"]
    ) <= MAX_TRANSLATION_VIEW_RESULTS

    def keys(value):
        if isinstance(value, dict):
            yield from value.keys()
            for child in value.values():
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    translation_keys = set(keys(parsed["nic_detail"]["translation"]))
    assert not {
        "intervals",
        "temporal_intervals",
        "values",
        "samples",
        "timestamps_ns",
        "display_values",
        "display_timestamps_ns",
        "raw_store",
        "uint64_values",
        "float64_values",
    }.intersection(translation_keys)
    assert len(payload.encode()) <= MAX_EXPORT_JSON_BYTES


def test_nic08_latency_distinguishes_missing_unsourced_and_reset():
    missing_view = build_view_model(make_store(), AnalysisFilterState(
        node_id=NODE_A.node_id,
        nic_id=NIC_A.nic_id,
        direction="all",
        start_ns=0,
        end_ns=60_000_000_000,
    ))
    missing_latency = missing_view["nic_detail"]["latency"]
    assert missing_latency["state"] == "missing"
    assert len(missing_latency["missing_counter_ids"]) == 48
    assert missing_latency["collector_analyses"] == ()

    view = build_view_model(make_latency_store(), latency_filters())
    latency = view["nic_detail"]["latency"]
    collector = latency["collector_analyses"][0]
    assert latency["state"] == LatencyStatus.NONZERO_NAMED_BINS_ONLY.value
    assert latency["bin_metadata_status"] == "unsourced"
    assert collector["population"]["direction"] == "not_directional"
    assert collector["bin_results"][0]["count"] == 15
    assert collector["bin_results"][1]["count"] == 7
    assert collector["total_observations"] is None
    assert collector["quantile_results"][0]["state"] == "unsupported"
    assert collector["mean_result"]["state"] == "unsupported"
    assert LatencyQualityFlag.MISSING_BIN_EDGES.value in collector["quality_flags"]
    assert LatencyQualityFlag.MISSING_EDGE_UNITS.value in collector["quality_flags"]

    reset_view = build_view_model(
        make_latency_store(
            duplicate_collectors=True,
            collector_b_overrides={"pct_req_rsp_latency_1": (7, 5, 0)}
        ),
        latency_filters(),
    )
    reset_collector = next(
        record
        for record in reset_view["nic_detail"]["latency"]["collector_analyses"]
        if record["population"]["collector_registry_id"]
        == LATENCY_COLLECTOR_B.collector_id
    )
    assert reset_collector["state"] == LatencyStatus.INVALID_DATA.value
    assert reset_collector["reset_excluded_interval_count"] == 2


def test_nic08_latency_uses_sourced_edges_without_inventing_atu_latency(monkeypatch):
    monkeypatch.setattr(
        analysis_app,
        "_LATENCY_BIN_DEFINITIONS",
        {
            LatencyHistogramGroup.REQUEST_RESPONSE: LatencyBinDefinition(
                definition_version=1,
                group=LatencyHistogramGroup.REQUEST_RESPONSE,
                bin_names=("latency-0", "latency-1"),
                encoding=LatencyBinEncoding.DISJOINT,
                source="hand-check fixture definition",
                bin_edges=(
                    LatencyBinEdge(lower=0, upper=10, open_tailed=False),
                    LatencyBinEdge(lower=10, upper=20, open_tailed=False),
                ),
                unit="nanoseconds",
            )
        },
    )

    view = build_view_model(
        make_latency_store(
            counter_names=("pct_req_rsp_latency_0", "pct_req_rsp_latency_1")
        ),
        latency_filters(),
    )
    collector = view["nic_detail"]["latency"]["collector_analyses"][0]

    assert collector["state"] == LatencyStatus.OBSERVED_NONZERO.value
    assert collector["total_observations"] == 22
    assert collector["bin_results"][0]["normalized_fraction"] == 15 / 22
    assert collector["quantile_results"][0]["bin"] == "latency-0"
    assert collector["quantile_results"][0]["lower_edge"] == 0
    assert collector["quantile_results"][0]["upper_edge"] == 10
    assert collector["mean_result"]["value"] == pytest.approx(90 / 11)
    assert collector["mean_result"]["unit"] == "nanoseconds"
    assert collector["atu_latency_substitute"] is False


def test_nic08_latency_preserves_duplicate_collectors_and_renders_bounded_summary():
    view = build_view_model(
        make_latency_store(duplicate_collectors=True), latency_filters()
    )
    latency = view["nic_detail"]["latency"]
    rendered = _render_nic_detail(view)
    rendered_ids = component_ids(rendered)
    rendered_json = str(rendered.to_plotly_json())

    assert latency["collector_count"] == 2
    assert len(latency["collector_analyses"]) == 2
    assert {
        record["population"]["collector_registry_id"]
        for record in latency["collector_analyses"]
    } == {
        LATENCY_COLLECTOR_A.collector_id,
        LATENCY_COLLECTOR_B.collector_id,
    }
    assert "nic-08-latency-heading" in rendered_ids
    assert "NIC-08 communication latency" in rendered_json
    assert "no ATU-latency or MPI-duration substitution" in rendered_json
    assert str(MAX_LATENCY_BIN_RESULTS) in rendered_json


def test_nic08_latency_export_is_bounded_without_raw_samples():
    view = build_view_model(
        make_latency_store(duplicate_collectors=True), latency_filters()
    )
    document = build_export_document(view)
    latency = document["nic_detail"]["latency"]
    payload = serialize_export_document(document)
    parsed = json.loads(payload)

    assert len(latency["collector_analyses"]) <= MAX_LATENCY_COLLECTORS
    assert sum(
        len(collector["bin_results"])
        for collector in latency["collector_analyses"]
    ) <= MAX_LATENCY_BIN_RESULTS
    assert sum(
        len(collector["quantile_results"])
        for collector in latency["collector_analyses"]
    ) <= MAX_LATENCY_QUANTILE_RESULTS

    def keys(value):
        if isinstance(value, dict):
            yield from value.keys()
            for child in value.values():
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    latency_keys = set(keys(parsed["nic_detail"]["latency"]))
    assert not {
        "intervals",
        "values",
        "samples",
        "timestamps_ns",
        "display_values",
        "display_timestamps_ns",
        "raw_store",
        "uint64_values",
        "float64_values",
    }.intersection(latency_keys)
    assert len(payload.encode()) <= MAX_EXPORT_JSON_BYTES
