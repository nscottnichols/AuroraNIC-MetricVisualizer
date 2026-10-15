"""Explicit capability contract for the catalog's 15 question families.

The contract separates what a dataset exposes from what an analysis can claim.
It is intentionally independent of Dash and trace-reader implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from counter_semantic_registry import (
    COUNTER_SEMANTIC_REGISTRY_VERSION,
    counter_semantic_definition,
)


CAPABILITY_MATRIX_VERSION = 1


@dataclass(frozen=True)
class CounterDescriptor:
    """Minimal semantic information needed to validate a required counter."""

    semantics: str
    unit: str
    registry_version: str = ""
    source: str = ""
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuestionCounterMetadata:
    """Counter and coverage metadata accepted in place of a TraceStore.

    ``counter_metadata`` is required to promote NIC-01 to supported when the
    input is not a TraceStore: names alone do not prove cumulative byte units.
    """

    requested_counter_names: Sequence[str] = ()
    available_counter_names: Sequence[str] = ()
    readable_counter_names: Sequence[str] = ()
    observed_counter_names: Sequence[str] = ()
    unreadable_counter_names: Sequence[str] = ()
    counter_metadata: Mapping[str, CounterDescriptor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "counter_metadata",
            MappingProxyType(dict(self.counter_metadata)),
        )


@runtime_checkable
class _CoverageLike(Protocol):
    requested_counter_names: Sequence[str]
    available_counter_names: Sequence[str]
    readable_counter_names: Sequence[str]
    observed_counter_names: Sequence[str]
    unreadable_counter_names: Sequence[str]


@runtime_checkable
class _TraceStoreLike(Protocol):
    coverage: _CoverageLike
    counters: Iterable[Any]


@dataclass(frozen=True)
class QuestionCapability:
    """One immutable capability row for a catalog question."""

    contract_version: int
    question_id: str
    question: str
    scope: str
    status: str
    required_counters: tuple[str, ...]
    available_required_counters: tuple[str, ...]
    missing_required_counters: tuple[str, ...]
    unreadable_required_counters: tuple[str, ...]
    implemented_analysis_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    scientific_limitations: tuple[str, ...]


_BYTES_TRANSFERRED = ("hni_sts_tx_ok_octets",)
_BYTES_RECEIVED = ("hni_sts_rx_ok_octets",)
_TX_FRAME_SIZES = (
    "hni_tx_ok_27",
    "hni_tx_ok_35",
    "hni_tx_ok_36_to_63",
    "hni_tx_ok_64",
    "hni_tx_ok_65_to_127",
    "hni_tx_ok_128_to_255",
    "hni_tx_ok_256_to_511",
    "hni_tx_ok_512_to_1023",
    "hni_tx_ok_1024_to_2047",
    "hni_tx_ok_2048_to_4095",
    "hni_tx_ok_4096_to_8191",
    "hni_tx_ok_8192_to_max",
)
_RX_FRAME_SIZES = (
    "hni_rx_ok_27",
    "hni_rx_ok_35",
    "hni_rx_ok_36_to_63",
    "hni_rx_ok_64",
    "hni_rx_ok_65_to_127",
    "hni_rx_ok_128_to_255",
    "hni_rx_ok_256_to_511",
    "hni_rx_ok_512_to_1023",
    "hni_rx_ok_1024_to_2047",
    "hni_rx_ok_2048_to_4095",
    "hni_rx_ok_4096_to_8191",
    "hni_rx_ok_8192_to_max",
)
_TX_PAUSED_CYCLES = tuple(f"hni_tx_paused_{priority}" for priority in range(8))
_RX_PAUSED_CYCLES = tuple(f"hni_rx_paused_{priority}" for priority in range(8))
_PAUSE_SENT = ("hni_pause_refresh", "hni_pause_sent")
_PAUSE_XOFF_SENT = tuple(f"hni_pause_xoff_sent_{priority}" for priority in range(8))
_PAUSE_RECV = tuple(f"hni_pause_recv_{priority}" for priority in range(8))
_FGFC = ("hni_fgfc_discard", "hni_fgfc_event_xoff")
_OXE_TX_STALL = tuple(f"oxe_stall_pbuf_bc_{index}" for index in range(10))
_IXE_HOST_STALL = ("ixe_dmawr_stall_np_cdt", "ixe_dmawr_stall_p_cdt")
_IXE_DISP_STALL = (
    "ixe_disp_stall_atu_cdt",
    "ixe_disp_stall_atu_fifo",
    "ixe_disp_stall_conflict",
    "ixe_disp_stall_mdid_cdts",
    "ixe_disp_stall_put_resp",
    "ixe_disp_stall_resp_fifo",
)
_PI_POSTED_PKTS = ("parbs_tarb_pi_posted_pkts",)
_PI_NONPOSTED_PKTS = ("parbs_tarb_pi_non_posted_pkts",)
_PI_POSTED_BLOCKED = ("parbs_tarb_pi_posted_blocked_cnt",)
_PI_NONPOSTED_BLOCKED = ("parbs_tarb_pi_non_posted_blocked_cnt",)
_CQ_DMA_CMD_COUNTS = tuple(f"cq_dma_cmd_counts_{index}" for index in range(16))
_CQ_NUM_DMA_CMDS = tuple(f"cq_num_dma_cmds_{index}" for index in range(4))
_CQ_NUM_IDC_CMDS = tuple(f"cq_num_idc_cmds_{index}" for index in range(4))
_OVERFLOW_MATCH = tuple(f"lpe_net_match_overflow_{priority}" for priority in range(4))
_PRIORITY_MATCH = tuple(f"lpe_net_match_priority_{priority}" for priority in range(4))
_ATU_MISS_BY_POOL = tuple(f"atu_cache_miss_{index}" for index in range(4))
_ATU_MISS_BY_CLIENT = ("atu_cache_miss_ee", "atu_cache_miss_ixe", "atu_cache_miss_oxe")
_ATU_HITS = (
    *(f"atu_cache_hit_base_page_size_{index}" for index in range(4)),
    *(f"atu_cache_hit_derivative1_page_size_{index}" for index in range(4)),
    *(f"atu_cache_hit_derivative2_page_size_{index}" for index in range(4)),
)
_ATU_EVICTIONS = ("atu_cache_evictions",)
_ATU_CLIENT_REQ = ("atu_client_req_ee", "atu_client_req_ixe", "atu_client_req_oxe")
_REQ_RSP_LATENCY = tuple(f"pct_req_rsp_latency_{index}" for index in range(32))
_HOST_ACCESS_LATENCY = tuple(f"pct_host_access_latency_{index}" for index in range(16))
_LINK_RELIABILITY = (
    "hni_llr_rx_replay_event",
    "hni_llr_tx_replay_event",
    "hni_pcs_corrected_cw",
    "hni_pcs_good_cw",
    "hni_pcs_uncorrected_cw",
)
_PCT_ERRORS = (
    "pct_no_mst_nacks",
    "pct_no_tct_nacks",
    "pct_no_trs_nacks",
    "pct_retry_srb_requests",
    "pct_sct_timeouts",
    "pct_spt_timeouts",
    "pct_trs_rsp_nack_drops",
)
_BAD_OCTETS = ("hni_sts_rx_bad_octets", "hni_sts_tx_bad_octets")
_DISCARDS = (
    *(f"hni_discard_cntr_{index}" for index in range(8)),
    "ixe_rx_pkt_drop_ixe_parser",
    "ixe_rx_pkt_drop_pct",
    "ixe_rx_pkt_drop_rmu_norsp",
    "ixe_rx_pkt_drop_rmu_wrsp",
)


_COUNTER_GROUPS: dict[str, tuple[str, ...]] = {
    "bytes_transferred": _BYTES_TRANSFERRED,
    "bytes_received": _BYTES_RECEIVED,
    "tx_frame_sizes": _TX_FRAME_SIZES,
    "rx_frame_sizes": _RX_FRAME_SIZES,
    "tx_paused_cycles": _TX_PAUSED_CYCLES,
    "rx_paused_cycles": _RX_PAUSED_CYCLES,
    "pause_sent": _PAUSE_SENT,
    "pause_xoff_sent": _PAUSE_XOFF_SENT,
    "pause_recv": _PAUSE_RECV,
    "fgfc": _FGFC,
    "oxe_tx_stall": _OXE_TX_STALL,
    "ixe_host_stall": _IXE_HOST_STALL,
    "ixe_disp_stall": _IXE_DISP_STALL,
    "pi_posted_pkts": _PI_POSTED_PKTS,
    "pi_nonposted_pkts": _PI_NONPOSTED_PKTS,
    "pi_posted_blocked": _PI_POSTED_BLOCKED,
    "pi_nonposted_blocked": _PI_NONPOSTED_BLOCKED,
    "cq_dma_cmd_counts": _CQ_DMA_CMD_COUNTS,
    "cq_num_dma_cmds": _CQ_NUM_DMA_CMDS,
    "cq_num_idc_cmds": _CQ_NUM_IDC_CMDS,
    "overflow_match": _OVERFLOW_MATCH,
    "priority_match": _PRIORITY_MATCH,
    "atu_miss_by_pool": _ATU_MISS_BY_POOL,
    "atu_miss_by_client": _ATU_MISS_BY_CLIENT,
    "atu_hits": _ATU_HITS,
    "atu_evictions": _ATU_EVICTIONS,
    "atu_client_req": _ATU_CLIENT_REQ,
    "req_rsp_latency": _REQ_RSP_LATENCY,
    "host_access_latency": _HOST_ACCESS_LATENCY,
    "link_reliability": _LINK_RELIABILITY,
    "pct_errors": _PCT_ERRORS,
    "bad_octets": _BAD_OCTETS,
    "discards": _DISCARDS,
}

_NIC_01_COUNTERS = _BYTES_TRANSFERRED + _BYTES_RECEIVED
_NIC_02_COUNTERS = (
    _TX_PAUSED_CYCLES
    + _RX_PAUSED_CYCLES
    + _PAUSE_SENT
    + _PAUSE_XOFF_SENT
    + _PAUSE_RECV
    + _FGFC
)
_NIC_03_COUNTERS = _PRIORITY_MATCH + _OVERFLOW_MATCH
_NIC_04_COUNTERS = (
    _PI_POSTED_PKTS
    + _PI_NONPOSTED_PKTS
    + _PI_POSTED_BLOCKED
    + _PI_NONPOSTED_BLOCKED
)
_NIC_05_COUNTERS = _CQ_DMA_CMD_COUNTS + _CQ_NUM_DMA_CMDS + _CQ_NUM_IDC_CMDS
_NIC_06_COUNTERS = _TX_FRAME_SIZES + _RX_FRAME_SIZES + _NIC_01_COUNTERS
_NIC_07_COUNTERS = (
    _ATU_CLIENT_REQ
    + _ATU_HITS
    + _ATU_MISS_BY_POOL
    + _ATU_MISS_BY_CLIENT
    + _ATU_EVICTIONS
    + _TX_FRAME_SIZES
    + _RX_FRAME_SIZES
    + _NIC_01_COUNTERS
)
_NIC_08_COUNTERS = _REQ_RSP_LATENCY + _HOST_ACCESS_LATENCY
_NIC_09_COUNTERS = _LINK_RELIABILITY + _PCT_ERRORS + _BAD_OCTETS + _DISCARDS


_ALL_DIAGNOSTIC_COUNTERS = (
    _NIC_01_COUNTERS
    + _NIC_02_COUNTERS
    + _NIC_03_COUNTERS
    + _NIC_04_COUNTERS
    + _NIC_05_COUNTERS
    + _NIC_06_COUNTERS
    + _ATU_CLIENT_REQ
    + _ATU_HITS
    + _ATU_MISS_BY_POOL
    + _ATU_MISS_BY_CLIENT
    + _ATU_EVICTIONS
    + _NIC_08_COUNTERS
    + _NIC_09_COUNTERS
)
_ALL_DIAGNOSTIC_COUNTERS = tuple(dict.fromkeys(_ALL_DIAGNOSTIC_COUNTERS))


@dataclass(frozen=True)
class _QuestionDefinition:
    question_id: str
    question: str
    scope: str
    required_counters: tuple[str, ...]
    implemented_analysis_ids: tuple[str, ...]
    base_reasons: tuple[str, ...]
    scientific_limitations: tuple[str, ...]


_QUESTION_DEFINITIONS: tuple[_QuestionDefinition, ...] = (
    _QuestionDefinition(
        question_id="NIC-01",
        question="How much traffic is the NIC injecting and receiving?",
        scope="NIC",
        required_counters=_NIC_01_COUNTERS,
        implemented_analysis_ids=(
            "nic01.traffic_totals",
            "nic01.weighted_rates",
            "nic01.rate_distribution",
            "nic01.time_series",
        ),
        base_reasons=(),
        scientific_limitations=(
            "Totals and rates use adjacent observed samples and measured intervals; gaps, resets, wrap suspects, duplicate or nonmonotonic timestamps, and partial windows are not treated as zero.",
            "Change-only or baseline-relative traces cannot establish a fixed cadence, the initial sample, unchanged samples, or sub-observation activity; temporal statistics are observation-interval statistics only.",
            "Zero traffic is an observed zero only when valid samples exist; missing counters or samples remain missing.",
            "TX/RX byte counters are wire-side octets, not MPI message or payload counts.",
            "FCS/replay accounting and register-specific reset domains remain unresolved.",
            "OK and BAD octet counters are separate; BAD octets are not traffic totals.",
            "Counter decreases are conservatively excluded; 56-bit wrap remains unverified without an independent signal.",
        ),
    ),
    _QuestionDefinition(
        question_id="NIC-02",
        question="Is the NIC experiencing network congestion or flow-control pressure?",
        scope="NIC",
        required_counters=_NIC_02_COUNTERS,
        implemented_analysis_ids=("nic02.flow_control_analysis",),
        base_reasons=(
            "Flow-control analysis is implemented, but authoritative metadata for every required counter is incomplete.",
            "The catalog has no explicitly identified ECN counter.",
            "Cycle-to-time conversion and clock provenance are not verified.",
        ),
        scientific_limitations=(
            "The PDF's 1 GHz flow-control cycle label must be verified for each capture and does not apply to every cycle counter.",
            "Summing per-priority pause fractions is not an NIC-wide wall-clock pause fraction.",
            "Pressure indicators are not proof of a unique causal network bottleneck, and ECN observations cannot be claimed without an identified ECN counter.",
        ),
    ),
    _QuestionDefinition(
        question_id="NIC-03",
        question="How are incoming messages being received?",
        scope="NIC",
        required_counters=_NIC_03_COUNTERS,
        implemented_analysis_ids=("nic03.matching_analysis",),
        base_reasons=(
            "Priority and overflow match analysis is implemented, but authoritative population-compatibility metadata is incomplete.",
            "Subcounter overlap and exact receive interpretation are not verified.",
        ),
        scientific_limitations=(
            "No matches gives an undefined fraction, not 0% or 100%.",
            "Priority/overflow matches are evidence about receive handling, not measured copy time or exact MPI message/receive semantics.",
            "Counters must be proved compatible before they are summed.",
        ),
    ),
    _QuestionDefinition(
        question_id="NIC-04",
        question="Is the host/PCIe path limiting NIC performance?",
        scope="NIC",
        required_counters=_NIC_04_COUNTERS,
        implemented_analysis_ids=("nic04.host_pressure_analysis",),
        base_reasons=(
            "Host-interface pressure analysis is implemented, but authoritative blocked-counter unit metadata is incomplete.",
            "Host blocked-counter units are unresolved.",
        ),
        scientific_limitations=(
            "The source describes blocked cycles and cycles/packets ambiguously; authoritative unit definitions are required before ratio or time labels.",
            "Correlation with throughput is not proof that the host/PCIe path is the limiting resource.",
            "DMA and dispatcher stalls must remain distinct from host/PCIe evidence.",
        ),
    ),
    _QuestionDefinition(
        question_id="NIC-05",
        question="What communication protocol/operation is the NIC using?",
        scope="NIC",
        required_counters=_NIC_05_COUNTERS,
        implemented_analysis_ids=("nic05.operation_analysis",),
        base_reasons=(
            "Command-queue analysis is implemented, but authoritative opcode mappings are incomplete.",
            "An authoritative opcode mapping is not supplied.",
            "Driver documentation and MPI/application traces are needed for eager/rendezvous interpretation.",
        ),
        scientific_limitations=(
            "Command queues are control path only and are not wire bytes or MPI payload volumes.",
            "Numeric opcode indices remain unmapped rather than assigned guessed operations.",
            "The catalog does not establish a universal eager/rendezvous ratio.",
        ),
    ),
    _QuestionDefinition(
        question_id="NIC-06",
        question="What is the packet-size behavior?",
        scope="NIC",
        required_counters=_NIC_06_COUNTERS,
        implemented_analysis_ids=("nic06.frame_analysis",),
        base_reasons=(
            "Wire frame-size distribution analysis is implemented, but authoritative histogram metadata is incomplete.",
            "Histogram bin edges and population compatibility are not sourced.",
        ),
        scientific_limitations=(
            "Wire frame-size distributions are not MPI message sizes or allreduce message sizes.",
            "Exact mean frame size requires compatible byte and frame populations; midpoint estimates are approximate.",
            "An open-ended maximum bin precludes an exact mean from bins alone, and zero frames leaves a ratio undefined rather than zero.",
        ),
    ),
    _QuestionDefinition(
        question_id="NIC-07",
        question="Is address translation becoming a communication bottleneck?",
        scope="NIC",
        required_counters=_NIC_07_COUNTERS,
        implemented_analysis_ids=("nic07.translation_analysis",),
        base_reasons=(
            "ATU analysis is implemented, but authoritative miss-population metadata is incomplete.",
            "No explicitly identified ATU-latency histogram group or bin mapping is supplied.",
            "Pool/client miss overlap and hit/miss population compatibility are not verified.",
        ),
        scientific_limitations=(
            "Pool and client miss views must not be summed without proof that they are disjoint.",
            "Hit/miss ratios require matching populations, and zero traffic makes misses per GB or packet unavailable rather than zero.",
            "Request/response and host-access latency histograms must not be relabeled as ATU latency.",
        ),
    ),
    _QuestionDefinition(
        question_id="NIC-08",
        question="What is the communication latency behavior?",
        scope="NIC",
        required_counters=_NIC_08_COUNTERS,
        implemented_analysis_ids=("nic08.latency_analysis",),
        base_reasons=(
            "Latency histogram analysis is implemented, but authoritative histogram metadata is incomplete.",
            "Histogram bin edges, bin semantics, and time conversion factors are not sourced.",
        ),
        scientific_limitations=(
            "Without verified edges and units, bin-index counts must be shown rather than invented nanoseconds or P99 values.",
            "Compatible cumulative bins must be differenced over time after confirming whether they are exclusive or threshold buckets.",
            "NIC histograms and MPI API durations are distinct populations, and per-NIC P99 values must not be averaged into a job P99.",
        ),
    ),
    _QuestionDefinition(
        question_id="NIC-09",
        question="Are there reliability or error problems?",
        scope="NIC",
        required_counters=_NIC_09_COUNTERS,
        implemented_analysis_ids=("nic09.reliability_analysis",),
        base_reasons=(
            "Reliability and error analysis is implemented, but authoritative counter-category metadata is incomplete.",
            "Counter definitions, corrected/uncorrected categories, and count overlap are not verified.",
        ),
        scientific_limitations=(
            "All-zero observations remain visible as invariants but are not a guarantee about future runs.",
            "Missing counters or samples are not zero-valued observations.",
            "hni_pcs_good_cw is not an error merely because of its supplied grouping; its definition must be verified.",
        ),
    ),
    _QuestionDefinition(
        question_id="NODE-01",
        question="Are the NICs load balanced?",
        scope="node",
        required_counters=_NIC_01_COUNTERS,
        implemented_analysis_ids=("node01.nic_balance",),
        base_reasons=(
            "NIC load-balance analysis is implemented for compatible TX/RX traffic results.",
            "Observed NIC inventory and common-window coverage are not established by this contract.",
        ),
        scientific_limitations=(
            "The effective-NIC-count formula 1/sum(share_i^2) is a project choice because the PDF does not define it.",
            "Unknown or missing NICs are not inactive, and a zero total leaves shares and effective count undefined.",
            "Comparisons require a common valid window and declared denominators; an idle denominator must not be hidden in max/min ratios.",
        ),
    ),
    _QuestionDefinition(
        question_id="NODE-02",
        question="Is the behavior consistent across NICs?",
        scope="node",
        required_counters=_ALL_DIAGNOSTIC_COUNTERS,
        implemented_analysis_ids=("node02.nic_traffic_consistency",),
        base_reasons=(
            "Cross-NIC traffic consistency is implemented; full diagnostic comparison is not.",
            "Comparable populations, windows, and per-metric units are not established.",
        ),
        scientific_limitations=(
            "A full-trace average on one NIC must not be compared with a partial window on another.",
            "A unitless combined bad-NIC score requires a defined, explainable model; visual correlation is not causation.",
            "Missing counters must remain visible as a missing-data pattern.",
        ),
    ),
    _QuestionDefinition(
        question_id="JOB-01",
        question="Are there spatial/topological hotspots?",
        scope="job",
        required_counters=_NIC_01_COUNTERS,
        implemented_analysis_ids=(
            "job01.node_hotspots",
            "job01.optional_topology_groups",
        ),
        base_reasons=(
            "Topology and communication-edge metadata are not available from counter coverage.",
            "Node-level hotspot analysis is implemented; peer/near-far topology claims are not.",
        ),
        scientific_limitations=(
            "Node names alone do not establish switch connectivity, physical placement, communication peers, or distance.",
            "Unknown topology and edges remain unknown; a node-only traffic view must not be presented as a near/far or peer claim.",
            "Optional cabinet/rack/switch/region grouping requires a user-provided validated mapping.",
        ),
    ),
    _QuestionDefinition(
        question_id="JOB-02",
        question="Is communication spatially balanced?",
        scope="job",
        required_counters=_NIC_01_COUNTERS,
        implemented_analysis_ids=("job02.node_balance",),
        base_reasons=(
            "Whole-job node balance and concentration analysis is implemented.",
            "Valid topology groups, complete NIC/node inventory, and common windows are not established.",
        ),
        scientific_limitations=(
            "Compatible traffic totals must be conserved without counting a second-format copy or overlapping group membership twice.",
            "Top-k concentration and every denominator must be defined in exports.",
            "Missing or unmapped nodes remain visible and are not treated as zero traffic.",
        ),
    ),
    _QuestionDefinition(
        question_id="JOB-03",
        question="Does NIC behavior change over time?",
        scope="job",
        required_counters=_NIC_01_COUNTERS,
        implemented_analysis_ids=("job03.manual_phase_analysis",),
        base_reasons=(
            "Manual phase exploration and change analysis are implemented.",
            "Verified clock alignment and optional MPI/benchmark event annotations are not available from counter coverage.",
        ),
        scientific_limitations=(
            "Sub-sample behavior, alignment, and MPI phases must not be invented.",
            "Automated phase boundaries are parameterized estimates with uncertainty.",
            "A legacy before/after pair cannot identify a time phase or explain a counter change.",
        ),
    ),
    _QuestionDefinition(
        question_id="JOB-04",
        question="What communication characteristics does the job exhibit?",
        scope="job",
        required_counters=_ALL_DIAGNOSTIC_COUNTERS,
        implemented_analysis_ids=(),
        base_reasons=(
            "The evidence-backed job communication profile is not implemented.",
            "Configurable thresholds, comparison rules, and application/MPI metadata are not established.",
        ),
        scientific_limitations=(
            "Wire-frame evidence is not true message-size metadata, and request counts are not measured translation time.",
            "Correlations are not causes, and the profile must expose unavailable dimensions rather than issue a black-box bottleneck diagnosis.",
            "High/low thresholds are configurable project choices; one allreduce is only a workflow smoke test for this application.",
        ),
    ),
)


def _semantic_name(value: Any) -> str:
    return str(getattr(value, "value", value))


def _counter_descriptors(counters: Any) -> dict[str, CounterDescriptor]:
    if isinstance(counters, Mapping):
        items = counters.items()
    else:
        items = ((counter.name, counter) for counter in counters)

    descriptors: dict[str, CounterDescriptor] = {}
    for name, counter in items:
        if not isinstance(name, str) or not name:
            raise ValueError("counter names must be nonempty strings")
        if name in descriptors:
            raise ValueError(f"duplicate counter metadata for {name!r}")
        if isinstance(counter, Mapping):
            semantics = counter["semantics"]
            unit = counter["unit"]
            registry_version = counter.get("registry_version", "")
            source = counter.get("source", "")
            caveats = tuple(counter.get("caveats", ()))
        else:
            semantics = counter.semantics
            unit = counter.unit
            registry_version = getattr(counter, "registry_version", "")
            source = getattr(counter, "source", "")
            caveats = tuple(getattr(counter, "caveats", ()))
        definition = counter_semantic_definition(name)
        if definition is not None:
            registry_version = definition.registry_version
            source = definition.source
            caveats = definition.caveats
        descriptors[name] = CounterDescriptor(
            semantics=_semantic_name(semantics),
            unit=str(unit),
            registry_version=str(registry_version),
            source=str(source),
            caveats=caveats,
        )
    return descriptors


def _metadata_from_source(
    source: _TraceStoreLike | _CoverageLike,
) -> QuestionCounterMetadata:
    if hasattr(source, "coverage"):
        coverage = source.coverage
        counters = getattr(source, "counters", ())
    else:
        coverage = source
        counters = getattr(source, "counter_metadata", {})

    return QuestionCounterMetadata(
        requested_counter_names=tuple(coverage.requested_counter_names),
        available_counter_names=tuple(coverage.available_counter_names),
        readable_counter_names=tuple(coverage.readable_counter_names),
        observed_counter_names=tuple(coverage.observed_counter_names),
        unreadable_counter_names=tuple(coverage.unreadable_counter_names),
        counter_metadata=_counter_descriptors(counters),
    )


def _counter_evidence_reasons(
    missing: tuple[str, ...], unreadable: tuple[str, ...]
) -> tuple[str, ...]:
    reasons: list[str] = []
    if missing:
        reasons.append("Missing required counters: " + ", ".join(missing) + ".")
    if unreadable:
        reasons.append("Unreadable required counters: " + ", ".join(unreadable) + ".")
    return tuple(reasons)


def _row(
    definition: _QuestionDefinition,
    metadata: QuestionCounterMetadata,
) -> QuestionCapability:
    available = set(metadata.available_counter_names)
    readable = set(metadata.readable_counter_names)
    observed = set(metadata.observed_counter_names)
    unreadable = set(metadata.unreadable_counter_names)
    required = definition.required_counters

    available_required = tuple(name for name in required if name in available)
    missing_required = tuple(
        name for name in required if name not in observed and name not in unreadable
    )
    unreadable_required = tuple(name for name in required if name in unreadable)
    evidence_reasons = _counter_evidence_reasons(missing_required, unreadable_required)

    if definition.question_id == "NIC-01":
        descriptors = [
            metadata.counter_metadata.get(name) for name in definition.required_counters
        ]
        cumulative_byte_counters = all(
            descriptor is not None
            and descriptor.semantics == "cumulative"
            and descriptor.unit == "bytes"
            and descriptor.registry_version == COUNTER_SEMANTIC_REGISTRY_VERSION
            for descriptor in descriptors
        )
        readable_and_observed = [
            name
            for name in definition.required_counters
            if name in readable and name in observed
        ]
        if cumulative_byte_counters and len(readable_and_observed) == len(required):
            status = "supported"
            reasons: tuple[str, ...] = (
                "Both required cumulative TX/RX byte counters are readable and observed.",
                "Supported by the HPE-sourced network-frame octet semantic registry.",
                *evidence_reasons,
            )
        else:
            status = "partial" if readable_and_observed else "unsupported"
            reasons = (
                "NIC-01 traffic analysis is implemented, but required counter evidence is incomplete.",
                *evidence_reasons,
            )
            if not metadata.counter_metadata:
                reasons += (
                    "Counter semantics were not supplied; cumulative byte-counter status is unverified.",
                )
            elif not cumulative_byte_counters:
                reasons += (
                    "Counter metadata does not identify both required counters as cumulative byte counters from the sourced octet registry.",
                )
    else:
        status = "partial" if any(name in observed for name in required) else "unsupported"
        reasons = definition.base_reasons
        if status == "partial":
            reasons += (
                "Partial state reflects observed counter evidence only; support requires all required counters and authoritative metadata.",
            )
        reasons += evidence_reasons

    return QuestionCapability(
        contract_version=CAPABILITY_MATRIX_VERSION,
        question_id=definition.question_id,
        question=definition.question,
        scope=definition.scope,
        status=status,
        required_counters=required,
        available_required_counters=available_required,
        missing_required_counters=missing_required,
        unreadable_required_counters=unreadable_required,
        implemented_analysis_ids=definition.implemented_analysis_ids,
        reasons=reasons,
        scientific_limitations=definition.scientific_limitations,
    )


def build_question_capability_matrix(
    source: _TraceStoreLike | _CoverageLike,
) -> tuple[QuestionCapability, ...]:
    """Return one immutable capability row for each catalog question ID.

    A TraceStore supplies both coverage and counter semantics. A metadata-only
    input can supply semantics through ``QuestionCounterMetadata.counter_metadata``;
    a name-only Coverage cannot promote NIC-01 to supported.
    """

    metadata = _metadata_from_source(source)
    return tuple(_row(definition, metadata) for definition in _QUESTION_DEFINITIONS)
