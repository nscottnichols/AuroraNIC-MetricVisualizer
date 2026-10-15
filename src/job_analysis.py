"""Pure job-level aggregation over validated per-node balance results."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import Enum
import math

from node_analysis import (
    MaxMinRatioDenominator,
    NodeBalanceStatus,
    NodeNicBalanceResult,
)
from traffic_analysis import TrafficDirection


TRAFFIC_SHARE_FORMULA = "node_total_bytes / job_total_bytes"
MAX_MIN_RATIO_FORMULA = "max(node_total_bytes) / min(node_total_bytes)"
EFFECTIVE_NODE_COUNT_FORMULA = "1 / sum((node_total_bytes / job_total_bytes)^2)"
SPATIAL_QUANTILE_INTERPOLATION = (
    "nearest-rank upper order statistic across node total bytes; "
    "temporal NIC rate quantiles are never averaged"
)
HOTSPOT_CONCENTRATION_FORMULA = (
    "sum of the largest max(1, ceil(0.01 * valid_node_count)) "
    "node traffic shares"
)
HOTSPOT_TOP_FRACTION = 0.01
COVERAGE_AGGREGATION = (
    "minimum node coverage_fraction; undefined when any expected node is missing"
)
TOPOLOGY_SHARE_FORMULA = "topology_group_total_bytes / job_total_bytes"
PEER_NEAR_FAR_REQUIREMENT = (
    "Communication-edge metadata with verified endpoints and clock alignment "
    "is required for peer or near/far claims."
)


class JobBalanceStatus(str, Enum):
    OK = "ok"
    ZERO_TRAFFIC = "zero_traffic"
    PARTIAL_DATA = "partial_data"
    MISSING_DATA = "missing_data"
    UNSUPPORTED_DATA = "unsupported_data"


class JobBalanceState(str, Enum):
    ZERO_TRAFFIC = "zero_traffic"
    MISSING_NODE = "missing_node"
    UNKNOWN_NODE = "unknown_node"
    MISSING_OR_INVALID_NODE = "missing_or_invalid_node"
    PARTIAL_NODE = "partial_node"
    PARTIAL_COVERAGE = "partial_coverage"
    UNSUPPORTED_NODE = "unsupported_node"
    BASELINE_RELATIVE = "baseline_relative"


class JobTopologyState(str, Enum):
    ABSENT = "absent"
    VALID = "valid"
    VALID_PARTIAL = "valid_partial"
    INVALID = "invalid"


@dataclass(frozen=True)
class JobNodeBalance:
    """One node's contribution to a single job direction."""

    node_id: str
    status: NodeBalanceStatus
    state_flags: frozenset
    status_reasons: tuple[str, ...]
    total_bytes: int | float | None
    traffic_share: float | None
    known_nic_count: int
    observed_nic_count: int
    active_nic_fraction: float | None
    coverage_fraction: float | None
    topology_group_id: str | None
    source_result: NodeNicBalanceResult


@dataclass(frozen=True)
class JobTopologyGroupBalance:
    """Traffic summary for one non-overlapping topology group."""

    topology_group_id: str
    node_ids: tuple[str, ...]
    missing_node_ids: tuple[str, ...]
    total_bytes: int | float | None
    traffic_share: float | None
    coverage_fraction: float | None
    node_count: int
    valid_node_count: int


@dataclass(frozen=True)
class JobNodeBalanceResult:
    """Explicit node-level job balance result for one traffic direction."""

    run_id: str | None
    direction: TrafficDirection | None
    source_counter: str | None
    status: JobBalanceStatus
    state_flags: frozenset[JobBalanceState]
    status_reasons: tuple[str, ...]
    common_window_ns: tuple[int, int] | None
    represented_window_ns: tuple[int, int] | None
    valid_interval_boundary_signature: tuple[int, ...] | None
    requested_duration_ns: int
    covered_duration_ns: int
    coverage_fraction: float | None
    known_node_ids: tuple[str, ...]
    observed_node_ids: tuple[str, ...]
    missing_node_ids: tuple[str, ...]
    unknown_observed_node_ids: tuple[str, ...]
    unavailable_node_ids: tuple[str, ...]
    partial_node_ids: tuple[str, ...]
    unsupported_node_ids: tuple[str, ...]
    known_node_count: int
    observed_node_count: int
    valid_node_count: int
    job_total_bytes: int | float | None
    per_node: tuple[JobNodeBalance, ...]
    max_traffic_bytes: int | float | None
    min_traffic_bytes: int | float | None
    max_min_ratio: float | None
    max_min_ratio_denominator: MaxMinRatioDenominator
    effective_node_count: float | None
    spatial_p50_bytes: int | float | None
    spatial_p99_bytes: int | float | None
    hotspot_concentration: float | None
    hotspot_node_count: int | None
    valid_interval_count: int
    invalid_interval_count: int
    baseline_relative_observation_count: int
    topology_state: JobTopologyState
    topology_status_reasons: tuple[str, ...]
    topology_supported: bool
    topology_groups: tuple[JobTopologyGroupBalance, ...]
    unknown_topology_node_ids: tuple[str, ...]
    duplicate_topology_node_ids: tuple[str, ...]
    peer_near_far_supported: bool
    peer_near_far_reason: str
    unsupported_dimensions: tuple[str, ...]
    traffic_share_formula: str
    max_min_ratio_formula: str
    effective_node_count_formula: str
    spatial_quantile_interpolation: str
    hotspot_concentration_formula: str
    coverage_aggregation: str
    topology_share_formula: str


TopologyGroups = Mapping[str, Collection[str]]


def _sum_bytes(values: list[int | float]) -> int | float:
    if any(type(value) is float for value in values):
        return math.fsum(float(value) for value in values)
    return sum(values, start=0)


def _nearest_rank_quantile(
    values: list[int | float], probability: float
) -> int | float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def _validate_topology(
    topology_groups: TopologyGroups | None,
) -> tuple[
    JobTopologyState,
    tuple[str, ...],
    tuple[tuple[str, tuple[str, ...]], ...],
    frozenset[str],
    tuple[str, ...],
]:
    if topology_groups is None:
        return (
            JobTopologyState.ABSENT,
            ("No topology mapping was supplied; node-only results are returned.",),
            (),
            frozenset(),
            (),
        )
    if not isinstance(topology_groups, Mapping):
        raise ValueError("topology_groups must be a mapping or None")

    normalized: list[tuple[str, tuple[str, ...]]] = []
    assignments: dict[str, list[str]] = {}
    for group_id, node_ids in topology_groups.items():
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("every topology group ID must be a nonempty string")
        if isinstance(node_ids, str) or not isinstance(node_ids, Collection):
            raise ValueError("topology group values must be node-ID collections")
        mapped_nodes = tuple(node_ids)
        if not mapped_nodes:
            return (
                JobTopologyState.INVALID,
                ("Topology groups must not be empty.",),
                (),
                frozenset(),
                (),
            )
        for node_id in mapped_nodes:
            if not isinstance(node_id, str) or not node_id:
                raise ValueError("every topology node ID must be a nonempty string")
            assignments.setdefault(node_id, []).append(group_id)
        normalized.append((group_id, mapped_nodes))

    duplicate_nodes = tuple(
        sorted(
            node_id
            for node_id, group_ids in assignments.items()
            if len(group_ids) > 1
        )
    )
    if duplicate_nodes:
        return (
            JobTopologyState.INVALID,
            (
                "Topology groups overlap; each node may be assigned to only one group.",
            ),
            (),
            frozenset(),
            duplicate_nodes,
        )

    mapped = frozenset(assignments)
    normalized.sort(key=lambda item: item[0])
    return JobTopologyState.VALID, (), tuple(normalized), mapped, ()


def _topology_group_results(
    normalized_groups: tuple[tuple[str, tuple[str, ...]], ...],
    node_results: Mapping[str, NodeNicBalanceResult],
    job_total: int | float | None,
    compatible: bool,
) -> tuple[JobTopologyGroupBalance, ...]:
    groups: list[JobTopologyGroupBalance] = []
    for group_id, mapped_nodes in normalized_groups:
        present_results = [
            node_results[node_id] for node_id in mapped_nodes if node_id in node_results
        ]
        usable_results = [
            result
            for result in present_results
            if result.node_total_bytes is not None
            and result.status != NodeBalanceStatus.UNSUPPORTED_DATA
        ]
        totals = [result.node_total_bytes for result in usable_results]
        group_total = _sum_bytes(totals) if totals and compatible else None
        missing = tuple(
            node_id
            for node_id in mapped_nodes
            if node_id not in node_results
            or node_results[node_id].status == NodeBalanceStatus.MISSING_DATA
        )
        coverage_values = [
            result.coverage_fraction
            for result in present_results
            if result.coverage_fraction is not None
        ]
        coverage = min(coverage_values) if coverage_values and not missing else None
        share = (
            float(group_total / job_total)
            if compatible and group_total is not None
            and job_total is not None
            and job_total > 0
            else None
        )
        groups.append(
            JobTopologyGroupBalance(
                topology_group_id=group_id,
                node_ids=mapped_nodes,
                missing_node_ids=missing,
                total_bytes=group_total,
                traffic_share=share,
                coverage_fraction=coverage,
                node_count=len(mapped_nodes),
                valid_node_count=len(usable_results),
            )
        )
    return tuple(groups)


def analyze_job_node_balance(
    node_results: Mapping[str, NodeNicBalanceResult],
    known_node_ids: Collection[str] | None = None,
    *,
    topology_groups: TopologyGroups | None = None,
) -> JobNodeBalanceResult:
    """Aggregate compatible node balances for one run and traffic direction.

    ``known_node_ids`` establishes the expected node population. Nodes absent
    from that population but present in ``node_results`` remain visible as
    unknown observations. ``topology_groups`` maps a topology-group ID to its
    node IDs; groups must not overlap. Missing nodes are never treated as zero.
    """

    if not isinstance(node_results, Mapping):
        raise ValueError("node_results must be a mapping keyed by node identity")
    if isinstance(known_node_ids, str):
        raise ValueError("known_node_ids must be a collection, not a string")

    topology_state, topology_reasons, normalized_groups, mapped_nodes, duplicates = (
        _validate_topology(topology_groups)
    )
    if topology_state is JobTopologyState.INVALID:
        normalized_groups = ()
        mapped_nodes = frozenset()

    supplied_known = [] if known_node_ids is None else list(known_node_ids)
    if any(not isinstance(node_id, str) or not node_id for node_id in supplied_known):
        raise ValueError("every known node identity must be a nonempty string")
    if len(set(supplied_known)) != len(supplied_known):
        raise ValueError("known_node_ids must not contain duplicates")

    for node_id, result in node_results.items():
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("every result key must be a nonempty node identity")
        if result.node_id != node_id:
            raise ValueError("every result key must match its embedded node identity")

    observed = tuple(sorted(node_results))
    known = tuple(sorted(set(supplied_known) | mapped_nodes | set(observed)))
    known_set = frozenset(known)
    missing_from_results = tuple(
        node_id for node_id in known if node_id not in node_results
    )
    missing_result_nodes = tuple(
        node_id
        for node_id in observed
        if node_results[node_id].status == NodeBalanceStatus.MISSING_DATA
    )
    missing = tuple(sorted(set(missing_from_results) | set(missing_result_nodes)))
    unknown = tuple(node_id for node_id in observed if node_id not in known_set)
    unavailable = tuple(
        node_id
        for node_id in observed
        if node_results[node_id].status == NodeBalanceStatus.MISSING_DATA
    )
    partial = tuple(
        node_id
        for node_id in observed
        if node_results[node_id].status == NodeBalanceStatus.PARTIAL_DATA
    )
    unsupported = tuple(
        node_id
        for node_id in observed
        if node_results[node_id].status == NodeBalanceStatus.UNSUPPORTED_DATA
    )

    reference = next(
        (
            result
            for result in node_results.values()
            if result.status != NodeBalanceStatus.MISSING_DATA
        ),
        None,
    )
    compatibility_reasons: list[str] = []
    compatible = True
    if reference is None:
        run_id = None
        direction = None
        source_counter = None
        common_window = None
        represented_window = None
        boundary_signature = None
    else:
        run_id = reference.run_id
        direction = reference.direction
        source_counter = reference.source_counter
        common_window = reference.common_window_ns
        represented_window = reference.represented_window_ns
        boundary_signature = reference.valid_interval_boundary_signature
        for result in node_results.values():
            if result.status == NodeBalanceStatus.MISSING_DATA:
                continue
            identity_or_window_incompatible = (
                result.run_id != run_id
                or result.direction != direction
                or result.source_counter != source_counter
                or result.common_window_ns != common_window
                or result.represented_window_ns != represented_window
            )
            boundary_signature_incompatible = (
                result.valid_interval_boundary_signature != boundary_signature
            )
            if identity_or_window_incompatible or boundary_signature_incompatible:
                compatible = False
                if identity_or_window_incompatible:
                    compatibility_reasons.append(
                        "All node results must share run, direction, counter, "
                        "and windows."
                    )
                if boundary_signature_incompatible:
                    compatibility_reasons.append(
                        "All node results must share exact valid interval "
                        "boundaries; their boundary signatures differ."
                    )
                compatibility_reasons.append(
                    "Job totals, shares, and topology sums are unsupported."
                )
                break

    if not compatible:
        run_id = None
        direction = None
        source_counter = None
        common_window = None
        represented_window = None
        boundary_signature = None

    usable_nodes = (
        tuple(
            result
            for result in node_results.values()
            if compatible
            and result.node_total_bytes is not None
            and result.status != NodeBalanceStatus.UNSUPPORTED_DATA
        )
        if node_results
        else ()
    )
    totals = [result.node_total_bytes for result in usable_nodes]
    job_total = _sum_bytes(totals) if totals else None

    group_for_node = {
        node_id: group_id
        for group_id, node_ids in normalized_groups
        for node_id in node_ids
    }
    per_node = tuple(
        JobNodeBalance(
            node_id=node_id,
            status=result.status,
            state_flags=result.state_flags,
            status_reasons=result.status_reasons,
            total_bytes=(
                result.node_total_bytes
                if compatible and result.node_total_bytes is not None
                else None
            ),
            traffic_share=(
                float(result.node_total_bytes / job_total)
                if compatible
                and result.node_total_bytes is not None
                and job_total is not None
                and job_total > 0
                else None
            ),
            known_nic_count=result.known_nic_count,
            observed_nic_count=result.observed_nic_count,
            active_nic_fraction=result.active_nic_fraction,
            coverage_fraction=result.coverage_fraction,
            topology_group_id=(
                group_for_node.get(node_id)
                if topology_state
                in {JobTopologyState.VALID, JobTopologyState.VALID_PARTIAL}
                else None
            ),
            source_result=result,
        )
        for node_id, result in sorted(node_results.items())
    )

    spatial_values = [
        result.node_total_bytes
        for result in usable_nodes
        if result.node_total_bytes is not None
    ]
    max_traffic = max(spatial_values, default=None)
    min_traffic = min(spatial_values, default=None)
    if not spatial_values:
        ratio_denominator = MaxMinRatioDenominator.MISSING
        max_min_ratio = None
    elif min_traffic == 0:
        ratio_denominator = MaxMinRatioDenominator.ZERO
        max_min_ratio = None
    else:
        ratio_denominator = MaxMinRatioDenominator.POSITIVE
        max_min_ratio = max_traffic / min_traffic

    shares = [
        result.traffic_share for result in per_node if result.traffic_share is not None
    ]
    effective_node_count = (
        1.0 / math.fsum(share**2 for share in shares)
        if job_total is not None and job_total > 0
        else None
    )
    hotspot_node_count = (
        max(1, math.ceil(HOTSPOT_TOP_FRACTION * len(shares))) if shares else None
    )
    hotspot_concentration = (
        math.fsum(sorted(shares, reverse=True)[:hotspot_node_count])
        if shares and hotspot_node_count is not None
        else None
    )

    coverage_values = [
        result.coverage_fraction
        for result in node_results.values()
        if compatible and result.coverage_fraction is not None
    ]
    coverage = (
        min(coverage_values)
        if coverage_values
        and not missing
        and len(coverage_values) == len(node_results)
        else None
    )
    covered_duration = (
        min(result.covered_duration_ns for result in node_results.values())
        if compatible and node_results
        else 0
    )
    requested_duration = (
        common_window[1] - common_window[0] if common_window is not None else 0
    )

    baseline_count = sum(
        result.baseline_relative_observation_count
        for result in node_results.values()
    )
    states: set[JobBalanceState] = set()
    if job_total == 0:
        states.add(JobBalanceState.ZERO_TRAFFIC)
    if missing:
        states.add(JobBalanceState.MISSING_NODE)
    if unknown:
        states.add(JobBalanceState.UNKNOWN_NODE)
    if unavailable:
        states.add(JobBalanceState.MISSING_OR_INVALID_NODE)
    if partial:
        states.add(JobBalanceState.PARTIAL_NODE)
    if coverage is not None and coverage < 1.0:
        states.add(JobBalanceState.PARTIAL_COVERAGE)
    if unsupported:
        states.add(JobBalanceState.UNSUPPORTED_NODE)
    if baseline_count:
        states.add(JobBalanceState.BASELINE_RELATIVE)

    reasons: list[str] = compatibility_reasons
    if JobBalanceState.ZERO_TRAFFIC in states:
        reasons.append(
            "Job total traffic is zero; shares, effective node count, "
            "and hotspot concentration are undefined."
        )
    if missing:
        reasons.append(
            "Expected nodes without valid node results are missing, not zero traffic."
        )
    if unknown:
        reasons.append("Observed nodes absent from the known inventory are unknown.")
    if unavailable:
        reasons.append("Some node results have missing or invalid source data.")
    if partial:
        reasons.append("Some node numerical results are partial.")
    if coverage is not None and coverage < 1.0:
        reasons.append("At least one node has incomplete observational coverage.")
    elif missing and coverage_values:
        reasons.append("Overall coverage is unknown because expected nodes are missing.")
    if unsupported:
        reasons.append("Some node results are unsupported and have no traffic total.")
    if baseline_count:
        reasons.append(
            "Baseline-relative totals cover observed increments only, "
            "not an absolute pre-window baseline."
        )

    if not compatible:
        status = JobBalanceStatus.UNSUPPORTED_DATA
    elif not usable_nodes:
        status = (
            JobBalanceStatus.UNSUPPORTED_DATA
            if unsupported
            else JobBalanceStatus.MISSING_DATA
        )
    elif job_total == 0:
        status = JobBalanceStatus.ZERO_TRAFFIC
    elif states - {JobBalanceState.ZERO_TRAFFIC}:
        status = JobBalanceStatus.PARTIAL_DATA
    else:
        status = JobBalanceStatus.OK

    unknown_topology = (
        tuple(sorted(set(known) - mapped_nodes))
        if topology_state
        in {JobTopologyState.VALID, JobTopologyState.VALID_PARTIAL}
        else ()
    )
    if topology_state is JobTopologyState.VALID and unknown_topology:
        topology_state = JobTopologyState.VALID_PARTIAL
        topology_reasons = (
            "Some nodes have no topology assignment and remain node-only.",
        )

    topology_supported = topology_state in {
        JobTopologyState.VALID,
        JobTopologyState.VALID_PARTIAL,
    }
    unsupported_dimensions = []
    if not topology_supported:
        unsupported_dimensions.append("topology")
    unsupported_dimensions.extend(("peer", "near_far"))

    topology_result_groups = (
        _topology_group_results(
            normalized_groups, node_results, job_total, compatible
        )
        if topology_supported
        else ()
    )

    return JobNodeBalanceResult(
        run_id=run_id,
        direction=direction,
        source_counter=source_counter,
        status=status,
        state_flags=frozenset(states),
        status_reasons=tuple(reasons),
        common_window_ns=common_window,
        represented_window_ns=represented_window,
        valid_interval_boundary_signature=boundary_signature,
        requested_duration_ns=requested_duration,
        covered_duration_ns=covered_duration,
        coverage_fraction=coverage,
        known_node_ids=known,
        observed_node_ids=observed,
        missing_node_ids=missing,
        unknown_observed_node_ids=unknown,
        unavailable_node_ids=unavailable,
        partial_node_ids=partial,
        unsupported_node_ids=unsupported,
        known_node_count=len(known),
        observed_node_count=len(observed),
        valid_node_count=len(usable_nodes),
        job_total_bytes=job_total,
        per_node=per_node,
        max_traffic_bytes=max_traffic,
        min_traffic_bytes=min_traffic,
        max_min_ratio=max_min_ratio,
        max_min_ratio_denominator=ratio_denominator,
        effective_node_count=effective_node_count,
        spatial_p50_bytes=_nearest_rank_quantile(spatial_values, 0.50),
        spatial_p99_bytes=_nearest_rank_quantile(spatial_values, 0.99),
        hotspot_concentration=hotspot_concentration,
        hotspot_node_count=hotspot_node_count,
        valid_interval_count=sum(
            result.valid_interval_count for result in node_results.values()
        ),
        invalid_interval_count=sum(
            result.invalid_interval_count for result in node_results.values()
        ),
        baseline_relative_observation_count=baseline_count,
        topology_state=topology_state,
        topology_status_reasons=topology_reasons,
        topology_supported=topology_supported,
        topology_groups=topology_result_groups,
        unknown_topology_node_ids=unknown_topology,
        duplicate_topology_node_ids=duplicates,
        peer_near_far_supported=False,
        peer_near_far_reason=PEER_NEAR_FAR_REQUIREMENT,
        unsupported_dimensions=tuple(unsupported_dimensions),
        traffic_share_formula=TRAFFIC_SHARE_FORMULA,
        max_min_ratio_formula=MAX_MIN_RATIO_FORMULA,
        effective_node_count_formula=EFFECTIVE_NODE_COUNT_FORMULA,
        spatial_quantile_interpolation=SPATIAL_QUANTILE_INTERPOLATION,
        hotspot_concentration_formula=HOTSPOT_CONCENTRATION_FORMULA,
        coverage_aggregation=COVERAGE_AGGREGATION,
        topology_share_formula=TOPOLOGY_SHARE_FORMULA,
    )
